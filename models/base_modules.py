"""Base neural network modules for the TrajGNN model."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool as gmp
from torch_geometric.nn import GENConv
from torch_scatter import scatter_mean, scatter_softmax

class T2V(nn.Module):
    """Time2Vec embedding module."""
    def __init__(self, in_features: int, out_features: int):
        super(T2V, self).__init__()
        self.out_features = out_features
        self.ori_layer = nn.Linear(in_features, 1, bias=True)
        self.act_layer = nn.Linear(in_features, out_features - 1, bias=True)
        self.act = torch.sin
        
    def forward(self, tau):
        x_ori = self.ori_layer(tau)
        x_sin = self.act(self.act_layer(tau))
        return torch.cat((x_ori, x_sin), dim=-1)

class Mlp(nn.Module):
    """Basic MLP module with dropout and activation."""
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, dropout_prob: float = 0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU(0.2)
        self.drop1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x

class Temporal_Embed(nn.Module):
    """
    Temporal embedding module.
    
    Processes time transition vectors between node pairs and applies learned weighting.
    Input:
        - x: Time transition vectors [batch_size, time_slots]
        - aux_info: Auxiliary information (freq, distance) [batch_size, 2]
        - pos: Batch position indices
    """
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, t_cls: int, dropout_prob: float = 0.):
        super(Temporal_Embed, self).__init__()
        self.init_linear = nn.Linear(in_features, out_features)
        self.mlp2 = Mlp(out_features, hidden_features, out_features, dropout_prob)
        self.mlp3 = Mlp(out_features, hidden_features, out_features, dropout_prob)
        self.norm = nn.LayerNorm(out_features)
        self.linear = nn.Linear(out_features, t_cls)
        self.drop = nn.Dropout(dropout_prob)
        
        # 深层权重学习网络
        self.weight_net = nn.Sequential(
            nn.Linear(out_features + 2, hidden_features),  # 结合时间特征和辅助信息
            nn.LayerNorm(hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x, aux_info, pos):
        """
        Args:
            x: 时间转移向量 [num_edges, time_slots]
            aux_info: 辅助信息 [num_edges, 2]
            pos: 位置信息 [num_edges]
        """
        # 处理时间转移向量
        x = self.init_linear(x)  # [num_edges, out_features]
        x = x + self.mlp2(self.norm(x))
        x = x + self.mlp3(self.norm(x))
        
        # 计算权重
        aux_expanded = aux_info.unsqueeze(1)  # [num_edges, 1, 2]
        x_reshaped = x.unsqueeze(-1)  # [num_edges, out_features, 1]
        
        # 将特征和辅助信息组合
        weight_input = torch.cat([
            x_reshaped.transpose(1, 2),  # [num_edges, 1, out_features]
            aux_expanded  # [num_edges, 1, 2]
        ], dim=-1)  # [num_edges, 1, out_features+2]
        
        # 生成权重
        weights = self.weight_net(weight_input)  # [num_edges, 1, 1]
        weights = weights.squeeze(-1).squeeze(-1)  # [num_edges]
        
        # 应用权重并聚合
        x_weighted = x * weights.unsqueeze(-1)  # [num_edges, out_features]
        x_weighted = gmp(x_weighted, pos)  # [batch_size, out_features]
        x_weighted = self.linear(self.norm(x_weighted))  # [batch_size, t_cls]
        
        return x, x_weighted

class EnhancedGATConv(nn.Module):
    """
    Enhanced GAT convolution layer that incorporates auxiliary information
    into the attention mechanism.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, dropout_prob: float = 0.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 特征转换
        self.linear = nn.Linear(in_channels, out_channels)
        
        # 注意力计算网络
        self.att_net = nn.Sequential(
            nn.Linear(out_channels * 2 + 2, hidden_channels),  # +2 for aux_info
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_channels, 1)
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x, edge_index, aux_info):
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            aux_info: Edge auxiliary info [num_edges, 2]
        """
        # 转换节点特征
        x = self.linear(x)  # [num_nodes, out_channels]
        
        # 获取源节点和目标节点的特征
        src, dst = edge_index
        src_feat = x[src]  # [num_edges, out_channels]
        dst_feat = x[dst]  # [num_edges, out_channels]
        
        # 计算注意力权重
        edge_feat = torch.cat([src_feat, dst_feat, aux_info], dim=-1)  # [num_edges, out_channels*2 + 2]
        att_logits = self.att_net(edge_feat)  # [num_edges, 1]
        
        # 对每个目标节点的边进行softmax
        att_weights = scatter_softmax(att_logits, dst, dim=0)  # [num_edges, 1]
        
        # Dropout
        att_weights = self.dropout(att_weights)
        
        # 加权聚合
        out = scatter_mean(src_feat * att_weights, dst, dim=0, dim_size=x.size(0))  # [num_nodes, out_channels]
        
        return out + x  # 残差连接

class LocationEmbed(nn.Module):
    """
    Location embedding module with enhanced attention mechanism.
    
    Uses auxiliary information (frequency and distance) to enhance the GAT's
    attention mechanism for better node feature aggregation.
    """
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, cls_num: int, dropout_prob: float = 0.):
        super(LocationEmbed, self).__init__()
        
        # 构建增强的GAT层
        self.conv1 = EnhancedGATConv(in_features, hidden_features, hidden_features, dropout_prob)
        self.conv2 = EnhancedGATConv(hidden_features, out_features, hidden_features, dropout_prob)
        
        # 输出层
        self.cls_linear = nn.Linear(out_features, cls_num)
        self.norm = nn.LayerNorm(out_features)
        
    def forward(self, x, edge_index, aux_info, batch):
        """
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            aux_info: Edge auxiliary information [num_edges, 2]
            batch: Batch indices [num_nodes]
        """
        # 通过增强的GAT层
        x = self.conv1(x, edge_index, aux_info)
        x = F.relu(x)
        x = self.conv2(x, edge_index, aux_info)
        x = self.norm(x)
        
        # 全局池化和分类
        fusion = gmp(x, batch)
        cls = self.cls_linear(fusion)
        
        return x, cls

class ContrastiveHead(nn.Module):
    """
    Contrastive learning head for trajectory representation learning.

    This module is designed for self-supervised learning by learning
    to distinguish between similar and dissimilar trajectory pairs.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """Project embeddings to contrastive space"""
        return self.projection(x)

class LinkPrediction(nn.Module):
    """
    Link prediction module for self-supervised learning on trajectory graphs.

    Performs link prediction task:
    1. Randomly remove some edges from the trajectory graph
    2. Predict whether edges exist between node pairs
    3. This teaches the model to understand trajectory structure and spatio-temporal relationships
    """
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()

        # Edge existence predictor
        self.edge_predictor = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),  # Concatenate source and target node features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Probability of edge existence
        )

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes] (node-level batch indices)
        Returns:
            edge_pred: Predicted probabilities for edge existence
            edge_labels: Ground truth edge labels (1 for existing, 0 for negative samples)
        """
        device = x.device
        num_edges = edge_index.size(1)

        # Positive edges prediction
        src_nodes = x[edge_index[0]]  # [num_edges, node_dim]
        dst_nodes = x[edge_index[1]]  # [num_edges, node_dim]
        pos_edge_features = torch.cat([src_nodes, dst_nodes], dim=-1)  # [num_edges, node_dim * 2]
        pos_edge_pred = self.edge_predictor(pos_edge_features)  # [num_edges, 1]

        # Generate negative samples
        negative_edges_list = []

        # Strategy 1: Sample negative edges within the same graph (trajectory)
        for i in range(num_edges):
            src_node = edge_index[0, i].item()
            src_batch = batch[src_node].item()

            # Find all nodes in the same graph
            same_graph_mask = batch == src_batch
            same_graph_nodes = torch.where(same_graph_mask)[0]

            # Remove source node and existing neighbors
            available_nodes = same_graph_nodes[same_graph_nodes != src_node]
            existing_neighbors = edge_index[1, edge_index[0] == src_node]
            available_nodes = available_nodes[~torch.isin(available_nodes, existing_neighbors)]

            if len(available_nodes) > 0:
                # Sample one negative edge within the same graph
                dst_node = available_nodes[torch.randint(0, len(available_nodes), (1,))].item()
                negative_edges_list.append([src_node, dst_node])

        # Strategy 2: Sample some negative edges across different graphs (optional)
        # This helps learn that edges shouldn't exist between different trajectories
        if len(negative_edges_list) < num_edges // 2:  # If we don't have enough intra-graph negatives
            unique_batches = torch.unique(batch)
            if len(unique_batches) > 1:
                for _ in range(min(num_edges // 4, 10)):  # Sample a few cross-graph negatives
                    # Randomly select two different graphs
                    batch1, batch2 = unique_batches[torch.randperm(len(unique_batches))[:2]]

                    # Sample one node from each graph
                    nodes1 = torch.where(batch == batch1)[0]
                    nodes2 = torch.where(batch == batch2)[0]

                    if len(nodes1) > 0 and len(nodes2) > 0:
                        src_node = nodes1[torch.randint(0, len(nodes1), (1,))].item()
                        dst_node = nodes2[torch.randint(0, len(nodes2), (1,))].item()
                        negative_edges_list.append([src_node, dst_node])

        # Convert negative edges to tensor
        if negative_edges_list:
            negative_edges = torch.tensor(negative_edges_list, device=device).t()  # [2, num_negative]

            # Get features for negative edges
            neg_src_nodes = x[negative_edges[0]]  # [num_negative, node_dim]
            neg_dst_nodes = x[negative_edges[1]]  # [num_negative, node_dim]
            neg_edge_features = torch.cat([neg_src_nodes, neg_dst_nodes], dim=-1)  # [num_negative, node_dim * 2]

            # Predict negative edge probabilities
            neg_edge_pred = self.edge_predictor(neg_edge_features)  # [num_negative, 1]

            # Combine positive and negative predictions
            all_edge_pred = torch.cat([pos_edge_pred, neg_edge_pred], dim=0)
            all_edge_labels = torch.cat([
                torch.ones(pos_edge_pred.size(0), 1, device=device),  # Positive labels
                torch.zeros(neg_edge_pred.size(0), 1, device=device)   # Negative labels
            ], dim=0)

            return all_edge_pred, all_edge_labels
        else:
            # Fallback: return only positive predictions
            return pos_edge_pred, torch.ones_like(pos_edge_pred, device=device)

class Spatio_Tmp_Embed(nn.Module):
    """
    Spatio-temporal embedding module with self-supervised capabilities.

    融合空间和时间特征的图神经网络模块。处理流程：
    1. 将时间特征和辅助信息（频率、距离）编码为边特征
    2. 通过GNN层更新节点的时空特征
    3. 使用注意力机制融合节点对的特征
    4. 支持对比学习和重建任务

    Input:
        - x: 已经过LocationEmbed更新的节点特征
        - edge_index: 图的连接关系
        - edge_attr: (time_attr, aux_info) 包含：
            * time_attr: 经过Temporal_Embed更新的时间特征 [batch_size, time_slots]
            * aux_info: 辅助信息 (freq, distance) [batch_size, 2]
        - batch: 批处理索引
    """
    def __init__(self, embed_dim: int, hidden_channels: int,
                 fuse_num: int, t_l_embed: int, edge_attr_dim: int = 64,
                 ssl_mode: str = "none"):
        super(Spatio_Tmp_Embed, self).__init__()

        self.edge_attr_dim = edge_attr_dim
        self.ssl_mode = ssl_mode
        self.use_ssl = ssl_mode != "none"
        
        # 边特征编码层
        self.time_proj = nn.Linear(64, edge_attr_dim // 2)  # 从time_embed输出的64维特征
        self.aux_proj = nn.Linear(2, edge_attr_dim // 2)    # 辅助信息投影
        self.edge_norm = nn.LayerNorm(edge_attr_dim)
        
        # GEN convolution layers for feature fusion
        self.conv1 = GENConv(
            in_channels=embed_dim, 
            out_channels=hidden_channels,
            edge_dim=edge_attr_dim,
            aggr="powermean",
            learn_p=True,
            msg_norm=True,
            learn_msg_scale=True,
            norm="layer"
        )
        
        self.conv2 = GENConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_attr_dim,
            aggr="powermean",
            learn_p=True,
            msg_norm=True,
            learn_msg_scale=True,
            norm="layer"
        )
        
        # 从辅助信息学习权重的网络
        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # 确保权重在0-1之间
        )
        
        # 输出层
        self.norm = nn.LayerNorm(hidden_channels * 2)  # 两个节点的特征拼接
        self.fuse_linear = nn.Linear(hidden_channels * 2, t_l_embed)

                    # Self-supervised learning modules - initialize all if SSL is enabled
        if self.ssl_mode != "none":
            self.contrastive_head = ContrastiveHead(
                embed_dim=hidden_channels,
                hidden_dim=hidden_channels,
                output_dim=128
            )
            self.link_prediction_head = LinkPrediction(
                node_dim=embed_dim,
                hidden_dim=hidden_channels
            )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: (time_attr, aux_info, pos)
            batch: Batch indices [num_nodes]
        """
      
        # 解析边属性
        time_attr, aux_info, _ = edge_attr
        
        # 编码边特征
        time_feat = self.time_proj(time_attr)  # [num_edges, edge_attr_dim//2]
        aux_feat = self.aux_proj(aux_info)     # [num_edges, edge_attr_dim//2]
        edge_features = torch.cat([time_feat, aux_feat], dim=-1)
        edge_features = self.edge_norm(edge_features)
        
        # 通过GNN更新节点特征
        x = self.conv1(x, edge_index, edge_features)
        x = self.conv2(x, edge_index, edge_features)
        
        # 获取节点对特征
        start_nodes = x[edge_index[0]]  # [num_edges, hidden_dim]
        end_nodes = x[edge_index[1]]    # [num_edges, hidden_dim]
        
        # 从辅助信息学习权重
        edge_weights = self.weight_net(aux_info)  # [num_edges, 1]
        
        # 融合特征
        node_pair_features = torch.cat([start_nodes, end_nodes], dim=-1)  # [num_edges, hidden_dim*2]
        weighted_features = node_pair_features * edge_weights  # [num_edges, hidden_dim*2]
        
        # 全局池化
        fusion = gmp(weighted_features, batch[edge_index[0]])  # 使用源节点的batch信息
        
        # 最终输出
        fusion = self.fuse_linear(self.norm(fusion))

        # Self-supervised outputs - generate all SSL outputs if enabled
        ssl_outputs = {}
        if self.ssl_mode != "none":
            # Contrastive learning representation
            contrastive_repr = self.contrastive_head(x)
            ssl_outputs['contrastive'] = contrastive_repr

            # Link prediction task
            edge_pred, edge_labels = self.link_prediction_head(x, edge_index, batch)
            ssl_outputs['edge_pred'] = edge_pred
            ssl_outputs['edge_labels'] = edge_labels

        return x, torch.sigmoid(fusion), ssl_outputs
