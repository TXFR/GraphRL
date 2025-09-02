"""Base neural network modules for the TrajGNN model."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool as gmp
from torch_geometric.nn.models import GAT
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
        print("\nLocationEmbed forward pass:")
        print(f"Input x shape: {x.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Aux info shape: {aux_info.shape}")
        
        # 通过增强的GAT层
        x = self.conv1(x, edge_index, aux_info)
        print(f"After conv1 shape: {x.shape}")
        x = F.relu(x)
        x = self.conv2(x, edge_index, aux_info)
        print(f"After conv2 shape: {x.shape}")
        x = self.norm(x)
        
        # 全局池化和分类
        fusion = gmp(x, batch)
        print(f"After pooling shape: {fusion.shape}")
        cls = self.cls_linear(fusion)
        print(f"Output cls shape: {cls.shape}")
        
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

class TrajectoryReconstruction(nn.Module):
    """
    Trajectory reconstruction module for self-supervised learning.

    Reconstructs trajectory sequences from corrupted inputs.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """Encode and reconstruct trajectory"""
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

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
                 use_self_supervised: bool = True):
        super(Spatio_Tmp_Embed, self).__init__()

        self.edge_attr_dim = edge_attr_dim
        self.use_self_supervised = use_self_supervised
        
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

        # Self-supervised learning modules
        if self.use_self_supervised:
            self.contrastive_head = ContrastiveHead(
                embed_dim=hidden_channels,
                hidden_dim=hidden_channels,
                output_dim=128
            )
            self.reconstruction_head = TrajectoryReconstruction(
                input_dim=embed_dim,
                hidden_dim=hidden_channels,
                output_dim=embed_dim
            )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: (time_attr, aux_info, pos)
            batch: Batch indices [num_nodes]
        """
        print("\nSpatio_Tmp_Embed forward pass:")
        print(f"x shape: {x.shape}")
        print(f"edge_index shape: {edge_index.shape}")
        
        # 解包边属性
        time_attr, aux_info, _ = edge_attr
        print(f"time_attr shape: {time_attr.shape}")
        print(f"aux_info shape: {aux_info.shape}")
        
        # 编码边特征
        time_feat = self.time_proj(time_attr)  # [num_edges, edge_attr_dim//2]
        aux_feat = self.aux_proj(aux_info)     # [num_edges, edge_attr_dim//2]
        edge_features = torch.cat([time_feat, aux_feat], dim=-1)
        edge_features = self.edge_norm(edge_features)
        print(f"edge_features shape: {edge_features.shape}")
        
        # 通过GNN更新节点特征
        x = self.conv1(x, edge_index, edge_features)
        x = self.conv2(x, edge_index, edge_features)
        print(f"After GNN x shape: {x.shape}")
        
        # 获取节点对特征
        start_nodes = x[edge_index[0]]  # [num_edges, hidden_dim]
        end_nodes = x[edge_index[1]]    # [num_edges, hidden_dim]
        
        # 从辅助信息学习权重
        edge_weights = self.weight_net(aux_info)  # [num_edges, 1]
        print(f"edge_weights shape: {edge_weights.shape}")
        
        # 融合特征
        node_pair_features = torch.cat([start_nodes, end_nodes], dim=-1)  # [num_edges, hidden_dim*2]
        weighted_features = node_pair_features * edge_weights  # [num_edges, hidden_dim*2]
        
        # 全局池化
        fusion = gmp(weighted_features, batch[edge_index[0]])  # 使用源节点的batch信息
        print(f"After pooling shape: {fusion.shape}")
        
        # 最终输出
        fusion = self.fuse_linear(self.norm(fusion))
        print(f"Final output shape: {fusion.shape}")

        # Self-supervised outputs
        ssl_outputs = {}
        if self.use_self_supervised:
            # 对比学习表示
            contrastive_repr = self.contrastive_head(x)
            ssl_outputs['contrastive'] = contrastive_repr

            # 重建任务输出
            reconstruction = self.reconstruction_head(x)
            ssl_outputs['reconstruction'] = reconstruction
            ssl_outputs['original'] = x  # 用于计算重建损失

        return x, torch.sigmoid(fusion), ssl_outputs
