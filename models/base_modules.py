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

class DenseGNN(nn.Module):
    """
    Dense Graph Neural Network with enhanced residual connections.
    
    Features:
    1. Dense connections for feature reuse
    2. Pre-norm design for stable training
    3. Dual residual paths (skip + transform) to prevent over-smoothing
    4. Elastic aggregation to preserve local structure
    5. Gated residual learning
    """
    def __init__(self, in_channels: int, hidden_channels: int, 
                 edge_dim: int, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Layer components
        self.convs = nn.ModuleList()
        self.norms1 = nn.ModuleList()  # Pre-norm for GNN
        self.norms2 = nn.ModuleList()  # Pre-norm for FFN
        self.ffns = nn.ModuleList()    # Feed-forward networks
        self.skips = nn.ModuleList()   # Skip connections
        self.gates = nn.ModuleList()   # Residual gates
        
        # First layer
        self.convs.append(self._build_conv(hidden_channels, hidden_channels, edge_dim))
        self.norms1.append(nn.LayerNorm(hidden_channels))
        self.norms2.append(nn.LayerNorm(hidden_channels))
        self.ffns.append(self._build_ffn(hidden_channels))
        self.skips.append(nn.Identity())
        self.gates.append(self._build_gate(hidden_channels))
        
        # Subsequent layers
        for i in range(1, num_layers):
            layer_in_channels = hidden_channels * (i + 1)  # Dense connection
            self.convs.append(self._build_conv(layer_in_channels, hidden_channels, edge_dim))
            self.norms1.append(nn.LayerNorm(layer_in_channels))
            self.norms2.append(nn.LayerNorm(hidden_channels))
            self.ffns.append(self._build_ffn(hidden_channels))
            self.skips.append(nn.Linear(layer_in_channels, hidden_channels))
            self.gates.append(self._build_gate(hidden_channels))
    
    def _build_ffn(self, hidden_channels: int) -> nn.Sequential:
        """Builds a position-wise feed-forward network."""
        return nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.Dropout(self.dropout)
        )
    
    def _build_gate(self, hidden_channels: int) -> nn.Sequential:
        """Builds a gating mechanism for residual learning."""
        return nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid()
        )
    
    @staticmethod
    def _build_conv(in_channels: int, out_channels: int, edge_dim: int) -> GENConv:
        """Builds a GENConv layer with specified configuration."""
        return GENConv(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            aggr="softmax",  # Use softmax aggregation for local sensitivity
            learn_p=True,
            msg_norm=True,
            learn_msg_scale=True,
            norm="layer"
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the dense GNN.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Final node representations [num_nodes, hidden_channels]
        """
        # Initial projection
        x = self.input_proj(x)
        xs = [x]  # Store all layer outputs for dense connections
        
        for i in range(self.num_layers):
            # Prepare input with dense connections
            current_input = x if i == 0 else torch.cat(xs, dim=-1)
            
            # 1. GNN sublayer with residual
            residual = x
            current_input = self.norms1[i](current_input)
            conv_out = self.convs[i](current_input, edge_index, edge_attr)
            skip_out = self.skips[i](current_input)
            
            # Gated residual connection for GNN
            gate_input = torch.cat([conv_out, residual], dim=-1)
            gate = self.gates[i](gate_input)
            x = gate * conv_out + (1 - gate) * skip_out
            
            # 2. FFN sublayer with residual
            residual = x
            x = self.norms2[i](x)
            x = self.ffns[i](x) + residual
            
            # Store output for dense connections
            xs.append(x)
        
        return xs[-1]  # Return final layer output

class Spatio_Tmp_Embed(nn.Module):
    """
    Spatio-temporal embedding module with self-supervised capabilities.
    
    Architecture:
    1. Edge Feature Encoder: Combines temporal and auxiliary information
    2. Deep Dense GNN: Processes node features with dense connections
    3. Node Pair Fusion: Combines node pairs and their edge features
    4. Self-supervised Learning: Optional SSL tasks (contrastive, link prediction)
    
    Args:
        embed_dim: Node feature dimension (128)
        hidden_channels: Hidden layer dimension (128)
        decoder_dim: Joint classification dimension (192)
        out_features_t: Temporal feature dimension (64)
        ssl_mode: Self-supervised learning mode
    """
    def __init__(self, embed_dim: int, hidden_channels: int,
                 decoder_dim: int, out_features_t: int = 64, ssl_mode: str = "none", gnn_num_layers: int = 4):
        super().__init__()
        self.ssl_mode = ssl_mode
        self.use_ssl = ssl_mode != "none"
        
        # Edge Feature Encoder
        self.edge_encoder = nn.ModuleDict({
            'time_proj': nn.Linear(out_features_t, out_features_t // 2),
            'aux_proj': nn.Linear(2, out_features_t // 2),
            'norm': nn.LayerNorm(out_features_t)
        })
        
        # Deep Dense GNN
        self.gnn = DenseGNN(
            in_channels=embed_dim,
            hidden_channels=hidden_channels,
            edge_dim=out_features_t,
            num_layers=gnn_num_layers
        )
        
        # Node Pair Feature Fusion
        self.fusion = nn.ModuleDict({
            'weight_net': nn.Sequential(
                nn.Linear(2, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            ),
            'norm': nn.LayerNorm(hidden_channels * 2 + out_features_t),
            'proj': nn.Linear(hidden_channels * 2 + out_features_t, decoder_dim)
        })
        
        # Self-supervised Learning Heads
        if self.use_ssl:
            self.ssl_heads = nn.ModuleDict({
                'contrastive': ContrastiveHead(
                    embed_dim=hidden_channels,
                    hidden_dim=hidden_channels,
                    output_dim=128
                ),
                'link_pred': LinkPrediction(
                    node_dim=embed_dim,
                    hidden_dim=hidden_channels
                )
            })

    def _encode_edge_features(self, time_attr, aux_info):
        """Encode temporal and auxiliary edge information."""
        time_feat = self.edge_encoder['time_proj'](time_attr)
        aux_feat = self.edge_encoder['aux_proj'](aux_info)
        edge_features = torch.cat([time_feat, aux_feat], dim=-1)
        return self.edge_encoder['norm'](edge_features)
    
    def _fuse_node_pairs(self, node_features, edge_index, edge_features, aux_info, batch):
        """Fuse node pair features with edge information."""
        # Extract node pairs
        start_nodes = node_features[edge_index[0]]
        end_nodes = node_features[edge_index[1]]
        
        # Learn edge weights from auxiliary information
        edge_weights = self.fusion['weight_net'](aux_info)
        
        # Combine node pair and edge features
        node_pair_features = torch.cat([
            start_nodes,      # [num_edges, hidden_dim]
            end_nodes,       # [num_edges, hidden_dim]
            edge_features    # [num_edges, out_features_t]
        ], dim=-1)
        
        # Apply learned weights and pool
        weighted_features = node_pair_features * edge_weights
        pooled_features = gmp(weighted_features, batch[edge_index[0]])
        
        # Project to output space
        return torch.sigmoid(
            self.fusion['proj'](
                self.fusion['norm'](pooled_features)
            )
        )
    
    def _compute_ssl_outputs(self, node_features, edge_index, batch):
        """Compute self-supervised learning outputs if enabled."""
        if not self.use_ssl:
            return {}
            
        ssl_outputs = {}
        
        # Contrastive learning
        ssl_outputs['contrastive'] = self.ssl_heads['contrastive'](node_features)
        
        # Link prediction
        edge_pred, edge_labels = self.ssl_heads['link_pred'](
            node_features, edge_index, batch
        )
        ssl_outputs.update({
            'edge_pred': edge_pred,
            'edge_labels': edge_labels
        })
        
        return ssl_outputs
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass through the spatio-temporal embedding module.
        
        Process:
        1. Encode edge features (temporal + auxiliary)
        2. Update node representations via Dense GNN
        3. Fuse node pairs with edge information
        4. Compute SSL outputs if enabled
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: (time_attr, aux_info, pos)
            batch: Batch indices [num_nodes]
            
        Returns:
            node_features: Updated node features
            fused_features: Node pair fusion results
            ssl_outputs: Self-supervised learning outputs (if enabled)
        """
        # Unpack edge attributes
        time_attr, aux_info, _ = edge_attr
        
        # Process edge features
        edge_features = self._encode_edge_features(time_attr, aux_info)
        
        # Update node representations
        node_features = self.gnn(x, edge_index, edge_features)
        
        # Fuse node pairs
        fused_features = self._fuse_node_pairs(
            node_features, edge_index, edge_features, aux_info, batch
        )
        
        # Compute SSL outputs
        ssl_outputs = self._compute_ssl_outputs(node_features, edge_index, batch)
        
        return node_features, fused_features, ssl_outputs
