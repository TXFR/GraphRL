"""Main model architecture for trajectory embedding."""

import torch
from torch import nn
from .base_modules import (
    T2V, Mlp, Temporal_Embed, LocationEmbed, 
    Spatio_Tmp_Embed
)

class Build_Model(nn.Module):
    """
    Main model architecture for trajectory prediction.
    
    The model takes trajectory data sampled at 30-minute intervals (48 slots per day)
    and predicts the probability distribution of a user being at different types of
    locations (8 categories) for each hour of the day (24 hours).
    
    Key dimensions:
    - Input time features: 48 (30-minute slots per day)
    - Output time dimension: 24 (hours per day)
    - Location categories: 8 (types of places)
    - Final output shape: (batch_size, 24, 8) - probability for each hour and location type
    """
    
    def __init__(self, config):
        """
        Initialize the model with configuration parameters.
        
        Args:
            config: ModelConfig object containing all model parameters
        """
        super(Build_Model, self).__init__()
        
        # Store dimensions for later use
        self.time_slots = config.time_slots
        self.hours_per_day = config.hours_per_day
        self.location_categories = config.location_categories
        self.decoder_dim = config.decoder_dim
        
        # Initialize embedding modules
        self.time_embed = Temporal_Embed(
            config.time_in,
            config.hidden_features,
            config.out_features_t,
            config.t_cls,
            config.dropout_prob
        )
        
        self.location_embed = LocationEmbed(
            config.location_in,
            config.hidden_features,
            config.out_features_l,
            config.loc_cls_num
        )
        
        self.spatio_tmp_embed = Spatio_Tmp_Embed(
            config.out_features_l,
            config.hidden_features,
            config.st_fuse_num,
            config.t_l_embed
        )
        
        # Additional components
        self.tv = T2V(config.time_in, config.time_in)
        self.location_linear = nn.Linear(config.location_in, config.location_in)
        self.fuse_linear = nn.Linear(config.decoder_dim, config.decoder_dim)
        
        # Normalization layers
        self.norm = nn.LayerNorm(config.location_in)
        self.norm_decode = nn.LayerNorm(config.decoder_dim)

    def ts_decoder(self, x):
        """
        Trajectory sequence decoder.
        
        Args:
            x: Input tensor to decode [batch_size, seq_len, decoder_dim]
        Returns:
            Decoded tensor [batch_size, decoder_dim]
        """
        print(f"\nts_decoder input shape: {x.shape}")
        
        # 确保在正确的设备上
        device = x.device
        x = nn.Flatten()(x)
        
        # 创建MLP并移到正确的设备
        mlp1 = Mlp(self.decoder_dim, self.decoder_dim * 2, self.decoder_dim, 0.5).to(device)
        mlp2 = Mlp(self.decoder_dim, self.decoder_dim * 2, self.decoder_dim, 0.5).to(device)
        
        # 前向传播
        x = x + mlp1(x)
        x = x + mlp2(self.norm_decode(x))
        x = self.norm_decode(x)
        
        print(f"ts_decoder output shape: {x.shape}")
        return x

    def spatio_temp_transfer(self, t_cls, l_cls, t_l_cls):
        """
        Transfer between spatial and temporal domains.
        
        Args:
            t_cls: Time classification logits [batch_size, time_dim]
            l_cls: Location classification logits [batch_size, location_dim]
            t_l_cls: Time-location classification logits [batch_size, time_dim]
        Returns:
            Transferred and reshaped tensor [batch_size, hours_per_day, location_categories]
        """
        # 确保所有输入在同一设备上
        device = t_cls.device
        t_l_cls = t_l_cls.to(device)
        
        # 转换为概率分布
        t_l_cls = torch.unsqueeze(t_l_cls, dim=-1)
        t_cls_ = torch.sigmoid(t_cls)
        l_cls_ = torch.sigmoid(l_cls)[:, 1:]  # 跳过第一个类别
        
        # 创建时空概率矩阵
        t_s_matrix = torch.unsqueeze(t_cls_, dim=-1) @ torch.unsqueeze(l_cls_, dim=1)
        t_s_matrix = torch.cat([t_l_cls, t_s_matrix], dim=-1)
        
        # 解码和重塑
        x = self.ts_decoder(self.norm(t_s_matrix))
        x = self.fuse_linear(x)
        
        # 重塑为最终输出形状
        output = x.reshape(-1, self.hours_per_day, self.location_categories)
        
        return output

    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass through the model."""        
        # Unpack edge attributes
        edge_attr, aux_info, pos = edge_attr
        print(f"Edge attr shape: {edge_attr.shape}")
        print(f"Aux info shape: {aux_info.shape}")
        print(f"Position shape: {pos.shape}")
        
        # Time embedding
        x_t, t_cls = self.time_embed(edge_attr, aux_info, pos)
        # Location embedding
        x_transformed = self.location_linear(x)
        x_l, l_cls = self.location_embed(x_transformed, edge_index, aux_info, batch)

        x_t_l, t_l_cls = self.spatio_tmp_embed(
            x=x_l,
            edge_index=edge_index,
            edge_attr=(x_t, aux_info, pos),
            batch=batch
        )
        
        # Transfer and return
        t_l_cls = self.spatio_temp_transfer(t_cls, l_cls, t_l_cls)
        print("\nFinal output shapes:")
        print(f"t_cls shape: {t_cls.shape}")
        print(f"l_cls shape: {l_cls.shape}")
        print(f"t_l_cls shape: {t_l_cls.shape}")
        
        return t_cls, l_cls, t_l_cls