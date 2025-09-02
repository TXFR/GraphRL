from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # Time-related parameters
    time_slots: int = 48  # 一天的时间槽数（30分钟一个槽）
    hours_per_day: int = 24  # 输出预测的小时数
    
    # Location-related parameters
    location_categories: int = 8  # 位置类别数
    location_ini_embeds: int = 128  # 初始节点嵌入维度
    
    # Feature dimensions
    out_features_t: int = 64  # 时间特征输出维度（边特征）
    hidden_channels: int = 128  # GNN隐藏层维度
    out_features_l: int = 128  # 空间特征输出维度（节点特征）
    gnn_num_layers: int = 4  # 深层GNN的层数（可调）
    
    # Edge attribute dimensions
    edge_time_dim: int = 48  # 边的时间转移向量维度（一天的时间槽数）
    
    # Derived parameters (will be set in __post_init__)
    time_in: int = None  # 时间特征输入维度（基于time_slots）
    location_in: int = None  # 位置特征输入维度
    loc_cls_num: int = None  # 位置分类数
    t_cls: int = None  # 时间分类数（基于hours_per_day）
    decoder_dim: int = None  # 时空联合分类维度（hours_per_day * location_categories）

    # Training Parameters
    batch_size: int = 256
    num_epochs: int = 57
    learning_rate: float = 0.002
    dropout_prob: float = 0.5
    scheduler_min_lr: float = 0.01
    
    # Model Paths
    pretrain_path: Optional[str] = "../model_save/prob_all_part_04_06_t_l_ep24.pkl"
    save_dir: str = "../model_save_temp"
    save_prefix: str = "prob_all_part_04_06_t_l_ep"
    save_interval: int = 4
    
    # Data Paths
    data_root: str = "../result/graphData/"
    
    # Loss Parameters
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0
    logit_init_bias: float = 0.05
    logit_neg_scale: float = 2.0
    map_alpha: float = 0.1
    map_beta: float = 10.0
    map_gamma: float = 0.9

    # Self-supervised Learning Parameters
    ssl_mode: str = "none"  # "none", "contrastive", "reconstruction", "combined"
    ssl_temperature: float = 0.5
    ssl_mask_ratio: float = 0.15
    ssl_queue_size: int = 65536
    ssl_momentum: float = 0.999
    ssl_weight: float = 1.0
    supervised_weight: float = 0.1

    def __post_init__(self):
        """Initialize derived parameters and validate configuration"""
        # Set derived parameters based on actual meanings
        self.time_in = self.time_slots  # 输入是30分钟一个槽的时间特征
        self.location_in = self.location_ini_embeds  # 位置特征维度
        self.loc_cls_num = self.location_categories  # 位置分类数
        self.t_cls = self.hours_per_day  # 输出预测24小时的时间分布
        self.decoder_dim = self.hours_per_day * self.location_categories  # 时空联合分类维度
        
        # Validate time-related parameters
        assert self.time_slots > 0, "time_slots must be positive"
        assert self.hours_per_day > 0, "hours_per_day must be positive"
        assert self.time_slots >= self.hours_per_day, "time_slots must be >= hours_per_day"
        
        # Validate location-related parameters
        assert self.location_categories > 0, "location_categories must be positive"
        
        # Validate training parameters
        assert 0 <= self.dropout_prob <= 1, "dropout_prob must be between 0 and 1"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        
        # Validate feature dimensions
        assert self.decoder_dim > 0, "decoder_dim must be positive"
        assert self.hidden_channels > 0, "hidden_channels must be positive"
        assert self.gnn_num_layers > 0, "gnn_num_layers must be positive"
        
        # Additional semantic validations
        assert self.time_slots % 2 == 0, "time_slots should be even (for 30-min intervals)"
        assert self.hours_per_day == 24, "hours_per_day should be 24"
