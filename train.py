import os
import torch
from torch.utils.data import DataLoader
import math
import torch.optim.lr_scheduler as lr_scheduler

from models.Traj_Embed import Build_Model
from DataProcess.dataMaker import MyOwnDataset
from Utils.util import (
    cal_traj_loss, cal_location_loss, cal_time_loss, evaluate,
    cal_self_supervised_loss
)
from Utils.logger import TrainingLogger
from config.model_config import ModelConfig

class TrainingPipeline:
    def __init__(self, config: ModelConfig, use_self_supervised: bool = False):
        self.config = config
        self.use_self_supervised = use_self_supervised
        self.ssl_mode = config.ssl_mode  # Use config.ssl_mode directly, it's already set appropriately
        self.logger = TrainingLogger()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialize model, dataset, and dataloaders
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
    def setup_data(self):
        """Setup dataset and dataloaders"""
        self.dataset = MyOwnDataset(root=self.config.data_root)
        self.train_loader = DataLoader(
            self.dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
    def setup_model(self):
        """Initialize model and load pretrained weights if available"""
        self.model = Build_Model(
            config=self.config,
            use_self_supervised=self.use_self_supervised
        ).to(self.device)
        
        if self.config.pretrain_path and os.path.exists(self.config.pretrain_path):
            self.model.load_state_dict(
                torch.load(self.config.pretrain_path, map_location=self.device)
            )
            self.logger.log_info(f"Loaded pretrained model from {self.config.pretrain_path}")
            
    def setup_training(self):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Cosine annealing scheduler
        lf = lambda x: ((1 + math.cos(x * math.pi / self.config.num_epochs)) / 2) * \
                      (1 - self.config.scheduler_min_lr) + self.config.scheduler_min_lr
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        
    def process_batch(self, data):
        """
        Process a single batch of data

        Edge attributes structure:
        - first_traj_edge_attr: 时间转移向量 [batch_size, time_slots]
        - freqs: 频率 [batch_size, 1]
        - edge_distances: 距离 [batch_size, 1]
        """
        x, edge_index, edge_attr, y, pos, batch, ptr = data.to(self.device)

        # Process edge attributes
        edge_attr_processed = edge_attr[1].to(torch.float32)
        time_attr = edge_attr_processed[:, :self.config.edge_time_dim]  # 时间转移向量
        freq = edge_attr_processed[:, -2].unsqueeze(-1)  # 频率
        distance = edge_attr_processed[:, -1].unsqueeze(-1)  # 距离

        # 组合辅助特征
        aux_attr = torch.cat([freq, distance], dim=-1)

        x_ = x[1].to(torch.float32)
        pos_batch = self.edge_batch(pos[1])

        # 返回处理后的边属性和原始边属性
        edge_attr = (time_attr, aux_attr, pos_batch)
        # Store original edge attributes for reconstruction task
        original_edge_attr = (edge_attr_processed[:, :self.config.edge_time_dim], aux_attr)

        return x_, edge_index[1], edge_attr, batch[1], y[1], original_edge_attr
        
    @staticmethod
    def edge_batch(pos):
        """Process edge batch information"""
        unique_pos, inverse_indices = torch.unique_consecutive(pos, return_inverse=True)
        return inverse_indices
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_p_loss = total_t_loss = total_n_loss = 0
        num_steps = 0
        
        for data in self.train_loader:
            self.optimizer.zero_grad()
            
            # Process batch
            x_, edge_index, edge_attr, batch, y, original_edge_attr = self.process_batch(data)

            # Forward pass
            model_outputs = self.model(x_, edge_index, edge_attr, batch, original_edge_attr)

            # Handle different return values based on SSL mode
            if self.ssl_mode != "none":
                time_l, location_l, t_l_cls, ssl_outputs = model_outputs
            else:
                time_l, location_l, t_l_cls = model_outputs
                ssl_outputs = None

            # Calculate supervised losses
            loss_pred = cal_traj_loss(t_l_cls, y)
            loss_l = cal_location_loss(location_l, y.to(torch.float))
            loss_t = cal_time_loss(time_l, y.to(torch.float))
            supervised_loss = loss_t + loss_l + loss_pred

            # Calculate self-supervised loss if enabled
            if self.ssl_mode != "none":
                ssl_loss, loss_components = cal_self_supervised_loss(
                    ssl_outputs,
                    supervised_loss if self.ssl_mode == "combined" else None,
                    ssl_mode=self.ssl_mode,
                    ssl_weight=self.config.ssl_weight,
                    supervised_weight=self.config.supervised_weight,
                    batch_indices=batch
                )
                # Use SSL loss for backward pass
                loss = ssl_loss
            else:
                loss = supervised_loss

            # Evaluate metrics
            accuracy, precision, recall, f1 = evaluate(t_l_cls, y)

            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_p_loss += float(loss_pred)
            total_t_loss += float(loss_t)
            total_n_loss += float(loss_l)
            num_steps += 1
            
        return (total_p_loss / num_steps, total_t_loss / num_steps, 
                total_n_loss / num_steps, accuracy, precision, recall, f1)
                
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        if epoch % self.config.save_interval == 0:
            save_path = os.path.join(
                self.config.save_dir, 
                f"{self.config.save_prefix}{epoch}.pkl"
            )
            os.makedirs(self.config.save_dir, exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            self.logger.log_info(f"Saved checkpoint to {save_path}")
            
    def train(self):
        """Main training loop"""
        self.logger.log_info("Starting training...")
        self.logger.log_info(f"Training on device: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            # Train one epoch
            metrics = self.train_epoch()
            p_loss, t_loss, n_loss, accuracy, precision, recall, f1 = metrics
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.logger.log_metrics(
                epoch, p_loss, t_loss, n_loss,
                accuracy, precision, recall, f1
            )
            
            # Save checkpoint
            self.save_checkpoint(epoch)
            
        self.logger.log_info("Training completed!")

def main(use_self_supervised=False, ssl_mode="none"):
    # Initialize config
    config = ModelConfig()

    # Set SSL mode based on parameters
    if use_self_supervised and ssl_mode != "none":
        config.ssl_mode = ssl_mode
    elif use_self_supervised:
        config.ssl_mode = "combined"  # Default to combined mode if SSL enabled but no specific mode
    else:
        config.ssl_mode = "none"

    # Create and run training pipeline
    pipeline = TrainingPipeline(config, use_self_supervised=use_self_supervised)
    pipeline.train()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train TrajGNN model')
    parser.add_argument('--supervised', action='store_true',
                       help='Use supervised learning only (no self-supervised)')
    parser.add_argument('--ssl', action='store_true',
                       help='Enable self-supervised learning (contrastive + link prediction)')
    parser.add_argument('--ssl-mode', type=str, choices=['none', 'contrastive', 'reconstruction', 'combined'],
                       default='combined', help='Self-supervised learning mode (legacy)')
    parser.add_argument('--ssl-weight', type=float, default=1.0,
                       help='Weight for self-supervised losses')
    parser.add_argument('--supervised-weight', type=float, default=0.1,
                       help='Weight for supervised loss in multi-task learning')

    args = parser.parse_args()

    # Initialize config first
    config = ModelConfig()

    # Determine SSL usage and mode
    if args.ssl:
        # New simplified SSL interface
        use_ssl = True
        ssl_mode = "combined"
    elif args.supervised:
        # Explicit supervised only
        use_ssl = False
        ssl_mode = "none"
    else:
        # Legacy mode: use ssl-mode parameter
        use_ssl = args.ssl_mode != "none"
        ssl_mode = args.ssl_mode

    # Update config with command line arguments
    config.ssl_mode = ssl_mode
    config.ssl_weight = args.ssl_weight
    config.supervised_weight = args.supervised_weight

    main(use_self_supervised=use_ssl, ssl_mode=ssl_mode)


