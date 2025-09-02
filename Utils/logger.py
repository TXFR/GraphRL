import logging
import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="../logs"):
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('TrajGNN')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f"{log_dir}/train_{timestamp}.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
    def log_metrics(self, epoch: int, p_loss: float, t_loss: float, n_loss: float, 
                   accuracy: float = None, precision: float = None, 
                   recall: float = None, f1: float = None):
        """Log training metrics"""
        metrics_msg = f"Epoch: {epoch:03d}, Pred Loss: {p_loss:.4f}, Time Loss: {t_loss:.4f}, Location Loss: {n_loss:.4f}"
        if accuracy is not None:
            metrics_msg += f", Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        self.logger.info(metrics_msg)
        
    def log_info(self, msg: str):
        """Log general information"""
        self.logger.info(msg)
        
    def log_error(self, msg: str):
        """Log error information"""
        self.logger.error(msg)
