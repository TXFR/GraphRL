import torch
# from flocalLoss import FocalLoss
from random import sample
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import mean, sum
from torch import nn
from Utils.ResampleLoss import ResampleLoss

# focalloss = FocalLoss(class_num=6, alpha=None, gamma=2, size_average=True)
def index_select(x, index):
    row_ = torch.index_select(x[0], 0, index)
    col_ = torch.index_select(x[1], 0, index)
    return (row_,col_)

def location_index_select(y):
    device = y.device
    y_ = torch.split(y, 1, dim=0)
    label_y = []
    for i in y_:
        y_ = torch.zeros(8, device=device)
        location = torch.unique(torch.masked_select(i, i > 0).to(torch.int64))
        for j in location:
            y_[j] = 1.0
        label_y.append(y_)
    return torch.stack(label_y)


def DB_LOSS(x,y):
    device = x.device
    x = x.to(device)
    y = y.to(device)
    
    # 创建掩码
    mask = torch.zeros_like(y, dtype=torch.int64, device=device)
    value = torch.ones(1, dtype=torch.int64, device=device)
    index_ = torch.where(y != 0)
    mask.index_put_(index_, value)
    new_y = torch.clone(mask).to(torch.float).to(device)

    # 计算类别频率
    count_sum = torch.sum(mask, dim=0)
    train_num = new_y.shape[0]
    class_freq = (count_sum.cpu().numpy() + 1) / train_num
    
    # 创建损失函数并计算损失
    loss_func = ResampleLoss(
        reweight_func='rebalance', 
        loss_weight=1.0,
        focal=dict(focal=True, alpha=0.5, gamma=2),
        logit_reg=dict(init_bias=0.05, neg_scale=2.0),
        map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
        class_freq=class_freq, 
        train_num=train_num
    ).to(device)
    
    return loss_func(x,new_y)

def cal_location_loss(x,y):
    device = x.device
    x = x.to(device)
    y = y.to(device)
    y_ = location_index_select(y).to(torch.float).to(device)
    loss = DB_LOSS(x, y_)
    return mean(sum(loss, dim=-1))

def cal_time_loss(x,y):
    device = x.device
    x = x.to(device)
    y = y.to(device)
    loss = DB_LOSS(x, y)
    return mean(sum(loss, dim=-1))

def cal_traj_loss(x,y):
    device = x.device
    x = x.to(device)
    y = y.to(device)
    y_ = torch.nn.functional.one_hot(y.to(torch.int64), 8).to(device)
    loss = DB_LOSS(x, y_)
    return mean(sum(sum(loss, dim=-1), dim=-1))

def contrastive_loss(embeddings, temperature=0.5):
    """
    NT-Xent loss for contrastive learning
    """
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Create labels for positive pairs (diagonal)
    batch_size = embeddings.shape[0]
    labels = torch.arange(batch_size).to(embeddings.device)

    # Compute cross-entropy loss
    loss = torch.nn.functional.cross_entropy(
        similarity_matrix,
        labels,
        reduction='mean'
    )

    return loss

def reconstruction_loss(reconstructed, original):
    """
    MSE loss for trajectory reconstruction
    """
    return torch.nn.functional.mse_loss(reconstructed, original)

def trajectory_masking_loss(x, masked_x, mask_ratio=0.15):
    """
    Masked prediction loss for trajectory sequences
    """
    # Randomly mask some trajectory points
    batch_size, seq_len, feat_dim = x.shape
    num_mask = int(seq_len * mask_ratio)

    # Create random mask
    mask_indices = torch.rand(batch_size, seq_len).argsort(dim=-1)[:, :num_mask]
    mask = torch.zeros_like(x)
    mask.scatter_(1, mask_indices.unsqueeze(-1).expand(-1, -1, feat_dim), 1)

    # Apply mask
    masked_x = x * (1 - mask) + mask * masked_x

    # Predict masked positions
    loss = torch.nn.functional.mse_loss(masked_x, x, reduction='none')
    loss = loss.mean(dim=-1)  # Average over feature dimension
    loss = (loss * mask.squeeze(-1)).sum() / mask.sum()  # Only masked positions

    return loss

def cal_self_supervised_loss(ssl_outputs, supervised_loss=None, alpha=0.1):
    """
    Combined self-supervised loss

    Args:
        ssl_outputs: Dict containing contrastive, reconstruction outputs
        supervised_loss: Optional supervised loss for multi-task learning
        alpha: Weight for supervised loss when doing multi-task
    """
    total_loss = 0

    # Contrastive loss
    if 'contrastive' in ssl_outputs:
        contrastive_embeddings = ssl_outputs['contrastive']
        cont_loss = contrastive_loss(contrastive_embeddings)
        total_loss += cont_loss

    # Reconstruction loss
    if 'reconstruction' in ssl_outputs and 'original' in ssl_outputs:
        recon_loss = reconstruction_loss(
            ssl_outputs['reconstruction'],
            ssl_outputs['original']
        )
        total_loss += recon_loss

    # Multi-task: combine with supervised loss
    if supervised_loss is not None:
        total_loss = total_loss + alpha * supervised_loss

    return total_loss

def evaluate(logits, labels):
    # 确保输入在同一设备上
    device = logits.device
    logits = logits.to(device)
    labels = labels.to(device)

    # 计算预测结果
    predictions = torch.argmax(logits, dim=-1)

    # 转移到CPU进行评估计算
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    acc = np.zeros(len(predictions))
    prec = np.zeros(len(predictions))
    rec = np.zeros(len(predictions))
    f1s = np.zeros(len(predictions))

    for i in range(len(predictions)):
        # 计算准确率
        accuracy = accuracy_score(labels[i], predictions[i])
        acc[i] = accuracy
        # 计算精确率
        precision = precision_score(labels[i], predictions[i], average='micro')
        prec[i] = precision
        # 计算召回率
        recall = recall_score(labels[i], predictions[i], average='micro')
        rec[i] = recall
        # 计算F1-score
        f1 = f1_score(labels[i], predictions[i], average="micro")
        f1s[i] = f1

    a = acc.mean()
    b = prec.mean()
    c = rec.mean()
    d = f1s.mean()
    return a, b, c, d


