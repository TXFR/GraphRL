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

def link_prediction_loss(edge_pred, edge_labels):
    """
    Loss for link prediction task

    Args:
        edge_pred: Predicted probabilities for edge existence [num_samples, 1]
        edge_labels: Ground truth edge labels (1 for existing, 0 for non-existing) [num_samples, 1]
    """
    if len(edge_pred) > 0 and len(edge_labels) > 0:
        # Binary cross entropy loss for link prediction
        loss = torch.nn.functional.binary_cross_entropy(edge_pred, edge_labels)
        return loss
    else:
        return torch.tensor(0.0, device=edge_pred.device if len(edge_pred) > 0 else torch.device('cpu'))


def augment_trajectory_graph(node_features, edge_index, edge_attr, augmentation_type="noise"):
    """
    Apply data augmentation to trajectory graph.

    Args:
        node_features: Node features [num_nodes, feature_dim]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge attributes [num_edges, attr_dim]
        augmentation_type: Type of augmentation ("noise", "dropout", "shuffle")

    Returns:
        augmented_node_features, augmented_edge_index, augmented_edge_attr
    """
    if augmentation_type == "noise":
        # Add small noise to node features
        noise = torch.randn_like(node_features) * 0.1
        augmented_node_features = node_features + noise

        # Keep edge structure same
        augmented_edge_index = edge_index
        augmented_edge_attr = edge_attr

    elif augmentation_type == "dropout":
        # Randomly dropout some nodes
        keep_prob = 0.8
        keep_mask = torch.rand(node_features.size(0)) < keep_prob
        keep_indices = torch.where(keep_mask)[0]

        if len(keep_indices) > 0:
            augmented_node_features = node_features[keep_indices]

            # Filter edges that connect kept nodes
            edge_mask = torch.isin(edge_index[0], keep_indices) & torch.isin(edge_index[1], keep_indices)
            augmented_edge_index = edge_index[:, edge_mask]
            augmented_edge_attr = edge_attr[edge_mask]

            # Remap edge indices
            old_to_new = torch.full_like(keep_mask, -1, dtype=torch.long)
            old_to_new[keep_indices] = torch.arange(len(keep_indices), device=edge_index.device)
            augmented_edge_index = old_to_new[augmented_edge_index]
        else:
            # Keep at least one node
            augmented_node_features = node_features[:1]
            augmented_edge_index = torch.empty(2, 0, dtype=torch.long, device=edge_index.device)
            augmented_edge_attr = torch.empty(0, edge_attr.size(-1), device=edge_attr.device)

    elif augmentation_type == "shuffle":
        # Shuffle node order
        perm = torch.randperm(node_features.size(0))
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm), device=perm.device)

        augmented_node_features = node_features[perm]
        augmented_edge_index = inv_perm[edge_index]
        augmented_edge_attr = edge_attr

    else:
        # No augmentation
        augmented_node_features = node_features
        augmented_edge_index = edge_index
        augmented_edge_attr = edge_attr

    return augmented_node_features, augmented_edge_index, augmented_edge_attr

def trajectory_contrastive_loss(node_embeddings, batch_indices, temperature=0.07):
    """
    Pure self-supervised contrastive loss for trajectory graphs.

    Uses a label-free approach where we simply encourage:
    - Similar embeddings to be close (positive pairs)
    - Dissimilar embeddings to be far apart (negative pairs)

    Since we don't have explicit positive/negative pairs, we use:
    - Positive pairs: Different random augmentations of the same graph embedding
    - Negative pairs: All other graph embeddings

    Args:
        node_embeddings: Node-level embeddings [num_nodes, embed_dim]
        batch_indices: Node-level batch indices [num_nodes]
        temperature: Temperature parameter (not used in this simplified version)
    """
    from torch_scatter import scatter_mean

    # Aggregate node embeddings to graph-level representations
    graph_embeddings = scatter_mean(node_embeddings, batch_indices, dim=0)  # [num_graphs, embed_dim]

    num_graphs = graph_embeddings.size(0)

    if num_graphs < 2:
        return torch.tensor(0.0, device=graph_embeddings.device)

    # Create two augmented views of each graph embedding
    # This simulates having two different augmentations of the same input
    noise1 = torch.randn_like(graph_embeddings) * 0.1
    noise2 = torch.randn_like(graph_embeddings) * 0.1

    view1 = graph_embeddings + noise1  # [num_graphs, embed_dim]
    view2 = graph_embeddings + noise2  # [num_graphs, embed_dim]

    # Normalize both views
    view1 = torch.nn.functional.normalize(view1, dim=-1)
    view2 = torch.nn.functional.normalize(view2, dim=-1)

    # Compute similarities between corresponding views (positive pairs)
    pos_similarities = torch.sum(view1 * view2, dim=-1)  # [num_graphs]

    # Compute similarities between different graphs (negative pairs)
    # For each graph, compute similarity with all other graphs
    neg_similarities = []
    for i in range(num_graphs):
        # Similarities between view1 of graph i and view1 of all other graphs
        neg_sim_i = torch.matmul(view1[i:i+1], view1.t())[0]  # [num_graphs]
        neg_sim_i = neg_sim_i[torch.arange(num_graphs) != i]  # Remove self-similarity
        neg_similarities.append(neg_sim_i)

    neg_similarities = torch.stack(neg_similarities)  # [num_graphs, num_graphs-1]

    # Compute contrastive loss for each graph
    loss = 0
    for i in range(num_graphs):
        pos_sim = pos_similarities[i]  # scalar
        neg_sims = neg_similarities[i]  # [num_graphs-1]

        # InfoNCE-style loss: -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sims))))
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sims))
        loss_i = -torch.log(numerator / denominator)
        loss += loss_i

    # Average loss over all graphs
    loss = loss / num_graphs
    return loss



def cal_self_supervised_loss(ssl_outputs, supervised_loss=None, ssl_mode="combined",
                           ssl_weight=1.0, supervised_weight=0.1, batch_indices=None):
    """
    Enhanced self-supervised loss function

    Args:
        ssl_outputs: Dict containing SSL task outputs
        supervised_loss: Optional supervised loss for multi-task learning
        ssl_mode: SSL mode ("contrastive", "reconstruction", "combined")
        ssl_weight: Weight for SSL losses
        supervised_weight: Weight for supervised loss in multi-task
        batch_indices: Batch indices for proper contrastive learning

    Returns:
        total_loss: Combined total loss
        loss_components: Dict containing individual loss components
    """
    total_loss = 0
    loss_components = {}

    # Always compute available SSL losses when SSL is enabled
    if ssl_mode != "none":
        # Contrastive loss (trajectory-level) - always compute if available
        if 'contrastive' in ssl_outputs:
            if batch_indices is not None:
                cont_loss = trajectory_contrastive_loss(
                    ssl_outputs['contrastive'],  # Node embeddings
                    batch_indices
                )
            else:
                # Fallback to original contrastive loss if no batch_indices
                cont_loss = contrastive_loss(ssl_outputs['contrastive'])
            total_loss += ssl_weight * cont_loss
            loss_components['contrastive_loss'] = cont_loss.item()

        # Link prediction loss - always compute if available
        if 'edge_pred' in ssl_outputs:
            link_loss = link_prediction_loss(
                ssl_outputs['edge_pred'],
                ssl_outputs['edge_labels']
            )
            total_loss += ssl_weight * link_loss
            loss_components['link_prediction_loss'] = link_loss.item()

    # Multi-task: combine with supervised loss
    if supervised_loss is not None:
        supervised_component = supervised_weight * supervised_loss
        total_loss = total_loss + supervised_component
        loss_components['supervised_component'] = supervised_component.item()

    return total_loss, loss_components

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


