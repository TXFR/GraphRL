import torch
import numpy as np
from torch_geometric.data import Data, Batch
from models.Traj_Embed import Build_Model
from config.model_config import ModelConfig
from Utils.util import cal_traj_loss, cal_location_loss, cal_time_loss, evaluate

def create_batch_data(batch_size=5, num_nodes=4, time_slots=48, location_dim=8, device=None):
    """
    创建一个批次的轨迹图数据
    
    Args:
        batch_size: 批次中的图数量
        num_nodes: 每个图中的节点数量
        time_slots: 时间槽数量（30分钟一个槽）
        location_dim: 位置类别数量
        device: 数据所在设备
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_node_features = []
    all_edge_indices = []
    all_edge_attrs = []
    all_labels = []
    all_pos = []
    
    total_nodes = 0
    
    for graph_idx in range(batch_size):
        # 1. 创建节点特征
        node_features = torch.randn(num_nodes, location_dim, device=device)
        all_node_features.append(node_features)
        
        # 2. 创建边的连接关系 (假设是一个完全图)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # 加上节点偏移量
                    edge_index.append([i + total_nodes, j + total_nodes])
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
        all_edge_indices.append(edge_index)
        
        num_edges = edge_index.size(1)
        
        # 3. 创建边属性
        # 3.1 时间特征：每条边的时间转移向量 [num_edges, time_slots]
        edge_time_attr = torch.zeros(num_edges, time_slots, device=device)
        for i in range(num_edges):
            # 随机选择几个时间点设置为1
            active_slots = torch.randint(0, time_slots, (3,), device=device)
            edge_time_attr[i, active_slots] = 1.0
        
        # 3.2 创建频率和距离
        frequencies = torch.rand(num_edges, 1, device=device)
        distances = torch.rand(num_edges, 1, device=device)
        
        # 3.3 组合边属性
        edge_attr = torch.cat([
            edge_time_attr,  # [num_edges, time_slots]
            frequencies,     # [num_edges, 1]
            distances       # [num_edges, 1]
        ], dim=1)
        all_edge_attrs.append(edge_attr)
        
        # 4. 创建标签：24小时内每个时刻的位置类别 [24]
        label = torch.zeros(24, device=device)
        # 随机选择几个时刻有活动
        active_times = torch.randint(0, 24, (5,), device=device)  # 假设每天有5个活动时刻
        for t in active_times:
            label[t] = torch.randint(1, location_dim, (1,), device=device)  # 1-7表示位置，0表示无活动
        all_labels.append(label)
        
        # 5. 创建position信息
        pos = torch.full((num_edges,), graph_idx, dtype=torch.long, device=device)
        all_pos.append(pos)
        
        total_nodes += num_nodes
    
    # 合并所有图的数据
    batch_data = Batch(
        x=torch.cat(all_node_features),
        edge_index=torch.cat(all_edge_indices, dim=1),
        edge_attr=torch.cat(all_edge_attrs),
        y=torch.stack(all_labels),
        pos=torch.cat(all_pos),
        batch=torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            repeats=torch.tensor([num_nodes] * batch_size, device=device)
        )
    )
    
    return batch_data

def test_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建配置
    config = ModelConfig()
    
    # 创建模型
    model = Build_Model(config).to(device)
    print("Model created successfully")
    
    # 创建一个批次的数据
    print("\nCreating batch data...")
    batch = create_batch_data(
        batch_size=5,
        num_nodes=4,
        time_slots=48,
        location_dim=8,
        device=device
    )
    
    print("\nBatch information:")
    print(f"- Number of graphs: {batch.num_graphs}")
    print(f"- Total nodes: {batch.num_nodes}")
    print(f"- Total edges: {batch.edge_index.size(1)}")
    
    print("\nInput shapes:")
    print(f"- Node features: {batch.x.shape}")
    print(f"- Edge index: {batch.edge_index.shape}")
    print(f"- Edge attr: {batch.edge_attr.shape}")
    print(f"- Labels: {batch.y.shape}")
    print(f"- Batch: {batch.batch.shape}")
    print(f"- Position: {batch.pos.shape}")
    
    # 设置模型为训练模式
    model.train()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        # 前向传播
        print("\nAttempting forward pass...")
        
        # 分离边属性中的时间特征和辅助信息
        edge_time_attr = batch.edge_attr[:, :48]  # 前48维是时间特征
        edge_aux_info = batch.edge_attr[:, -2:]   # 最后2维是频率和距离
        
        t_cls, l_cls, t_l_cls = model(
            batch.x,
            batch.edge_index,
            (edge_time_attr, edge_aux_info, batch.pos),  # 正确的边属性格式
            batch.batch
        )
        
        # 计算损失
        device = t_l_cls.device
        batch_y = batch.y.to(device)
        
        try:
            # 使用与训练相同的损失函数
            loss_pred = cal_traj_loss(t_l_cls, batch_y)
            loss_l = cal_location_loss(l_cls, batch_y.to(torch.float))
            loss_t = cal_time_loss(t_cls, batch_y.to(torch.float))
            
            # 计算评估指标
            accuracy, precision, recall, f1 = evaluate(t_l_cls, batch.y)
            
            # 总损失
            loss = loss_t + loss_l + loss_pred
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            print("\nTraining metrics:")
            print(f"Total Loss: {loss.item():.4f}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            return False
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\nModel test completed successfully!")
    else:
        print("\nModel test failed!")
