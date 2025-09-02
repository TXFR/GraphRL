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
    all_batch = []

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
            # 随机选择2个时间点（两个点之间有转移）设置为1
            active_slots = torch.randint(0, time_slots, (2,), device=device)
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
        # 创建batch索引：标识每个节点属于哪个图
        batch_idx = torch.full((num_nodes,), graph_idx, dtype=torch.long, device=device)
        all_batch.append(batch_idx)

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
        batch=torch.cat(all_batch)  # 正确的batch索引
    )
    
    return batch_data

def test_model(ssl_mode="none"):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Testing SSL mode: {ssl_mode}")

    # 创建配置
    config = ModelConfig()
    config.ssl_mode = ssl_mode

    # 创建模型
    use_ssl = ssl_mode != "none"
    model = Build_Model(config, use_self_supervised=use_ssl).to(device)

    # 创建一个批次的数据
    batch = create_batch_data(
        batch_size=5,
        num_nodes=4,
        time_slots=48,
        location_dim=8,
        device=device
    )

    # 设置模型为训练模式
    model.train()

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    try:
        
        # 分离边属性中的时间特征和辅助信息
        edge_time_attr = batch.edge_attr[:, :48]  # 前48维是时间特征
        edge_aux_info = batch.edge_attr[:, -2:]   # 最后2维是频率和距离
        
        # Forward pass - handle different SSL modes
        # Prepare original edge attributes for reconstruction task
        original_edge_attr = (edge_time_attr, edge_aux_info)

        model_outputs = model(
            batch.x,
            batch.edge_index,
            (edge_time_attr, edge_aux_info, batch.pos),  # 正确的边属性格式
            batch.batch,
            original_edge_attr
        )

        # Handle different return values based on SSL mode
        if ssl_mode != "none":
            t_cls, l_cls, t_l_cls, ssl_outputs = model_outputs
        else:
            t_cls, l_cls, t_l_cls = model_outputs
            ssl_outputs = None

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
            supervised_loss = loss_t + loss_l + loss_pred

            if ssl_mode != "none":
                from Utils.util import cal_self_supervised_loss
                ssl_loss, loss_components = cal_self_supervised_loss(
                    ssl_outputs,
                    supervised_loss if ssl_mode == "combined" else None,
                    ssl_mode=ssl_mode,
                    batch_indices=batch.batch
                )
                total_loss = ssl_loss

                # Print individual SSL loss components
                if 'contrastive_loss' in loss_components:
                    print(f"Contrastive Loss: {loss_components['contrastive_loss']:.4f}")
                if 'link_prediction_loss' in loss_components:
                    print(f"Link Prediction Loss: {loss_components['link_prediction_loss']:.4f}")
                print(f"SSL Total Loss: {ssl_loss.item():.4f}")
            else:
                total_loss = supervised_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            print(f"Total Loss: {total_loss.item():.4f}")
            print(f"Supervised Loss: {supervised_loss.item():.4f}")
            print(f"Accuracy: {accuracy:.4f}")

            return True

        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return False

def test_all_modes():
    """Test all SSL modes to ensure backward compatibility"""
    modes_to_test = ["none", "contrastive", "reconstruction", "combined"]

    print("Testing all SSL modes for backward compatibility...")

    for mode in modes_to_test:
        print(f"\n--- Testing {mode} mode ---")
        success = test_model(ssl_mode=mode)
        if not success:
            print(f"✗ {mode} mode test failed!")
            return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test TrajGNN model')
    parser.add_argument('--ssl', action='store_true',
                       help='Enable self-supervised learning (contrastive + link prediction)')
    parser.add_argument('--ssl-mode', type=str, choices=['none', 'contrastive', 'reconstruction', 'combined'],
                       default='none', help='Self-supervised learning mode to test (legacy)')
    parser.add_argument('--all-modes', action='store_true',
                       help='Test all SSL modes for backward compatibility')
    parser.add_argument('--explain', action='store_true',
                       help='Show detailed explanation of SSL outputs')

    args = parser.parse_args()

    # Determine SSL mode
    if args.ssl:
        ssl_mode = "combined"  # Enable both contrastive and link prediction
    else:
        ssl_mode = args.ssl_mode

    if args.all_modes:
        success = test_all_modes()
        if success:
            print("\n🎉 All SSL modes tested successfully! Backward compatibility maintained.")
        else:
            print("\n❌ Some SSL mode tests failed!")
    else:
        success = test_model(ssl_mode=ssl_mode)
        if success:
            if ssl_mode == "none":
                print("\n✓ Supervised learning test completed successfully!")            
            else:
                print(f"\n✓ SSL ({ssl_mode}) test completed successfully!")
        else:
            print(f"\n✗ {ssl_mode} mode test failed!")
