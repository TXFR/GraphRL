import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader,NeighborLoader
from torch_geometric.nn import GraphSAGE,SAGEConv
from torch_geometric.nn.models import rect,GAT
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import degree,to_undirected
import torch
from torch import nn
from sklearn.preprocessing import PowerTransformer

'''
This is a pre-training for constructing nodes embedding.
计算geo_cells之间的空间位置关系和语义特征（源于格网中的POI）
两个点之间有边则说明它们在空间上存在一定关系（有关联）
则它们在低维空间上的特征编码应该是相似的
'''

def data_scale(data):
    w, h = data.shape
    new_d = np.reshape(data, newshape=(-1, 1))
    ss = PowerTransformer(method="yeo-johnson")
    s_d = ss.fit_transform(new_d)
    new_d = np.reshape(s_d, newshape=(w, h))
    return new_d

def unsp_graphsage():
    cell_feature_flow = np.load("../data/adjMatrix/allpart_index_Flowfeatures.npy", allow_pickle=True).item()
    cell_feature_poi = np.load("../data/adjMatrix/allpart_index_POIfeatures.npy", allow_pickle=True).item()
    # 分别获取flow和poi特征
    cell_feature_flow = pd.DataFrame(cell_feature_flow).T.values
    cell_feature_poi = pd.DataFrame(cell_feature_poi).T.values
    flow_dim = cell_feature_flow.shape[1]
    poi_dim = cell_feature_poi.shape[1]
    # 拼接特征
    cell_feature = np.concatenate([cell_feature_flow, cell_feature_poi], axis=1)
    edge_weight = np.load("../data/adjMatrix/allpart_edge_weight.npy", allow_pickle=True)
    edges = edge_weight[0:2]
    # weights = edge_weight[-1]

    node_feature = torch.tensor(cell_feature, dtype=torch.float32)
    edge_index = torch.LongTensor(edges)
    # edge_weight = torch.tensor(weights, dtype=torch.float32)
    data = Data(x=node_feature, edge_index=edge_index)
    # b = to_undirected(edge_index)
    # a = degree(b[0]).numpy()

    train_loader = LinkNeighborLoader(
        data,
        batch_size=1000,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[5, 5],
    )

    class GraphSageModel_(torch.nn.Module):
        def __init__(self, num_feature, hidden_channels, num_layers, flow_dim=None, poi_dim=None):
            super(GraphSageModel_, self).__init__()
            self.flow_dim = flow_dim
            self.poi_dim = poi_dim
            
            # 第一个GraphSAGE只处理flow特征的30%
            self.graph1 = GraphSAGE(int(flow_dim*0.3),
                                   hidden_channels=hidden_channels,
                                   out_channels=int(flow_dim*0.3),
                                   num_layers=num_layers)

            self.graph2 = GraphSAGE(num_feature,
                                    hidden_channels=hidden_channels,
                                    out_channels=hidden_channels,
                                    num_layers=num_layers)
            self.input_embed = nn.Linear(num_feature, num_feature)
            self.mid_norm = nn.LayerNorm(num_feature,num_feature)
            self.out_linear = nn.Linear(hidden_channels, num_feature)
            # 掩码token只需要补充flow特征的缺失部分
            self.mask_token = nn.Parameter(torch.zeros(1, flow_dim-int(flow_dim*0.3)))
            self.init_weights()

        def creat_id_posEmbeddings(self, n_pos, dim):
            n_pos_vec = torch.arange(n_pos, dtype=torch.float)
            assert dim % 2 == 0, "wrong dimension"
            position_embedding = torch.zeros(len(n_pos_vec), dim, dtype=torch.float)
            omega = torch.arange(dim // 2., dtype=torch.float)
            omega /= dim / 2.
            omega = 1. / (10000 ** omega)
            out = torch.unsqueeze(n_pos_vec, -1) @ torch.unsqueeze(omega, -1).T
            emb_sin = torch.sin(out)
            emb_cos = torch.cos(out)
            # 将emb_sin赋值给偶数位，emb_cos赋值给奇数位
            position_embedding[:, 0::2] = emb_sin
            position_embedding[:, 1::2] = emb_cos

            return position_embedding

        def random_masking(self, x, mask_ratio):
            """
            Perform per-sample random masking by per-sample shuffling.
            Per-sample shuffling is done by argsort random noise.
            x: [N, L, D], sequence
            """
            N, L = x.shape  
            len_keep = int(L * (1 - mask_ratio)) 

            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

            # sort noise for each sample
            # 排序后返回下标，即index
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep)

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            return x_masked, mask, ids_restore

        def init_weights(self, mode=''):
            nn.init.trunc_normal_(self.mask_token, std=0.02)
            self.apply(self._init_weights)

        def _init_weights(self, m):
            # this fn left here for compat with downstream users
            # init_weights_vit_timm(m)
            """
                ViT weight initialization
                :param m: module
                """
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

        def forward_loss(self, y, pred, mask):

            loss = torch.abs(pred - y)
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            return loss

        def forward(self, x, edge_index):
            y = x
            x = self.input_embed(x)
            
            # 分离flow和poi特征
            flow_features = x[:, :self.flow_dim]
            poi_features = x[:, self.flow_dim:]
            
            # 只对flow特征进行掩码
            flow_masked, mask, ids_restore = self.random_masking(flow_features, 0.7)
            
            # 第一阶段只处理被掩码的flow特征
            flow_encoded = self.graph1(flow_masked, edge_index)
            
            # 补充掩码并重组特征
            mask_tokens = self.mask_token.repeat(flow_encoded.shape[0], 1)
            flow_full = torch.cat([flow_encoded, mask_tokens], dim=1)
            flow_restored = torch.gather(flow_full, dim=1, index=ids_restore)
            
            # 将处理后的flow特征与原始poi特征拼接
            x = torch.cat([flow_restored, poi_features], dim=1)
            
            # 继续后续处理
            x = self.graph2(self.mid_norm(x), edge_index)
            pred = self.out_linear(x)
            
            loss = self.forward_loss(y, pred, mask)
            return x, loss


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSageModel_(data.num_node_features, hidden_channels=128, num_layers=2, flow_dim=flow_dim, poi_dim=poi_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    # x, edge_index = data.x.to(device), data.edge_index.to(device)


    def train():
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            h,loss = model(batch.x,batch.edge_index)
            total_loss+=loss
            loss.backward()
            optimizer.step()
            # total_loss += float(loss) * pred.size(0)

        return total_loss/batch.num_nodes


    @torch.no_grad()
    def test():
        model.eval()
        out,l = model(data.x.to(device), data.edge_index.to(device))
        np.save("../result/SL_Model_feature.npy",out.cpu().numpy())
        # clf = LogisticRegression()
        # clf.fit(out[data.train_mask], data.y[data.train_mask])
        #
        # val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
        # test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])
        #
        # return val_acc, test_acc


    for epoch in range(1, 50):
        loss = train()
        scheduler.step()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test()
        # val_acc, test_acc = test()
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
        #       f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')


if __name__ == '__main__':
    unsp_graphsage()
