import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
import pandas as pd
from collections import Counter
from torch_geometric.utils import degree


class TrajDistri_Process_withEdge():
    def __init__(self,traj,cell_index,path,cell_feature,distance):
        self.path = path
        self.traj = traj
        self.cell_index = cell_index
        self.cell_feature = cell_feature
        self.distance = distance
        self.id_groups = traj.groupby("id")
        self.save_feature = dict()
        self.save_label = dict()
        self.save_edge = dict()
        self.save_edge_attr = dict()

        self.build_graphs()
        # first traj of mul_trajs

    def cal_distribution(self, data):
        arr = np.zeros((24, 8))
        for key in data:
            freq = Counter(data[key])
            if len(freq) == 0:
                arr[key][0] = 1
            else:
                total = sum(freq.values())
                prob = {k: v / total for k, v in freq.items()}
                keys = np.array(list(prob.keys()))
                values = np.array(list(prob.values()))
                arr[key][keys] = values
        return arr

    def mul_trajs_firts(self, id_trace):
        # self.fcls = np.zeros((24,))

        self.first_traj_edge_attr = []
        self.first_edge_cls = []
        # self.first_traj_feature = []
        self.first_traj_feature_no_time = []
        self.first_time_index = []
        self.spatial_temporal = {}
        for i in range(24):
            self.spatial_temporal[i] = []
        first_contain_cells = []
        first_stay_time = []

        trace_0 = id_trace.iloc[0] # 第一条轨迹
        coor = trace_0["trace"].split("#")[0:-1]
        times = trace_0["time_hour"].split("#")[0:-1]
        first_t_weight = np.array(trace_0["distance"].split("#")[0:-1],dtype=float)
        for k in range(len(coor)):
            cell = coor[k]
            stay_time = np.zeros((24,))
            cls = self.cell_index[self.cell_index["index_"] == cell]["cls"].values[0]
            first_contain_cells.append(cls + 1)
            self.spatial_temporal[int(times[k])].append(cls + 1)
            # Get the actual node features from cell_feature
            node_features = self.cell_feature[cell]
            stay_time[int(times[k])] = 1
            self.first_traj_feature_no_time.append(node_features)
            first_stay_time.append(stay_time)

        # 以cls为标识创建第一条轨迹的edge_index shape=[(1,3),(3,3),(2,1)...]
        self.first_i_edge = np.zeros((2, len(first_contain_cells) - 1))
        for i_cls in range(len(first_contain_cells)-1):
            start_ = first_contain_cells[i_cls]
            end_ = first_contain_cells[i_cls+1]
            self.first_edge_cls.append((start_,end_))
            start_time = times[i_cls]
            end_time = times[i_cls + 1]
            self.first_time_index.append((start_time, end_time))
            self.first_i_edge[0][i_cls] = i_cls
            self.first_i_edge[1][i_cls] = i_cls + 1

        # 创建第一条轨迹的边属性列表
        for i_time in range(len(first_stay_time) - 1):
            e_attr_ = first_stay_time[i_time] + first_stay_time[i_time+1]
            a = first_t_weight[i_time].reshape(1,)
            e_attr = np.concatenate([e_attr_, a], axis=0)
            self.first_traj_edge_attr.append(e_attr)


    def mul_trajs_n_1(self, id_trace):
        self.n_1_edge_cls = []
        self.n_1_time_feature = []
        self.n_1_traj_edge_attr = []
        self.n_1_time_index = []

        tmp_edge_time = []
        # 剩下的n-1条轨迹，只需要获得cls_edge_index和edge_attr
        for t in range(1, len(id_trace)):
            trace_t = id_trace.iloc[t]
            coor = trace_t["trace"].split("#")[0:-1]
            times = trace_t["time_hour"].split("#")[0:-1]
            n_t_weight = np.array(trace_t["distance"].split("#")[0:-1], dtype=float)
            for k in range(len(coor)):
                cell = coor[k]
                cls = self.cell_index[self.cell_index["index_"] == cell]["cls"].values[0]
                # self.fcls[int(times[k])] = cls + 1
                self.spatial_temporal[int(times[k])].append(cls + 1)
                if k+1 < len(coor):
                    cell_next = coor[k+1]
                    cls_next = self.cell_index[self.cell_index["index_"] == cell_next]["cls"].values[0]
                    start_edge = cls + 1
                    end_edge = cls_next + 1
                    a = (start_edge, end_edge)
                    start_t_index = times[k]
                    end_t_index = times[k + 1]
                    b = (start_t_index, end_t_index)
                    comb_a_b = (a, b)

                    tmp_edge_time.append(comb_a_b)
                    self.n_1_edge_cls.append(a)
                    self.n_1_time_index.append(b)
                    start_time = np.zeros((24,))
                    start_time[int(start_t_index)] = 1
                    end_time = np.zeros((24,))
                    end_time[int(end_t_index)] = 1
                    e_n_attr_ = start_time + end_time
                    e_n_attr = np.concatenate([e_n_attr_, n_t_weight[k].reshape(1, )], axis=0)
                    self.n_1_traj_edge_attr.append(e_n_attr)


    def updata_mul_trajs(self, name):
        # 将其余N条轨迹图加入到第一条轨迹中
        # 首先遍历 n_1_edge_cls中的邻接边在 first_edge_cls 中是否已经存在
        # 如不存在，则将n_1_edge_index加入到原有的edge_index中，并在第一条traj_feature中加入n_1的feature

        self.freq = [1 for i in range(len(self.first_edge_cls))]
        for step in range(len(self.n_1_edge_cls)):
            iter_first_edge_cls = enumerate(self.first_edge_cls)
            e_cls = self.n_1_edge_cls[step]
            e_time_index = self.n_1_time_index[step]
            co_idnex = [i for i,v in iter_first_edge_cls if v==e_cls]
            # 如果co_idnex！=0，则执行try的内容
            try:
                co_edge_cls = False
                for j in co_idnex:
                    # 遍历完所有co_index才能进行判断 （如果edge_cls和time_index都相等，则频率加一）
                    if self.first_time_index[j] == e_time_index:
                        self.freq[j]+=1
                        co_edge_cls = True
                # 如果edge_cls相同，但是time_index不同，则把time_index加入其中
                # 只选取co_index种的第一个索引；co_index[0]也对应traj_feature中的索引，作为更新边索引的起点
                if co_edge_cls == False:
                    updated_start, updated_end = self.first_edge_cls[co_idnex[0]]
                    # Get the cell index for the end node
                    end_cell = self.cell_index[self.cell_index["cls"] == (updated_end - 1)]["index_"].values[0]
                    # Get actual node features from cell_feature
                    node_features = self.cell_feature[end_cell]
                    start_end_idx = np.array([co_idnex[0], len(self.first_traj_feature_no_time)]).reshape(2, 1)
                    self.first_i_edge = np.concatenate([self.first_i_edge, start_end_idx], axis=1)
                    self.first_traj_feature_no_time.append(node_features)
                    self.first_traj_edge_attr.append(self.n_1_traj_edge_attr[step])
                    self.first_edge_cls.append(self.first_edge_cls[co_idnex[0]])
                    self.first_time_index.append(e_time_index)
                    self.freq.append(1)
            # 如果co_idnex=0，则说明n_1的e_cls之前不存在于第一条轨迹的cls中，则需要将新的e_cls添加进去
            except:
                # s_t_feature, n_t_feature = self.n_1_time_feature[step]
                s_node, e_node = e_cls
                # Get the cell indices for both start and end nodes
                start_cell = self.cell_index[self.cell_index["cls"] == (s_node - 1)]["index_"].values[0]
                end_cell = self.cell_index[self.cell_index["cls"] == (e_node - 1)]["index_"].values[0]
                # Get actual node features from cell_feature
                start_features = self.cell_feature[start_cell]
                end_features = self.cell_feature[end_cell]

                idx = False
                for num in range(len(self.first_traj_feature_no_time)):
                    if np.array_equal(start_features, self.first_traj_feature_no_time[num]):
                        idx = True
                        startnode_idx_be_add = num
                        endnode_idx_be_add = len(self.first_traj_feature_no_time)
                        start_end_idx = np.array([startnode_idx_be_add, endnode_idx_be_add]).reshape(2, 1)
                        self.first_i_edge = np.concatenate([self.first_i_edge, start_end_idx], axis=1)
                        self.first_traj_feature_no_time.append(end_features)
                        self.first_edge_cls.append(e_cls)
                        self.first_traj_edge_attr.append(self.n_1_traj_edge_attr[step])
                        self.first_time_index.append(self.n_1_time_index[step])
                        self.freq.append(1)
                    if idx:
                        idx = False
                        break
        freqs = np.array(self.freq).reshape([-1,1]) #接着对用户的freq做一个softmax作为训练的注意力
        # freqs -= np.max(freqs, axis = 1, keepdims = True)
        # freqs_att = np.exp(freqs) / np.sum(np.exp(freqs), keepdims = True)
        first_traj_feature_np = np.array(self.first_traj_feature_no_time)
        
        # Get distances for all edges
        edge_distances = []
        for i in range(self.first_i_edge.shape[1]):

            # Get the cell indices for the start and end nodes
            start_cell = self.cell_index[self.cell_index["cls"] == (self.first_edge_cls[i][0] - 1)]["index_"].values[0]
            end_cell = self.cell_index[self.cell_index["cls"] == (self.first_edge_cls[i][1] - 1)]["index_"].values[0]
            # Get the distance from the distance matrix
            edge_distance = self.distance[start_cell][end_cell]
            edge_distances.append(edge_distance)
        
        edge_distances = np.array(edge_distances).reshape([-1,1])
        
        # self.save_label[name] = self.fcls
        self.save_label[name] = self.cal_distribution(self.spatial_temporal)
        self.save_feature[name] = first_traj_feature_np
        self.save_edge[name] = self.first_i_edge
        # Concatenate time features, frequencies and distances
        self.save_edge_attr[name] = np.concatenate([np.array(self.first_traj_edge_attr), freqs, edge_distances], axis=-1)

    def build_graphs(self):
        for name, id_trace in self.id_groups:
            if len(id_trace) == 1:
                continue
                # single_traj_graph = self.single_traj(name=name,id_trace=id_trace)
            else:
                # tmp_trace = self.id_groups.get_group('001385e921098b4aa13427301a952656')
                m_first_traj = self.mul_trajs_firts(id_trace=id_trace)
                m_n_1_traj = self.mul_trajs_n_1(id_trace=id_trace)
                updated_trajs = self.updata_mul_trajs(name=name)

        np.save('../data/trajFeatureDistri_cellsCls/train0%d/nodeFeature0%d.npy'%(self.path,self.path), self.save_feature)
        np.save('../data/trajFeatureDistri_cellsCls/train0%d/nodeLabel0%d.npy'%(self.path,self.path), self.save_label)
        np.save('../data/trajFeatureDistri_cellsCls/train0%d/edgeIndex0%d.npy'%(self.path,self.path), self.save_edge)
        np.save('../data/trajFeatureDistri_cellsCls/train0%d/edgeAttr0%d.npy'%(self.path,self.path), self.save_edge_attr)



def load_data():
    traj_edge = []
    traj_graph = []
    traj_label = []
    traj_edge_attr = []
    num=0
    root_path = '../data/trajFeatureDistri_cellsCls/'
    for i in range(1,8):
        traj_edge_0 = np.load(root_path+'train0%d/edgeIndex0%d.npy'%(i,i),
                               allow_pickle=True).item()
        traj_graph_0 = np.load(root_path+'train0%d/nodeFeature0%d.npy'%(i,i),
                                allow_pickle=True).item()

        traj_label_0 = np.load(root_path+'train0%d/nodeLabel0%d.npy'%(i,i),
                                allow_pickle=True).item()
        traj_edge_attr_0 = np.load(root_path+'train0%d/edgeAttr0%d.npy'%(i,i),
                                    allow_pickle=True).item()
        traj_edge.append(traj_edge_0)
        traj_graph.append(traj_graph_0)
        num+=len(traj_graph_0)
        traj_label.append(traj_label_0)
        traj_edge_attr.append(traj_edge_attr_0)

    print(num)
    return traj_edge, traj_graph, traj_label, traj_edge_attr

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['traj_data_countPeople.pt']

    def process(self):
        edge_values, traj_values, label_values, edge_attr_values = load_data()
        # Read data into huge `Data` list.
        data_list = []
        index = 0
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]
        for i in range(7):
            for key in traj_values[i]:
                if len(traj_values[i][key]):
                    g1 = traj_values[i][key]
                    e1 = edge_values[i][key]
                    attr = edge_attr_values[i][key]
                    n1 = np.reshape(label_values[i][key], newshape=[1, -1])
                    node_feature = torch.tensor(g1, dtype=torch.float32)
                    edge_index = torch.LongTensor(e1)
                    node_label = torch.tensor(n1)
                    edge_attr = torch.tensor(attr, dtype=torch.float32)
                    index_tensor = torch.full((edge_attr.shape[0], ), index)
                    data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr, y=node_label, pos=index_tensor)
                    outD = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
                    data.edge_attr[:,-1]/= outD[data.edge_index[0]]
                    data_list.append(data)
                    index+=1

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__=="__main__":
    print("dataMaking...")
    # load_data()
    cell_index = pd.read_csv("../data/adjMatrix/highfreq_cell_cls.csv")
    cell_feature = np.load("../result/SL_Model_feature.npy", allow_pickle=True).item() # ["index_","node_features"]
    distance = np.load("../data/adjMatrix/allpart_edge_distance.npy", allow_pickle=True) # ["index_start","index_end"]
    for i in range(2,8):
        print(i)
        traj = pd.read_csv("../data/highfreq_traj/train0%d_Nove.csv"%(i))
        a = TrajDistri_Process_withEdge(traj=traj, cell_index=cell_index, path=i, cell_feature=cell_feature, distance=distance)