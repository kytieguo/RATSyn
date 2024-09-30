from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
import numpy as np


class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, cell_graph=None):
        super(DTADataset, self).__init__()

        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        self.cell_graph = cell_graph

    def read_drug_info(self, drug_id, labels):

        c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, deg = self.smile_graph[str(drug_id)]  ##drug——id是str类型的，不是int型的，这点要注意
        subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, deg = self.sub_graph[str(drug_id)]

        data_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                              edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                              y=torch.LongTensor([labels]),
                              rel_index=torch.Tensor(np.array(rel_index, dtype=int)),
                              sp_edge_index=torch.LongTensor(sp_edge_index).transpose(1, 0),
                              sp_value=torch.Tensor(np.array(sp_value, dtype=int)),
                              sp_edge_rel=torch.LongTensor(np.array(sp_rel, dtype=int))
                              )
        data_mol.__setitem__('c_size', torch.LongTensor([c_size]))

        data_graph = DATA.Data(x=torch.LongTensor(subset),
                                edge_index=torch.LongTensor(subgraph_edge_index).transpose(1,0),
                                y=torch.LongTensor([labels]),
                                id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
                                rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)),
                                sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
                                sp_value=torch.Tensor(np.array(s_value, dtype=int)),
                                sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int))
                                )

        return data_mol, data_graph

    def read_cell_info(self, cell_id, labels):
        subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, deg = self.cell_graph[str(cell_id)]

        data_graph = DATA.Data(x=torch.LongTensor(subset),
                               edge_index=torch.LongTensor(subgraph_edge_index).transpose(1, 0),
                               y=torch.LongTensor([labels]),
                               id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
                               rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)),
                               sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
                               sp_value=torch.Tensor(np.array(s_value, dtype=int)),
                               sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int))
                               )
        return data_graph

    def __len__(self):
        return len(self.drug_ID)

    def __getitem__(self, idx):
        drug1_id = self.drug_ID[idx, 0]
        drug2_id = self.drug_ID[idx, 1]
        cell_id = self.drug_ID[idx, 2]
        labels = int(self.labels[idx])

        drug1_mol, drug1_subgraph = self.read_drug_info(drug1_id, labels)
        drug2_mol, drug2_subgraph = self.read_drug_info(drug2_id, labels)
        cell_subgraph = self.read_cell_info(cell_id, labels)

        return drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph, cell_subgraph


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    batchE = Batch.from_data_list([data[4] for data in data_list])

    return batchA, batchB, batchC, batchD, batchE
