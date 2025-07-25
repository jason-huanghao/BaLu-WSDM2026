import torch
import torch.nn as nn
# from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv, SAGEConv, RGCNConv #, global_sort_pool, global_add_pool
import torch.nn.functional as F


class GNN_IF(nn.Module):
    def __init__(self, data, in_dim, gconv=GCNConv, node_dims=[64, 64], dropout=0.1):
        super(GNN_IF, self).__init__()
        self.n_rel_types = data.n_rel_types
        self.dropout = dropout
        self.gconv = gconv  # Store the conv type for use in forward

        self.gnn_layers = nn.ModuleList()  # Fixed: was using undefined self.rconvs
        prev_hidden_size = in_dim
        for next_hidden_size in node_dims:
            if gconv == GCNConv:
                self.gnn_layers.append(gconv(prev_hidden_size,  # Fixed: was self.rconvs
                                        next_hidden_size,
                                        improved=True))
            elif gconv == SAGEConv:
                self.gnn_layers.append(gconv(prev_hidden_size,  # Fixed: was self.rconvs
                                        next_hidden_size))
            elif gconv == RGCNConv:
                self.gnn_layers.append(gconv(prev_hidden_size,  # Fixed: was self.rconvs
                                        next_hidden_size,
                                        num_relations=self.n_rel_types))
            prev_hidden_size = next_hidden_size

    def forward(self, t_rep, data):  # Fixed: proper indentation
        rel_edge_index, rel_edge_type = data.rel_edge_index, data.rel_edge_type
        x_gnn = t_rep

        for gnn_net in self.gnn_layers:
            if self.gconv == RGCNConv:  # Fixed: was self.gnn_fun == 'RGCNConv'
                x_gnn = gnn_net(x_gnn, rel_edge_index, rel_edge_type)
            else:
                x_gnn = gnn_net(x_gnn, rel_edge_index)
            x_gnn = F.dropout(F.relu(x_gnn), p=self.dropout, training=self.training)
        return x_gnn

