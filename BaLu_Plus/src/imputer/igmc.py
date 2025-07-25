
import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge
# from torch_geometric.nn import GCNConv, SAGEConv #, RGCNConv, global_sort_pool, global_add_pool
import torch.nn.functional as F
from src.imputer.conv_modules import get_conv, forward_conv


class IGMC(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout. [64, 64, 64, 64]
    def __init__(self, data, gconv:str, node_dims=[64, 64, 64], out_dim=64, adj_dropout=0.2, dropout=0.1):
        super(IGMC, self).__init__()

        self.gconv = gconv
        
        self.convs = nn.ModuleList()
        self.adj_dropout = adj_dropout
        self.dropout = dropout
        
        prev_hidden_size = data.x.shape[1]
        for next_hidden_size in node_dims:
            self.convs.append(get_conv(gconv, 
                                       prev_hidden_size,
                                       next_hidden_size)) 
            prev_hidden_size = next_hidden_size

        # self.out = nn.Sequential(*[nn.Linear(sum(node_dims), out_dim),       # imputed_rep  output is relued into [0,1]
                                #    nn.ReLU()])
        self.out = nn.Sequential(*[nn.Linear(node_dims[-1], out_dim),       # imputed_rep  output is relued into [0,1]
                                   nn.ReLU()])

    def forward(self, data):
        edge_index, edge_attr, x = data.edge_index, data.edge_attr, data.x 
        
        if self.adj_dropout > 0 and self.training:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.adj_dropout, training=self.training, force_undirected=True)
            # edge_attr = edge_attr[edge_mask]
        
        concat_states = []
        for conv_net in self.convs:
            # x = F.dropout(F.relu(conv(x, edge_index, edge_attr)), training=self.training, p=self.dropout)
            x = forward_conv(self.gconv, conv_net, x, edge_index)
            x = F.dropout(F.relu(x), training=self.training, p=self.dropout)
            
            concat_states.append(x)
        return self.out(x)


        # concat_states = torch.cat(concat_states, 1)

        # return self.out(concat_states)
        # node_emb = concat_states[:data.n_units]
        # attr_emb = concat_states[data.n_units:]
        
        # N, M, K = node_emb.size(0), attr_emb.size(0), node_emb.size(1) 
        # node_expanded = node_emb.unsqueeze(1).expand(N, M, K)
        # attr_expanded = attr_emb.unsqueeze(0).expand(N, M, K)
        # combined = torch.cat([node_expanded, attr_expanded], dim=2)
        # x = combined.view(N * M, 2 * K)
        
        # return self.out(x)



