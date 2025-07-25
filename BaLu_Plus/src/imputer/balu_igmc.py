
import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge
# from torch_geometric.nn import GCNConv, SAGEConv , RGCNConv #, global_sort_pool, global_add_pool
import torch.nn.functional as F
from src.imputer.conv_modules import get_conv, forward_conv #, EGraphSage


class BaLu_IGMC_Imp(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.    # [64, 64, 64, 64]
    def __init__(self, data, gconv:str, rconv:str, use_rel_type=True, node_dims=[64, 64, 64], out_dim=64, adj_dropout=0.2, dropout=0.1):
        super(BaLu_IGMC_Imp, self).__init__()

        self.use_rel_type = use_rel_type
        self.gconv = gconv
        self.rconv = rconv

        self.convs = nn.ModuleList()
        self.rconvs = nn.ModuleList()
        self.n_rel_types = data.n_rel_types
        
        self.adj_dropout = adj_dropout
        self.dropout = dropout
        
        prev_hidden_size = data.x.shape[1]
        for next_hidden_size in node_dims:
            self.convs.append(get_conv(gconv, 
                                       prev_hidden_size,
                                       next_hidden_size)) 
            prev_hidden_size = next_hidden_size

        for next_hidden_size in node_dims:
            self.rconvs.append(get_conv(rconv, 
                                       next_hidden_size,
                                       next_hidden_size,
                                       self.n_rel_types))

        # self.out = nn.Sequential(*[nn.Linear(sum(node_dims), out_dim),       # imputed_rep  output is relued into [0,1]
        #                            nn.ReLU()])
        self.out = nn.Sequential(*[nn.Linear(node_dims[-1], out_dim),       # imputed_rep  output is relued into [0,1]
                                   nn.ReLU()])

    def forward(self, data):
        edge_index, edge_attr, rel_edge_index, rel_edge_type, x= data.edge_index, data.edge_attr, data.rel_edge_index, data.rel_edge_type, data.x
        
        if self.adj_dropout > 0 and self.training:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.adj_dropout, training=self.training, force_undirected=True)
            # edge_attr = edge_attr[edge_mask]          # do not use edge_attr
        
        concat_states = []
        for l, conv_net in enumerate(self.convs):
            x = forward_conv(self.gconv, conv_net, x, edge_index)
            x = F.relu(x)          # connection between units and attributes

            rconv_net = self.rconvs[l]
            x = forward_conv(self.rconv, rconv_net, x, rel_edge_index, rel_edge_type if self.use_rel_type else None)
            x = F.dropout(F.relu(x), training=self.training, p=self.dropout)      # connections between units
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



