import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from src.imputer.conv_modules import get_conv, forward_conv, EGraphSage

class BaLu_GRAPE_Imp(nn.Module):
    def __init__(self, data, rconv:str, use_rel_type=True,  node_dims=[64, 64, 64], edge_dim=16, out_dim=64, adj_dropout=0.2, dropout=0.1):
        super(BaLu_GRAPE_Imp, self).__init__()

        self.adj_dropout = adj_dropout
        self.dropout = dropout
        self.rconv = rconv
        self.use_rel_type = use_rel_type

        ################## node updating ##################
        self.node_convs = nn.ModuleList()
        prev_node_dim = data.x.shape[1]
        for l, next_node_dim in enumerate(node_dims):
            self.node_convs.append(EGraphSage(prev_node_dim,
                                            next_node_dim,
                                            edge_channels=1 if l == 0 else edge_dim))
            prev_node_dim = next_node_dim

        ################## edge updating ##################
        self.edge_nns = nn.ModuleList()
        for l, node_dim in enumerate(node_dims):
            if l == 0:
                prev_node_dim = 2*node_dim + 1
            else:
                prev_node_dim = 2*node_dim + edge_dim 
            self.edge_nns.append(nn.Sequential(nn.Linear(prev_node_dim, edge_dim),
                                            nn.ReLU(),))
        
        ################## unit node updating ##################
        self.rconv_nets = nn.ModuleList()
        self.n_rel_types = data.n_rel_types
        for next_node_dim in node_dims:
            self.rconv_nets.append(get_conv(self.rconv, next_node_dim, next_node_dim, self.n_rel_types))            # can have relations

        ################## output nn ##################
        self.out = nn.Sequential(*[nn.Linear(node_dims[-1], out_dim),       # imputed_rep  output is relued into [0,1]
                                   nn.ReLU()])
    
    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat((x_i,x_j,edge_attr),dim=-1))
        return edge_attr

    def forward(self, data):
        edge_index, edge_attr, rel_edge_index, rel_edge_type, x = data.edge_index, data.edge_attr, data.rel_edge_index, data.rel_edge_type, data.x
        if self.adj_dropout > 0 and self.training:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.adj_dropout, training=self.training, force_undirected=True)
            edge_attr = edge_attr[edge_mask]

        edge_attr_emb = edge_attr.unsqueeze(1)      # to shape of [E, 1]
        
        for l, conv_net in enumerate(self.node_convs):
            x = conv_net(x, edge_attr_emb, edge_index)      # edge_index unit-attr
            rconv_net = self.rconv_nets[l]
            x = forward_conv(self.rconv, rconv_net, x, rel_edge_index, rel_edge_type if self.use_rel_type else None)
            x = F.dropout(F.relu(x), training=self.training, p=self.dropout)        
            edge_attr_emb = self.update_edge_attr(x, edge_attr_emb, edge_index, self.edge_nns[l])
        
        return self.out(x)


