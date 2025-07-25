import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge
from src.imputer.conv_modules import EGraphSage

class GRAPE_Imp(nn.Module):
    def __init__(self, data, node_dims=[64, 64, 64], edge_dim=16, out_dim=64, adj_dropout=0.2, dropout=0.1):
        super(GRAPE_Imp, self).__init__()

        self.adj_dropout = adj_dropout
        self.dropout = dropout

        ################## node updating ##################
        self.node_convs = nn.ModuleList()
        prev_hidden_size = data.x.shape[1]
        for l, next_hidden_size in enumerate(node_dims):
            self.node_convs.append(EGraphSage(prev_hidden_size,
                                            next_hidden_size,
                                            edge_channels=1 if l == 0 else edge_dim))
            prev_hidden_size = next_hidden_size

        ################## edge updating ##################
        self.edge_nns = nn.ModuleList()
        for l, node_dim in enumerate(node_dims):
            if l == 0:
                prev_hidden_size = 2*node_dim + 1
            else:
                prev_hidden_size = 2*node_dim + edge_dim 
            self.edge_nns.append(nn.Sequential(nn.Linear(prev_hidden_size, edge_dim),
                                            nn.ReLU(),))
        
        ################## output nn ##################
        self.out = nn.Sequential(*[nn.Linear(node_dims[-1], out_dim),       # imputed_rep  output is relued into [0,1]
                                   nn.ReLU()])
    
    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat((x_i,x_j,edge_attr),dim=-1))
        return edge_attr
        
    def forward(self, data):
        edge_index, edge_attr, x = data.edge_index, data.edge_attr, data.x
        if self.adj_dropout > 0 and self.training:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.adj_dropout, training=self.training, force_undirected=True)
            edge_attr = edge_attr[edge_mask]

        edge_attr_emb = edge_attr.unsqueeze(1)      # to shape of [E, 1]

        for l, conv in enumerate(self.node_convs):
            x = conv(x, edge_attr_emb, edge_index)      # edge_index unit-attr
            edge_attr_emb = self.update_edge_attr(x, edge_attr_emb, edge_index, self.edge_nns[l])

        return self.out(x)

    
