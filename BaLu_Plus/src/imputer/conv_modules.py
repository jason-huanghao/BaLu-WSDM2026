from torch_geometric.nn import GCNConv, GATConv, SAGEConv , RGCNConv, RGATConv
import torch
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F

# conv_map= {'GCN': GCNConv, 
#            'GraphSAGE': SAGEConv, 
#            'RGCN': RGCNConv, 
#            'GAT': GATConv, 
#            'RGAT': RGATConv}

convs = ['GCN', 'GraphSAGE', 'RGCN', 'GAT', 'RGAT']

def get_conv(conv, in_dim, out_dim, n_rel_types=None):
    if conv == 'GCN':
        return GCNConv(in_dim, out_dim, improved=True)
    elif conv == 'GraphSAGE':
        return SAGEConv(in_dim, out_dim)
    elif conv == 'GAT':
        return GATConv(in_dim, out_dim)
    elif conv == "RGCN":
        assert n_rel_types is not None
        return RGCNConv(in_dim, out_dim, num_relations=n_rel_types)
    elif conv == "RGAT":
        assert n_rel_types is not None
        return RGATConv(in_dim, out_dim, n_rel_types)
    else:
        raise Exception(f"has no such conv option {conv}")
    
def forward_conv(conv, conv_net, x, edge_index, edge_type=None):
    if conv in ['GCN', 'GraphSAGE', 'GAT']:
        return conv_net(x, edge_index)
    elif conv in ["RGCN", 'RGAT']:
        return conv_net(x, edge_index, edge_type)

class EGraphSage(MessagePassing):
    """Concise Non-minibatch GraphSage with edge features."""
    
    def __init__(self, in_channels, out_channels, edge_channels, aggr='mean'):
        super().__init__(aggr=aggr)
        self.message_lin = nn.Linear(in_channels + edge_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)
    
    def forward(self, x, edge_attr, edge_index):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        m_j = torch.cat((x_j, edge_attr),dim=-1)
        return F.relu(self.message_lin(m_j))
    
    def update(self, aggr_out, x):
        out = F.relu(self.agg_lin(torch.cat((aggr_out, x),dim=-1)))
        return out
