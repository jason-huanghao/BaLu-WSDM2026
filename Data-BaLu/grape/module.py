import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing

from torch.nn.init import xavier_uniform_, zeros_

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils
import numpy as np

# def get_gnn(#data, args):
#     n_features,
#     node_dim=64,
#     edge_dim=16,
#     msg_dim=64,
#     n_layers=2,
#     dropout=0.1,
#     activation='relu', 
#     model_types = 'EGSAGE_EGSAGE'):
    
#     model_types = model_types.split('_')
#     norm_embs = [True,]*len(model_types)
#     post_hiddens = [node_dim]
        
#     print(model_types, norm_embs, post_hiddens)
#     # build model
#     model = GNNStack(n_features, 
#                      1,
#                     node_dim, 
#                     edge_dim, 
#                     1,
#                     model_types, 
#                     dropout, 
#                     activation,
#                     False, 
#                     post_hiddens,
#                     norm_embs, 
#                     'mean')
#     return model

class MLPNet(torch.nn.Module):
    def __init__(self, 
         		input_dims, output_dim,
         		hidden_layer_sizes=(64,),
         		hidden_activation='relu',
         		output_activation=None,
                dropout=0.):
        super(MLPNet, self).__init__()

        layers = nn.ModuleList()

        input_dim = np.sum(input_dims)

        for layer_size in hidden_layer_sizes:
            hidden_dim = layer_size
            layer = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        get_activation(hidden_activation),
                        nn.Dropout(dropout),
                        )
            layers.append(layer)
            input_dim = hidden_dim

        layer = nn.Sequential(
        				nn.Linear(input_dim, output_dim),
        				get_activation(output_activation),
        				)
        layers.append(layer)
        self.layers = layers

    def forward(self, inputs):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        input_var = torch.cat(inputs,-1)
        for layer in self.layers:
            input_var = layer(input_var)
        return input_var



class GNNStack(torch.nn.Module):
    def __init__(self, 
                node_input_dim, edge_input_dim,
                node_dim, edge_dim, edge_mode,
                model_types, dropout, activation,
                concat_states, node_post_mlp_hiddens,
                normalize_embs, aggr
                ):
        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.concat_states = concat_states
        self.model_types = model_types
        self.gnn_layer_num = len(model_types)

        # convs
        self.convs = self.build_convs(node_input_dim, edge_input_dim,
                                    node_dim, edge_dim, edge_mode,
                                    model_types, normalize_embs, activation, aggr)

        # post node update
        if concat_states:
            self.node_post_mlp = self.build_node_post_mlp(int(node_dim*len(model_types)), int(node_dim*len(model_types)), node_post_mlp_hiddens, dropout, activation)
        else:
            self.node_post_mlp = self.build_node_post_mlp(node_dim, node_dim, node_post_mlp_hiddens, dropout, activation)

        self.edge_update_mlps = self.build_edge_update_mlps(node_dim, edge_input_dim, edge_dim, self.gnn_layer_num, activation)

    def build_node_post_mlp(self, input_dim, output_dim, hidden_dims, dropout, activation):
        if 0 in hidden_dims:
            return get_activation('none')
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layer = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            get_activation(activation),
                            nn.Dropout(dropout),
                            )
                layers.append(layer)
                input_dim = hidden_dim
            layer = nn.Linear(input_dim, output_dim)
            layers.append(layer)
            return nn.Sequential(*layers)

    def build_convs(self, node_input_dim, edge_input_dim,
                     node_dim, edge_dim, edge_mode,
                     model_types, normalize_embs, activation, aggr):
        convs = nn.ModuleList()
        conv = self.build_conv_model(model_types[0],node_input_dim,node_dim,
                                    edge_input_dim, edge_mode, normalize_embs[0], activation, aggr)
        convs.append(conv)
        for l in range(1,len(model_types)):
            conv = self.build_conv_model(model_types[l],node_dim, node_dim,
                                    edge_dim, edge_mode, normalize_embs[l], activation, aggr)
            convs.append(conv)
        return convs

    def build_conv_model(self, model_type, node_in_dim, node_out_dim, edge_dim, edge_mode, normalize_emb, activation, aggr):
        #print(model_type)
        if model_type == 'GCN':
            return pyg_nn.GCNConv(node_in_dim,node_out_dim)
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv(node_in_dim,node_out_dim)
        elif model_type == 'GAT':
            return pyg_nn.GATConv(node_in_dim,node_out_dim)
        elif model_type == 'EGCN':
            return EGCNConv(node_in_dim,node_out_dim,edge_dim,edge_mode)
        elif model_type == 'EGSAGE':
            return EGraphSage(node_in_dim,node_out_dim,edge_dim,activation,edge_mode,normalize_emb, aggr)
    
    def build_edge_update_mlps(self, node_dim, edge_input_dim, edge_dim, gnn_layer_num, activation):
        edge_update_mlps = nn.ModuleList()
        edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_input_dim,edge_dim),
                get_activation(activation),
                )
        edge_update_mlps.append(edge_update_mlp)
        for l in range(1,gnn_layer_num):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_dim+node_dim+edge_dim,edge_dim),
                get_activation(activation),
                )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0],:]
        x_j = x[edge_index[1],:]
        edge_attr = mlp(torch.cat((x_i,x_j,edge_attr),dim=-1))
        return edge_attr

    def forward(self, data):
        edge_index, edge_attr, x = data.edge_index, data.edge_attr, data.x
        # print("edge shape:", edge_attr.shape)
        
            # print(f"reshape from to {edge_attr.shape}")

        if self.concat_states:
            concat_x = []
        for l,(conv_name,conv) in enumerate(zip(self.model_types,self.convs)):
            # self.check_input(x,edge_attr,edge_index)
            if conv_name == 'EGCN' or conv_name == 'EGSAGE':
                x = conv(x, edge_attr, edge_index)
            else:
                x = conv(x, edge_index)
            if self.concat_states:
                concat_x.append(x)
            edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
            #print(edge_attr.shape)
        if self.concat_states:
            x = torch.cat(concat_x, 1)
        x = self.node_post_mlp(x)
        # self.check_input(x,edge_attr,edge_index)
        
        return x

    def check_input(self, xs, edge_attr, edge_index):
        Os = {}
        for indx in range(128):
            i=edge_index[0,indx].detach().numpy()
            j=edge_index[1,indx].detach().numpy()
            xi=xs[i].detach().numpy()
            xj=list(xs[j].detach().numpy())
            eij=list(edge_attr[indx].detach().numpy())
            if str(i) not in Os.keys():
                Os[str(i)] = {'x_j':[],'e_ij':[]}
            Os[str(i)]['x_i'] = xi
            Os[str(i)]['x_j'] += xj
            Os[str(i)]['e_ij'] += eij

        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1,3,1)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_i'],label=str(i))
            plt.title('x_i')
        plt.legend()
        plt.subplot(1,3,2)
        for i in Os.keys():
            plt.plot(Os[str(i)]['e_ij'],label=str(i))
            plt.title('e_ij')
        plt.legend()
        plt.subplot(1,3,3)
        for i in Os.keys():
            plt.plot(Os[str(i)]['x_j'],label=str(i))
            plt.title('x_j')
        plt.legend()
        plt.show()





def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError


class EGraphSage(MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels,
                 edge_channels, activation='relu', edge_mode=1,
                 normalize_emb=True,
                 aggr='mean'):
        super(EGraphSage, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.edge_mode = edge_mode

        if edge_mode == 0:
            self.message_lin = nn.Linear(in_channels, out_channels)
            self.attention_lin = nn.Linear(2*in_channels+edge_channels, 1)
        elif edge_mode == 1:
            self.message_lin = nn.Linear(in_channels+edge_channels, out_channels)
        elif edge_mode == 2:
            self.message_lin = nn.Linear(2*in_channels+edge_channels, out_channels)
        elif edge_mode == 3:
            self.message_lin = nn.Sequential(
                    nn.Linear(2*in_channels+edge_channels, out_channels),
                    get_activation(activation),
                    nn.Linear(out_channels, out_channels),
                    )
        elif edge_mode == 4:
            self.message_lin = nn.Linear(in_channels, out_channels*edge_channels)
        elif edge_mode == 5:
            self.message_lin = nn.Linear(2*in_channels, out_channels*edge_channels)

        self.agg_lin = nn.Linear(in_channels+out_channels, out_channels)

        self.message_activation = get_activation(activation)
        self.update_activation = get_activation(activation)
        self.normalize_emb = normalize_emb

    def forward(self, x, edge_attr, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(num_nodes, num_nodes))

    def message(self, x_i, x_j, edge_attr, edge_index, size):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]
        if self.edge_mode == 0:
            attention = self.attention_lin(torch.cat((x_i,x_j, edge_attr),dim=-1))
            m_j = attention * self.message_activation(self.message_lin(x_j))
        elif self.edge_mode == 1:
            
            m_j = torch.cat((x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 2 or self.edge_mode == 3:
            m_j = torch.cat((x_i,x_j, edge_attr),dim=-1)
            m_j = self.message_activation(self.message_lin(m_j))
        elif self.edge_mode == 4:
            E = x_j.shape[0]
            w = self.message_lin(x_j)
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        elif self.edge_mode == 5:
            E = x_j.shape[0]
            w = self.message_lin(torch.cat((x_i,x_j),dim=-1))
            w = self.message_activation(w)
            w = torch.reshape(w, (E,self.out_channels,self.edge_channels))
            m_j = torch.bmm(w, edge_attr.unsqueeze(-1)).squeeze(-1)
        return m_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x),dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class EGCNConv(MessagePassing):
    # form https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv
    def __init__(self, in_channels, out_channels,
                 edge_channels, edge_mode,
                 improved=False, cached=False,
                 bias=True, **kwargs):
        super(EGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.edge_mode = edge_mode

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if edge_mode == 0:
            self.attention_lin = nn.Linear(2*out_channels+edge_channels, 1)
        elif self.edge_mode == 1:
            self.message_lin = nn.Linear(2*out_channels+edge_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.weight)
        zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_attr, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)


    def message(self, x_i, x_j, edge_attr, norm):
        if self.edge_mode == 0:
            attention = self.attention_lin(torch.cat((x_i,x_j, edge_attr),dim=-1))
            m_j = attention * x_j
        elif self.edge_mode == 1:
            m_j = torch.cat((x_i, x_j, edge_attr),dim=-1)
            m_j = self.message_lin(m_j)
        return norm.view(-1, 1) * m_j

    def update(self, aggr_out, x):
        #print(aggr_out)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.edge_mode == 0:
            aggr_out = aggr_out + x
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)