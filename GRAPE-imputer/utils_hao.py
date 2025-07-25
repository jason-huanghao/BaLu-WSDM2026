import torch
import numpy as np
from torch_geometric.data import Data

def to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return obj

def to_device(data, device=None):
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_device = Data()   
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            setattr(data_device, key, value.to(device))
        else:
            setattr(data_device, key, value)
    return data_device

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


def get_known_mask(known_prob, edge_num):
    known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < known_prob).view(-1)
    return known_mask

def mask_edge(edge_index,edge_attr,mask,remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:,mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.
    return edge_index, edge_attr


def create_data_edge_index_mask(X, n_units, n_features, device, used_mask):
    """
    Create data edge indices for the bipartite graph.
    
    Args:
        X: Feature matrix
        n_units: Number of units
        n_features: Number of features
        
    Returns:
        Edge index and edge attribute tensors, and observed mask
    """
    source_nodes = []
    target_nodes = []
    data_edge_attr_list = []
    attribute_index = []

    used_mask = to_numpy(used_mask) if used_mask is not None else used_mask
    
    for i in range(n_units):
        for j in range(n_features):

            if not np.isnan(X[i, j]):
                if not used_mask[i]:        # chosen used units
                    attribute_index.append(False)
                else:
                    source_nodes.append(i)
                    target_nodes.append(n_units + j)
                    data_edge_attr_list.append(X[i, j])
                    attribute_index.append(True)
            else:
                attribute_index.append(False)
    
    # from attribute to unit (bidirection)
    source_nodes_new = source_nodes + target_nodes
    target_nodes_new = target_nodes + source_nodes
    data_edge_attr_list += data_edge_attr_list

    # print(len(source_nodes_new), len(target_nodes_new), len(data_edge_attr_list))
    # Convert to tensors
    if data_edge_attr_list:
        data_edge_index = torch.tensor([source_nodes_new, target_nodes_new], dtype=torch.long)
        data_edge_attr = torch.tensor(data_edge_attr_list, dtype=torch.float)
    else:
        data_edge_index = torch.zeros((2, 0), dtype=torch.long)
        data_edge_attr = torch.zeros((0,), dtype=torch.float)
    
    return data_edge_index.to(device), data_edge_attr.to(device), torch.tensor(attribute_index, dtype=torch.bool).to(device)

