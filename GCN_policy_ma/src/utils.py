import math
import torch
import itertools
import numpy as np
import torch as tr
import torch.nn as nn

# from copy import deepcopy
from torch.autograd import Variable
# from scipy.sparse import coo_matrix
from scipy.special import gamma
from torch_geometric.data import Data
import copy 


def filter_and_remap_edges(rel_edge_index, non_null_indicator):
    """
    Most optimized version - combines operations and minimizes memory allocations.
    """
    valid_mask = non_null_indicator
    
    mapping = torch.cumsum(valid_mask, dim=0) - 1
    mapping[~valid_mask] = -1  # Mark invalid nodes
    
    # Filter and remap in one step
    edge_mask = valid_mask[rel_edge_index[0]] & valid_mask[rel_edge_index[1]]
    filtered_edges = rel_edge_index[:, edge_mask]
    
    # Apply mapping directly
    new_rel_edge_index = mapping[filtered_edges]
    
    return new_rel_edge_index, mapping

def filter_and_remap_edges_balu(edge_index, unit_mask, n_units, n_attrs):
    """
    Filter and remap edges in bipartite graph (units <-> attributes).
    
    Args:
        edge_index: torch.Tensor of shape [2, E] 
        unit_mask: torch.Tensor of shape [n_units] with boolean values for units to keep
        n_units: Original number of units
        n_attrs: Number of attributes (unchanged)
    
    Returns:
        new_edge_index: torch.Tensor with filtered and remapped edges
    """
    # Create mapping for unit nodes (0 to n_units-1)
    unit_mapping = torch.cumsum(unit_mask, dim=0) - 1
    unit_mapping[~unit_mask] = -1
    
    # Create mapping for attribute nodes (n_units to n_units+n_attrs-1)
    # Attribute indices need to be shifted down by the number of removed units
    n_units_kept = unit_mask.sum().item()
    attr_mapping = torch.arange(n_attrs) + n_units_kept  # New attribute indices
    
    # Create full node mapping
    full_mapping = torch.full((n_units + n_attrs,), -1, dtype=torch.long)
    full_mapping[:n_units] = unit_mapping
    full_mapping[n_units:] = attr_mapping
    
    # Filter edges: keep only edges involving kept units
    unit_nodes_in_edges = edge_index[0] < n_units  # Source is unit
    attr_nodes_in_edges = edge_index[0] >= n_units  # Source is attribute
    
    # For unit->attr edges: keep if unit is in mask
    unit_to_attr = unit_nodes_in_edges & unit_mask[edge_index[0].clamp(max=n_units-1)]
    # For attr->unit edges: keep if target unit is in mask  
    attr_to_unit = attr_nodes_in_edges & unit_mask[edge_index[1].clamp(max=n_units-1)]
    
    edge_mask = unit_to_attr | attr_to_unit
    
    # Apply mapping to filtered edges
    filtered_edges = edge_index[:, edge_mask]
    new_edge_index = full_mapping[filtered_edges]
    
    return new_edge_index


def filter_dataset(data: Data, mask: torch.Tensor) -> Data:
    """
    Filter dataset to keep only units where mask[i] = True.
    Keeps ALL attribute nodes but updates their indices.
    
    Args:
        data: PyTorch Geometric Data object with bipartite structure
        mask: Boolean tensor of shape [n_units] for units to keep
    
    Returns:
        train_data: New Data object with filtered units and remapped structure
    """
    
    # Ensure mask is boolean and on CPU for numpy operations
    mask = mask.bool()
    mask_np = mask.cpu().numpy()
    n_units_kept = mask.sum().item()
    
    # Create new data object
    train_data = copy.deepcopy(data)
    
    # Copy metadata (n_attrs doesn't change)
    for attr in ['n_attrs', 'n_rel_types', 'node_feature_dim', 'edge_attr_dim']:
        if hasattr(data, attr):
            setattr(train_data, attr, getattr(data, attr))
    
    # Update n_units to filtered count
    # n_units, n_attrs = data.n_units, data.n_attrs
    train_data.n_units = n_units_kept

    
    # 1. Filter node features: keep filtered units + all attributes
    if hasattr(data, 'x') and data.x is not None:
        unit_features = data.x[:data.n_units][mask]  # Filter unit nodes
        attr_features = data.x[data.n_units:]        # Keep all attribute nodes
        train_data.x = torch.cat([unit_features, attr_features], dim=0)
    
    # 2. Filter is_unit: keep filtered units + all attributes  
    if hasattr(data, 'is_unit') and data.is_unit is not None:
        unit_is_unit = data.is_unit[:data.n_units][mask]  # Filter unit flags
        attr_is_unit = data.is_unit[data.n_units:]        # Keep attribute flags
        train_data.is_unit = torch.cat([unit_is_unit, attr_is_unit], dim=0)
    
    # 3. Filter unit-level tensors
    if hasattr(data, 'treatment') and data.treatment is not None:
        train_data.treatment = data.treatment[mask]
    
    if hasattr(data, 'outcome') and data.outcome is not None:
        train_data.outcome = data.outcome[mask]
    
    if hasattr(data, 'true_effect') and data.true_effect is not None:
        train_data.true_effect = data.true_effect[mask]
    
    # 4. Filter unit-level masks
    if hasattr(data, 'treatment_mask') and data.treatment_mask is not None:
        train_data.treatment_mask = data.treatment_mask[mask]
    
    if hasattr(data, 'outcome_mask') and data.outcome_mask is not None:
        train_data.outcome_mask = data.outcome_mask[mask]
    
    # 5. Filter split masks (train/val/test)
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        train_data.train_mask = data.train_mask[mask]
    
    if hasattr(data, 'val_mask') and data.val_mask is not None:
        train_data.val_mask = data.val_mask[mask]
    
    if hasattr(data, 'test_mask') and data.test_mask is not None:
        train_data.test_mask = data.test_mask[mask]
    
    # 6. Filter observed_mask (reshape, filter, flatten)
    if hasattr(data, 'observed_mask') and data.observed_mask is not None:
        # Reshape to [n_units, n_attrs], filter units, then flatten
        observed_reshaped = data.observed_mask.view(data.n_units, data.n_attrs)
        observed_filtered = observed_reshaped[mask]  # Shape: [n_units_kept, n_attrs]
        train_data.observed_mask = observed_filtered.flatten()
    
    # 7. Filter and remap bipartite edges (unit <-> attribute)
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        train_data.edge_index = filter_and_remap_edges_balu(
            data.edge_index, mask, data.n_units, data.n_attrs
        )
        
        # Filter edge attributes based on which edges were kept
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # Need to determine which edges were kept
            unit_nodes_in_edges = data.edge_index[0] < data.n_units
            attr_nodes_in_edges = data.edge_index[0] >= data.n_units
            
            unit_to_attr = unit_nodes_in_edges & mask[data.edge_index[0].clamp(max=data.n_units-1)]
            attr_to_unit = attr_nodes_in_edges & mask[data.edge_index[1].clamp(max=data.n_units-1)]
            
            edge_mask = unit_to_attr | attr_to_unit
            train_data.edge_attr = data.edge_attr[edge_mask]
    
    # 8. Filter relational edges (unit <-> unit only)
    if hasattr(data, 'rel_edge_index') and data.rel_edge_index is not None:
        # Relational edges only involve units (indices 0 to n_units-1)
        rel_edge_mask = mask[data.rel_edge_index[0]] & mask[data.rel_edge_index[1]]
        filtered_rel_edges = data.rel_edge_index[:, rel_edge_mask]
        
        # Remap unit indices (0 to n_units_kept-1)
        unit_mapping = torch.cumsum(mask, dim=0) - 1
        train_data.rel_edge_index = unit_mapping[filtered_rel_edges]
        
        # Filter relational edge types
        if hasattr(data, 'rel_edge_type') and data.rel_edge_type is not None:
            train_data.rel_edge_type = data.rel_edge_type[rel_edge_mask]
    
    # 9. Filter adjacency matrix (for network baselines)
    # if hasattr(data, 'A') and data.A is not None:
    #     # Handle sparse tensor - only involves units
    #     if hasattr(data.A, 'to_dense'):
    #         adj_dense = data.A.to_dense().cpu().numpy()
    #     else:
    #         adj_dense = data.A.cpu().numpy()
        
    #     # Filter adjacency matrix (units only)
    #     filtered_adj = adj_dense[np.ix_(mask_np, mask_np)]
        
    #     # Convert back to same format as original
    #     if hasattr(data.A, 'to_dense'):
    #         indices = torch.nonzero(torch.tensor(filtered_adj)).t()
    #         values = torch.tensor(filtered_adj)[torch.nonzero(torch.tensor(filtered_adj), as_tuple=True)]
    #         train_data.A = torch.sparse_coo_tensor(
    #             indices=indices,
    #             values=values,
    #             size=filtered_adj.shape
    #         ).to(data.A.device)
    #     else:
    #         train_data.A = torch.tensor(filtered_adj, device=data.A.device, dtype=data.A.dtype)
    
    # 10. Filter tabular data (numpy arrays) - units only
    if hasattr(data, 'arr_X') and data.arr_X is not None:
        train_data.arr_X = data.arr_X[mask_np]
    
    if hasattr(data, 'arr_YF') and data.arr_YF is not None:
        train_data.arr_YF = data.arr_YF[mask_np]
    
    if hasattr(data, 'arr_Y1') and data.arr_Y1 is not None:
        train_data.arr_Y1 = data.arr_Y1[mask_np]
    
    if hasattr(data, 'arr_Y0') and data.arr_Y0 is not None:
        train_data.arr_Y0 = data.arr_Y0[mask_np]
    
    # Filter multi-dimensional adjacency matrix (units only)
    if hasattr(data, 'arr_Adj') and data.arr_Adj is not None:
        if len(data.arr_Adj.shape) == 2:  # Single adjacency matrix
            train_data.arr_Adj = data.arr_Adj[np.ix_(mask_np, mask_np)]
        elif len(data.arr_Adj.shape) == 3:  # Multi-relational adjacency matrices
            # Convert boolean mask to indices first:
            node_indices = np.where(mask_np)[0]
            # For 3D arrays [R, N, N], filter the last two dimensions:
            train_data.arr_Adj = data.arr_Adj[:, node_indices, :][:, :, node_indices]
            # train_data.arr_Adj = data.arr_Adj[:, np.ix_(mask_np, mask_np)]
    
    # # 11. Filter dataframes (units only)
    # if hasattr(data, 'df_full') and data.df_full is not None:
    #     train_data.df_full = data.df_full[mask_np].reset_index(drop=True)
    
    # if hasattr(data, 'df_miss') and data.df_miss is not None:
    #     train_data.df_miss = data.df_miss[mask_np].reset_index(drop=True)
    
    # if hasattr(data, 'df_imputed') and data.df_imputed is not None:
    #     train_data.df_imputed = data.df_imputed[mask_np].reset_index(drop=True)
    
    return train_data

# If GPU is to be used
dtype_int = tr.cuda.IntTensor if tr.cuda.is_available() else tr.IntTensor            # tr.int32
dtype_long = tr.cuda.LongTensor if tr.cuda.is_available() else tr.LongTensor         # tr.int64
dtype_float = tr.cuda.FloatTensor if tr.cuda.is_available() else tr.FloatTensor      # tr.float
dtype_double = tr.cuda.DoubleTensor if tr.cuda.is_available() else tr.DoubleTensor   # tr.double


def weighted_mse_loss(inp, target, weights):

    return tr.mean(((inp - target)**2) * weights)


def to_float(t):
    # Map to torch FloatTensor, check CUDA
    # Input t is already torch DoubleTensor
    if isinstance(t, np.ndarray):
        t = tr.from_numpy(t)
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    return t.type(tr.cuda.FloatTensor).to(device) if tr.cuda.is_available() else t.type(tr.FloatTensor).to(device)


def to_double(t):
    if isinstance(t, np.ndarray):
        t = tr.from_numpy(t)
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    return t.type(tr.cuda.DoubleTensor).to(device) if tr.cuda.is_available() else t.type(tr.DoubleTensor)


def to_int(t):
    if isinstance(t, np.ndarray):
        t = tr.from_numpy(t)
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    return t.type(tr.cuda.IntTensor).to(device) if tr.cuda.is_available() else t.type(tr.IntTensor).to(device)


def to_long(t):
    # return t
    if isinstance(t, np.ndarray):
        t = tr.from_numpy(t)
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    return t.type(tr.cuda.LongTensor).to(device) if tr.cuda.is_available() else t.type(tr.LongTensor).to(device)


def coo_to_gnn_tensor(adj):
    # Convert coo adjacency to torch tensor for GNN
    # Notice that pytorch geometric requires the edge_index to be LongTensor
    adj = tr.stack([tr.tensor(adj.row), tr.tensor(adj.col)], dim=0)
    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    return adj.type(tr.cuda.LongTensor).to(device) if tr.cuda.is_available() else adj.type(tr.LongTensor).to(device)


def coo_to_sparse_tensor(adj, ent_ones):
    # Convert coo adjacency to torch sparse tensor in coo

    indices = to_long(np.vstack([adj.row, adj.col]))
    if ent_ones:
        data = to_float(np.ones(adj.data.shape[0]))
    else:
        data = to_float(adj.data)

    device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
    adj_sparse = tr.sparse_coo_tensor(indices, data, size=adj.shape, device=device)
    return adj_sparse


# ===================================================================================================================


def subs_tc_mmd_lin(rep, tc, sample_size=5000):

    subs_ind = to_long(np.random.choice(rep.shape[0], size=sample_size, replace=False))
    subs_rep = rep[subs_ind]
    subs_tc = tc[subs_ind]

    return tc_mmd_lin(subs_rep, subs_tc)


def subs_tc_mmd_rbf(rep, tc, sample_size=5000):

    subs_ind = to_long(np.random.choice(rep.shape[0], size=sample_size, replace=False))
    subs_rep = rep[subs_ind]
    subs_tc = tc[subs_ind]

    return tc_mmd_rbf(subs_rep, subs_tc)


def subs_tc_mmd_hsic(rep, tc, sample_size=5000):
    actual_sample_size = min(rep.shape[0], sample_size)
    subs_ind = to_long(np.random.choice(rep.shape[0], size=actual_sample_size, replace=False))
    
    # subs_ind = to_long(np.random.choice(rep.shape[0], size=sample_size, replace=False))
    subs_rep = rep[subs_ind].float()
    subs_tc = tc[subs_ind].float()

    
    return tc_mmd_hsic(subs_rep, subs_tc)


def tc_mmd_lin(rep, tc):

    t_idx = (tc == 1).nonzero().flatten()
    c_idx = (tc == 0).nonzero().flatten()
    treated_rep = rep[t_idx]
    control_rep = rep[c_idx]

    return mmd_lin(treated_rep, control_rep)


def tc_mmd_rbf(rep, tc, sigma=None):

    t_idx = (tc == 1).nonzero().flatten()
    c_idx = (tc == 0).nonzero().flatten()
    treated_rep = rep[t_idx]
    control_rep = rep[c_idx]

    return mmd_rbf(treated_rep, control_rep, sigma=sigma)


def tc_mmd_hsic(rep, tc, sigma_x=None, sigma_y=None):

    return mmd_hsic(rep, tc, sigma_x=sigma_x, sigma_y=sigma_y)


def mmd_lin(rep1, rep2):
    mmd = ((rep1.mean(0) - rep2.mean(0)) ** 2).mean()
    return mmd


def mmd_rbf(rep1, rep2, sigma=None):

    if sigma is None:
        # Define sigma as in code: https://github.com/romain-lopez/HCV/blob/master/scVI/scVIgenqc.py
        sigma = 2 * math.sqrt(gamma(0.5 * (rep1.shape[1] + 1)) / gamma(0.5 * rep1.shape[1]))

    k12 = tr.exp(- pdist(rep1, rep2) / sigma ** 2)
    k11 = tr.exp(- pdist(rep1, rep1) / sigma ** 2)
    k22 = tr.exp(- pdist(rep2, rep2) / sigma ** 2)

    m1 = rep1.size()[0]
    m2 = rep2.size()[0]
    if m1 > 1:
        d11 = 1 / (m1 * (m1-1)) * (tr.sum(k11) - m1)  # Remove diagonal, see Lemma 6 in 'A Kernel Two-Sample Test'
    else:
        d11 = 0
    if m2 > 1:
        d22 = 1 / (m2 * (m2-1)) * (tr.sum(k22) - m2)
    else:
        d22 = 0
    d12 = - 2.0 / (m1 * m2) * tr.sum(k12)
    mmd = d11 + d22 + d12

    return mmd


def mmd_hsic(x, y, sigma_x=None, sigma_y=None):
    # print("jason 1")
    # print(type(x[0]), type[y[0]])
    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    # Define sigma as in code: https://github.com/romain-lopez/HCV/blob/master/scVI/scVIgenqc.py
    if sigma_x is None:
        sigma_x = 2 * math.sqrt(gamma(0.5 * (x.shape[1] + 1)) / gamma(0.5 * x.shape[1]))
    if sigma_y is None:
        sigma_y = 2 * math.sqrt(gamma(0.5 * (y.shape[1] + 1)) / gamma(0.5 * y.shape[1]))
    # print("jason 2")
    k_xx = tr.exp(- pdist(x, x) / sigma_x ** 2)
    k_yy = tr.exp(- pdist(y, y) / sigma_y ** 2)

    # print("jason 3")
    hsic = 0
    hsic += (k_xx * k_yy).mean()
    hsic += k_xx.mean() * k_yy.mean()
    hsic -= 2 * tr.mean(k_xx.mean(1) * k_yy.mean(1))

    return hsic


def pdist(x, y):
    # Compute squared Euclidean distance between all pairs x in X, y in Y, namely (x - y)^2
    m = -2 * tr.matmul(x, tr.transpose(y, 0, 1))
    sqx = tr.sum(x ** 2, 1, keepdim=True)
    sqy = tr.sum(y ** 2, 1, keepdim=True)
    d = m + tr.transpose(sqy, 0, 1) + sqx
    return d


# ===================================================================================================================


# Differential discrete sampling
# https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/conditional_computation.py
def sample_gumbel_like(template_tensor, eps=1e-10):

    uniform_samples_tensor = template_tensor.clone().uniform_()
    gumble_samples_tensor = - tr.log(eps - tr.log(uniform_samples_tensor + eps))
    return gumble_samples_tensor


def gumbel_softmax_sample(logits, dim, tau=1):

    gumble_samples_tensor = sample_gumbel_like(logits.data)
    gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
    soft_samples = nn.functional.softmax(gumble_trick_log_prob_samples / tau, dim)
    return soft_samples


def gumbel_softmax(logits, dim, hard=False, tau=1):

    samples_soft = gumbel_softmax_sample(logits, tau=tau, dim=dim)
    if hard:
        _, max_value_indexes = samples_soft.data.max(dim, keepdim=True)
        samples_hard = logits.data.clone().zero_().scatter_(dim, max_value_indexes, 1.0)
        return_samples = Variable(samples_hard - samples_soft.data) + samples_soft
    else:
        return_samples = samples_soft
    return return_samples


# ===================================================================================================================


def process_wave_data(dt, train=True, train_mask=None):
    # dt: id, t, exp, y, ycf, mu0, delta_mu, spillover

    if train:
        t, exp, y, y_cf = to_float(dt[:, 1]), to_float(dt[:, 2]), to_float(dt[:, 3]), to_float(dt[:, 4])
        features = to_float(dt[:, 8:])
        y_mean, y_std = tr.mean(y[train_mask]), tr.std(y[train_mask])
        y_standardized = (y - y_mean) / y_std
        return t, exp, y_standardized, features
    else:
        t, exp, y, y_cf = to_float(dt[:, 1]), to_float(dt[:, 2]), to_float(dt[:, 3]), to_float(dt[:, 4])
        delta_mu = to_float(dt[:, 6])
        features = to_float(dt[:, 8:])
        return t, exp, y, delta_mu, features


def process_pokec_data(dt, train=True, train_mask=None):
    # dt: id, t, exp, y, ycf, mu0, delta_mu, spillover, id, features

    assert (dt[:, 0] == dt[:, 8]).all(), 'Simulation and features ids not match'

    if train:
        t, exp, y, y_cf = to_float(dt[:, 1]), to_float(dt[:, 2]), to_float(dt[:, 3]), to_float(dt[:, 4])
        features = to_float(dt[:, 9:])
        y_mean, y_std = tr.mean(y[train_mask]), tr.std(y[train_mask])
        y_standardized = (y - y_mean) / y_std
        return t, exp, y_standardized, features
    else:
        t, exp, y, y_cf = to_float(dt[:, 1]), to_float(dt[:, 2]), to_float(dt[:, 3]), to_float(dt[:, 4])
        delta_mu = to_float(dt[:, 6])
        features = to_float(dt[:, 9:])
        return t, exp, y, delta_mu, features


def process_amazon_data(dt, train=True, train_mask=None):
    # dt: id, t, exp, y, ycf, spillover
    # Notice in Amazon simulation data, spillover effect is not added
    # Notice that even the y is standardized in semi simulation, y + spillover is not yet

    if train:
        t, exp, y, y_cf = to_float(dt[:, 1]), to_float(dt[:, 2]), to_float(dt[:, 3]), to_float(dt[:, 4])
        spillover, features = to_float(dt[:, 5]), to_float(dt[:, 6:])
        ts_gt = y + spillover

        ts_gt_mean, ts_gt_std = tr.mean(ts_gt[train_mask]), tr.std(ts_gt[train_mask])
        ts_gt = (ts_gt - ts_gt_mean) / ts_gt_std
        return t, exp, ts_gt, features
    else:
        t, exp, y, y_cf, spillover, features = dt[:, 1], dt[:, 2], dt[:, 3], dt[:, 4], dt[:, 5], dt[:, 6:]
        mu1, mu0 = y.copy(), y.copy()
        mu1[np.argwhere(t == 0).flatten()] = y_cf[np.argwhere(t == 0).flatten()]
        mu0[np.argwhere(t == 1).flatten()] = y_cf[np.argwhere(t == 1).flatten()]
        delta_mu = mu1 - mu0
        ts_gt = y + spillover
        return to_float(t), to_float(exp), to_float(ts_gt), to_float(delta_mu), to_float(features)


def reassign_adj_gnn(dt, adj_gnn_gt):

    node_id = dt[:, 0]
    id2idx = {node: i for i, node in enumerate(node_id)}

    row = adj_gnn_gt[0].cpu().numpy()
    col = adj_gnn_gt[1].cpu().numpy()
    new_row = [id2idx[row_id] for row_id in row]
    new_col = [id2idx[col_id] for col_id in col]

    new_adj = tr.stack([tr.tensor(new_row), tr.tensor(new_col)], dim=0)
    if tr.cuda.is_available():
        new_adj = new_adj.type(tr.cuda.LongTensor).to('cuda')
    else:
        new_adj = new_adj.type(tr.LongTensor).to('cpu')

    return new_adj


def shuffle_train_data(dt_gt, num_train):
    # Randomly shuffle training data
    dt = dt_gt.copy()

    train_data = dt[:num_train]
    valid_test_data = dt[num_train:]
    np.random.shuffle(train_data)

    return np.vstack((train_data, valid_test_data))


# ===================================================================================================================


def flatten_nested_tuple(tup):

    for t in tup:
        if type(t) is not tuple:
            yield t
        else:
            for element in flatten_nested_tuple(t):
                yield element


def load_configurations(configs):

    cd = {k: getattr(configs, k) for k in dir(configs) if not callable(getattr(configs, k)) and not k.startswith("__")}
    multi = {k: v for k, v in cd.items() if type(v) is tuple}
    single = {k: v for k, v in cd.items() if type(v) is not tuple}
    config_list = []

    if len(multi) == 0:
        config_list.append(single)
    elif len(multi) == 1:
        for v in list(multi.values())[0]:
            conf = {list(multi.keys())[0]: v}
            for k in single.keys():
                conf[k] = single[k]
            config_list.append(conf)
    else:
        multi_confs = list(multi.values())[0]
        for i in range(1, len(multi)):
            multi_confs = list(itertools.product(multi_confs, list(multi.values())[i]))
        for multi_conf in multi_confs:
            multi_conf = list(flatten_nested_tuple(multi_conf))
            conf = {}
            for i, v in enumerate(multi_conf):
                conf[list(multi.keys())[i]] = v
            for k in single.keys():
                conf[k] = single[k]
            config_list.append(conf)

    return config_list


def conf_dict_to_class(config_dict, config_class):
    # Change the dictionary of config list elements into class instance

    for k in dir(config_class):
        if not callable(getattr(config_class, k)) and not k.startswith("__"):
            setattr(config_class, k, config_dict[str(k)])

    return config_class


def write_conf_into_text(config, path):
    # config: a class instance

    config_dict = {k: getattr(config, k) for k in dir(config) if not callable(getattr(config, k))
                   and not k.startswith("__")}

    f = open(path + 'config.txt', 'w')
    for k, v in config_dict.items():
        f.write(str(k) + ' = ' + str(v) + '\n')
    f.close()




# ===================================================================================================================
# class CheckPoint:
#     def __init__(self, config):
#         self.config = config
#         self.best_valid_loss = np.infty
#         self.valid_loss_list = []
#
#     def check(self, epoch, valid_loss, saving, test_loss=None):
#         check_break = False
#
#         if valid_loss < self.best_valid_loss:
#             self.best_valid_loss = valid_loss
#             print('Epoch ', epoch, 'Saving model with valid loss, test loss', valid_loss, test_loss)
#             for s in saving:
#                 callable(s)
#
#         if len(self.valid_loss_list) == self.config.num_tolerance:
#             early_stopping_threshold = None
#             if self.config.early_stopping_op == 'mean':
#                 early_stopping_threshold = np.mean(np.array(self.valid_loss_list))
#             elif self.config.early_stopping_op == 'max':
#                 early_stopping_threshold = np.max(np.array(self.valid_loss_list))
#             if valid_loss > early_stopping_threshold:
#                 check_break = True
#             self.valid_loss_list = []
#         self.valid_loss_list.append(valid_loss)
#
#         return check_break
#
# ===================================================================================================================
# Cyclical learning schedule
# https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
# def cyclical_lr(step_size, min_lr=0.0002, max_lr=0.0005):
#
#     # Additional function to see where on the cycle we are
#     def relative(it, _step_size):
#         cycle = math.floor(1 + it / (2 * _step_size))
#         x = abs(it / _step_size - 2 * cycle + 1)
#         return max(0, (1 - x)) * cycle
#
#     lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, step_size)
#
#     return lr_lambda
#
# ===================================================================================================================
# def decay_lr(step_size, lr, decay):
#
#     lr_lambda = lambda it: decay ** (it / step_size) * lr
#
#     return lr_lambda
#
# ===================================================================================================================
# Fit old individual and exposure policies
# class OldPolicyNetwork(nn.Module):
#     def __init__(self, config):
#         super(OldPolicyNetwork, self).__init__()
#         self.config = config
#
#         # Individual treatment estimator
#         ind_layers = []
#         prev_hidden_size = self.config.features_dim
#         for next_hidden_size in self.config.ind_pol_dims:
#             ind_layers.extend([
#                 nn.Linear(prev_hidden_size, next_hidden_size),
#                 nn.ReLU(),
#             ])
#             prev_hidden_size = next_hidden_size
#         ind_layers.append(
#             nn.Linear(prev_hidden_size, 1)
#         )
#         self.ind_est = nn.Sequential(*ind_layers)
#
#         # Exposure estimator, use treatment assignment for exposure estimation
#         exp_layers = []
#         prev_hidden_size = self.config.features_dim + 1
#         for next_hidden_size in self.config.exp_pol_dims:
#             exp_layers.extend([
#                 nn.Linear(prev_hidden_size, next_hidden_size),
#                 nn.ReLU(),
#             ])
#             prev_hidden_size = next_hidden_size
#         exp_layers.extend([
#             nn.Linear(prev_hidden_size, 1),
#             nn.Sigmoid()
#         ])
#         self.exp_est = nn.Sequential(*exp_layers)
#
#     def forward(self, rep, treatment):
#         est_ind = self.ind_est(rep)
#         est_exp = self.exp_est(tr.cat((rep, treatment[:, None]), dim=1))
#         return tr.squeeze(est_ind), tr.squeeze(est_exp)
#
# ===================================================================================================================
# if self.config.discretize_exposure:
#     z = tr.zeros(tc.shape[0], self.config.num_intervals).to(self.device)
#     g = to_long(g * (self.config.num_intervals - 1))
#     g = z.scatter_(1, g.unsqueeze_(-1), tr.ones((g.shape[0], 1)).to(self.device))
#     # rep = tr.cat((rep, g), dim=1)
#     rep = tr.cat((rep, h_rep, g), dim=1)
#
# ===================================================================================================================
# Find treatment or control response in the neighbors of adj with the same t/c but lowest exposure level
# t, exp = dt[:, 0], dt[:, 1]
# pair_i, pair_j = [], []
# for i in range(t.shape[0]):
#     ngb_idx = adj.col[np.argwhere(adj.row == i).flatten()]
#     ngb_t = t[ngb_idx]
#     ngb_e = exp[ngb_idx]
#     if t[i] in ngb_t:
#         ngb_min_exp = np.min(ngb_e[np.squeeze(np.argwhere(ngb_t == t[i]))])
#         ind_match_t = np.argwhere(ngb_t == t[i]).flatten()
#         ind_match_e = np.argwhere(ngb_e == ngb_min_exp).flatten()
#         ind_match = list(set(list(ind_match_t)) & set(list(ind_match_e)))
#         ind_match_ngb = ngb_idx[ind_match]
#         for j in ind_match_ngb:
#             pair_i.append(i)
#             pair_j.append(j)
# pair_i, pair_j = np.array(pair_i), np.array(pair_j)
# pairs = [pair_i, pair_j]
#
# Balancing treatment / control effects via similar nodes without interference
# pair_i, pair_j = pairs
# bal_loss = tr.mean((t_eff[pair_i] - y[pair_j]) ** 2)
# ===================================================================================================================
# Individual treatment effect estimator
# class TreatmentEffectEstimator(nn.Module):
#     def __init__(self, config):
#         super(TreatmentEffectEstimator, self).__init__()
#         self.config = config
#         self.device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
#
#         if self.config.sep_nets_eff_esti:
#             # Separate estimators for treatment and control
#             t_layers = []
#             prev_hidden_size = self.config.rep_hidden_dims[-1]
#             for next_hidden_size in self.config.treat_eff_dims:
#                 t_layers.extend([
#                     nn.Linear(prev_hidden_size, next_hidden_size),
#                     nn.ReLU(),
#                 ])
#                 prev_hidden_size = next_hidden_size
#             t_layers.extend([
#                 nn.Linear(prev_hidden_size, 1)  # Not normalized treatment effect
#             ])
#             self.ind_t_eff = nn.Sequential(*t_layers)
#
#             c_layers = []
#             prev_hidden_size = self.config.rep_hidden_dims[-1]
#             for next_hidden_size in self.config.treat_eff_dims:
#                 c_layers.extend([
#                     nn.Linear(prev_hidden_size, next_hidden_size),
#                     nn.ReLU(),
#                 ])
#                 prev_hidden_size = next_hidden_size
#             c_layers.extend([
#                 nn.Linear(prev_hidden_size, 1)  # Not normalized control effect
#             ])
#             self.ind_c_eff = nn.Sequential(*c_layers)
#         else:
#             # Estimator for individual treatment and control effect, input [repr, t/c]
#             eff_layers = []
#             prev_hidden_size = self.config.rep_hidden_dims[-1] + 1
#             for next_hidden_size in self.config.treat_eff_dims:
#                 eff_layers.extend([
#                     nn.Linear(prev_hidden_size, next_hidden_size),
#                     nn.ReLU(),
#                 ])
#                 prev_hidden_size = next_hidden_size
#             eff_layers.extend([
#                 nn.Linear(prev_hidden_size, 1)  # Not normalized effects
#             ])
#             self.ind_eff = nn.Sequential(*eff_layers)
#
#     def forward(self, rep, tc):
#         # Input: representations and treatment / control assignments
#         assert rep.shape[0] == tc.shape[0], 'Unmatched size'
#
#         if self.config.sep_nets_eff_esti:
#             ind_eff = tr.zeros((tc.shape[0], 1)).to(self.device)
#             t_idx = (tc == 1).nonzero().flatten()
#             c_idx = (tc == 0).nonzero().flatten()
#             if len(t_idx) != 0:
#                 ind_eff[t_idx] = self.ind_t_eff(rep[t_idx])
#             if len(c_idx) != 0:
#                 ind_eff[c_idx] = self.ind_c_eff(rep[c_idx])
#         else:
#             ind_eff = self.ind_eff(tr.cat((rep, tc.unsqueeze(-1)), dim=1))
#         return tr.squeeze(ind_eff)
#
# ===================================================================================================================
# SpilloverEffectEstimator with sep_spill_esti, separate treatment and control nets
# Later default true, hence delete this option
# class SpilloverEffectEstimator(nn.Module):
#     def __init__(self, config):
#         super(SpilloverEffectEstimator, self).__init__()
#         self.config = config
#         self.device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
#
#         init_dim = self.config.rep_hidden_dims[-1]
#         if self.config.discretize_exposure:
#             init_dim += self.config.num_intervals
#         else:
#             init_dim += 1
#
#         if self.config.sep_spill_esti:
#             spill_t_layers = []
#             prev_hidden_size = init_dim
#             for next_hidden_size in self.config.spillover_dims:
#                 spill_t_layers.extend([
#                     nn.Linear(prev_hidden_size, next_hidden_size),
#                     nn.ReLU(),
#                 ])
#                 prev_hidden_size = next_hidden_size
#             spill_t_layers.extend([
#                 nn.Linear(prev_hidden_size, 1)  # Not normalized spillover effect with T=1 (under treatment)
#             ])
#             self.spill_t = nn.Sequential(*spill_t_layers)
#
#             spill_c_layers = []
#             prev_hidden_size = init_dim
#             for next_hidden_size in self.config.spillover_dims:
#                 spill_c_layers.extend([
#                     nn.Linear(prev_hidden_size, next_hidden_size),
#                     nn.ReLU(),
#                 ])
#                 prev_hidden_size = next_hidden_size
#             spill_c_layers.extend([
#                 nn.Linear(prev_hidden_size, 1)  # Not normalized spillover effect with T=0 (under control)
#             ])
#             self.spill_c = nn.Sequential(*spill_c_layers)
#         else:
#             spill_eff_layers = []
#             prev_hidden_size = init_dim + 1
#             for next_hidden_size in self.config.spillover_dims:
#                 spill_eff_layers.extend([
#                     nn.Linear(prev_hidden_size, next_hidden_size),
#                     nn.ReLU(),
#                 ])
#                 prev_hidden_size = next_hidden_size
#             spill_eff_layers.extend([
#                 nn.Linear(prev_hidden_size, 1)  # Not normalized effects
#             ])
#             self.spill_eff = nn.Sequential(*spill_eff_layers)
#
#     def forward(self, rep, tc, g):
#         # Input: shared representations, treatment / control assignments, exposures
#
#         if self.config.discretize_exposure:
#             z = tr.zeros(tc.shape[0], self.config.num_intervals).to(self.device)
#             g = to_long(g * (self.config.num_intervals - 1))
#             g = z.scatter_(1, g.unsqueeze_(-1), tr.ones((g.shape[0], 1)).to(self.device))
#             rep = tr.cat((rep, g), dim=1)
#         else:
#             rep = tr.cat((rep, g.unsqueeze(-1)), dim=1)
#
#         if self.config.sep_spill_esti:
#             spill_eff = tr.zeros(tc.shape[0], 1).to(self.device)
#             t_idx = (tc == 1).nonzero().flatten()
#             c_idx = (tc == 0).nonzero().flatten()
#             if len(t_idx) != 0:
#                 spill_eff[t_idx] = self.spill_t(rep[t_idx])
#             if len(c_idx) != 0:
#                 spill_eff[c_idx] = self.spill_c(rep[c_idx])
#         else:
#             spill_eff = self.spill_eff(tr.cat((rep, tc.unsqueeze(-1)), dim=1))
#
#         return tr.squeeze(spill_eff)
#
# ===================================================================================================================
# def parse_data(d):
#     par = dict()
#     par['t'] = to_float(d[:, 0])
#     par['exposure'] = to_float(d[:, 1])
#     par['y'] = to_float(d[:, 2])
#     par['y_cf'] = to_float(d[:, 3])
#     par['ind_spill_f'] = to_float(d[:, 4])
#     par['ind_spill_cf'] = to_float(d[:, 5])
#     par['features'] = to_float(d[:, 6:])
#     return par
#
# ===================================================================================================================
# class WeightedMatching(nn.Module):
#     def __init__(self, config):
#         super(WeightedMatching, self).__init__()
#         self.config = config
#
#         # Simple trainable parameters for weighting
#         if self.config.match_mode == 'simple':
#             self.w = tr.nn.Parameter(tr.ones())
#
#     def forward(self):
#         return self.w
#
# ===================================================================================================================
# First predict new individual policy from features, derive exposure policy from new individual policy
# Map to representations based on new individual policy
# new_ind_prob, new_ind, _ = new_pol_net(features)
# rep = rep_net(features, new_ind)
# deg = np.array(adj.sum(axis=1)).flatten()
# deg[(deg == 0.).nonzero()] = 1.
# deg = to_float(deg)
# adj_sparse = coo_to_sparse_tensor(adj, ent_ones=True)
# new_exp = adj_sparse.mm(new_ind.unsqueeze(1)).flatten() / deg
# ts_eff = ts_net(rep, new_ind, new_exp, adj_gnn)
#
# ================================================================================================================
# def adj_see_treated(adj, tc):
#     # Return adj, one node's neighbors are only treated
#
#     new_row, new_col = [], []
#     row, col = adj.row, adj.col
#
#     for i in range(adj.shape[0]):
#         ind = np.argwhere(row == i).flatten()
#         if len(ind) > 0:
#             ngb = col[ind]
#             ngb_t_ind = np.argwhere(tc[ngb] == 1).flatten()
#             if len(ngb_t_ind) > 0:
#                 new_row += [i] * len(ngb_t_ind)
#                 new_col += [ngb[i] for i in ngb_t_ind]
#     new_row = np.array(new_row)
#     new_col = np.array(new_col)
#     new_data = np.ones(new_row.shape)
#     new_adj = coo_matrix((new_data, (new_row, new_col)), shape=(adj.shape[0], adj.shape[0]))
#
#     return new_adj
#
# ================================================================================================================
# lag_optimizer.zero_grad()
# loss_lag.backward(retain_graph=True)
# lag_theta.grad *= -1
# lag_optimizer.step()
#
# loss.backward()
# lr = config.new_pol_lr
# for name, param in new_pol_net.named_parameters():
#     param.data *= (epoch + 1)      # cannot start from 0, otherwise ruin the initialization
#     param.data -= lr * param.grad.data
#     param.data /= (epoch + 2)
#
# ================================================================================================================
# # MSE of new policy, simple individual treatment matching
# rep = rep_net(features, new_ind)
# if config.gnn_fun == 'NN':
#     ts_eff = ts_net(rep, new_ind, new_exp)
# else:
#     ts_eff, _ = ts_net(rep, rep, new_ind, new_exp, adj_gnn)
#
# matched_ind_pol_idx = (old_ind == new_ind).nonzero().flatten()
# tmp_p = np.mean(new_ind[matched_ind_pol_idx].detach().numpy())
# print('Total matched, control and treated in matched ', matched_ind_pol_idx.shape[0] / np.float(old_ind.shape[0]),
#       1 - tmp_p, tmp_p)
#
# ind_pol_weight = to_float(1. / old_ind_prob[to_long(new_ind[matched_ind_pol_idx])])
# kernel_width = config.kernel_width
# weight = tr.exp(- ((new_exp - old_exp) / (2 * kernel_width))**2)
# exp_pol_weight_den = to_float(old_exp_prob[to_long(new_exp * (config.num_intervals - 1))])[matched_ind_pol_idx]
# exp_pol_weight = weight[matched_ind_pol_idx] / (kernel_width * exp_pol_weight_den)
# se = ((ts_eff - y)[matched_ind_pol_idx]) ** 2
# mse_new_pol = tr.mean(ind_pol_weight * exp_pol_weight * se)
#
# ================================================================================================================
# def prepare_ts_estimator(config, model_path, dt_gt, adj):
#     # Return trained rep net adn ts estimator if exists, otherwise train them first
#
#     device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
#     rep_net = Rep(config).to(device)
#     if config.gnn_fun == 'NN':
#         ts_net = SpilloverEffectEstimator(config).to(device)
#     else:
#         ts_net = GraphSpilloverEffectEstimator(config).to(device)
#
#     if not os.path.isfile(model_path + 'rep_net.pt') or not os.path.isfile(model_path + 'ts_net.pt'):
#
#         num_train = int(dt_gt.shape[0] * 0.9)
#         num_valid = int((dt_gt.shape[0] - num_train) / 2)
#         train_mask = tr.zeros(dt_gt.shape[0], dtype=tr.uint8)
#         train_mask[:num_train] = 1
#         valid_mask = tr.zeros(dt_gt.shape[0], dtype=tr.uint8)
#         valid_mask[num_train: num_train + num_valid] = 1
#         test_mask = tr.zeros(dt_gt.shape[0], dtype=tr.uint8)
#         test_mask[num_train + num_valid:] = 1
#         masks = [train_mask, valid_mask, test_mask]
#
#         # Ground truth of adj_gnn before shuffling
#         if config.only_see_treated_ngb:
#             adj_gnn_gt = coo_to_gnn_tensor(adj_see_treated(adj=adj, tc=dt_gt[:, 1]))
#         else:
#             adj_gnn_gt = coo_to_gnn_tensor(adj)
#
#         param = [{'params': rep_net.parameters(), 'lr': config.lr}, {'params': ts_net.parameters(), 'lr': config.lr}]
#         optimizer = optim.Adam(param, weight_decay=config.l2_reg)
#         valid_loss_list = []
#         best_valid_loss = np.infty
#
#         for epoch in range(config.num_epochs + 1):
#             if epoch % 2 == 0:
#                 dt = shuffle_train_data(dt_gt, num_train=num_train)
#                 adj_gnn = reassign_adj_gnn(dt, adj_gnn_gt)
#             ts_effect_train(adj_gnn, dt, train_mask, rep_net, ts_net, optimizer, config)
#
#             if epoch % config.save_epoch == 0:
#                 train_loss, valid_loss, test_loss, pehe_train, pehe_valid, pehe_test, eate_train, eate_valid, eate_test = \
#                     ts_effect_test(adj_gnn_gt, dt_gt, masks, rep_net, ts_net, config)
#                 if valid_loss < best_valid_loss:
#                     best_valid_loss = valid_loss
#                     print('Epoch ', epoch, 'Saving model with valid loss, test loss', valid_loss, test_loss)
#                     tr.save(rep_net.state_dict(), model_path + 'rep_net.pt')
#                     tr.save(ts_net.state_dict(), model_path + 'ts_net.pt')
#                     with open(model_path + 'loss_metrics.txt', 'a') as metrics_file:
#                         metrics_file.write('epoch %d, train %f, valid %f, test %f, pehe train %f, pehe valid %f, '
#                                            'pehe test %f, eate train %f, eate valid %f, eate test %f \n' %
#                                            (epoch, train_loss, valid_loss, test_loss, pehe_train, pehe_valid, pehe_test,
#                                             eate_train, eate_valid, eate_test))
#                 if len(valid_loss_list) == config.num_tolerance:
#                     early_stopping_threshold = np.mean(np.array(valid_loss_list))
#                     valid_loss_list = []
#                     if valid_loss <= early_stopping_threshold:
#                         print('epoch, train loss, valid loss, test loss ', epoch, train_loss, valid_loss, test_loss)
#                     else:
#                         break
#                 valid_loss_list.append(valid_loss)
#
#     rep_net.load_state_dict(tr.load(model_path + 'rep_net.pt'))
#     ts_net.load_state_dict(tr.load(model_path + 'ts_net.pt'))
#
#     return rep_net, ts_net
#
# ================================================================================================================











































