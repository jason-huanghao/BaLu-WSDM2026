import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
import copy

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=25, verbose=False, delta=0, device=None, trace_func=print):
        if patience <= 0:
            raise ValueError("Patience must be positive")
            
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
        self.best_state_dict = None
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return
        
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.best_state_dict = copy.deepcopy(model.state_dict())
            if self.verbose:
                self.trace_func(f'Initial best validation loss: {val_loss:.6f}')
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0  # Reset counter since improvement occurred
            if self.verbose:
                self.trace_func(f'New best validation loss: {val_loss:.6f}')
        else:
            # No significant improvement
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func('Early stopping triggered')
    
    def get_best_model(self, model):
        
        if self.best_state_dict is None:
            raise ValueError("No best model state available. Train the model first.")
        
        # Create a copy of the model to avoid modifying the original
        best_model = copy.deepcopy(model)
        best_model.load_state_dict(self.best_state_dict)
        best_model.to(self.device)
        
        return best_model


def filter_and_remap_edges(edge_index, unit_mask, n_units, n_attrs):
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
        train_data.edge_index = filter_and_remap_edges(
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

#######################################
# Loss Functions
#######################################

def compute_imputation_loss(outputs, data, **params):
    use_index = params.get('unit_indexes')

    if use_index is not None:
        counter = 0
        all_edge_idx = torch.zeros_like(data.observed_mask)
        for i in range(len(data.observed_mask)):
            if data.observed_mask[i]:
                all_edge_idx[i] = counter
                counter += 1
        
        # all_edge_idx = torch.tensor(all_edge_idx, dtype=torch.long)
        all_edge_idx = all_edge_idx.clone().to(torch.long)

        data_use_mask = use_index.unsqueeze(1).expand(-1, data.n_attrs).flatten()   # N (len(use_index)) * M (n_attrs)
        use_observe_mask = data.observed_mask & data_use_mask
        observed_edge_idx = all_edge_idx[use_observe_mask]
        
    else:
        use_observe_mask = data.observed_mask
        observed_edge_idx = torch.ones(len(data.edge_attr)//2).to(torch.bool)
    
    true_vals = data.edge_attr[:len(data.edge_attr)//2][observed_edge_idx]
    pred_vals = outputs['imputed_attrs'][use_observe_mask]
    
    if true_vals.numel() == 0 or pred_vals.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=pred_vals.device)
    
    return F.mse_loss(pred_vals, true_vals)



def compute_treatment_loss(outputs, data, unit_indexes=None):
    """
    Compute loss for treatment prediction.
    
    Args:
        outputs: Model outputs containing pred_treatments
        data: Data from load_heter_graph
    
    Returns:
        Treatment prediction loss
    """
    device = data.treatment.device
    treatment = data.treatment
    treatment_mask = data.treatment_mask
    if unit_indexes is not None:
        treatment_mask = data.treatment_mask & unit_indexes
    
    if treatment_mask.sum() > 0:
        # print(outputs['pred_treatments'].dtype, treatment.dtype)
        t_loss = F.binary_cross_entropy(
            outputs['pred_treatments'][treatment_mask],
            treatment[treatment_mask].float()
        )
    else:
        t_loss = torch.tensor(0.0, device=device)
    
    return t_loss


def compute_outcome_loss(outputs, data, unit_indexes=None):
    """
    Compute loss for outcome prediction.
    
    Args:
        outputs: Model outputs containing pred_outcomes
        data: Data from load_heter_graph
    
    Returns:
        Outcome prediction loss
    """
    device = data.outcome.device
    outcome = data.outcome
    outcome_mask = data.outcome_mask

    if unit_indexes is not None:       # train, validataion, test
        outcome_mask = data.outcome_mask & unit_indexes
    
    if outcome_mask.sum() > 0:
        y_loss = F.mse_loss(
            outputs['pred_outcomes'][outcome_mask],
            outcome[outcome_mask]
        )
    else:
        y_loss = torch.tensor(0.0, device=device)
    
    return y_loss


def compute_balance_loss(outputs, data, unit_indexes=None):
    """
    Compute representation balancing loss.
    
    Args:
        outputs: Model outputs
        data: Data from load_heter_graph
        hidden_dim: Dimension of hidden representations
        version: BaLu version (affects how balance loss is computed)
    
    Returns:
        Balance loss measuring representation divergence between treatment groups
    """
    device = data.treatment.device
    treatment = data.treatment
    treatment_mask = data.treatment_mask
    
    # Identify treated and control units
    treated_mask = (treatment > 0.5) & treatment_mask
    control_mask = (treatment <= 0.5) & treatment_mask
    
    if unit_indexes is not None:
        treated_mask &= unit_indexes
        control_mask &= unit_indexes
    
    if treated_mask.sum() > 0 and control_mask.sum() > 0:
        # For Version 2, use representation interference
        treated_rep = outputs['unit_embeddings'][treated_mask]
        # torch.cat([
        #     outputs['interference'][treated_mask],
        #     outputs['unit_embeddings'][treated_mask]
        # ], dim=1)
        
        control_rep = outputs['unit_embeddings'][control_mask] 
        # torch.cat([
        #     outputs['interference'][control_mask],
        #     outputs['unit_embeddings'][control_mask]
        # ], dim=1)
        
        b_loss, _ = wasserstein(treated_rep, control_rep)
        
    else:
        b_loss = torch.tensor(0.0, device=device)
    
    return b_loss

def wasserstein1(x,y,p=0.5,lam=10,its=10,sq=False,backpropT=False,cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]
    
    x = x.squeeze()
    y = y.squeeze()
    
#    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x,y) #distance_matrix(x,y,p=2)
    
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M,10.0/(nx*ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta*torch.ones(M[0:1,:].shape)
    col = torch.cat([delta*torch.ones(M[:,0:1].shape),torch.zeros((1,1))],0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1))/nx,(1-p)*torch.ones((1,1))],0)
    b = torch.cat([(1-p)*torch.ones((ny,1))/ny, p*torch.ones((1,1))],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K/a

    u = a

    for i in range(its):
        u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b/(torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def wasserstein(x, y, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """return W dist between x and y"""
    nx = x.shape[0]
    ny = y.shape[0]
    device = x.device
    
    x = x.squeeze().contiguous()
    y = y.squeeze().contiguous()
    
    # Use torch.cdist for faster distance calculation
    M = torch.cdist(x, y, p=2.0)
    
    # Estimate lambda and delta
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, min(10.0/(nx*ny), 0.5))  # Cap dropout probability
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()
    
    # Pre-allocate augmented distance matrix
    Mt = torch.zeros(nx+1, ny+1, device=device)
    Mt[:nx, :ny] = M
    Mt[nx, :ny] = delta
    Mt[:nx, ny] = delta
    
    # Compute marginals
    a = torch.zeros(nx+1, 1, device=device)
    a[:nx, 0] = p / nx
    a[nx, 0] = 1-p
    
    b = torch.zeros(ny+1, 1, device=device)
    b[:ny, 0] = (1-p) / ny
    b[ny, 0] = p
    
    # Compute kernel
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + 1e-6
    
    # Sinkhorn iterations
    u = a.clone()
    
    for _ in range(its):
        KTu = K.t() @ u
        v = b / KTu.clamp_min(1e-6)
        Kv = K @ v
        u = a / Kv.clamp_min(1e-6)
    
    # Compute transport cost
    P = u @ v.t() * K
    D = 2 * torch.sum(P * Mt)
    
    return D, Mlam

