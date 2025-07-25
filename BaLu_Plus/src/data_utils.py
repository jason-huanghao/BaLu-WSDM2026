import os
import torch
import numpy as np
# from torch_geometric.data import Data
# import pandas as pd
# import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
import scipy.sparse as sp
import scipy.io as sio

def z_normalize(data, epsilon=1e-8):
    """
    Performs z-normalization (standardization) on each dimension of a numpy array.
    
    Args:
        data: numpy array of shape (n_samples, n_dimensions)
        epsilon: small constant to avoid division by zero (default: 1e-8)
        
    Returns:
        normalized_data: numpy array of the same shape as input, with each dimension normalized
                        to have zero mean and unit variance
    """
    # Calculate mean along each dimension (axis=0)
    mean = np.mean(data, axis=0, keepdims=True)
    
    # Calculate standard deviation along each dimension (axis=0)
    std = np.std(data, axis=0, keepdims=True)
    
    # Add epsilon to avoid division by zero
    std = np.maximum(std, epsilon)
    
    # Perform z-normalization
    normalized_data = (data - mean) / std
    
    return normalized_data

def generate_mask(n:int, miss_p:float):
    k = int(n*miss_p)
    random_scores = torch.rand(n)
    _, topk_indices = torch.topk(random_scores, k)
    mask = torch.ones(n, dtype=torch.bool)
    mask[topk_indices] = False
    return mask

def create_node_feature(n_units, n_attrs):
    """
    Create node features for the bipartite graph.
    
    Args:
        n_units: Number of unit nodes
        n_attrs: Number of attribute nodes
        
    Returns:
        x: Node feature tensor
        is_unit: Boolean tensor indicating if a node is a unit node
    """
    # Unit nodes: ones
    # Attribute nodes: one-hot encoding
    x = torch.zeros((n_units + n_attrs, n_attrs))
    x[:n_units] = torch.ones((n_units, n_attrs))  # Unit nodes
    for i in range(n_attrs):
        x[n_units + i, i] = 1.0  # One-hot encoding for attribute nodes

    is_unit = torch.zeros(n_units + n_attrs, dtype=torch.bool)
    is_unit[:n_units] = True

    return x, is_unit


def adjs_to_edge_index(arr_Adj):
    adjacency_bool = arr_Adj.astype(bool)
    aggregated_bool = np.logical_or.reduce(adjacency_bool, axis=0).astype(arr_Adj.dtype)
    adj_t = torch.from_numpy(aggregated_bool)
    edge_index, weights = dense_to_sparse(adj_t)
    return edge_index, weights


def edge_index_to_sparse_coo(edge_index, size, device=None):
    """
    Convert edge_index back to a sparse COO tensor
    
    Args:
        edge_index: 2D tensor of shape [2, nnz] containing the indices
        size: tuple of the output tensor size, default (5000, 5000)
        device: device to place the tensor on, if None uses edge_index device
    
    Returns:
        torch.Tensor: sparse COO tensor
    """
    if device is None:
        device = edge_index.device
    
    # Create values (all ones)
    nnz = edge_index.shape[1]  # nnz = 2298896
    values = torch.ones(nnz, device=device)
    
    # Create sparse COO tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices=edge_index,
        values=values,
        size=size,
        device=device
    )
    
    return sparse_tensor.coalesce()


def convert_data2mat(data, save_path: str, impute: str='no'):
    mat_dict = {}

    mat_dict['Attributes'] = sp.csc_matrix(data.arr_X)
    mat_dict['X_100'] = sp.csc_matrix(data.arr_X)

    adjacency_bool = data.arr_Adj.astype(bool)
    aggregated_adj = np.logical_or.reduce(adjacency_bool, axis=0).astype(data.arr_Adj.dtype)
    
    mat_dict['Network'] = sp.csc_matrix(aggregated_adj)
    
    # sparse_mx = mat_dict['Network']
    # sparse_mx = sparse_mx.tocoo().astype(np.float32)
    
    mat_dict['T'] = data.treatment.numpy().reshape(-1, 1)       # data.treatment is torch.tensor with shape [n_units]
    mat_dict['Y1'] = data.arr_Y1.reshape(-1, 1)                 # data.arr_Y1 is a numpy.array with shape [n_units]
    mat_dict['Y0'] = data.arr_Y0.reshape(-1, 1)                 # data.arr_Y0 is a numpy.array with shape [n_units]

    # Save the file
    output_path = save_path if save_path.endswith('.mat') else save_path + '.mat'
    sio.savemat(output_path, mat_dict)
    return output_path

def process_edge_index_from_adj(adj: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, int]:
    # Ensure input is a 3D NumPy array
    if not isinstance(adj, np.ndarray) or adj.ndim != 3:
        raise ValueError("Input 'adj' must be a 3-dimensional NumPy array.")

    num_relations = adj.shape[0]

    rel_indices, src_indices, tgt_indices = np.nonzero(adj > 0)

    # Check if any edges were found
    if rel_indices.size > 0:
        rel_edge_index = torch.from_numpy(np.stack([src_indices, tgt_indices])).long()
        rel_edge_type = torch.from_numpy(rel_indices).long()
    else:
        # Handle the case with no edges found
        rel_edge_index = torch.zeros((2, 0), dtype=torch.long)
        rel_edge_type = torch.zeros((0,), dtype=torch.long)

    return rel_edge_index, rel_edge_type, num_relations

def sparse_mx_to_torch_sparse_tensor(sparse_mx, cuda=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)

    shape = torch.Size(sparse_mx.shape)

    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor


def edge_index2adj(edge_index, n_nodes):
    """More efficient version avoiding dense conversion for symmetry and binarization."""
    device = edge_index.device if torch.is_tensor(edge_index) else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Ensure edge_index is torch tensor and on the correct device
    if not torch.is_tensor(edge_index):
        edge_index = torch.from_numpy(edge_index).to(device)
    else:
        edge_index = edge_index.to(device)

    # Add reverse edges for symmetry
    row, col = edge_index
    edge_index_symmetric = torch.cat([edge_index, torch.stack([col, row], dim=0)], dim=1)

    # Create values for each edge (all ones)
    values = torch.ones(edge_index_symmetric.shape[1], dtype=torch.float32, device=device)

    adj_symmetric = torch.sparse_coo_tensor(
        edge_index_symmetric,
        values,
        size=(n_nodes, n_nodes),
        dtype=torch.float32,
        device=device
    )

    adj_symmetric = adj_symmetric.coalesce()

    # Binarize by setting values > 0 to 1.
    # This needs to be done on the sparse values directly.
    # We can get the indices and values after coalescing
    coalesced_indices = adj_symmetric.indices()
    coalesced_values = adj_symmetric.values()

    # Binarize values: any value > 0 becomes 1
    binarized_values = torch.ones_like(coalesced_values)

    # Create the final binarized and symmetric sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        coalesced_indices,
        binarized_values,
        size=(n_nodes, n_nodes),
        dtype=torch.float32,
        device=device
    )

    return sparse_tensor

# def process_edge_index_from_adj(adj):
#     source_nodes = []
#     target_nodes = []
#     rel_edge_type_list = []
    
#     # Process each relation type
#     for k in range(adj.shape[0]):
#         for i in range(adj.shape[1]):
#             for j in range(adj.shape[2]):
#                 if adj[k, i, j] > 0:
#                     # i, j is symmetric
#                     source_nodes.append(i)
#                     target_nodes.append(j)
#                     rel_edge_type_list.append(k)
                    
#     # Convert to tensors
#     if rel_edge_type_list:
#         rel_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
#         rel_edge_type = torch.tensor(rel_edge_type_list, dtype=torch.long)
#     else:
#         rel_edge_index = torch.zeros((2, 0), dtype=torch.long)
#         rel_edge_type = torch.zeros((0,), dtype=torch.long)
    
#     return rel_edge_index, rel_edge_type, adj.shape[0]

def update_observed_mask(test_mask, observed_mask, n_units, n_attrs):
    """
    Efficiently update the observed_mask by setting all attributes for test units as observed.
    
    Args:
        test_mask: Boolean tensor of shape [n_units] indicating which units are in test set
        observed_mask: Original boolean tensor of shape [n_units * n_attrs]
        n_units: Number of units
        n_attrs: Number of attributes per unit
        
    Returns:
        Updated observed_mask with all attributes for test units set to True
    """
    extended_mask = test_mask.unsqueeze(1).expand(n_units, n_attrs)
    reshaped_mask = extended_mask.reshape(-1)
    updated_mask = torch.logical_or(reshaped_mask, observed_mask)
    return updated_mask

def create_data_edge_index(X:np.array, n_units, n_features):
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
    
    # Create edges for observed features
    for i in range(n_units):
        for j in range(n_features):
            # from unit to attribute
            if not np.isnan(X[i, j]):
                source_nodes.append(i)
                target_nodes.append(n_units + j)
                data_edge_attr_list.append(X[i, j])
    
    # from attribute to unit (bidirection)
    source_nodes_new = source_nodes + target_nodes
    target_nodes_new = target_nodes + source_nodes
    data_edge_attr_list += data_edge_attr_list

    # Convert to tensors
    if data_edge_attr_list:
        data_edge_index = torch.tensor([source_nodes_new, target_nodes_new], dtype=torch.long)
        data_edge_attr = torch.tensor(data_edge_attr_list, dtype=torch.float)
    else:
        data_edge_index = torch.zeros((2, 0), dtype=torch.long)
        data_edge_attr = torch.zeros((0,), dtype=torch.float)
    
    return data_edge_index, data_edge_attr


def get_known_mask(missing_prob, edge_num):
    """
    Generate a random mask for known edges.
    
    Args:
        missing_prob: Probability of an edge being known
        edge_num: Number of edges
        
    Returns:
        A boolean mask indicating which edges are known
    """
    known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < missing_prob).view(-1)
    return known_mask


def mask_edge(edge_index, edge_attr, mask, remove_edge):
    """
    Apply a mask to edges.
    
    Args:
        edge_index: Edge indices
        edge_attr: Edge attributes
        mask: Boolean mask
        remove_edge: Whether to remove edges or zero out attributes
        
    Returns:
        Masked edge_index and edge_attr
    """
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.
    return edge_index, edge_attr


def process_path_by_dataset(dataset_name, k, root_dir="datasets/graph"):
    """
    Process file paths based on dataset name and realization number.
    
    Args:
        dataset_name: Name of the dataset (AMZS, Flicker, Youtube)
        k: Simulation realization number (0-9)
        root_dir: Root directory containing the graph datasets
        
    Returns:
        File paths for the datasets
    """
    if dataset_name == "Flickr":
        x_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_{k}_x.npy")
        t_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_{k}_T.npy")
        yf_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_{k}_yf.npy")
        adj_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_{k}_adjs.npy")
        y1_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_{k}_y1_spe.npy")
        y0_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_{k}_y0_spe.npy")
    elif dataset_name == "Youtube":
        x_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_x_{k}.npy")
        t_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_T_{k}.npy")
        yf_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_yf_{k}.npy")
        adj_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_adjs_{k}.npy")
        y1_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_y1_spe_{k}.npy")
        y0_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_y0_spe_{k}.npy")
    elif dataset_name == "AMZS" or dataset_name == "Syn":
        x_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_x.npy")
        t_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_t.npy")
        yf_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_yf.npy")
        adj_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_adjs.npy")
        y1_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_y1_spe.npy")
        y0_path = os.path.join(root_dir, dataset_name, f"{dataset_name}_y0_spe.npy")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return x_path, t_path, yf_path, adj_path, y1_path, y0_path



def edge_dropout(graph_data, dropout_rate=0.3, random_seed=42):
    """
    Apply edge dropout to data edges for model robustness.
    
    Args:
        graph_data: Graph data from construct_balu_graph
        dropout_rate: Dropout rate
        random_seed: Random seed for reproducibility
        
    Returns:
        Graph data with dropped edges
    """
    np.random.seed(random_seed)
    
    assert isinstance(graph_data, Data)
    # PyTorch Geometric Data object
    data_edge_index = graph_data.edge_index
    data_edge_attr = graph_data.edge_attr
    observed_mask = graph_data.observed_mask
    
    # Randomly select edges to keep
    n_edges = data_edge_index.size(1)
    keep_mask = np.random.rand(n_edges) > dropout_rate
    
    # Update data edges
    new_data_edge_index = data_edge_index[:, keep_mask]
    new_data_edge_attr = data_edge_attr[keep_mask]
    new_observed_mask = [observed_mask[i] for i, keep in enumerate(keep_mask) if keep]
    
    # Create a new Data object with the updated edges
    new_graph_data = Data.from_dict(graph_data.to_dict())
    new_graph_data.edge_index = new_data_edge_index
    new_graph_data.edge_attr = new_data_edge_attr
    new_graph_data.observed_mask = new_observed_mask
    
    return new_graph_data


def create_train_test_split(data, test_ratio=0.2, random_seed=42):
    """
    Create train/test split for the graph data.
    
    Args:
        data: PyTorch Geometric Data object
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
        
    Returns:
        A tuple of train and test indices
    """
    np.random.seed(random_seed)
    
    n = data.n_units
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    split_idx = int(n * (1 - test_ratio))
    train_idx = torch.tensor(indices[:split_idx], dtype=torch.long)
    test_idx = torch.tensor(indices[split_idx:], dtype=torch.long)
    
    return train_idx, test_idx


# def load_df2graph(csv_path, treatment_col, outcome_col, covariates_cols=None, 
#                   relation_cols=None, missing_rate=0.0):
#     """
#     Load a CSV file and convert it directly to a PyTorch Geometric Data object.
    
#     Args:
#         csv_path: Path to the CSV file
#         treatment_col: Column name for treatment
#         outcome_col: Column name for outcome
#         covariates_cols: List of columns to use as covariates (if None, use all except treatment and outcome)
#         relation_cols: List of columns to use for creating relations between units
#         missing_rate: Additional missing rate to simulate
        
#     Returns:
#         A PyTorch Geometric Data object
#     """
#     # Load the dataframe
#     df = pd.read_csv(csv_path)
    
#     # Determine covariates columns if not specified
#     if covariates_cols is None:
#         covariates_cols = [col for col in df.columns if col != treatment_col and col != outcome_col]
#         if relation_cols is not None:
#             covariates_cols = [col for col in covariates_cols if col not in relation_cols]
    
#     # Extract treatment and outcome
#     T = df[treatment_col].values
#     YF = df[outcome_col].values
    
#     # Extract covariates
#     X = df[covariates_cols].values
    
#     # Get dimensions
#     n_units = len(df)
#     n_features = len(covariates_cols)
    
#     # Create node features
#     x = create_node_feature(n_units, n_features)
    
#     # Create data edge index
#     data_edge_index, data_edge_attr, observed_mask = create_data_edge_index(X, n_units, n_features)
    
#     # Process relations if specified
#     rel_edge_index_list = []
#     rel_edge_type_list = []
    
#     if relation_cols is not None:
#         for rel_idx, rel_col in enumerate(relation_cols):
#             # Group by relation values
#             for val in df[rel_col].dropna().unique():
#                 # Find units with this value
#                 matching_indices = df[df[rel_col] == val].index.tolist()
                
#                 # Create relations between all pairs
#                 for i in range(len(matching_indices)):
#                     for j in range(i+1, len(matching_indices)):
#                         idx1 = matching_indices[i]
#                         idx2 = matching_indices[j]
                        
#                         # Add bidirectional relations
#                         rel_edge_index_list.append([idx1, idx2])
#                         rel_edge_type_list.append(rel_idx)
#                         rel_edge_index_list.append([idx2, idx1])
#                         rel_edge_type_list.append(rel_idx)
    
#     # Convert to tensors
#     if rel_edge_index_list:
#         rel_edge_index = torch.tensor(rel_edge_index_list, dtype=torch.long).t()
#         rel_edge_type = torch.tensor(rel_edge_type_list, dtype=torch.long)
#     else:
#         rel_edge_index = torch.zeros((2, 0), dtype=torch.long)
#         rel_edge_type = torch.zeros((0,), dtype=torch.long)
    
#     # Prepare treatment and outcome tensors
#     treatment = torch.tensor(T, dtype=torch.float)
#     outcome = torch.tensor(YF, dtype=torch.float)
    
#     # Create masks for observed treatments and outcomes
#     treatment_mask = torch.tensor([not np.isnan(t) for t in T], dtype=torch.bool)
#     outcome_mask = torch.tensor([not np.isnan(y) for y in YF], dtype=torch.bool)
    
#     # Create PyTorch Geometric Data object
#     data = Data(
#         x=x,
#         edge_index=data_edge_index,
#         edge_attr=data_edge_attr,
#         rel_edge_index=rel_edge_index,
#         rel_edge_type=rel_edge_type,
#         treatment=treatment,
#         outcome=outcome,
#         treatment_mask=treatment_mask,
#         outcome_mask=outcome_mask,
#         observed_mask=observed_mask,
#         n_units=n_units,
#         n_attrs=n_features,
#         n_rel_types=len(relation_cols) if relation_cols else 0
#     )
    
#     # Introduce additional missing values if requested
#     if missing_rate > 0:
#         data = edge_dropout(data, dropout_rate=missing_rate, random_seed=42)
    
#     return data
