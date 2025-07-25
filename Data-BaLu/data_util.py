import scipy.io as sio
import torch
import os
import numpy as np
from torch_geometric.data import Data
import pandas as pd
from hyperimpute.plugins.utils.simulate import simulate_nan
import scipy.sparse as sp
from hyperimpute.plugins.imputers import Imputers
from sklearn.impute import KNNImputer
from missforest import MissForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from balu_grape import balu_grape_imputer
from grape import grape_imputer

###############################################################################################
#                             Load Data
###############################################################################################

def load_multi_relational_data(dataset_name, k, root_dir="multi-relations"):
    """
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
    
    X = np.load(x_path)
    T = np.load(t_path).reshape(-1)     # from [n, 1] to [n]
    Y = np.load(yf_path).reshape(-1)   # from [n, 1] to [n]
    A = np.load(adj_path)
    Y1 = np.load(y1_path)
    Y0 = np.load(y0_path)
    
    return X, A, T, Y, Y1, Y0

def load_network_data(f_path):
    data = sio.loadmat(f_path)
    A = data['Network'] # csr matrix
    X = data['Attributes'].todense().A      # csc matrix: .A is to return np.array, otherwise np.matrix
    Y1 = np.squeeze(data['Y1'])
    Y0 = np.squeeze(data['Y0'])
    T = np.squeeze(data['T'])
    Y = np.where(T > 0.5, Y1, Y0)

    # A1 = sparse_mx_to_torch_sparse_tensor(A)
    # A2 = numpy_adj_to_binary_sparse_tensor(A_3d[0])
    A_dense = A.toarray()                   # This gives you a (N, N) numpy.ndarray
    A_3d = np.expand_dims(A_dense, axis=0)  # Shape becomes (1, N, N)
    
    return X, A_3d, T, Y, Y1, Y0


###############################################################################################
#                   Missing Data and Imputing Data
###############################################################################################

def generate_miss_mask(n: int, miss_p: float, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    mask = np.ones(n, dtype=bool)
    indices_to_mask = np.random.choice(n, size=int(n * miss_p), replace=False)
    mask[indices_to_mask] = False
    return ~mask

def missing_pattern_simulate(X, seed, use2D=False, missing_prob=0.0, miss_pattern='MCAR'):
    if missing_prob == 0.0:
        miss_mask = np.zeros(X.shape, dtype=bool)
        miss_mask = miss_mask if use2D else miss_mask.flatten()
        miss_T = np.zeros(X.shape[0], dtype=bool)
        miss_Y = np.zeros(X.shape[0], dtype=bool)
        
        return miss_mask.astype(bool), miss_T, miss_Y
    np.random.seed(seed)
    if miss_pattern == 'MCAR':
        res = simulate_nan(X, missing_prob, "MCAR")
    elif miss_pattern == "MAR":
        res = simulate_nan(X, missing_prob, "MAR")
    elif miss_pattern == "MNAR":
        res = simulate_nan(X, missing_prob, "MNAR")

    miss_mask = res['mask'] 
    miss_mask = miss_mask if use2D else miss_mask.flatten()
    
    miss_T = generate_miss_mask(int(X.shape[0]), missing_prob) 
    miss_Y = generate_miss_mask(int(X.shape[0]), missing_prob) 
    
    return miss_mask.astype(bool), miss_T, miss_Y 

def create_full_miss_dfs(X, T, Y, x_miss, t_miss, y_miss):
    n_units, n_features = X.shape[0], X.shape[1]
    feature_columns = {f"X{i+1}": X[:, i] for i in range(n_features)}
    df_full = pd.DataFrame({"T": T, "Y": Y, **feature_columns})

    df_miss = df_full.copy()
    X_masked = X.copy()
    X_masked[x_miss.reshape((n_units, n_features))] = np.nan
    
    for j in range(n_features):
        df_miss[f"X{j+1}"] = X_masked[:, j]
    df_miss.loc[t_miss, "T"] = np.nan
    df_miss.loc[y_miss, "Y"] = np.nan

    return df_full, df_miss

def extract_arrays_from_df(df_full):
    feature_cols = [col for col in df_full.columns if col.startswith('X')]
    X = df_full[feature_cols].values
    T = df_full['T'].values
    Y = df_full['Y'].values
    return X, T, Y

def train_val_test(n, tr:0.6, ts:0.2):
    n_train = int(n * tr)
    n_test = int(n * ts)
    idx = np.random.permutation(n)
    idx_train, idx_test, idx_val = idx[:n_train], idx[n_train:n_train+n_test], idx[n_train+n_test:]
    return idx_train, idx_test, idx_val

    # train_mask = np.zeros(n, dtype=bool)
    # test_mask = np.zeros(n, dtype=bool)
    # val_mask = np.zeros(n, dtype=bool)
    
    # train_mask[idx_train] = True
    # test_mask[idx_test] = True
    # val_mask[idx_val] = True

    # return train_mask, test_mask, val_mask

def impute_missing(
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    impute_test: bool,
    method: str = 'mean',
    train_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    test_idx: np.ndarray = None,
) -> pd.DataFrame:
    print("imputation method:", method)
    if method in ['no', 'full'] or not df_missing.isna().any().any():
        return df_missing.copy()

    df_imputed = df_missing.copy()
    
    # --- Handle HyperImpute methods ---
    if method in ['mean', 'mice', 'gain']:
        imputer = Imputers().get(method)
        imputer._fit(df_missing.loc[train_idx, :])
        
        # Transform each split
        for idx in [train_idx, val_idx]:
            X_transformed = imputer._transform(df_missing.loc[idx, :])
            
            target_dtype = df_imputed.dtypes.iloc[0]
            df_imputed.loc[idx, :] = pd.DataFrame(
                X_transformed.astype(target_dtype),
                index=df_missing.loc[idx].index, 
                columns=df_missing.columns
            )
        if impute_test:
            X_te = imputer._transform(df_missing.loc[test_idx, :])
            
            df_imputed.loc[test_idx, :] = pd.DataFrame(
                X_te, 
                index=df_missing.loc[test_idx].index, 
                columns=df_missing.columns
            )

    elif method == 'ori_grape':
        imputer = grape_imputer.GrapeImputation(n_features=df_missing.shape[1])
        imputer.fit(df_missing, idx_train=train_idx, idx_val=val_idx)

        for idx in [train_idx, val_idx]:
            X_transformed = imputer.transform(df_missing.loc[idx, :])
            target_dtype = df_imputed.dtypes.iloc[0]
            df_imputed.loc[idx, :] = pd.DataFrame(
                X_transformed.astype(target_dtype),
                index=df_missing.loc[idx].index, 
                columns=df_missing.columns
            )
        if impute_test:
            X_te = imputer.transform(df_missing.loc[test_idx, :])
            df_imputed.loc[test_idx, :] = pd.DataFrame(
                X_te, 
                index=df_missing.loc[test_idx].index, 
                columns=df_missing.columns
            )

    elif method == 'grape':
        print("graph methods used")
        imputer = balu_grape_imputer.GrapeImputation(n_features=df_missing.shape[1])
        imputer.fit(df_missing, idx_train=train_idx, idx_val=val_idx)

        for idx in [train_idx, val_idx]:
            X_transformed = imputer.transform(df_missing.loc[idx, :])
            target_dtype = df_imputed.dtypes.iloc[0]
            df_imputed.loc[idx, :] = pd.DataFrame(
                X_transformed.astype(target_dtype),
                index=df_missing.loc[idx].index, 
                columns=df_missing.columns
            )
        if impute_test:
            X_te = imputer.transform(df_missing.loc[test_idx, :])
            df_imputed.loc[test_idx, :] = pd.DataFrame(
                X_te, 
                index=df_missing.loc[test_idx].index, 
                columns=df_missing.columns
            )

    elif method == 'missforest':
        imputer = MissForest(verbose=0)
        
        # Fit on training data
        train_data = df_missing.loc[train_idx, :].astype(float)
        imputer.fit(train_data)
        
        # Transform each split
        for idx in [train_idx, val_idx]:
            input_data = df_missing.loc[idx, :].astype(float)
            X_transformed = imputer.transform(input_data)
            
            # Create a DataFrame with the transformed data
            transformed_df = pd.DataFrame(
                X_transformed,
                index=df_missing.loc[idx].index,
                columns=df_missing.columns
            )
            
            # Ensure the transformed DataFrame has the same dtype as df_imputed
            transformed_df = transformed_df.astype(df_imputed.dtypes.to_dict())
            
            # Now assign the values
            df_imputed.loc[idx, :] = transformed_df
        
        # Transform test if requested
        if impute_test:
            test_data = df_missing.loc[test_idx, :].astype(float)
            X_te = imputer.transform(test_data)
            
            # Create a DataFrame with the transformed test data
            transformed_test_df = pd.DataFrame(
                X_te,
                index=df_missing.loc[test_idx].index,
                columns=df_missing.columns
            )
            
            # Ensure the transformed DataFrame has the same dtype as df_imputed
            transformed_test_df = transformed_test_df.astype(df_imputed.dtypes.to_dict())
            
            # Now assign the values
            df_imputed.loc[test_idx, :] = transformed_test_df

    # --- Handle KNN imputation using sklearn ---
    elif method == 'knn':
        knn = KNNImputer(n_neighbors=5)
        
        # Fit on train data
        knn.fit(df_missing.loc[train_idx, :])
        
        # Transform each split
        for idx in [train_idx, val_idx]:
            X_transformed = knn.transform(df_missing.loc[idx, :])
            target_dtype = df_imputed.dtypes.iloc[0]
            df_imputed.loc[idx, :] = pd.DataFrame(
                X_transformed.astype(target_dtype),
                index=df_missing.loc[idx].index, 
                columns=df_missing.columns
            )
        
        # Transform test if requested
        if impute_test:
            X_te = knn.transform(df_missing.loc[test_idx, :])
            df_imputed.loc[test_idx, :] = pd.DataFrame(
                X_te, 
                index=df_missing.loc[test_idx].index, 
                columns=df_missing.columns
            )
    else:
        raise ValueError(
            f"Imputation method '{method}' not recognized. "
            "Choose from 'no', 'mean', 'mice', 'gain', 'missforest', 'knn'."
        )
    
    # Restore complete truth for test if not imputing test set
    if not impute_test:
        df_imputed.loc[test_idx, :] = df_complete.loc[test_idx, :]
    df_imputed['T'] = (df_imputed['T'] > 0.5).astype(int)
    return df_imputed


###############################################################################################
#                             Export Data
###############################################################################################

def to_device(**obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (ob.to(device) if isinstance(ob, torch.Tensor) else ob for ob in obs)

def adjs2adj(A):
    """ 
    A.shape: [k, N, N]
    """
    return np.logical_or.reduce(A.astype(bool), axis=0).astype(A.dtype)

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


def numpy_adj_to_torch_sparse_tensor(adj_matrix):
    """
    checked!
    For data in .mat
    """
    rows, cols = np.nonzero(adj_matrix)
    indices = np.stack((rows, cols), axis=0)
    indices = torch.from_numpy(indices.astype(np.int64))
    num_edges = indices.shape[1]
    values = torch.ones(num_edges, dtype=torch.float32)
    
    shape = torch.Size(adj_matrix.shape)
    # sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    
    # if cuda:
    #     sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy.sparse._coo.coo_matrix (represent bi-directed edges) to a torch sparse tensor. 
    checked!
    For data in .mat
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    
    # # sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    # if cuda:
    #     sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor

def train_val_test_miss(n, m, tr, ts, x_miss, t_miss, y_miss):
    if x_miss.ndim == 2:
        complete_cases_miss_x = np.all(~x_miss, axis=1)
    else:
        complete_cases_miss_x = np.all(~x_miss.reshape((n, m)), axis=1)
    
    complete_cases_overall = complete_cases_miss_x & (~t_miss) & (~y_miss)
    all_indices = np.arange(n)
    complete_indices = all_indices[complete_cases_overall]
    incomplete_indices = all_indices[~complete_cases_overall]

    idx_train1, idx_test1, idx_val1 = train_val_test(len(complete_indices), tr=tr, ts=ts)
    idx_train0, idx_test0, idx_val0 = train_val_test(len(incomplete_indices), tr=tr, ts=ts)

    train_idx = np.concatenate([incomplete_indices[idx_train0],complete_indices[idx_train1]])
    test_idx = np.concatenate([incomplete_indices[idx_test0],complete_indices[idx_test1]])
    val_idx = np.concatenate([incomplete_indices[idx_val0],complete_indices[idx_val1]])
    
    print(f"#complete in tr: {len(idx_train1)}, ts:{len(idx_test1)}, val:{len(idx_val1)}")
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    np.random.shuffle(val_idx)

    assert n == len(train_idx) + len(test_idx) + len(val_idx)
    return train_idx, test_idx, val_idx


def create_dataset(X, A, T, Y, Y1, Y0, seed, missing_p,miss_pattern='MCAR', impute_method='no', tr=0.6, ts=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # train_idx, val_idx, test_idx = train_val_test(n=X.shape[0], tr=tr, ts=ts)
    x_miss, t_miss, y_miss = missing_pattern_simulate(X, seed=seed, missing_prob=missing_p, miss_pattern=miss_pattern)

    train_idx, test_idx, val_idx = train_val_test_miss(n=X.shape[0], m=X.shape[1], tr=tr, ts=ts, x_miss=x_miss, t_miss=t_miss, y_miss=y_miss)

    df_full, df_miss = create_full_miss_dfs(X, T, Y, x_miss, t_miss, y_miss)
    
    df_imputed = impute_missing(df_miss, df_full, impute_test=True, method=impute_method, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    
    n_units, n_features = X.shape[0], X.shape[1]

    #####################################   balu baselines: do not include imputed values back to data   ################################################
    x, is_unit = create_node_feature(n_units, n_features)
    observed_mask, treatment_mask, outcome_mask = ~torch.tensor(x_miss, dtype=torch.bool), ~torch.tensor(t_miss, dtype=torch.bool), ~torch.tensor(y_miss, dtype=torch.bool)
    
    rel_edge_index, rel_edge_type, n_rel_types = process_edge_index_from_adj(A)                 # relationships
    data_edge_index, data_edge_attr = create_data_edge_index(X, n_units, n_features)            # attribute values
    assert data_edge_attr.shape[0]//2 == n_units * n_features 
    
    data_edge_mask = torch.cat((observed_mask, observed_mask), dim=0)     # bi-directed: i -> j and j -> i
    obs_data_edge_index, obs_data_edge_attr = mask_edge(data_edge_index, data_edge_attr, data_edge_mask, True)

    treatment, outcome, true_effect = torch.tensor(T, dtype=torch.int), torch.tensor(Y, dtype=torch.float), torch.tensor(Y1 - Y0, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=obs_data_edge_index,
        edge_attr=obs_data_edge_attr
    )
    
    data.rel_edge_index = rel_edge_index
    data.rel_edge_type = rel_edge_type

    data.treatment, data.outcome, data.true_effect = treatment, outcome, true_effect

    data.is_unit = is_unit
    data.n_units, data.n_attrs, data.n_rel_types = n_units, n_features, n_rel_types
    data.node_feature_dim, data.edge_attr_dim = n_features, 1

    data.train_mask, data.val_mask, data.test_mask = torch.zeros(n_units, dtype=torch.bool), torch.zeros(n_units, dtype=torch.bool), torch.zeros(n_units, dtype=torch.bool)
    data.train_mask[train_idx], data.val_mask[val_idx], data.test_mask[test_idx] = True, True, True

    data.observed_mask, data.treatment_mask, data.outcome_mask = observed_mask, treatment_mask, outcome_mask
    
    #####################################  network baselines ################################################
    aggregated_A = np.logical_or.reduce(A.astype(bool), axis=0).astype(A.dtype)
    data.A = numpy_adj_to_torch_sparse_tensor(aggregated_A)
    
    #####################################  tabular baselines (numpy.array) ################################################
    data.arr_X = X
    data.arr_YF = Y
    data.arr_Adj = A
    data.arr_Y1 = Y1
    data.arr_Y0 = Y0
    data.df_full = df_full
    data.df_miss = df_miss
    data.df_imputed = df_imputed
    return data



def reduce_dimensions_pca(X, n_components=2, standardize=False):
    """
    Reduce the dimensionality of array X to n_components dimensions using PCA.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input array of shape (n_samples, n_features)
    n_components : int, default=2
        Number of dimensions to reduce to
    standardize : bool, default=False
        Whether to standardize the data before PCA
        
    Returns:
    --------
    X_reduced : numpy.ndarray
        Reduced array of shape (n_samples, n_components)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    X_copy = X.copy()
    
    if standardize:
        scaler = StandardScaler()
        X_copy = scaler.fit_transform(X_copy)
    
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_copy)
    
    # explained_variance = sum(pca.explained_variance_ratio_)
    # print(f"Explained variance: {explained_variance:.2%}")
    
    return X_reduced

###############################################################################################
#                             Other functions
###############################################################################################

def output_shape(X, A, T, Y, Y1, Y0):
    variables = [X, A, T, Y, Y1, Y0]
    names = ['X', 'A', 'T', 'Y', 'Y1', 'Y0']
    for name, var in zip(names, variables):
        print(f"{name}.shape = {var.shape}, datatype:{var.dtype} {name}.type = {type(var)}, ")


def compare_sparse_tensors(tensor1, tensor2):
    """
    Compare two PyTorch sparse tensors to check if they are equal
    Returns: Dictionary with comparison results
    """
    results = {}
    
    # Check if shapes match
    results['shapes_match'] = (tensor1.shape == tensor2.shape)
    
    # Convert to dense for direct comparison (only for small tensors)
    if np.prod(tensor1.shape) < 100_000_000:  # Only for reasonably sized tensors
        dense1 = tensor1.to_dense()
        dense2 = tensor2.to_dense()
        results['values_equal'] = torch.allclose(dense1, dense2)
        results['max_diff'] = torch.max(torch.abs(dense1 - dense2)).item()
    else:
        # For large tensors, compare indices and values
        # Sort both tensors' indices for comparison
        indices1, values1 = tensor1.coalesce().indices(), tensor1.coalesce().values()
        indices2, values2 = tensor2.coalesce().indices(), tensor2.coalesce().values()
        
        # Check if number of non-zeros match
        results['nnz_match'] = (indices1.shape[1] == indices2.shape[1])
        
        # Check structure (more complex, we can use a sampling approach)
        # Sample some random positions to check
        sample_size = min(1000, indices1.shape[1])
        if indices1.shape[1] > 0:
            random_indices = torch.randint(0, indices1.shape[1], (sample_size,))
            sampled_indices1 = indices1[:, random_indices]
            
            # Check if these indices exist in indices2
            matches = 0
            for i in range(sample_size):
                idx = sampled_indices1[:, i]
                # Check if this index exists in indices2
                # This is a simplification and might be slow for large tensors
                exists = False
                for j in range(indices2.shape[1]):
                    if torch.all(idx == indices2[:, j]):
                        exists = True
                        break
                if exists:
                    matches += 1
            
            results['structure_sample_match_percent'] = matches / sample_size * 100
    
    # Check if tensor1 has ones where tensor2 has ones (binary equality)
    if np.prod(tensor1.shape) < 100_000_000:
        binary1 = (dense1 > 0).float()
        binary2 = (dense2 > 0).float()
        results['binary_structure_equal'] = torch.allclose(binary1, binary2)
    
    return results
