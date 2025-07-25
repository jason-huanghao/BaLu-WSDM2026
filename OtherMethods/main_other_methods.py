import argparse
import numpy as np
import json
import os # Added for path creation
import torch
from models.ml_models import estimate, training
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from collections import defaultdict


def mkdir(path=[]):
    path = [e for e in path if e]
    print(path)
    for i in range(len(path)):
        tmp = os.path.join(*path[:i+1])
        os.makedirs(tmp, exist_ok=True)


def parse():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--nocuda', type=int, default=0,
                        help='Disables CUDA training.')
    parser.add_argument('--dataset', type=str, default='Syn')
    parser.add_argument("--missing_p", type=str, default='0.0')
    parser.add_argument('--em', dest='estimation_model', type=str, choices=["tl", "xl", 'rl', "cf", "dr", 'dml'], default='rl')
    parser.add_argument('--ebm', dest='estimation_base_model', type=str, default='forest') 
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument('--cv', type=int, default=5)
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    args = parser.parse_args()
    return args


def to_device(*obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fixed: Return tuple instead of generator
    return (ob.to(device) if isinstance(ob, torch.Tensor) else ob for ob in obs)

def extract_arrays_from_df(df_full):
    feature_cols = [col for col in df_full.columns if col.startswith('X')]
    X = df_full[feature_cols].values
    T = df_full['T'].values
    Y = df_full['Y'].values
    return X, T, Y



def create_frequency_weighted_adj(rel_edge_index):
    num_nodes = rel_edge_index.max().item() + 1
    # Count occurrences of each edge
    edge_counts = defaultdict(int)
    
    for i in range(rel_edge_index.size(1)):
        src = rel_edge_index[0, i].item()
        dst = rel_edge_index[1, i].item()
        edge_counts[(src, dst)] += 1
    
    # Create indices and values for sparse tensor
    indices = []
    values = []
    
    for (src, dst), count in edge_counts.items():
        indices.append([src, dst])
        values.append(float(1.0))       # weight is 1.0
    
    if indices:
        indices = torch.tensor(indices).t()  # Shape: [2, num_unique_edges]
        values = torch.tensor(values)
    else:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty(0)
    
    adj = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(num_nodes, num_nodes)
    )
    
    return adj

def filter_and_remap_edges(adj_matrix, keep_mask):
    """Improved version of the filter function"""
    keep_mask = keep_mask.bool()
    adj_matrix = adj_matrix.coalesce()
    
    valid_indices = torch.where(keep_mask)[0]
    num_valid_nodes = len(valid_indices)
    
    mapping = torch.full((len(keep_mask),), -1, dtype=torch.long, device=keep_mask.device)
    mapping[valid_indices] = torch.arange(num_valid_nodes, device=keep_mask.device)
    
    adj_indices = adj_matrix.indices()
    adj_values = adj_matrix.values()
    
    valid_edges_mask = keep_mask[adj_indices[0]] & keep_mask[adj_indices[1]]
    
    if valid_edges_mask.sum() == 0:
        filtered_indices = torch.empty((2, 0), dtype=torch.long, device=keep_mask.device)
        filtered_values = torch.empty(0, dtype=adj_values.dtype, device=keep_mask.device)
    else:
        filtered_indices = adj_indices[:, valid_edges_mask]
        filtered_values = adj_values[valid_edges_mask]
        filtered_indices[0] = mapping[filtered_indices[0]]
        filtered_indices[1] = mapping[filtered_indices[1]]
    
    filtered_adj = torch.sparse_coo_tensor(
        indices=filtered_indices,
        values=filtered_values,
        size=(num_valid_nodes, num_valid_nodes),
        dtype=adj_matrix.dtype,
        device=adj_matrix.device
    )
    
    return filtered_adj



def prepare_data(dataset_path, imputed_version=True):
    data = torch.load(dataset_path, weights_only=False)
    # numpy.array
    Y1, Y0 = data.arr_Y1, data.arr_Y0
    assert len(Y1.shape) == len(Y0.shape) == 1    # 2D, 1D, 1D

    if not imputed_version: # or "/no/" in dataset_path:
        X, T, Y = extract_arrays_from_df(data.df_full)
    else:
        X, T, Y = extract_arrays_from_df(data.df_imputed)

    assert len(X.shape) == 2 
    X = torch.tensor(X, dtype=torch.float)
    Y1 = torch.tensor(Y1, dtype=torch.float)
    Y0 = torch.tensor(Y0, dtype=torch.float)
    T = torch.tensor(T, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.float)

    # torch.tensor
    # A = create_frequency_weighted_adj(data.rel_edge_index)
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask

    if imputed_version:
        keep_non_null = ~torch.tensor(data.df_imputed.isnull().any(axis=1).to_numpy(), dtype=torch.bool)
        keep_non_null_and_test = keep_non_null | idx_test

        X = X[keep_non_null_and_test]
        Y1 = Y1[keep_non_null_and_test]
        Y0 = Y0[keep_non_null_and_test]
        T = T[keep_non_null_and_test]
        Y = Y[keep_non_null_and_test]

        # A = filter_and_remap_edges(A, keep_non_null_and_test)

        # Apply indicator to index masks
        idx_train = idx_train[keep_non_null_and_test]
        idx_val = idx_val[keep_non_null_and_test]
        idx_test = idx_test[keep_non_null_and_test]

    assert X.shape[0] == Y0.shape[0] == T.shape[0] == Y.shape[0] == idx_train.shape[0]      #  A.shape[0] ==


    print(f"train complete {idx_train.sum()}\tvalidating complete {idx_val.sum()}")
    # X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = to_device(X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test)
    A = None
    return X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test


def prepare(dataset_path, imputed_version=True):
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_path, imputed_version)
    
    X, T, Y, Y1, Y0, idx_train, idx_val, idx_test = X.numpy(), T.numpy(), Y.numpy(), Y1.numpy(), Y0.numpy(), idx_train.numpy(), idx_val.numpy(), idx_test.numpy()

    # Fixed: Ensure T is binary integer
    T = (T > 0.5).astype(int)
    
    # Separate train, validation, and test data
    X_train = X[idx_train]
    T_train = T[idx_train] 
    Y_train = Y[idx_train]
    
    X_val = X[idx_val]
    T_val = T[idx_val]
    Y_val = Y[idx_val]

    X_test = X[idx_test]
    T_test = T[idx_test]
    Y_test = Y[idx_test]

    # Create data structures for training and validation
    train_data = [X_train, T_train, Y_train]
    val_data = [X_val, T_val, Y_val]
    test_data = [X_test, T_test, Y_test]

    return train_data, val_data, test_data, Y1[idx_test]-Y0[idx_test]


def metics(te_gold, te_test):
    mae_te = np.mean(np.abs(te_test - te_gold))
    mse_te = np.mean((te_test - te_gold) ** 2)
    pehe = np.sqrt(mse_te)
    results = {
        'effect_pehe': pehe,        # RMSE of individual treatment effects
        'effect_mae': mae_te,      # MAE of average treatment effect
    }
    return results

def any_nan(data):
    has_nan = False
    for d_i in data:
        if np.any(np.isnan(d_i)):
            print(True)
            has_nan = True
        else:
            print(False)
    return has_nan

def train_model(dataset_path, one_result_fn, args):
    if os.path.exists(one_result_fn+"_test_results.json"):
    # if os.path.exists(one_result_fn+"_complete_test_results.json"):
        print(f"Skip {one_result_fn}, because existing results")
        return
    
    
    train_data, val_data, test_data, te_gold = prepare(dataset_path)

    # print("train:")
    # if any_nan(train_data):
    #     print("Warning: NaN values in training data")
    # print("validation:")
    # if any_nan(val_data):
    #     print("Warning: NaN values in validation data")
    

    model = training(train_data, val_data, args)

    ############################################## complete test ##############################################
    _, _, test_data, _ = prepare(dataset_path, imputed_version=False)
    print("test:")
    if any_nan(test_data):
        print("Warning: NaN values in test data")
    te_test = estimate(model, test_data, args)
    
    test_results = metics(te_gold, te_test)

    results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}
    with open(one_result_fn+"_test_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print(f'save {one_result_fn+"_test_results.json"}')

    ############################################## incomplete test ##############################################
    # _, _, test_data_imputed, _ = prepare(dataset_path, imputed_version=True)
    # te_test_imputed = estimate(model, test_data_imputed, args)

    # test_results = metics(te_gold, te_test_imputed)
    
    # results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}
    # with open(one_result_fn+"_complete_test_results.json", 'w') as f:
    #     json.dump(results_serializable, f, indent=4)
    # print("Done for one dataset ------------------>")



# if __name__ == '__main__':
#     args = parse()
#     args.em = 'rl'
#     data_path = '/Users/jason/Documents/Coding Projects/2025_Claude/NetDeconf_main_hao/datasets/exps/BlogCatalog/no_p=0.1_k=3_seed=699.pt'
#     one_result_fn = 'results/p=0.1_k=3_seed=699.pt'
#     train_model(data_path, one_result_fn, args)



if __name__ == '__main__':
# if False:
    # import os
    args = parse()

    ########################################## Method Settting  ################################
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    method = args.estimation_model
    dataset_dir = args.dataset
    missing_p = args.missing_p
    ############################################################################################

    balu_dir = '/mnt/vast-kisski/projects/kisski-tib-activecl/BaLu'
    source_dir = 'datasets/exps/'
    if os.path.exists(balu_dir):
        source_dir = os.path.join(balu_dir, source_dir)
    src_dataset = os.path.join(source_dir, dataset_dir)   # exps/dataset

    for impute in os.listdir(src_dataset)[::-1]:
        impute_result_dir = os.path.join(results_dir, dataset_dir, method, impute)  # results/dataset/method/impute/
        mkdir([results_dir, dataset_dir, method, impute])
        for fn in os.listdir(os.path.join(src_dataset, impute)): 
            if f"p={missing_p}" not in fn:                              # filter out file under other missing_p
                continue
            data_path = os.path.join(src_dataset, impute, fn)
            one_result_fn = os.path.join(impute_result_dir, fn)   
            try:
                train_model(data_path, one_result_fn, args)
            except Exception as e:
                print(f"An error occurred: {e}")
            print(f"data:{data_path}\nresult:{one_result_fn}")