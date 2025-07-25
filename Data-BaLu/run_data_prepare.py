import torch
from data_util import *
import os
import argparse
from sim_relations import *

def mkdir(path=[]):
    path = [e for e in path if e]
    print(path)
    for i in range(len(path)):
        tmp = os.path.join(*path[:i+1])
        os.makedirs(tmp, exist_ok=True)

def drop_edges(A, drop_rate=0.4):
    A_copy = A.copy()
    K, N, _ = A.shape
    
    for k in range(K):
        # Get upper triangle indices with existing edges
        i, j = np.triu_indices(N, k=1)
        mask = A_copy[k, i, j] != 0
        existing_i, existing_j = i[mask], j[mask]
        
        # Randomly select edges to drop
        n_drop = int(len(existing_i) * drop_rate)
        if n_drop > 0:
            drop_idx = np.random.choice(len(existing_i), n_drop, replace=False)
            drop_i, drop_j = existing_i[drop_idx], existing_j[drop_idx]
            
            # Remove edges (maintain symmetry)
            A_copy[k, drop_i, drop_j] = 0
            A_copy[k, drop_j, drop_i] = 0
    
    return A_copy

def process_dataset(dataset, missing_p, miss_pattern, k, seed, src_dir, tar_dir, impute_method, n_attr, relation_sim=0, n_rel=1, drop_rel_ratio=0.0):
    """Process a single dataset configuration"""
    
    if "graph" in src_dir:
        X, A, T, Y, Y1, Y0 = load_multi_relational_data(dataset, k=k, root_dir=src_dir)
    elif "network" in src_dir:
        fn = os.path.join(src_dir, dataset, f"BlogCatalog{k}.mat" if "BlogCatalog" in dataset else f"Flickr{k}.mat")
        X, A, T, Y, Y1, Y0 = load_network_data(fn)
    else:
        raise Exception(f"No such directory {src_dir}")
    
    # dimension reduction
    M = n_attr if X.shape[1] > n_attr else None 
    if M is not None and M < X.shape[1]:
        print(f"reduce dimension of {dataset} from {X.shape[1]} to {M}")
        X = reduce_dimensions_pca(X, n_components=M)

    # simulate relationships and outcomes based on X, relation_sim, n_rel
    if relation_sim > 0:
        X, A, T, Y, Y1, Y0 = sim_rel_outcomes(relation_sim, X, T, n_rel)

    if drop_rel_ratio > 1e-6:   # drop certain edges
        A = drop_edges(A, drop_rate=drop_rel_ratio)

    new_data_name = dataset+f"_M={M}_SimRel={relation_sim}_Rel={n_rel}_{miss_pattern}"
    print("save to dir: ",new_data_name)
    save_fn = os.path.join(tar_dir, new_data_name, impute_method, f'p={missing_p}_k={k}_seed={seed}.pt')
    if os.path.exists(save_fn):
        print(f"Already exists: {save_fn}")
        return
    # Create dataset object
    data_obj = create_dataset(X, A, T, Y, Y1, Y0, seed, missing_p, miss_pattern, impute_method)
    
    # Ensure directory exists and save
    mkdir(os.path.split(tar_dir) + (new_data_name, impute_method))
    torch.save(data_obj, save_fn)
    print(f"Saved: {save_fn}")

def main(dataset, missing_p_values=None, miss_pattern='MCAR', root_dir=None, seeds=None, n_attr=200, imputer=None, relation_sim=0, n_rel=1, drop_rel_ratio=0.0):
    # Set default values if not provided
    if missing_p_values is None:
        missing_p_values = [0.0, 0.1, 0.3, 0.5]
    
    if root_dir is None:
        root_dir = '/mnt/vast-kisski/projects/kisski-tib-activecl/BaLu/datasets'
    
    if seeds is None:
        seeds = [726, 925, 115, 699, 929, 293, 740, 209, 532, 194]
                #  ,460, 667, 230, 497, 656, 592, 390, 311, 201, 959]
    
    # Determine source and target directories
    src_dir = 'graph' if dataset in ['AMZS', "Flickr", "Syn", "Youtube"] else "network"
    tar_dir = 'exps'
    
    if os.path.exists(root_dir):
        src_dir = os.path.join(root_dir, src_dir)
        tar_dir = os.path.join(root_dir, tar_dir)
    
    print(f"Processing dataset: {dataset}")
    print(f"Source directory: {src_dir}")
    print(f"Target directory: {tar_dir}")
    
    # Process each missing probability value
    for missing_p in missing_p_values:
        print(f"Processing missing probability: {missing_p}")
        
        if missing_p < 1e-6:  # Handle complete data case
            impute_methods = ['full']
        else:
            impute_methods = ['ori_grape', 'no', 'mean', 'knn', 'mice', 'missforest', 'gain', 'grape']
        
        if imputer is not None:
            impute_methods = [imputer]

        for impute_method in impute_methods:
            print(f"  Imputation method: {impute_method}")
            
            for i, seed in enumerate(seeds):
                k = i % 10  # Use modulo to cycle through k values
                try:
                    process_dataset(dataset, missing_p, miss_pattern, k, seed, src_dir, tar_dir, impute_method, n_attr, relation_sim, n_rel, drop_rel_ratio)
                except Exception as e:
                    print(f"Exists excpetion {e}")

# if __name__ == "__main__":
#     process_dataset(dataset='Flickr', 
#                     missing_p=0.0, 
#                     miss_pattern='MAR', 
#                     k=1, seed=312, 
#                     src_dir='datasets/graph', tar_dir='', 
#                     impute_method='mean', 
#                     n_attr=20, relation_sim=0, n_rel=4)


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process datasets with various missing value configurations')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Flickr1, BlogCatalog)')
    parser.add_argument('--missing_p', type=float, nargs='+', default=[0.0, 0.1, 0.3], #c, 0.5], 
                        help='Missing probabilities (default: 0.0 0.1 0.3 0.5)')
    parser.add_argument("--miss_pattern", type=str, choices=['MCAR', 'MAR', "MNAR"], default='MCAR')
    parser.add_argument('--root_dir', type=str, 
                        default='/mnt/vast-kisski/projects/kisski-tib-activecl/BaLu/datasets',
                        help='Root directory for datasets')
    parser.add_argument('--seeds', type=int, nargs='+', 
                        default=[726, 925, 115, 699, 929, 293, 740, 209, 532, 194], 
                                # 460, 667, 230, 497, 656, 592, 390, 311, 201, 959],
                        help='Random seeds to use')
    parser.add_argument('--imputer', type=str, default=None)
    parser.add_argument("--n_attr", type=int, default=200)
    parser.add_argument("--relation_sim", type=int, default=0)
    parser.add_argument("--n_rel", type=int, default=1)
    parser.add_argument("--drop_rel_ratio", type=float, default=0.0)
    
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args.dataset, args.missing_p, args.miss_pattern, args.root_dir, args.seeds, args.n_attr, args.imputer, args.relation_sim, args.n_rel, args.drop_rel_ratio)