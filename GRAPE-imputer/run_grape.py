# import time
import argparse
# import sys
# import os
# import os.path as osp

import numpy as np
import torch
# import pandas as pd

from training.gnn_mdi import train_gnn_mdi, transform

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 1 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--aggr', type=str, default='mean',)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=16)  # 64
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.7) # 1 - edge dropout rate
    parser.add_argument('--auto_known', action='store_true', default=False)
    parser.add_argument('--loss_mode', type=int, default = 0) # 0: loss on all train edge, 1: loss only on unknown train edge
    parser.add_argument('--valid', type=float, default=0.) # valid-set ratio
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='0')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_prediction', action='store_true', default=False)
    parser.add_argument('--transfer_dir', type=str, default=None)
    parser.add_argument('--transfer_extra', type=str, default='')
    parser.add_argument('--mode', type=str, default='train') # debug

    parser.add_argument('--dataset', type=str, default='Syn', help='Dataset to use for the experiment')
    parser.add_argument('--missing_p', type=float, default=0.3, help='Proportion of missing data (between 0 and 1)')
    
    args = parser.parse_args()
    return args

def main(args, data, save_fn):
    # select device
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(0))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, impute_model = train_gnn_mdi(data, args, device)
    data = transform(data, model, impute_model)
    torch.save(data, save_fn)
    print(f"successfully save {save_fn}")

import os

# if __name__ == '__main__':
#     args = parse()
#     args.epochs = 4
#     print(args)

#     missing_p = args.missing_p
#     data = torch.load("/Users/jason/Documents/Coding Projects/2025_Claude/GRAPE-imputer/datasets/p=0.1_k=9_seed=959.pt", weights_only=False)
#     save_fn='tmp.pt'
#     main(args, data, save_fn=save_fn)


if __name__ == '__main__':
    args = parse()
    # args.epochs = 2
    print(args)

    missing_p = args.missing_p

    random_seed = [919, 930, 70, 213, 526, 706, 36, 569, 294, 300] # Note: Seed is now handled per data file
    data_save_dir = "datasets/exps"
    dataset_data_dir = os.path.join(data_save_dir, args.dataset)
    # Optional: Adjust path if running on specific infrastructure
    balu_dir = '/mnt/vast-kisski/projects/kisski-tib-activecl/BaLu'
    if os.path.exists(balu_dir):
        print(f"Adjusting data directory to: {balu_dir}")
        dataset_data_dir = os.path.join(balu_dir, dataset_data_dir)
    
    incomplete_dir = os.path.join(dataset_data_dir, 'no')
    saved_data_files = [f for f in os.listdir(incomplete_dir) if f.startswith(f"p={missing_p}_") and f.endswith(".pt")]

    if not saved_data_files:
        print(f"Warning: No data files found matching pattern 'p={missing_p}_*.pt' in {incomplete_dir}")
    
    save_dir = os.path.join(dataset_data_dir, 'grape_rel=0.0')
    os.makedirs(save_dir, exist_ok=True) 

    for data_file in saved_data_files:
        save_fn = os.path.join(save_dir, data_file)

        if os.path.exists(save_fn):
            print(f"exists, skip {save_fn}!")
            continue
        
        full_data_path = os.path.join(incomplete_dir, data_file)
        data = torch.load(full_data_path, weights_only=False)

        print(f"load data {full_data_path}")
        # print(f"save data to {save_fn}")
        try :
            main(args, data, save_fn=save_fn)
        except Exception as e:
            print(f"An error occurred: {e}")