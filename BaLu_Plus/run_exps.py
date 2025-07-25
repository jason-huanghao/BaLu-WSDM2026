import os
import torch
import json
# import shutil
import argparse
import logging # Optional: for better logging if needed

# Import ModelTrainer and specific model classes
from src.train_module import ModelTrainer
from src.balu import BaLu
from src.train_utils import filter_dataset

# Enable memory-efficient attention if using PyTorch 2.0+ and applicable
# torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cudnn.benchmark = False # Set False for reproducibility if needed
# torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='BaLu Model Arguments')
    
    parser.add_argument('--dataset', type=str, default='Syn', required=True,
                       help='Dataset name to use for training')
    parser.add_argument('--missing_p', type=float, default=0.0, required=True,
                       help='Missing data probability/percentage')
    parser.add_argument('--model_name', type=str, default='BaLu_Plus',
                       choices=['BaLu_Plus'],  # Add other choices as needed
                       help='Model name to use')

    # Graph convolution types
    parser.add_argument('--gconv', type=str, default='GCN',
                       choices=['GCN', 'GraphSAGE', 'GAT'],
                       help='Graph convolution type for general use')
    parser.add_argument('--rconv', type=str, default='GCN',
                       choices=['GCN', 'GraphSAGE', 'RGCN', 'GAT', 'RGAT'],
                       help='Graph convolution type for relational use')
    
    # Imputer settings
    parser.add_argument('--imputer', type=str, default='BaLu_GRAPE',
                       choices=['GRAPE', 'BaLu_GRAPE', 'IGMC', 'BaLu_IGMC'],
                       help='Type of imputer to use')
    parser.add_argument('--imputer_node_dims', type=int, nargs='+',
                       default=[64, 64],
                       help='List of hidden dimensions for imputer layers')
    
    # Network dimensions and dropout
    parser.add_argument('--edge_dim', type=int, default=16,
                       help='Edge feature dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='General dropout rate')
    parser.add_argument('--rel_dropout', type=float, default=0.0,
                       help='Relational/adjacency dropout rate')
    
    # Interference settings
    parser.add_argument('--interference', type=str, default='GNN',
                       choices=['GNN'],
                       help='Type of interference module')
    parser.add_argument('--interference_node_dims', type=int, nargs='+',
                       default=[64],
                       help='List of hidden dimensions for interference network')
    
    # Outcome representation
    parser.add_argument('--outcome_rep', type=str, default='h_r+X*+h_t',
                       help='Outcome representation components separated by +. ' +
                            'Available: h_r, X*, h_t')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--n_epochs', type=int, default=2000,
                       help='Number of training epochs')
    
    # Data normalization
    parser.add_argument('--norm_y', type=bool, default=True,
                       help='Whether to normalize outcome values')
    
    # Loss function weights
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for outcome loss (y_loss)')
    parser.add_argument('--beta', type=float, default=1e-4,
                       help='Weight for treatment loss (t_loss)')
    parser.add_argument('--gamma', type=float, default=1e-4,
                       help='Weight for balance loss (b_loss)')
    parser.add_argument('--eta', type=float, default=1e-4,
                       help='Weight for imputation loss (impute_loss)')
    
    # Early stopping
    parser.add_argument('--early_stop', type=bool, default=True,
                       help='Whether to use early stopping')
    parser.add_argument('--patience', type=int, default=25,
                       help='Patience for early stopping')
    
    return parser.parse_args()


def make_serializable(data_dict):
    serializable_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            serializable_dict[key] = value.item() if value.numel() == 1 else value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
             serializable_dict[key] = [v.item() for v in value]
        else:
            serializable_dict[key] = value
    return serializable_dict


import torch
import os

def setup_optimal_training(num_threads=None):
    """Setup optimal training configuration for available hardware"""
    
    if num_threads is None:
        num_threads = min(os.cpu_count(), 8)  # Cap at 8 to avoid overhead
    print("use", num_threads, "cpu")
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    
    # Enable MKL-DNN optimizations (helps CPU ops even during GPU training)
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    
    # GPU-specific optimizations
    if torch.cuda.is_available():
        print(f"GPU detected - Using GPU with {num_threads} CPU threads for hybrid operations")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # Find best algorithms
        torch.backends.cudnn.deterministic = False 
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
    else:
        print(f"No GPU detected - Using {num_threads} CPU threads for training")
        # CPU-only optimizations
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
            torch.backends.mkl.enabled = True


if __name__ == '__main__':
    setup_optimal_training()
    args = parse_args()
    
    # 
    dataset = args.dataset
    missing_p = args.missing_p

    random_seed = [919, 930, 70, 213, 526, 706, 36, 569, 294, 300] # Note: Seed is now handled per data file
    data_save_dir = "datasets/exps"
    output_dir = 'results' # Use a different output dir to avoid conflicts
    os.makedirs(output_dir, exist_ok=True)

    dataset_data_dir = os.path.join(data_save_dir, dataset)
    
    # Optional: Adjust path if running on specific infrastructure
    balu_dir = '/mnt/vast-kisski/projects/kisski-tib-activecl/BaLu'
    if os.path.exists(balu_dir):
        print(f"Adjusting data directory to: {balu_dir}")
        dataset_data_dir = os.path.join(balu_dir, dataset_data_dir)

    if args.model_name == 'BaLu_Plus':
        model_class = BaLu

    # full: BaLu/exps/dataset/full
    complete_dir = os.path.join(dataset_data_dir, 'full')
    saved_data_files = [f for f in os.listdir(complete_dir) if f.startswith(f"p={missing_p}_") and f.endswith(".pt")]
    
    # no: BaLu/exps/dataset/no
    incomplete_dir = os.path.join(dataset_data_dir, 'no')
    saved_data_files += [f for f in os.listdir(incomplete_dir) if f.startswith(f"p={missing_p}_") and f.endswith(".pt")]
    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)


    ########################################################  Save_Folder  ############################################################

    # method_name = f"{args.model_name.capitalize()}_imp={args.imputer}" #_inf={args.interference}_gconv={args.gconv}_rconv={args.rconv}" 

    method_name = f"{args.model_name.capitalize()}_imp={args.imputer}L={'-'.join([str(e) for e in args.imputer_node_dims])}_K={'-'.join([str(e) for e in args.interference_node_dims])}_gconv={args.gconv}_rconv={args.rconv}_reldrop={args.rel_dropout}_beta={args.beta}_gamma={args.gamma}_eta={args.eta}" #_inf={args.interference}_gconv={args.gconv}_rconv={args.rconv}" 
    
    ###################################################################################################################################
    

    print(f"\n--- Processing Method: {method_name} ---")
    method_output_dir = os.path.join(output_dir, dataset, method_name)
    os.makedirs(method_output_dir, exist_ok=True)

    for data_file in saved_data_files:
        data_fname_base = data_file.replace(".pt", "") # e.g., p=0.3_k=0_seed=919
        try:
            parts = data_fname_base.split('_')
            k_part = [p for p in parts if p.startswith('k=')][0]
            seed_part = [p for p in parts if p.startswith('seed=')][0]
            seed_from_fname = int(seed_part.split('=')[1])
        except IndexError:
            print(f"  Warning: Could not extract seed from filename {data_file}. Using default seed 0.")
            seed_from_fname = 0 

        ##################################################  BaLu/exps/dataset/full or no #############################################
        full_data_path = os.path.join(complete_dir, data_file) if 'p=0.0' in data_file else os.path.join(incomplete_dir, data_file)
        print(f"\n  Processing data file: {full_data_path} (Seed: {seed_from_fname})")
        ##############################################################################################################################

        print(f"run on dataset: {data_fname_base}")
        # model_save_path = os.path.join(method_output_dir, f"{data_fname_base}_model.pt")
        # history_save_path = os.path.join(method_output_dir, f"{data_fname_base}_history.json")
        # train_results_save_path = os.path.join(method_output_dir, f"{data_fname_base}_train_results.json")
        test_results_save_path = os.path.join(method_output_dir, f"{data_fname_base}_test_results.json")
        complete_test_results_save_path = os.path.join(method_output_dir, f"{data_fname_base}_complete_test_results.json")
        
        
        # Check if final results for this specific data file already exist
        if os.path.exists(complete_test_results_save_path):
            print(f"{complete_test_results_save_path} Results for {method_name} on data {data_file} already exist. Skipping.")
            continue
            
        data = torch.load(full_data_path, weights_only=False) # Set weights_only=False if loading models/data with code
        try:
            print(f"  Instantiating ModelTrainer for {model_class.__name__}...")
            params = vars(args)
            trainer = ModelTrainer(model_class, params=params, seed=seed_from_fname)
            
            ######################################## only use train and validation set in training phase ########################################
            val_train_mask = data.train_mask | data.val_mask
            train_data = filter_dataset(data, val_train_mask)
            trained_model = trainer.train(train_data)  # Returns the best model based on validation loss (if ES enabled)
            #####################################################################################################################################
            
            test_results = trainer.evaluate(data, unit_indexes=data.test_mask, on_complete_test=False)
            test_results_complete = trainer.evaluate(data, unit_indexes=data.test_mask, on_complete_test=True)
            
            # print(test_results, test_results_complete, train_results)
            # history = trainer.get_history()
            
            print(f"  Evaluation completed. Saving results...")

            # serializable_history = make_serializable(history)
            # with open(history_save_path, 'w') as f:
            #     json.dump(serializable_history, f, indent=4)
            
            serializable_test_results = make_serializable(test_results)
            with open(test_results_save_path, 'w') as f:
                json.dump(serializable_test_results, f, indent=4)
                print(f"  Test incomplete Results: {serializable_test_results}")
            
            serializable_complete_test_results = make_serializable(test_results_complete)
            with open(complete_test_results_save_path, 'w') as f:
                json.dump(serializable_complete_test_results, f, indent=4)
                print(f"  Test complete Results: {serializable_complete_test_results}")

            print(f"  Results saved successfully.")

        except Exception as e:
            logging.exception(f"  Error during training/evaluation for {method_name} on {data_file}: {e}") # Log traceback
            print(f"  Error during training/evaluation for {method_name} on {data_file}: {e}")

        finally:
            # Clean up data object and clear CUDA cache
            if 'data' in locals():
                del data
            if 'trainer' in locals():
                del trainer # Remove trainer instance
            if 'trained_model' in locals():
                del trained_model
            torch.cuda.empty_cache()
