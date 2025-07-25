# from src.utils import *
from src.utils import *
import torch 
import os
import argparse
import json 
import time
from src.gcn_policy import GCN_HSIC 
import torch.optim as optim
import copy

loss = torch.nn.MSELoss()
if torch.cuda.is_available():
    loss = loss.cuda()

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

def get_rep_loss(rep, t, args):

    t = t.float()
    # Balancing representation !
    rep_loss = 0
    if args.rep_loss_fun == 'lin':
        rep_loss += tc_mmd_lin(rep, tc=t)
    elif args.rep_loss_fun == 'rbf':
        rep_loss += tc_mmd_rbf(rep, tc=t)
    elif args.rep_loss_fun == 'hsic':
        rep_loss += tc_mmd_hsic(rep, tc=t)
    elif args.rep_loss_fun == 'subs_lin':
        rep_loss += subs_tc_mmd_lin(rep, tc=t, sample_size=5000)
    elif args.rep_loss_fun == 'subs_rbf':
        rep_loss += subs_tc_mmd_rbf(rep, tc=t, sample_size=5000)
    elif args.rep_loss_fun == 'subs_hsic':
        rep_loss += subs_tc_mmd_hsic(rep, tc=t, sample_size=5000)

    return rep_loss


def to_device(*obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (ob.to(device) if isinstance(ob, torch.Tensor) else ob for ob in obs)

def extract_arrays_from_df(df_full):
    feature_cols = [col for col in df_full.columns if col.startswith('X')]
    X = df_full[feature_cols].values
    T = df_full['T'].values
    Y = df_full['Y'].values
    return X, T, Y

def create_edge_weights_from_frequency(rel_edge_index):
    """Convert to edge_index and edge_weights for PyG"""
    # Count frequency of each unique edge
    edge_counts = {}
    unique_edges = []
    
    for i in range(rel_edge_index.size(1)):
        edge = (rel_edge_index[0, i].item(), rel_edge_index[1, i].item())
        if edge not in edge_counts:
            edge_counts[edge] = 0
            unique_edges.append(edge)
        edge_counts[edge] += 1
    
    # Create PyG format
    edge_index = torch.tensor(unique_edges).t()  # [2, num_unique_edges]
    edge_weights = torch.tensor([edge_counts[tuple(edge)] for edge in unique_edges])
    
    return edge_index, edge_weights.float()


def prepare_data(dataset_path, imputed_version=True):
    data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    Y1, Y0 = data.arr_Y1, data.arr_Y0

    if not imputed_version: 
        X, T, Y = extract_arrays_from_df(data.df_full)
    else:
        X, T, Y = extract_arrays_from_df(data.df_imputed)

    X = torch.tensor(X, dtype=torch.float)
    Y1 = torch.tensor(Y1, dtype=torch.float)
    Y0 = torch.tensor(Y0, dtype=torch.float)
    T = torch.tensor(T, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.float)
    # A, edge_weights = create_edge_weights_from_frequency(data.rel_edge_index)
    A = data.rel_edge_index
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask

    if imputed_version: 
        # (1) if full: df_imputed = df_full 
        # (2) if no: df_imputed != df_full, and df_imputed has null values
        # (3) if imputed: df_imputed != df_full, and df_imputed has no null values
        print("imputed version!!!!")

        keep_non_null = ~torch.tensor(data.df_imputed.isnull().any(axis=1).to_numpy(), dtype=torch.bool)
        keep_non_null_and_test = keep_non_null | idx_test

        X = X[keep_non_null_and_test]
        Y1 = Y1[keep_non_null_and_test]
        Y0 = Y0[keep_non_null_and_test]
        T = T[keep_non_null_and_test]
        Y = Y[keep_non_null_and_test]

        A, _ = filter_and_remap_edges(A, keep_non_null_and_test)

        # Apply indicator to index masks
        idx_train = idx_train[keep_non_null_and_test]
        idx_val = idx_val[keep_non_null_and_test]
        idx_test = idx_test[keep_non_null_and_test]

    assert X.shape[0] == Y0.shape[0] == T.shape[0] == Y.shape[0] == torch.max(A)+1 == idx_train.shape[0]

    print(f"train complete {idx_train.sum()}\tvalidating complete {idx_val.sum()}")
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = to_device(X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test)
    print(X.shape, Y0.shape, T.shape, Y.shape, A.shape, idx_train.shape)
    return X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test


def prepare(dataset_path, args):
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_path)
    # Model and optimizer

    model = GCN_HSIC(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    return X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer

def train_epoch(epoch, X, A, T, Y, Y1, Y0, idx_train, idx_val, model, optimizer, history, early_stopping, args):
    t = time.time()
    model.train()
    # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.zero_grad()

    init_rep, tc, edge_index = X, T, A
    # print(init_rep.shape, tc.shape, torch.max(edge_index), torch.min(edge_index))
    cf_y1, cf_y0, rep, graph_rep = model(init_rep, tc, edge_index)

    rep_balance = rep[idx_train]
    T_balance = T[idx_train]
    rep_loss = get_rep_loss(rep_balance, T_balance, args)       # only train set used

    yf_pred = torch.where(T > 0, cf_y1, cf_y0) 
    loss_train = loss(Y[idx_train], yf_pred[idx_train]) + args.alpha * rep_loss
    loss_train.backward()
    optimizer.step()

    # print("Y:", Y[idx_train],"\n Y_pred:",yf_pred[idx_train])
    # print("loss train:", loss(Y[idx_train], yf_pred[idx_train]).item(), "balance:", rep_loss.item())

    # ites = Y1 - Y0
    # ites1 = cf_y1 - cf_y0
    # ites = ites[idx_train|idx_val]
    # ites1 = ites1[idx_train|idx_val]
    # print("estimate loss: {:.4f}".format(torch.mean(torch.abs(ites - ites1)).item()), torch.sum(idx_train|idx_val).item())

    # print("after loss")
    loss_val = loss(Y[idx_val], yf_pred[idx_val]) + args.alpha * rep_loss
    
    # --- Update History ---
    history['epoch'].append(epoch)
    history['train_loss'].append(loss_train.item())
    history['val_loss'].append(loss_val.item())

    # --- Early Stopping Check ---
    early_stopping(loss_val.item(), model) # Pass validation loss to early stopping
    stopped = early_stopping.early_stop

    if epoch % 20 == 0:
        
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
    
    return history, stopped, model # Return updated history and stopped status

def eva(X, A, T, Y1, Y0, idx_train, idx_test, model, on_test=True):
    model.eval()

    init_rep, tc, edge_index = X, T, A
    cf_y1, cf_y0, rep, graph_rep = model(init_rep, tc, edge_index)

    y1_pred, y0_pred = cf_y1, cf_y0

    ite = Y1 - Y0
    ite_pred = y1_pred - y0_pred
    if on_test:
        pehe_ts = torch.sqrt(loss(ite_pred[idx_test],ite[idx_test]))
        mae_ate_ts = torch.abs(torch.mean(ite_pred[idx_test])-torch.mean(ite[idx_test]))
    else:
        pehe_ts = torch.sqrt(loss(ite_pred[idx_train],ite[idx_train]))
        mae_ate_ts = torch.abs(torch.mean(ite_pred[idx_train])-torch.mean(ite[idx_train]))
    print("Test set results:",
          "pehe_ts= {:.4f}".format(pehe_ts.item()),
          "mae_ate_ts= {:.4f}".format(mae_ate_ts.item()))
    
    results = {}
    results['effect_pehe'] = pehe_ts
    results['effect_mae'] = mae_ate_ts
    return results


def train_model(dataset_path, one_result_fn, args, impute):
    if os.path.exists(one_result_fn+"_complete_test_results.json"):
        print(f"Skip {one_result_fn}, because existing results")
        return
    
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer = prepare(dataset_path=dataset_path, args=args)
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    early_stopping = EarlyStopping(patience=args.patience, verbose=True) #, path=checkpoint_path)

    mask = idx_train | idx_val
    X, T, Y, Y1, Y0 = X[mask], T[mask], Y[mask], Y1[mask], Y0[mask]
    idx_train, idx_val = idx_train[mask], idx_val[mask]
    A, _ = filter_and_remap_edges(A, mask)

    t_total = time.time()
    for epoch in range(args.epochs):
        history, stopped, model = train_epoch(epoch, X, A, T, Y, Y1, Y0, idx_train, idx_val, model, optimizer, history, early_stopping, args)
        if stopped:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break # Exit the training loop for this experiment
    model = early_stopping.get_best_model(model)

    print("Optimization Finished!")
    print("Total training time: {:.4f}s".format(time.time() - t_total))

    history_save_path = one_result_fn+"_history.json"
    with open(history_save_path, 'w') as f:
        json.dump(history, f, indent=4)

    ############################################## complete test ##############################################
    # X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_path=dataset_path, imputed_version=False)
    test_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model, on_test=True) 
    results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}
    with open(one_result_fn+"_test_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print(f'save {one_result_fn+"_test_results.json"}')
    
    # print(results_serializable)
    # input()

    # ############################################## incomplete test ##############################################
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_path=dataset_path, imputed_version=True)
    test_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model, on_test=True) 
    results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}
    with open(one_result_fn+"_complete_test_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print("Done for one dataset ------------------>")

############ test main #####################

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
    parser.add_argument('--dataset', type=str, default='Syn_M=None_SimRel=1_Rel=4_MCAR')
    parser.add_argument("--missing_p", type=str, default='0.0')
    # parser.add_argument('--extrastr', type=str, default='1')

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 loss on parameters).')
   
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=1e-4,
                        help='trade-off of representation balancing.')
   
    parser.add_argument('--tr', type=float, default=0.6)
    parser.add_argument('--path', type=str, default='./datasets/')
    parser.add_argument('--normy', type=int, default=1)

    ##################### --- New Arguments ---
    parser.add_argument('--rep_loss_fun', type=str, default='subs_hsic')
    parser.add_argument("--gnn_fun", type=str, choices=['GraphSAGE', 'GCN'], default="GCN")
    parser.add_argument('--features_dim', type=int, default=20)
    parser.add_argument('--rep_hidden_dims', type=list, default=[64, 64])
    parser.add_argument('--gnn_dims', type=list, default=[64, 64])      # 64, 32
    parser.add_argument("--ind_pol_dims", type=list, default=[64, 64])  # 64, 32
    
    #####################  -- Ablation ---
    parser.add_argument('--ablation', type=str, choices=['no', 'no_rep', '1y_fun'], default='no')

    # --- New Arguments ---
    parser.add_argument('--patience', type=int, default=25, # Default patience for early stopping
                        help='Patience for early stopping.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                        help='Directory to save early stopping checkpoints.')

    args = parser.parse_args()
    args.cuda = not args.nocuda and torch.cuda.is_available()
    print(args)
    return args


def setup_optimal_training(num_threads=None):
    """Setup optimal training configuration for available hardware"""
    
    if num_threads is None:
        num_threads = min(os.cpu_count(), 8)  # Cap at 8 to avoid overhead
    
    # These settings are beneficial for BOTH CPU and GPU training
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    
    # Enable MKL-DNN optimizations (helps CPU ops even during GPU training)
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    
    # GPU-specific optimizations
    if torch.cuda.is_available():
        print(f"GPU detected - Using GPU with {num_threads} CPU threads for hybrid operations")
        # Enable CuDNN for optimal GPU performance
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # Find best algorithms
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Optional: Enable optimized attention (if using PyTorch 2.0+)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
    else:
        print(f"No GPU detected - Using {num_threads} CPU threads for training")
        # CPU-only optimizations
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        
        # Additional CPU optimizations
        if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
            torch.backends.mkl.enabled = True


# if __name__ == '__main__':
#     args = parse()
#     train_model('/Users/jason/Documents/Coding Projects/2025_Claude/NetDeconf_main_hao/datasets/exps/BlogCatalog/no_p=0.1_k=3_seed=699.pt', 'result/p=0.3_k=0_seed=194', args, 'no')
# #    
# #     # train_model('/Users/jason/Documents/Coding Projects/2025_Claude/NetDeconf_main_hao/datasets/exps/Syn/no/p=0.1_k=9_seed=959.pt', 'result/p=0.3_k=0_seed=919', args)

if __name__ == '__main__':
# if False:
    # import os
    setup_optimal_training()
    args = parse()

    ########################################## Method Settting  ################################
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    method = f'{args.gnn_fun}_{args.ablation}_drop={args.dropout}_HSIC'

    dataset_dir = args.dataset
    missing_p = args.missing_p
    ############################################################################################

    balu_dir = '/mnt/vast-kisski/projects/kisski-tib-activecl/BaLu'
    source_dir = 'datasets/exps/'
    if os.path.exists(balu_dir):
        source_dir = os.path.join(balu_dir, source_dir)
    src_dataset = os.path.join(source_dir, dataset_dir)   # exps/dataset

    for impute in os.listdir(src_dataset):
        impute_result_dir = os.path.join(results_dir, dataset_dir, method, impute)  # results/dataset/method/impute/
        mkdir([results_dir, dataset_dir, method, impute])
        for fn in os.listdir(os.path.join(src_dataset, impute)): 
            # print(impute, fn)
            if f"p={missing_p}" not in fn:                              # filter out file under other missing_p
                print("skip", fn, f"has no p={missing_p}")
                continue
            data_path = os.path.join(src_dataset, impute, fn)
            one_result_fn = os.path.join(impute_result_dir, fn)   
            try:
                train_model(data_path, one_result_fn, args, impute)
            except Exception as e:
                print(f"An error occurred: {e}")
            print(f"data:{data_path}\nresult:{one_result_fn}")

    
