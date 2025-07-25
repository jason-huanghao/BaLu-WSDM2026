import time
import argparse
import numpy as np
import json
import os # Added for path creation

import torch
# import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from models.netdeconf_hao import GCN_DECONF 
import utils
import copy 
# Import EarlyStopping
# Ensure you have installed it: pip install pytorch-early-stopping
# from early_stopping_pytorch import EarlyStopping


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



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0,
                    help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='Syn')
parser.add_argument("--missing_p", type=str, default='0.0')
# parser.add_argument('--extrastr', type=str, default='1')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=1e-4,
                    help='trade-off of representation balancing.')
parser.add_argument('--clip', type=float, default=100.,
                    help='gradient clipping')
parser.add_argument('--nout', type=int, default=2)
parser.add_argument('--nin', type=int, default=2)

parser.add_argument('--tr', type=float, default=0.6)
parser.add_argument('--path', type=str, default='./datasets/')
parser.add_argument('--normy', type=int, default=0)
# --- New Arguments ---
parser.add_argument('--patience', type=int, default=25, # Default patience for early stopping
                    help='Patience for early stopping.')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                    help='Directory to save early stopping checkpoints.')

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor

# alpha = Tensor([args.alpha])
alpha = torch.tensor([args.alpha], dtype=torch.float, device=device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    alpha = alpha.cuda()
    loss = loss.cuda()
    bce_loss = bce_loss.cuda()

# --- Ensure checkpoint directory exists ---
os.makedirs(args.checkpoint_dir, exist_ok=True)


def to_device(*obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    data = torch.load(dataset_path, map_location='cpu', weights_only=False)

    # numpy.array
    Y1, Y0 = data.arr_Y1, data.arr_Y0

    if not imputed_version: 
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
    # A = data.A
    A = create_frequency_weighted_adj(data.rel_edge_index)
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask

    if imputed_version:
        keep_non_null = ~torch.tensor(data.df_imputed.isnull().any(axis=1).to_numpy(), dtype=torch.bool)
        keep_non_null_and_test = keep_non_null | idx_test

        X = X[keep_non_null_and_test]
        Y1 = Y1[keep_non_null_and_test]
        Y0 = Y0[keep_non_null_and_test]
        T = T[keep_non_null_and_test]
        Y = Y[keep_non_null_and_test]

        A = filter_and_remap_edges(A, keep_non_null_and_test)

        # Apply indicator to index masks
        idx_train = idx_train[keep_non_null_and_test]
        idx_val = idx_val[keep_non_null_and_test]
        idx_test = idx_test[keep_non_null_and_test]

    assert X.shape[0] == Y0.shape[0] == T.shape[0] == Y.shape[0] == A.shape[0] == idx_train.shape[0]

    print(f"train complete {idx_train.sum()}\tvalidating complete {idx_val.sum()}")
    
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = to_device(X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test)

    return X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test


def prepare(dataset_path):
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_path)
    # Model and optimizer
    model = GCN_DECONF(nfeat=X.shape[1],
                nhid=args.hidden,
                dropout=args.dropout,n_out=args.nout,n_in=args.nin,cuda=args.cuda)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    return X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer


def train_epoch(epoch, X, A, T, Y, Y1, Y0, idx_train, idx_val, model, optimizer, history, early_stopping):
    t = time.time()
    model.train()
    
    optimizer.zero_grad()
    yf_pred, rep, p1 = model(X, A, T)

    # ycf_pred, _, p1 = model(X, A, 1-T)
    # cf_y1, cf_y0 = torch.where(T>0,yf_pred, ycf_pred), torch.where(T>0,ycf_pred,yf_pred)
    # ites = Y1 - Y0
    # ites1 = cf_y1 - cf_y0
    # ites = ites[idx_train|idx_val]
    # ites1 = ites1[idx_train|idx_val]
    # print("estimate loss: {:.4f}".format(torch.mean(torch.abs(ites - ites1)).item()), torch.sum(idx_train|idx_val).item())
    # print(eva(X, A, T, Y1, Y0, idx_train, None, model, on_test=False))

    # rep_t1, rep_t0 = rep[idx_train][(T[idx_train] > 0).nonzero()], rep[idx_train][(T[idx_train] < 1).nonzero()]
    rep_t1, rep_t0 = rep[idx_train][T[idx_train] > 0.5], rep[idx_train][T[idx_train] < 0.5]
    
    dist, _ = utils.wasserstein(rep_t1, rep_t0, cuda=args.cuda)
    
    YF = Y
    # YF = torch.where(T>0,Y1,Y0)
    # YCF = torch.where(T>0,Y0,Y1)

    if args.normy:
        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        ys = ys if ys > 1e-6 else torch.tensor(1.0, dtype=torch.float, device=ys.device)
        YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys
    else:
        YFtr = YF[idx_train]
        YFva = YF[idx_val]
    loss_train = loss(yf_pred[idx_train], YFtr) + alpha * dist

    loss_train.backward()
    optimizer.step()

    loss_val = loss(yf_pred[idx_val], YFva) + alpha * dist 
    
    # --- Update History ---
    history['epoch'].append(epoch)
    history['train_loss'].append(loss_train.item())
    history['val_loss'].append(loss_val.item())

    # --- Early Stopping Check ---
    early_stopping(loss_val.item(), model) # Pass validation loss to early stopping
    stopped = early_stopping.early_stop

    if epoch % 10 == 0:
        # y1_pred, y0_pred = torch.where(T>0,yf_pred,ycf_pred), torch.where(T>0,ycf_pred,yf_pred)
        # Y1, Y0 = torch.where(T>0, YF, YCF), torch.where(T>0, YCF, YF)
        # if args.normy:
        #     y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

        # # in fact, you are not supposed to do model selection w. pehe and mae_ate
        # # but it is possible to calculate with ITE ground truth (which often isn't available)

        # pehe_val = torch.sqrt(loss((y1_pred - y0_pred)[idx_val],(Y1 - Y0)[idx_val]))
        # mae_ate_val = torch.abs(
        #     torch.mean((y1_pred - y0_pred)[idx_val])-torch.mean((Y1 - Y0)[idx_val]))

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
            #   'pehe_val: {:.4f}'.format(pehe_val.item()),
            #   'mae_ate_val: {:.4f}'.format(mae_ate_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
    
    return history, stopped, model # Return updated history and stopped status

def eva(X, A, T, Y1, Y0, idx_train, idx_test, model, on_test=True):
    model.eval()
    yf_pred, rep, p1 = model(X, A, T) # p1 can be used as propensity scores
    # yf = torch.where(T>0, Y1, Y0)
    ycf_pred, _, _ = model(X, A, 1-T)

    YF = torch.where(T>0,Y1,Y0)
    # YCF = torch.where(T>0,Y0,Y1)

    ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
    # YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys

    y1_pred, y0_pred = torch.where(T>0,yf_pred,ycf_pred), torch.where(T>0,ycf_pred,yf_pred)

    if args.normy:
        y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

    # Y1, Y0 = torch.where(T>0, YF, YCF), torch.where(T>0, YCF, YF)
    if on_test:
        pehe_ts = torch.sqrt(loss((y1_pred - y0_pred)[idx_test],(Y1 - Y0)[idx_test]))
        mae_ate_ts = torch.abs(torch.mean((y1_pred - y0_pred)[idx_test])-torch.mean((Y1 - Y0)[idx_test]))
    else:
        pehe_ts = torch.sqrt(loss((y1_pred - y0_pred)[idx_train],(Y1 - Y0)[idx_train]))
        mae_ate_ts = torch.abs(torch.mean((y1_pred - y0_pred)[idx_train])-torch.mean((Y1 - Y0)[idx_train]))
    print("Test set results:",
          "pehe_ts= {:.4f}".format(pehe_ts.item()),
          "mae_ate_ts= {:.4f}".format(mae_ate_ts.item()))
    
    results = {}
    results['effect_pehe'] = pehe_ts
    results['effect_mae'] = mae_ate_ts
    return results

def save_model(model, save_path, save_weights=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_weights:
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model, save_path)


def train_model(dataset_path, one_result_fn, impute):
    if os.path.exists(one_result_fn+"_complete_test_results.json"):
        print(f"Skip {one_result_fn}, because existing results")
        return
    
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer = prepare(dataset_path=dataset_path)
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # prepare for tain
    # mask = idx_train | idx_val 
    mask = idx_train | idx_val | idx_test   # should be delete latter
    X, T, Y, Y1, Y0 = X[mask], T[mask], Y[mask], Y1[mask], Y0[mask]
    idx_train, idx_val = idx_train[mask], idx_val[mask]
    A = filter_and_remap_edges(A, mask)
    
    if impute.lower() == 'no':  # complete test
        data = torch.load(dataset_path, map_location='cpu', weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        keep_non_null = ~torch.tensor(data.df_imputed.isnull().any(axis=1).to_numpy(), dtype=torch.bool)
        keep_non_null_and_test = keep_non_null | data.test_mask 
        X1 = torch.tensor(data.arr_X, dtype=torch.float)[keep_non_null_and_test]
        X1 = X1.to(device)

        print("tr : val : test:", torch.sum(idx_train), torch.sum(idx_val), torch.sum(idx_test))
        print(X.shape, X1.shape)
        X[idx_test] = X1[idx_test]  # for no, we test on full datasets

    t_total = time.time()
    for epoch in range(args.epochs):
        history, stopped, model = train_epoch(epoch, X, A, T, Y, Y1, Y0, idx_train, idx_val, model, optimizer, history, early_stopping)

        if stopped:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break 
    model = early_stopping.get_best_model(model)

    print("Optimization Finished!")
    print("Total training time: {:.4f}s".format(time.time() - t_total))
    

    history_save_path = one_result_fn+"_history.json"
    with open(history_save_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    ############################################## complete test ##############################################
    # X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_path=dataset_path, imputed_version=False)
    # test_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model)
    test_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model, on_test=True)
    
    results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}
    print(results_serializable)
    
    with open(one_result_fn+"_test_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print(f'save {one_result_fn+"_test_results.json"}')
    ############################################## incomplete test ##############################################
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_path=dataset_path, imputed_version=True)
    test_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model)
    results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}

    with open(one_result_fn+"_complete_test_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print("Done for one dataset ------------------>")

    # if impute.lower() == 'no':
    #     print("OK")
    #     data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #     keep_non_null = ~torch.tensor(data.df_imputed.isnull().any(axis=1).to_numpy(), dtype=torch.bool)
    #     keep_non_null_and_test = keep_non_null | data.test_mask 
    #     X1 = torch.tensor(data.arr_X, dtype=torch.float)[keep_non_null_and_test]
    #     X1 = X1.to(device)

    #     print("tr : val : test:", torch.sum(idx_train), torch.sum(idx_val), torch.sum(idx_test))
    #     print(X.shape, X1.shape)
    #     X[idx_test] = X1[idx_test]  # for no, we test on full datasets
    # X2 = X[idx_test]
    # print("X has nan:",torch.isnan(X2).any(), "T has nan:",torch.isnan(T).any(), "Y0 has nan:",torch.isnan(Y0).any(), )

    
############ test main #####################

def mkdir(path=[]):
    path = [e for e in path if e]
    print(path)
    for i in range(len(path)):
        tmp = os.path.join(*path[:i+1])
        os.makedirs(tmp, exist_ok=True)


# if __name__ == '__main__':
#     train_model('/Users/jason/Documents/Coding Projects/2025_Claude/NetDeconf_main_hao/datasets/exps/BlogCatalog/no_p=0.1_k=3_seed=699.pt', 'result/p=0.3_k=0_seed=194', 'no')
#     train_model('/Users/jason/Documents/Coding Projects/2025_Claude/NetDeconf_main_hao/datasets/exps/BlogCatalog/p=0.0_k=9_seed=194.pt', 'result/p=0.3_k=0_seed=194', 'no')
#     # train_model('/Users/jason/Documents/Coding Projects/2025_Claude/NetDeconf_main_hao/datasets/exps/Syn/no/p=0.1_k=9_seed=959.pt', 'result/p=0.3_k=0_seed=919')
    


if __name__ == '__main__':
# if False:
    # import os

    ########################################## Method Settting  ################################
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    method = 'NetDeconf'
    # dataset_dir = 'Flickr'
    # missing_p = '0.3'

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
            if f"p={missing_p}" not in fn:                              # filter out file under other missing_p
                continue
            data_path = os.path.join(src_dataset, impute, fn)
            one_result_fn = os.path.join(impute_result_dir, fn)   
            try:
                train_model(data_path, one_result_fn, impute)
            except Exception as e:
                print(f"An error occurred: {e}")
            print(f"data:{data_path}\nresult:{one_result_fn}")

        
                
