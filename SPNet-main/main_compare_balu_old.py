import time
import argparse
import numpy as np
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
# import torch.nn.functional as F
import torch.optim as optim
import gc
from models.spnet_hao import GCN_DECONF
import utils

# Import EarlyStopping
# Ensure you have installed it: pip install pytorch-early-stopping
from early_stopping_pytorch import EarlyStopping

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', type=int, default=0,
                    help='Disables CUDA training...................')
parser.add_argument('--dataset', type=str, default='Syn')
# parser.add_argument('--extrastr', type=str, default='1')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=400,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=1e-4,
                    help='trade-off of representation balancing.')
parser.add_argument('--clip', type=float, default=100.,
                    help='gradient clipping')
parser.add_argument('--gama', type=float, default=1e-4)
parser.add_argument('--nout', type=int, default=2)
parser.add_argument('--nin', type=int, default=2)

parser.add_argument('--tr', type=float, default=0.6)
# Changed default path as it was used in the original for load_data
# Data path will be handled by the main loop's directory traversal
# parser.add_argument('--path', type=str, default='./datasets_3/')
parser.add_argument('--normy', type=int, default=1)
# --- New Arguments for Adaptation ---
parser.add_argument('--patience', type=int, default=25, # Default patience for early stopping
                    help='Patience for early stopping.')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                    help='Directory to save early stopping checkpoints.')
parser.add_argument("--missing_p", type=str, default='0.0',
                    help='Filter for missing percentage in filenames.')

args = parser.parse_args()
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
# LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor

# alpha = Tensor([args.alpha])
alpha = torch.tensor([args.alpha], dtype=torch.float)
# gama = Tensor([args.gama])
gama = torch.tensor([args.gama], dtype=torch.float)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()
Crossentropy_loss = torch.nn.CrossEntropyLoss()

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    alpha = alpha.cuda()
    gama = gama.cuda() # Added gama to cuda transfer
    loss = loss.cuda()
    bce_loss = bce_loss.cuda()
    Crossentropy_loss = Crossentropy_loss.cuda()

print("cuda", args.cuda)

# --- Ensure checkpoint directory exists ---
os.makedirs(args.checkpoint_dir, exist_ok=True)


def prepare_data(dataset_fn):
    """Loads data and initializes model and optimizer from a file path."""
    # Load data using load_data_mat or equivalent based on the file path
    # Assuming utils.load_data_mat exists and works like in the previous adopted code
    # Load data and init models
    # X, A, T, Y1, Y0 = utils.load_data(args.path, name=args.dataset, original_X=False, exp_id=str(i_exp), extra_str=args.extrastr)

    X, A, T, Y, Y1, Y0, idx_train, idx_test, idx_val = utils.load_data_mat(dataset_fn)

    if idx_train is None:
        n = X.shape[0]
        n_train = int(n * args.tr)
        n_test = int(n * 0.2)
        n_valid = n_test

        idx = np.random.permutation(n)
        idx_train, idx_test, idx_val = idx[:n_train], idx[n_train:n_train+n_test], idx[n_train+n_test:]

    # X = utils.normalize(X) #row-normalize
    # A = utils.normalize(A+sp.eye(n)) # Assuming sp is imported if uncommented

    X = X.todense()
    # X = Tensor(X)
    X = torch.tensor(X, dtype=torch.float, device=device)

    # Y1 = Tensor(np.squeeze(Y1))
    # Y0 = Tensor(np.squeeze(Y0))
    Y1 = torch.tensor(np.squeeze(Y1), dtype=torch.float, device=device)
    Y0 = torch.tensor(np.squeeze(Y0), dtype=torch.float, device=device)
    Y = torch.tensor(np.squeeze(Y), dtype=torch.float, device=device)
    # T = LongTensor(np.squeeze(T))
    T = torch.tensor(np.squeeze(T), dtype=torch.long, device=device)

    A = utils.sparse_mx_to_torch_sparse_tensor(A,cuda=args.cuda)

    # idx_train = LongTensor(idx_train)
    # idx_val = LongTensor(idx_val)
    # idx_test = LongTensor(idx_test)
    idx_train = torch.tensor(idx_train, dtype=torch.long, device=device)
    idx_val = torch.tensor(idx_val, dtype=torch.long, device=device)
    idx_test = torch.tensor(idx_test, dtype=torch.long, device=device)
    return X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test


# Modified prepare function to accept dataset file path
def prepare(dataset_fn):
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_fn)

    n = X.shape[0] # Get n after loading data

    # Model and optimizer
    # Pass n to the model constructor as in the original code
    model = GCN_DECONF(nfeat=X.shape[1],
                       nhid=args.hidden,
                       dropout=args.dropout, n_out=args.nout, n_in=args.nin, cuda=args.cuda, n=n)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    return X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer


def train_epoch(epoch, X, A, T, Y, Y1, Y0, idx_train, idx_val, model, optimizer, history, early_stopping):
    t = time.time()
    model.train()
    # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.zero_grad()
    yf_pred, rep, p1 = model(X, A, T)
    # ycf_pred, _, p1 = model(X, A, 1-T)

    # representation balancing, you can try different distance metrics such as MMD
    rep_t1, rep_t0 = rep[idx_train][(T[idx_train] > 0).nonzero()], rep[idx_train][(T[idx_train] < 1).nonzero()]
    dist, _ = utils.wasserstein(rep_t1, rep_t0, cuda=args.cuda)

    YF = Y
    # YF = torch.where(T > 0, Y1, Y0)
    # YCF = torch.where(T>0,Y0,Y1)

    if args.normy:
        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        ys = ys if ys > 1e-6 else torch.tensor([1.0], dtype=torch.float, device=device) #Tensor([1.0])
        YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys
    else:
        YFtr = YF[idx_train]
        YFva = YF[idx_val]
    loss_train = loss(yf_pred[idx_train], YFtr) + alpha * dist + gama*Crossentropy_loss(p1[idx_train],T[idx_train])

    loss_train.backward()
    optimizer.step()

    # validation
    #print(model.att.cpu().detach().numpy())
    loss_val = loss(yf_pred[idx_val], YFva) + alpha * dist + gama*Crossentropy_loss(p1[idx_val],T[idx_val])

    if epoch % 10 == 0:
        # y1_pred, y0_pred = torch.where(T > 0, yf_pred, ycf_pred), torch.where(T > 0, ycf_pred, yf_pred)
        # # Y1, Y0 = torch.where(T>0, YF, YCF), torch.where(T>0, YCF, YF)
        # if args.normy:
        #     y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

        # in fact, you are not supposed to do model selection w. pehe and mae_ate
        # but it is possible to calculate with ITE ground truth (which often isn't available)

        # pehe_val = torch.sqrt(loss((y1_pred - y0_pred)[idx_val],(Y1 - Y0)[idx_val]))
        # mae_ate_val = torch.abs(
        #      torch.mean((y1_pred - y0_pred)[idx_val])-torch.mean((Y1 - Y0)[idx_val]))
        # pehe_train = torch.sqrt(loss((y1_pred - y0_pred)[idx_train], (Y1 - Y0)[idx_train]))
        # mae_ate_train = torch.abs(
        #     torch.mean((y1_pred - y0_pred)[idx_train]) - torch.mean((Y1 - Y0)[idx_train]))
        print('Epoch: {:04d}'.format(epoch + 1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                # 'pehe_train: {:.4f}'.format(pehe_train.item()),
                # 'mae_ate_trainn: {:.4f}'.format(mae_ate_train.item()),
                # 'pehe_val: {:.4f}'.format(pehe_val.item()),
                # 'mae_ate_val: {:.4f}'.format(mae_ate_val.item()),
                'time: {:.4f}s'.format(time.time() - t))

    # --- Update History ---
    history['epoch'].append(epoch)
    history['train_loss'].append(loss_train.item())
    history['val_loss'].append(loss_val.item())
    
    # --- Early Stopping Check ---
    early_stopping(loss_val.item(), model)
    stopped = early_stopping.early_stop
    return history, stopped, model # Return updated history and stopped status

def eva(X, A, T, Y1, Y0, idx_train, idx_test, model, on_test=True):
    model.eval()
    with torch.no_grad():
        yf_pred, rep, p1 = model(X, A, T)  # p1 can be used as propensity scores
        # yf = torch.where(T>0, Y1, Y0)
        ycf_pred, _, _ = model(X, A, 1 - T)

        YF = torch.where(T > 0, Y1, Y0)
        # YCF = torch.where(T > 0, Y0, Y1)

        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        # YFtr, YFva = (YF[idx_train] - ym) / ys, (YF[idx_val] - ym) / ys

        y1_pred, y0_pred = torch.where(T > 0, yf_pred, ycf_pred), torch.where(T > 0, ycf_pred, yf_pred)

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

# New function to orchestrate training and evaluation for a single file

def save_model(model, save_path, save_weights=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_weights:
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model, save_path)

def create_on_test_imputed_path(base_path, j):
    parts = base_path.split('mats/')
    new_path = parts[0] + 'mats/' + f'test_imputed={j}/' + parts[1]
    return new_path

def train_model(dataset_fn, one_result_fn):
    dataset_fn_test_imputed = create_on_test_imputed_path(dataset_fn, 1)
    # print("----", dataset_fn_test_imputed)
    # X, A, T, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer = prepare(dataset_fn)
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer = prepare(dataset_fn_test_imputed)

    # Setup early stopping
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{os.path.basename(one_result_fn)}.pt')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=checkpoint_path)

    t_total = time.time()
    for epoch in range(args.epochs):
        # Pass history and early_stopping instances to the training function
        history, stopped, model = train_epoch(epoch, X, A, T, Y, Y1, Y0, idx_train, idx_val, model, optimizer, history, early_stopping)
        if stopped:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break # Exit the training loop for this experiment
    
    print("Optimization Finished!")
    print("Total training time: {:.4f}s".format(time.time() - t_total))
    
    save_model(model, one_result_fn+"_model.pt")

    history_save_path = one_result_fn+"_history.json"
    with open(history_save_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved history to {history_save_path}")

    # --- Evaluate the best model ---
    print("evaluate on test set")
    test_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model) # Pass the loaded best model
    results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}
    with open(one_result_fn+"_test_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=4)

    print("evaluate on train set")
    # train_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model, on_test=False) # Pass the loaded best model
    # results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in train_results.items()}
    # with open(one_result_fn+"_train_results.json", 'w') as f:
    #     json.dump(results_serializable, f, indent=4)

    print("evaluate on complete test set")
    dataset_fn_test_unimputed = create_on_test_imputed_path(dataset_fn, 0)
    del X, A, T, Y1, Y0, idx_train, idx_val, idx_test
    gc.collect()
    torch.cuda.empty_cache()
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_fn_test_unimputed)
    test_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model) # Pass the loaded best model
    results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}
    with open(one_result_fn+"_complete_test_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print("Done for one dataset ------------------>")

############ test main #####################
    
# if __name__ == '__main__':
    # train_model('/home/jason/coding/NetDeconf_main_hao/datasets/Flickr1/Flickr0.mat', 'result/p=0.3_k=0_seed=919')
    
    # train_model('/home/jason/coding/NetDeconf_main_hao/datasets/Flickr3/Flickr0.mat', 'result/p=0.3_k=0_seed=919')
    # train_model('/home/jason/coding/BaLu/Syn10.mat', 'result/p=0.3_k=0_seed=919')
    # train_model('/home/jason/coding/BaLu/Syn20.mat', 'result/p=0.3_k=0_seed=919')
    
    # train_model("/home/jason/coding/BaLu/datasets/mats/Syn/mice/p=0.0_k=0_seed=919.pt.mat", 'results/Syn/p=0.0_k=0_seed=919')
    # train_model("/home/jason/coding/BaLu/datasets/mats/Syn/knn/p=0.3_k=0_seed=919.pt.mat", 'results/Syn/p=0.3_k=0_seed=919')
    # train_model('/home/jason/coding/NetDeconf_main_hao/datasets/mats/Syn/mice/p=0.0_k=2_seed=70.pt.mat', 'results/Syn/p=0.0_k=2_seed=70')

    # train_model("/home/jason/coding/BaLu/datasets/mats/Flickr/mice/p=0.0_k=0_seed=919.pt.mat", 'results/Flickr/p=0.0_k=0_seed=919')
    # train_model("/home/jason/coding/BaLu/datasets/mats/Flickr/knn/p=0.3_k=9_seed=300.pt.mat", 'results/Flickr/p=0.3_k=0_seed=919')
                  


if __name__ == '__main__':
# if False:
    ########################################## Method Settting  ################################
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    method = 'SPNet'
    # dataset_dir = 'Flickr'
    # missing_p = '0.3'

    dataset_dir = args.dataset
    missing_p = args.missing_p
    ############################################################################################

    balu_dir = '/mnt/vast-kisski/projects/kisski-tib-activecl/BaLu'
    source_dir = 'datasets/exps/'
    target_dir = 'datasets/mats/'
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(balu_dir):
        source_dir = os.path.join(balu_dir, source_dir)
        target_dir = os.path.join(balu_dir, target_dir)

    src_dataset = os.path.join(source_dir, dataset_dir)   # source dataset dir
    tar_dataset = os.path.join(target_dir, dataset_dir)   # target dataset dir

    dataset_result_dir = os.path.join(results_dir, dataset_dir)
    os.makedirs(dataset_result_dir, exist_ok=True)

    method_result_dir = os.path.join(dataset_result_dir, method)
    os.makedirs(method_result_dir, exist_ok=True)

    for fn in os.listdir(src_dataset):
        if f"p={missing_p}" not in fn:
            continue
        for impute in ['no', 'mean', 'knn', 'mice', 'missforest', 'gain']:
            method_impute_dir = os.path.join(method_result_dir, impute)
            os.makedirs(method_impute_dir, exist_ok=True)
            
            tar_dataset_impute = os.path.join(tar_dataset, impute)
            data_mat_fn = os.path.join(tar_dataset_impute, fn+".mat")
            parts = data_mat_fn.split("mats/")[1].split("/")
            # dataset = parts[0]; method = parts[1]
            identifier = parts[2].split(".pt")[0]
            one_result_fn = os.path.join(method_impute_dir, identifier)

            if os.path.exists(one_result_fn+"_train_results.json"): # results are ready
                print(f"skip {one_result_fn}, exists!")
                continue
            
            try:
                train_model(data_mat_fn, one_result_fn)
            except Exception as e:
                print(f"An error occurred: {e}")
            print(data_mat_fn)
            print(one_result_fn)
