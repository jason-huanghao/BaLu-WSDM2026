import time
import argparse
import numpy as np
import json
import os # Added for path creation

import torch
# import torch.nn.functional as F
import torch.optim as optim
import gc
from models.netdeconf_hao import GCN_DECONF 
import utils

# Import EarlyStopping
# Ensure you have installed it: pip install pytorch-early-stopping
from early_stopping_pytorch import EarlyStopping

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
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100,
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
parser.add_argument('--normy', type=int, default=1)
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


def to_device(**obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (ob.to(device) if isinstance(ob, torch.Tensor) else ob for ob in obs)

def extract_arrays_from_df(df_full):
    feature_cols = [col for col in df_full.columns if col.startswith('X')]
    X = df_full[feature_cols].values
    T = df_full['T'].values
    Y = df_full['Y'].values
    return X, T, Y

def prepare_data(dataset_path, imputed_version=True):
    data = torch.load(dataset_path, weights_only=False)
    # numpy.array
    Y1, Y0 = data.arr_Y1, data.arr_Y0
    assert len(Y1.shape) == len(Y0.shape) == 1    # 2D, 1D, 1D

    if not imputed_version:
        X, T, Y = extract_arrays_from_df(data.df_full)
    else:
        X, T, Y = extract_arrays_from_df(data.df_imputed)

    assert len(X.shape) == 2 
    X = torch.tensor(X, dtype=torch.float)
    Y1 = torch.tensor(Y1, dtype=torch.float)
    Y1 = torch.tensor(Y0, dtype=torch.float)

    # torch.tensor
    A, T, Y = data.A, data.treatment, data.outcome 
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask

    # update train and val
    if imputed_version:     # impact only when impute = no
        has_complete = ~data.df_imputed.isnull().any(axis=1)
        has_complete = torch.tensor(has_complete.values, dtype=torch.bool)
        idx_train &= has_complete
        idx_val &= has_complete
    print(f"train complete {idx_train.sum()}\tvalidating complete {idx_val.sum()}")

    # to device
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
    # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.zero_grad()
    yf_pred, rep, p1 = model(X, A, T)
    # ycf_pred, _, p1 = model(X, A, 1-T)

    # representation balancing, you can try different distance metrics such as MMD
    # print("max train index:", idx_train.max(), "min:", idx_train.min())
    # print("rep.shape:", rep.shape)
    # print("T.shape:", T.shape)
    # print("A.shape:", A.shape)
    # print("Y1.shape:", Y1.shape)
    # print("Y0.shape:", Y0.shape)
    # print("idx_train.shape:", idx_train.shape)
    # print("idx_val.shape:", idx_val.shape)
    
    rep_t1, rep_t0 = rep[idx_train][(T[idx_train] > 0).nonzero()], rep[idx_train][(T[idx_train] < 1).nonzero()]
    dist, _ = utils.wasserstein(rep_t1, rep_t0, cuda=args.cuda)
    
    YF = Y
    # YF = torch.where(T>0,Y1,Y0)
    # YCF = torch.where(T>0,Y0,Y1)

    if args.normy:
        ym, ys = torch.mean(YF[idx_train]), torch.std(YF[idx_train])
        ys = ys if ys > 1e-6 else torch.tensor([1.0], dtype=torch.float)
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


def train_model(dataset_path, one_result_fn):
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer = prepare(dataset_path=dataset_path)
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model.pt')
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
    
    # save_model(model, one_result_fn+"_model.pt")

    history_save_path = one_result_fn+"_history.json"
    with open(history_save_path, 'w') as f:
        json.dump(history, f, indent=4)

    # --- Evaluate the best model ---
    # train_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model, on_test=False) # Pass the loaded best model
    # results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in train_results.items()}
    # with open(one_result_fn+"_train_results.json", 'w') as f:
    #     json.dump(results_serializable, f, indent=4)

    ############################################## complete test ##############################################
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_path=dataset_path, imputed_version=False)
    test_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model) # Pass the loaded best model
    results_serializable = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in test_results.items()}
    with open(one_result_fn+"_test_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    ############################################## incomplete test ##############################################
    X, A, T, Y, Y1, Y0, idx_train, idx_val, idx_test = prepare_data(dataset_path=dataset_path, imputed_version=True)
    test_results = eva(X, A, T, Y1, Y0, idx_train, idx_test, model) # Pass the loaded best model
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


if __name__ == '__main__':
# if False:
    import os

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
                train_model(data_path, one_result_fn)
            except Exception as e:
                print(f"An error occurred: {e}")
            print(f"data:{data_fn}\nresult:{one_result_fn}")

        
                

    
# if __name__ == '__main__':
    # train_model('/home/jason/coding/NetDeconf_main_hao/datasets/Flickr1/Flickr0.mat', 'result/p=0.3_k=0_seed=919')
    
    # train_model('/home/jason/coding/NetDeconf_main_hao/datasets/Flickr3/Flickr0.mat', 'result/p=0.3_k=0_seed=919')
    # train_model('/home/jason/coding/BaLu/Syn10.mat', 'result/p=0.3_k=0_seed=919')
    # train_model('/home/jason/coding/BaLu/Syn20.mat', 'result/p=0.3_k=0_seed=919')
    
    # train_model("/home/jason/coding/BaLu/datasets/mats/Syn/mice/p=0.0_k=0_seed=919.pt.mat", 'results/Syn/p=0.0_k=0_seed=919')
    # train_model("/home/jason/coding/BaLu/datasets/mats/Syn/knn/p=0.3_k=0_seed=919.pt.mat", 'results/Syn/p=0.3_k=0_seed=919')
    # print("OK")
    # # prepare_data("/home/jason/coding/network-deconfounder-wsdm20-master/datasets/BlogCatalog1/BlogCatalog0.mat")
    # prepare_data("/home/jason/coding/NetDeconf_main_hao/datasets/mats/test_imputed=1/Syn/mice/p=0.0_k=0_seed=919.pt.mat")

    # train_model('/home/jason/coding/NetDeconf_main_hao/datasets/mats/Syn/mice/p=0.0_k=0_seed=919.pt.mat', 'results/Syn/p=0.0_k=2_seed=70')

    # train_model("/home/jason/coding/BaLu/datasets/mats/Flickr/mice/p=0.0_k=0_seed=919.pt.mat", 'results/Flickr/p=0.0_k=0_seed=919')
    # train_model("/home/jason/coding/BaLu/datasets/mats/Flickr/knn/p=0.3_k=9_seed=300.pt.mat", 'results/Flickr/p=0.3_k=0_seed=919')
                                                              
# if __name__ == '__main__':
#     for i_exp in range(0,10):

#         # Train model
#         X, A, T, Y1, Y0, idx_train, idx_val, idx_test, model, optimizer = prepare(i_exp)
#         t_total = time.time()
#         for epoch in range(args.epochs):
#             train(epoch, X, A, T, Y1, Y0, idx_train, idx_val, model, optimizer)
#         print("Optimization Finished!")
#         print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

#         # Testing sudo rm /mnt/vast-kisski/projects/kisski-tib-activecl/BaLu

#         eva(X, A, T, Y1, Y0, idx_train, idx_test, model, i_exp)
