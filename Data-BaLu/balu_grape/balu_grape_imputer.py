from balu_grape.grape import GRAPE
from torch_geometric.data import Data
import pandas as pd
import torch 
import numpy as np
# from early_stopping_pytorch import EarlyStopping
import torch.optim as optim
from balu_grape.Modules import compute_imputation_loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            # self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            # self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


def create_node_feature(n_units, n_attrs):
    x = torch.zeros((n_units + n_attrs, n_attrs))
    x[:n_units] = torch.ones((n_units, n_attrs))  # Unit nodes
    for i in range(n_attrs):
        x[n_units + i, i] = 1.0  # One-hot encoding for attribute nodes

    is_unit = torch.zeros(n_units + n_attrs, dtype=torch.bool)
    is_unit[:n_units] = True

    return x, is_unit

def train_val_test(n, tr:0.6, ts:0.2):
    n_train = int(n * tr)
    n_test = int(n * ts)
    idx = np.random.permutation(n)
    idx_train, idx_test, idx_val = idx[:n_train], idx[n_train:n_train+n_test], idx[n_train+n_test:]
    return idx_train, idx_test, idx_val

def create_data_edge_index(X:np.array, n_units, n_features):
    source_nodes = []
    target_nodes = []
    data_edge_attr_list = []
    
    # Step 1: Create edges for observed features (unit -> feature)
    for i in range(n_units):
        for j in range(n_features):
            # Only create edges for non-NaN values
            if not np.isnan(X[i, j]):
                # Edge from unit i to feature j
                # Features are indexed after units (n_units + j)
                source_nodes.append(i)
                target_nodes.append(n_units + j)
                data_edge_attr_list.append(X[i, j])
    
    # Step 2: Create bidirectional edges by adding reverse edges (feature -> unit)
    # Concatenate reverse edges to the existing lists
    source_nodes_bidirectional = source_nodes + target_nodes
    target_nodes_bidirectional = target_nodes + source_nodes
    data_edge_attr_bidirectional = data_edge_attr_list + data_edge_attr_list
    
    # Step 3: Convert to PyTorch tensors
    if data_edge_attr_bidirectional:  # Check if we have any edges
        data_edge_index = torch.tensor(
            [source_nodes_bidirectional, target_nodes_bidirectional], 
            dtype=torch.long
        )
        data_edge_attr = torch.tensor(data_edge_attr_bidirectional, dtype=torch.float)
    else:
        # Create empty tensors if no edges
        data_edge_index = torch.zeros((2, 0), dtype=torch.long)
        data_edge_attr = torch.zeros((0,), dtype=torch.float)
    
    return data_edge_index, data_edge_attr


def df2data(df:pd.DataFrame):
    X = df.values
    n_units, n_features = X.shape[0], X.shape[1]
    x, is_unit = create_node_feature(n_units, n_features)

    unobserved_mask = torch.tensor(np.isnan(X).flatten(), dtype=torch.bool)
    observed_mask = ~unobserved_mask

    data_edge_index, data_edge_attr = create_data_edge_index(X, n_units, n_features)

    data = Data(
        x=x,
        edge_index=data_edge_index,
        edge_attr=data_edge_attr
    )
    data.is_unit = is_unit
    data.n_units, data.n_attrs = n_units, n_features
    data.observed_mask = observed_mask
    data = to_device(data)
    return data
    
def to_device(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_device = Data()   
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            setattr(data_device, key, value.to(device))
        else:
            setattr(data_device, key, value)
    return data_device

class GrapeImputation():
    def __init__(self,
                n_features,
                node_dim=64,
                edge_dim=16,
                msg_dim=64,
                n_layers=2,
                dropout=0.1,
                activation='relu'):
        self.model = GRAPE(n_features, node_dim, edge_dim, msg_dim, n_layers, dropout, activation)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    
    def fit(self, df_miss:pd.DataFrame, idx_train=None, idx_val=None, n_epochs = 1000, lr = 1e-3, weight_decay = 1e-4):
        data = df2data(df_miss)

        if idx_train is None:
            idx_train, idx_test, idx_val = train_val_test(df_miss.shape[0]) # row number
        # if not isinstance(idx_train, torch.Tensor):
        #     idx_train = torch.tensor(idx_train, dtype=torch.long if idx_train.dtype != bool else torch.bool, device=self.device)
        #     idx_val = torch.tensor(idx_val, dtype=torch.long if idx_train.dtype != bool else torch.bool, device=self.device)
        
        train_mask = torch.zeros(data.n_units, dtype=torch.bool, device=self.device)
        val_mask = torch.zeros(data.n_units, dtype=torch.bool, device=self.device)
        train_mask[idx_train] = True
        val_mask[idx_val] = True
        idx_train = train_mask
        idx_val = val_mask
        
        early_stopping = EarlyStopping(patience=25, path='checkpoint.pt', verbose=False)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(n_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(data)
            
            use_idx = idx_train.to(self.device).unsqueeze(1).expand(-1, data.n_attrs).flatten()  # use data of train set
            tr_loss = compute_imputation_loss(outputs=outputs, data=data, use_index=use_idx)
            tr_loss.backward()
            optimizer.step()

            # evaluation
            self.model.eval()
            with torch.no_grad():
                use_idx = idx_val.to(self.device).unsqueeze(1).expand(-1, data.n_attrs).flatten()
                val_loss = compute_imputation_loss(outputs=outputs, data=data, use_index=use_idx)
            
            early_stopping(val_loss.cpu(), self.model)
            if early_stopping.early_stop:
                print("Early stopping!")
                break
    
    def transform(self, df_miss:pd.DataFrame):
        data = df2data(df_miss)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            imputd_attributes = outputs['imputed_attrs']
            imputd_attributes[data.observed_mask] = data.edge_attr[: len(data.edge_attr)//2]
        
        n_units, n_features = df_miss.shape
        imputed_array = imputd_attributes.reshape(n_units, n_features).cpu().numpy()
        return imputed_array
    
        # df_imputed = pd.DataFrame(
        #     imputed_array,
        #     index=df_miss.index,
        #     columns=df_miss.columns
        # )
        
        # return df_imputed


