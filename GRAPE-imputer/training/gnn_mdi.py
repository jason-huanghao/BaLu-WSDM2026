import numpy as np
import torch
import torch.nn.functional as F
# import pickle
import pandas as pd
from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
import torch.optim as optim
from utils_hao import *
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Data


def train_gnn_mdi(data1, args, device=torch.device('cpu')):
    data = df2data(data1.df_miss)
    data = to_device(data, device=device) 

    model = get_gnn(data, args).to(device)
    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int,args.impute_hiddens.split('_')))
    
    input_dim = args.node_dim * 2
    output_dim = 1
    impute_model = MLPNet(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)
    
    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters())
    print("total trainable_parameters: ",len(trainable_parameters))

    filter_fn = filter(lambda p : p.requires_grad, trainable_parameters)
    opt = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)

    x = data.x
    arr_TYX = data1.df_miss.to_numpy()
    train_mask = data1.train_mask
    n_row = arr_TYX.shape[0]
    n_column = arr_TYX.shape[1]

    train_edge_index, train_edge_attr, train_attr_indexs = create_data_edge_index_mask(arr_TYX, n_row, n_column, device, used_mask=train_mask)
    assert train_edge_index.shape[1] == train_edge_attr.shape[0]
    
    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        opt.zero_grad()
        
        # known_edge_index, edge_mask = dropout_edge(train_edge_index, p=0.3, training=True, force_undirected=True)
        # known_edge_attr = train_edge_attr[edge_mask]

        known_edge_index = train_edge_index
        known_edge_attr = train_edge_attr
        
        x_embd = model(x, known_edge_attr.unsqueeze(1), known_edge_index)       # the attr_emb should be shape of [E, 1]

        # pred_train = impute_model([x_embd[train_edge_index[0][:int(train_edge_attr.shape[0] / 2)]], 
        #                            x_embd[train_edge_index[1][:int(train_edge_attr.shape[0] / 2)]]]).squeeze()
        # Should be:
        pred_train = impute_model(torch.cat([x_embd[train_edge_index[0][:int(train_edge_attr.shape[0] / 2)]], 
                                             x_embd[train_edge_index[1][:int(train_edge_attr.shape[0] / 2)]]], dim=1)).squeeze()
        
        label_train = train_edge_attr[:int(train_edge_attr.shape[0]/2)]      # values 

        loss = F.mse_loss(pred_train, label_train)
        loss.backward()
        opt.step()
        print(f"Train loss: {loss.item():.4f}")
    
    return model, impute_model

def transform(data1, model, impute_model, device=None):
    data = df2data(data1.df_miss)
    data = to_device(data, device=device)
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    arr_TYX = data1.df_miss.to_numpy()
    n_row = arr_TYX.shape[0]
    n_column = arr_TYX.shape[1]
    mask = data1.train_mask | data1.val_mask | data1.test_mask
    
    edge_index, edge_attr, attr_indexs = create_data_edge_index_mask(arr_TYX, n_row, n_column, device, used_mask=mask)
    
    x = data.x
    with torch.no_grad():
        x_embd = model(x, edge_attr.unsqueeze(1), edge_index)       # the attr_emb should be shape of [E, 1]

        h_x, attr_emb = x_embd[:n_row], x_embd[n_row:]
        N, M, K = h_x.size(0), attr_emb.size(0), h_x.size(1)
        node_expanded = h_x.unsqueeze(1).expand(N, M, K)
        attr_expanded = attr_emb.unsqueeze(0).expand(N, M, K)
        combined = torch.cat([node_expanded, attr_expanded], dim=2)

        pred_attrs = impute_model(combined.view(N * M, 2 * K)).squeeze()

        # print(train_attr_indexs.shape, pred_attrs.shape)
        assert attr_indexs.shape[0] == pred_attrs.shape[0]
         
        observed_attrs = edge_attr[: int(edge_attr.shape[0] / 2)]
        pred_attrs[attr_indexs] = observed_attrs
    
    X_transformed = pred_attrs.reshape(n_row, n_column).cpu().numpy()
    df_missing = data1.df_miss
    df_imputed = pd.DataFrame(
                X_transformed, 
                index=df_missing.index, 
                columns=df_missing.columns
            )
    
    data1.df_imputed = df_imputed
    return data1 

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
    data.n_units, data.n_attrs, data.node_feature_dim, data.edge_attr_dim = n_units, n_features, n_features, 1
    
    data.observed_mask = observed_mask
    data = to_device(data)
    return data


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



def create_node_feature(n_units, n_attrs):
    x = torch.zeros((n_units + n_attrs, n_attrs))
    x[:n_units] = torch.ones((n_units, n_attrs))  # Unit nodes
    for i in range(n_attrs):
        x[n_units + i, i] = 1.0  # One-hot encoding for attribute nodes

    is_unit = torch.zeros(n_units + n_attrs, dtype=torch.bool)
    is_unit[:n_units] = True

    return x, is_unit
