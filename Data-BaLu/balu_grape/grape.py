import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.data import Data
from balu_grape.Modules import (
    MLPNet,
    NodeUpdateV1,
    EdgeUpdating,
    compute_imputation_loss
)

class GRAPE(nn.Module):
    """
    Reimplementation from https://github.com/maxiaoba/GRAPE  
    Instead of training 20000 steps, we used the early stop technique
    Note that the edge_drop during training phase is not applied in this implementation
    """
    def __init__(
        self, 
        n_features,
        node_dim=64,
        edge_dim=16,
        msg_dim=64,
        n_layers=2,
        dropout=0.1,
        activation='relu'
    ):
        super(GRAPE, self).__init__()
        
        # Initialize node update modules for each layer
        self.node_update_layers = nn.ModuleList()
        # Initialize edge update modules for each layer
        self.edge_update_layers = nn.ModuleList()

        first_layer = True
        for _ in range(n_layers):
            # Version 1 of node updating
            self.node_update_layers.append(
                NodeUpdateV1(n_features if first_layer else node_dim, node_dim, 1 if first_layer else edge_dim, msg_dim, n_rel_types=0, dropout=dropout)
            )
            # Add edge update module for each layer
            self.edge_update_layers.append(
                EdgeUpdating(node_dim, 1 if first_layer else edge_dim, edge_dim)
            )
            first_layer = False 

        # Edge prediction for attribute imputation
        self.edge_predictor = MLPNet(input_dim=2 * node_dim, 
                                     hidden_dims=[edge_dim, ],
                                     output_dim=1,
                                     output_activation=None,
                                     activation=activation,
                                     dropout=dropout)
        
    def forward(self, data:Data):
        """
        Forward pass through the BaLu model.
        
        Args:
            data: Data object from load_heter_graph
            
        Returns:
            Dictionary of model outputs
        """
        
        edge_index, edge_attr, x = data.edge_index, data.edge_attr, data.x
        
        # Extract graph components
        data_edge_index = edge_index
        data_edge_attr = edge_attr.unsqueeze(1)     # [E, 1]

        # Initialize edge embeddings (scalar values)
        edge_emb = data_edge_attr
        
        # Apply node update and edge update layers
        for l in range(len(self.node_update_layers)):
            
            x = self.node_update_layers[l](
                x,
                data_edge_index, 
                edge_emb
            )
            edge_emb = self.edge_update_layers[l](data_edge_index, edge_emb, x)
        
        
        # Split embeddings into unit and attribute nodes
        node_emb = x[:data.n_units]
        attr_emb = x[data.n_units:]
        
        # print("node embedding 1")
        node_emb_expanded = node_emb.unsqueeze(1)  # Shape: [n_units, 1, embedding_dim]
        attr_emb_expanded = attr_emb.unsqueeze(0)  # Shape: [1, n_attrs, embedding_dim]

        # Repeat and reshape to get all combinations
        node_emb_repeated = node_emb_expanded.repeat(1, data.n_attrs, 1)  # Shape: [n_units, n_attrs, embedding_dim]
        attr_emb_repeated = attr_emb_expanded.repeat(data.n_units, 1, 1)  # Shape: [n_units, n_attrs, embedding_dim]
        
        # Concatenate embeddings
        combined_emb = torch.cat([
            node_emb_repeated.reshape(-1, node_emb.size(1)),  # Shape: [n_units*n_attrs, embedding_dim]
            attr_emb_repeated.reshape(-1, attr_emb.size(1))   # Shape: [n_units*n_attrs, embedding_dim]
        ], dim=1)  # Final 
        del node_emb_expanded, attr_emb_expanded, node_emb_repeated, attr_emb_repeated
        
        imputed_attrs = self.edge_predictor(combined_emb).squeeze()  # Shape: [n_units*n_attrs]
        
        return {
            'imputed_attrs': imputed_attrs,
            'edge_embeddings': edge_emb
        }
    
    def loss(self, outputs, data):
        """
        Compute the combined loss for the model.
        
        Args:
            outputs: Forward pass outputs
            data: Data object from load_heter_graph
            alpha: Weight for outcome loss
            beta: Weight for treatment loss
            gamma: Weight for balancing loss
            unit_indexes: mask of train, validation, test
            
        Returns:
            Total loss and component losses
        """
        # Calculate component losses
        imp_loss = compute_imputation_loss(outputs, data, on_observed=True)
        # imp_loss_unobs = compute_imputation_loss(outputs, data, on_observed=False)
        
        # Return total loss and components
        imp_loss.item() 
    