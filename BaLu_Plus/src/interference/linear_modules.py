# import torch
# from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F

def get_activation(activation):
    if activation is None:
        return nn.Identity()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU()

class MLPNet(nn.Module):
    """
    Multi-layer perceptron network with configurable architecture.
    
    This class can be used for various prediction tasks (edge prediction,
    treatment prediction, outcome prediction) with customizable layers.
    """
    def __init__(
        self, 
        input_dim, 
        output_dim=1, 
        hidden_dims=[64,],
        out_activation=None,
        dropout=0.0,
    ):
        super(MLPNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Build MLP layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            
            self.layers.append(nn.Sequential(
                linear,
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = hidden_dim
        
        # Add output layer
        output_linear = nn.Linear(prev_dim, output_dim)
        self.layers.append(nn.Sequential(
            output_linear,
            get_activation(out_activation)
        ))
    
    def forward(self, x):
        """Forward pass through the MLP."""
        input_var = x
        for layer in self.layers:
            input_var = layer(input_var)
        return input_var
