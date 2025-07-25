import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
# import copy
from torch_scatter import scatter_softmax
from torch_scatter import scatter
# from torch import scatter_add
import math
from torch.nn.parameter import Parameter

#######################################
# Helper Functions
#######################################

def init_testing(module, testing=False):
    """Initialize module parameters for testing with ones for weights and zeros for biases."""
    if testing:
        if hasattr(module, 'weight'):
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

#######################################
# Message Passing Base Components
#######################################

# class AttributeMessagePassing(nn.Module):
#     """Efficient message passing along attribute edges using pure PyTorch."""
#     def __init__(self, node_dim, edge_dim, msg_dim_out, testing=False):
#         super().__init__()
#         self.msg_nn = nn.Linear(node_dim + edge_dim, msg_dim_out)
#         init_testing(self.msg_nn, testing)

#     def forward(self, node_emb, edge_index, edge_emb, chunk_size=500000):
#         """
#         node_emb: [num_nodes, node_dim]
#         edge_index: [2, num_edges]
#         edge_emb: [num_edges, edge_dim]
#         chunk_size: edges per chunk for memory control
#         """
#         num_nodes = node_emb.size(0)
#         device = node_emb.device
#         result = torch.zeros(num_nodes, self.msg_nn.out_features, device=device)
#         counts = torch.zeros(num_nodes, 1, device=device)

#         # Unpack edge indices
#         src_all, dst_all = edge_index
#         num_edges = src_all.size(0)

#         # Process in chunks
#         for i in range(0, num_edges, chunk_size):
#             # print(f"Modules.py.AttributeMessagePassing.forward()--------------- chunk {i} --> {num_edges}")
#             src = src_all[i:i+chunk_size]
#             dst = dst_all[i:i+chunk_size]
#             edge_feat = edge_emb[i:i+chunk_size]

#             # Compute messages
#             inputs = torch.cat([node_emb[src], edge_feat], dim=1)        # can lead to OOM
#             del edge_feat
#             torch.cuda.empty_cache()

#             msgs = F.relu(self.msg_nn(inputs))
#             del inputs 
#             torch.cuda.empty_cache()

#             # Vectorized aggregation
#             result.index_add_(0, dst, msgs)
#             ones = torch.ones((dst.size(0), 1), device=device)
#             counts.index_add_(0, dst, ones)
#             del msgs, ones
#             torch.cuda.empty_cache()

#         # Mean aggregation
#         mask = counts > 0
#         result[mask.squeeze()] = result[mask.squeeze()] / counts[mask.squeeze()]
#         del counts 
#         torch.cuda.empty_cache()
#         return result


class AttributeMessagePassing(MessagePassing):
    """Message passing along attribute edges"""
    def __init__(self, node_dim, edge_dim, msg_dim_out, testing=False):
        super(AttributeMessagePassing, self).__init__(aggr="mean")
        self.msg_nn = nn.Linear(node_dim + edge_dim, msg_dim_out)  # Source node, edge attr, and destination node
        init_testing(self.msg_nn, testing)

    def forward(self, node_emb, edge_index, edge_emb):
        return self.propagate(edge_index, x=node_emb, edge_emb=edge_emb)
    
    def message(self, x_j, edge_emb):
        # print(f"x_j on {x_j.device}, edge_emb on {edge_emb.device}, weight on {self.msg_nn.weight.device}")
        # Create message from source node features, destination features and edge attribute
        return F.relu(self.msg_nn(torch.cat([x_j, edge_emb], dim=1)))

# class RelationalMessagePassing(nn.Module):
#     """
#     Memory-efficient message passing along edges of one relation type using pure PyTorch.
#     This class replaces torch_geometric's MessagePassing for a single relation.
#     """
#     def __init__(self, node_dim, msg_dim_out, testing=False):
#         super().__init__()
#         # Learnable linear for relation-specific transformation
#         self.msg_nn = nn.Linear(node_dim, msg_dim_out)
#         init_testing(self.msg_nn, testing)


#     def forward(self, node_emb, edge_index, chunk_size=500000, aggr="mean"):
#         """
#         node_emb: Tensor [num_nodes, node_dim]
#         edge_index: LongTensor [2, num_edges]
#         chunk_size: edges per chunk for memory control
#         aggr: aggregation method ('mean' or 'sum')
#         """
#         num_nodes = node_emb.size(0)
#         device = node_emb.device

#         # Prepare output tensors
#         result = torch.zeros(num_nodes, self.msg_nn.out_features, device=device)
#         counts = torch.zeros(num_nodes, 1, device=device) if aggr == "mean" else None

#         # Unpack edge indices
#         src_all, dst_all = edge_index
#         num_edges = src_all.size(0)

#         # Process in chunks
#         for i in range(0, num_edges, chunk_size):
#             src = src_all[i:i+chunk_size]
#             dst = dst_all[i:i+chunk_size]

#             # print("see vram 1")
#             # Compute messages: Q_r * x_j + ReLU
#             msgs = F.relu(self.msg_nn(node_emb[src]))  # [chunk, msg_dim]
#             # print("see vram 2")

#             # Sum aggregation
#             result.index_add_(0, dst, msgs)
#             del msgs 
#             torch.cuda.empty_cache()

#             # Count for mean aggregation
#             if aggr == "mean":
#                 ones = torch.ones((dst.size(0), 1), device=device)
#                 counts.index_add_(0, dst, ones)

#         # Finalize mean aggregation if needed
#         if aggr == "mean":
#             mask = counts > 0
#             result[mask.squeeze()] = result[mask.squeeze()] / counts[mask.squeeze()]

#         return result


class RelationalMessagePassing(MessagePassing):
    """
    Message passing along relational edges for ONE specific relation type.
    Contains its own learnable linear layer (Q_r).
    """
    def __init__(self, node_dim, msg_dim_out, aggr="mean", testing=False):
        super().__init__(aggr=aggr)
        # Each instance of this class will have its own Q_r weight matrix
        self.msg_nn = nn.Linear(node_dim, msg_dim_out)
        init_testing(self.msg_nn, testing)

    def forward(self, node_emb, edge_index):
        return self.propagate(edge_index, x=node_emb) #, size=node_emb.size(0))

    def message(self, x_j):
        return F.relu(self.msg_nn(x_j))


#######################################
# Node Updating Modules (V1 and V2)
#######################################


class NodeUpdateV1(nn.Module):
    """
    Version 1: Models similarity via attributes only (Homogeneous Graph).
    Update: h = sigma( Q_node * (x || m_attr) ) applied to all nodes.
    """
    def __init__(self, node_dim, node_dim_out, edge_dim, msg_dim, n_rel_types, dropout=0.1, testing=False):
        super().__init__()
        self.attr_message = AttributeMessagePassing(node_dim, edge_dim, msg_dim, testing)
        # Update NN input: Original features (hidden_dim) + attr messages (hidden_dim)
        self.update_nn = nn.Linear(node_dim + msg_dim, node_dim_out)
        # self.norm = nn.LayerNorm(node_dim) # Add normalization
        self.dropout_layer = nn.Dropout(dropout)
        
        init_testing(self.update_nn, testing)
        # init_testing(self.norm, testing)

    def forward(self, node_emb, data_edge_index, edge_emb, normalize=False, **kwargs):
        # node_emb: Node features [N, hidden_dim]
        # data_edge_index: Attribute edge index [2, E_attr]
        # edge_emb: Attribute edge attribute [E_attr, edge_dim]
        attr_messages = self.attr_message(node_emb, data_edge_index, edge_emb)
        update_input = torch.cat([node_emb, attr_messages], dim=1)
        h = F.relu(self.update_nn(update_input))
        h = self.dropout_layer(h)
        if normalize:
            h = F.normalize(h, p=2, dim=-1)
        return h



class NodeUpdateV2(nn.Module):
    """
    Version 2: Models similarity via attributes and relationships (Homogeneous Graph).
    Requires node type information to apply correct update rules.
    Update (Unit):    h = σ( Q_node_unit * (x || m_attr || m_rel) )
    Update (Attribute): h = σ( Q_node_attr * (x || m_attr) )
    """
    def __init__(self, node_dim, node_dim_out, edge_dim, msg_dim, n_rel_types, dropout=0.1, testing=False):
        super().__init__()
        self.node_dim = node_dim
        self.node_dim_out = node_dim_out
        self.msg_dim = msg_dim
        self.n_rel_types = n_rel_types

        # Message Passing Layers
        self.attr_message = AttributeMessagePassing(node_dim, edge_dim, msg_dim, testing)
        # Create n_rel_types instances of RelationalMessagePassing
        self.rel_message_passers = nn.ModuleList([
            RelationalMessagePassing(node_dim, msg_dim, testing=testing) for _ in range(n_rel_types)
        ])

        # Update Networks (Separate for unit/attribute)
        self.update_nn_unit = nn.Linear(node_dim + 2 * msg_dim, node_dim_out)
        self.update_nn_attr = nn.Linear(node_dim + msg_dim, node_dim_out)
        self.dropout_layer = nn.Dropout(dropout)
        
        init_testing(self.update_nn_unit, testing)
        init_testing(self.update_nn_attr, testing)

    def forward(self, node_emb, data_edge_index, edge_emb, rel_edge_index=None, 
                rel_edge_type=None, is_unit=None, normalize=False, **kwargs):
        # node_emb: Node features [N, hidden_dim]
        # is_unit_node: Boolean tensor [N]
        # data_edge_index: [2, E_attr]
        # edge_emb:        [E_attr, edge_dim]
        # rel_edge_index:  [2, E_rel]
        # rel_edge_type:   [E_rel] indicating relation type (0 to n_rel_types-1)

        N = node_emb.size(0)
        device = node_emb.device
        attr_messages = self.attr_message(node_emb, data_edge_index, edge_emb) # Shape [N, hidden_dim]
        # 2. Compute Relational Messages (accumulated only for unit nodes)
        rel_messages_accum = torch.zeros(N, self.msg_dim, device=device)
        if rel_edge_index is not None and rel_edge_type is not None and self.n_rel_types > 0:
            if not isinstance(rel_edge_type, torch.LongTensor):
                 rel_edge_type = rel_edge_type.long()

            for r in range(self.n_rel_types):
                type_mask = (rel_edge_type == r)
                if type_mask.sum() > 0:
                    r_edge_index = rel_edge_index[:, type_mask]
                    rel_passer_instance = self.rel_message_passers[r]
                    r_messages = rel_passer_instance(node_emb, r_edge_index)
                    rel_messages_accum += r_messages
        
        # 3. Node Update (Conditional based on node type)
        h_final = torch.zeros((N, self.node_dim_out), device=device)   # [n_node, node_dim_out]
        # identity = x

        # Indices for unit and attribute nodes
        unit_indices = torch.where(is_unit)[0]
        attr_indices = torch.where(~is_unit)[0]

        # Update Unit Nodes: σ( Q_node_unit * (x || m_attr || m_rel) )
        if len(unit_indices) > 0:
            x_unit = node_emb[unit_indices]
            m_attr_unit = attr_messages[unit_indices]
            m_rel_unit = rel_messages_accum[unit_indices]
            update_input_unit = torch.cat([x_unit, m_attr_unit, m_rel_unit], dim=1)
            h_unit = F.relu(self.update_nn_unit(update_input_unit))

            h_final[unit_indices] = h_unit
            # Add residual connection from original unit features
            # h_final[unit_indices] = self.norm_unit(h_unit + identity[unit_indices])
        
        # Update Attribute Nodes: σ( Q_node_attr * (x || m_attr) )
        if len(attr_indices) > 0:
            x_attr = node_emb[attr_indices]
            m_attr_attr = attr_messages[attr_indices]
            # Note: m_rel is implicitly zero for attributes here as we don't use rel_messages_accum
            update_input_attr = torch.cat([x_attr, m_attr_attr], dim=1)
            h_attr = F.relu(self.update_nn_attr(update_input_attr))
            h_final[attr_indices] = h_attr
            # Add residual connection from original attribute features
            # h_final[attr_indices] = self.norm_attr(h_attr + identity[attr_indices])

        h_final = self.dropout_layer(h_final)
        if normalize:
            h_final = F.normalize(h_final, p=2, dim=-1)
        return h_final

#######################################
# Edge Updating Modeling Module
#######################################


class EdgeUpdating(nn.Module):
    """Updates edge embeddings based on node embeddings."""
    def __init__(self, node_dim, edge_dim, edge_dim_out, testing=False):
        super(EdgeUpdating, self).__init__()
        self.edge_update = nn.Linear(2 * node_dim + edge_dim, edge_dim_out)  # Previous edge attr + source node + target node
        init_testing(self.edge_update, testing)
        
    def forward(self, edge_index, edge_emb, node_emb):
        """
        Args:
            edge_index: Edge indices [2, E]
            edge_attr: Previous edge attributes [E, 1]
            x: Node embeddings [N, hidden_dim]
        Returns:
            Updated edge attributes [E, hidden_dim]
        """
        src, dst = edge_index
        src_emb = node_emb[src]  # [E, hidden_dim]
        dst_emb = node_emb[dst]  # [E, hidden_dim]
        # Concatenate previous edge attr with source and target embeddings
        edge_input = torch.cat([edge_emb, src_emb, dst_emb], dim=1)
        # Apply update function
        updated_edge_attr = F.relu(self.edge_update(edge_input))
        return updated_edge_attr

#######################################
# Augmented Contextual Modeling (ACM) Modules
#######################################

class AugContextualModelingV1(nn.Module):
    """
    Version 1: Imputation via predicted attributes.
    
    This implementation directly works with the attribute values (both observed and imputed).
    The process is:
    1. Create attribute vectors for each unit by combining observed and imputed values
    2. Embed these vectors into a latent space
    3. Project them to treatment and outcome specific spaces
    
    Formula from specification:
    x_u^t = P_t · x_u
    x_u^y = P_y · x_u
    x_u = f_embed(AGG_attr({(1-M_u,p)×ê_u,p + M_u,p×D_u,p | p ∈ P_C^ctx}))
    """
    def __init__(self, node_dim, edge_dim, n_attrs, dropout=0.1, testing=False):
        """
        Initialize the contextual modeling component.
        
        Args:
            hidden_dim: Dimension of the hidden representations
            n_attrs: Number of attributes in the data
            testing: Whether to initialize parameters for testing
        """
        super(AugContextualModelingV1, self).__init__()
        
        # Network layers for embedding and projections
        self.attr_embed = nn.Linear(n_attrs, node_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.proj_t = nn.Linear(node_dim, node_dim)  # Project to treatment space
        self.proj_y = nn.Linear(node_dim, node_dim)  # Project to outcome space
        
        # Initialize for testing if needed
        init_testing(self.attr_embed, testing)
        init_testing(self.proj_t, testing)
        init_testing(self.proj_y, testing)
    
    def forward(self, unit_emb, attr_emb, imputed_attrs, data_edge_index, data_edge_attr, observed_mask, n_units, n_attrs):
        """
        Forward pass to compute treatment and outcome representations.
        
        Args:
            unit_emb: Unit node embeddings [n_units, hidden_dim]
            attr_emb: Attribute node embeddings [n_attrs, hidden_dim]
            imputed_attrs: Imputed attributes [n_units * n_attrs]
            data_edge_index: Edge indices [2, E]
            data_edge_attr: Edge attributes [E, 1]
            n_units: Number of unit nodes
            n_attrs: Number of attribute nodes
            
        Returns:
            Tuple of (x_t, x_y) where:
            - x_t: Treatment-specific representations [n_units, hidden_dim]
            - x_y: Outcome-specific representations [n_units, hidden_dim]
        """
        # 1. Create attribute vectors by combining observed and imputed values

        # print(imputed_attrs.shape)
        # print(data_edge_attr.shape)
        obs_attr_vector = data_edge_attr.squeeze()      # unit -> attr, attr -> unit
        obs_attr_vector = obs_attr_vector[:len(obs_attr_vector)//2]

        attr_vectors = imputed_attrs.clone()            # n_units * n_attrs imputed values

        # print("attr_vectors shape", attr_vectors.shape)
        # print("observed_mask shape", observed_mask.shape, "\t#obsered", observed_mask.sum())
        # print("obs_attr_vector", obs_attr_vector.shape)
        attr_vectors[observed_mask] = obs_attr_vector   # observed values

        # 2. Embed attribute vectors into a latent space
        x = F.relu(self.attr_embed(attr_vectors.view(n_units, n_attrs)))
        x1 = self.dropout_layer(x)

        # 3. Project to treatment and outcome spaces
        x_t = self.proj_t(x1)
        x_y = self.proj_y(x1)
        
        return x, x_t, x_y


class AugContextualModelingV2(nn.Module):
    def __init__(self, node_dim, edge_dim, n_attrs, dropout=0.1, testing=False):
        """
        Initialize the contextual modeling component.
        
        Args:
            node_dim: Dimension of the node representations
            edge_dim: Dimension of the edge representations
            dropout: Dropout rate
            testing: Whether to initialize parameters for testing
        """
        super(AugContextualModelingV2, self).__init__()
        
        # Network layers for projections
        self.proj_imp = nn.Linear(2 * node_dim, node_dim)   # Project concatenated node embeddings
        self.proj_obs = nn.Linear(edge_dim, node_dim)       # Project edge attributes
        
        # Projection for final unit representation
        self.proj_final = nn.Linear(node_dim + n_attrs*node_dim, node_dim)  # +1 for the unit's own embedding
        
        # Projections for treatment and outcome
        self.proj_t = nn.Linear(node_dim, node_dim)  # Project to treatment space
        self.proj_y = nn.Linear(node_dim, node_dim)  # Project to outcome space
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize for testing if needed
        if testing:
            init_testing(self.proj_imp, testing)
            init_testing(self.proj_obs, testing)
            init_testing(self.proj_final, testing)
            init_testing(self.proj_t, testing)
            init_testing(self.proj_y, testing)
        
        self.node_dim = node_dim
    
    def forward(self, unit_emb, attr_emb, imputed_attrs, data_edge_index, edge_emb, observed_mask, n_units, n_attrs):
        """
        edge_emb: [E, node_dim]
        """
        device = unit_emb.device
        hidden_dim = self.node_dim
        
        # Create edge source and destination masks for vectorized operations
        src, dst = data_edge_index
        unit_to_attr_mask = (src < n_units) & (dst >= n_units)
        attr_to_unit_mask = (dst < n_units) & (src >= n_units)
        
        # Prepare edge information for observed attributes
        unit_attr_edges = torch.zeros((n_units, n_attrs), dtype=torch.bool, device=device)
        edge_features = torch.zeros((n_units, n_attrs, hidden_dim), device=device)
        
        # Process unit-to-attribute edges
        if unit_to_attr_mask.any():
            u2a_src = src[unit_to_attr_mask]
            u2a_dst = dst[unit_to_attr_mask] - n_units
            u2a_edge_idx = torch.arange(data_edge_index.size(1), device=device)[unit_to_attr_mask]
            unit_attr_edges[u2a_src, u2a_dst] = True
            edge_features[u2a_src, u2a_dst] = self.proj_obs(edge_emb[u2a_edge_idx])
        
        # Process attribute-to-unit edges
        if attr_to_unit_mask.any():
            a2u_src = src[attr_to_unit_mask] - n_units
            a2u_dst = dst[attr_to_unit_mask]
            a2u_edge_idx = torch.arange(data_edge_index.size(1), device=device)[attr_to_unit_mask]
            unit_attr_edges[a2u_dst, a2u_src] = True
            edge_features[a2u_dst, a2u_src] = self.proj_obs(edge_emb[a2u_edge_idx])
        
        # Vectorized computation of all unit-attribute pairs
        unit_expanded = unit_emb.unsqueeze(1).expand(-1, n_attrs, -1)  # [n_units, n_attrs, hidden_dim]
        attr_expanded = attr_emb.unsqueeze(0).expand(n_units, -1, -1)  # [n_units, n_attrs, hidden_dim]
        
        # Concatenate along feature dimension
        concat_unit_attr = torch.cat([unit_expanded, attr_expanded], dim=2)  # [n_units, n_attrs, 2*hidden_dim]
        
        # Reshape for batch linear projection
        reshaped_concat = concat_unit_attr.view(-1, 2 * hidden_dim)  # [n_units * n_attrs, 2*hidden_dim]
        imp_features = self.proj_imp(reshaped_concat).view(n_units, n_attrs, hidden_dim)  # [n_units, n_attrs, hidden_dim]
        
        # Use observed features where available, imputed features elsewhere
        combined_features = torch.where(
            unit_attr_edges.unsqueeze(-1),
            edge_features,
            imp_features
        )
        # Aggregate attribute features for each unit
        aggregated_attr_features = combined_features.reshape(n_units, n_attrs * hidden_dim)
        
        # Concatenate with unit embeddings
        final_concat = torch.cat([unit_emb, aggregated_attr_features], dim=1)  # [n_units, hidden_dim + n_attrs*hidden_dim]
        
        # Apply final projection and activations
        x_u = F.relu(self.proj_final(final_concat))
        x1 = self.dropout_layer(x_u)  # Fixed incorrect reference to dropout
        
        # Project to treatment and outcome spaces
        x_t = self.proj_t(x1)
        x_y = self.proj_y(x1)
        
        return x_u, x_t, x_y

class AugContextualModelingV3(nn.Module):
    """
    Version 3: Imputation via representation learning - Efficient Implementation.
    
    This implementation uses batch processing and vectorized operations to achieve better performance:
    1. Pre-computes all possible unit-attribute pair embeddings in a batch
    2. Uses sparse tensor operations to handle observed vs imputed values
    3. Performs aggregation in a vectorized manner
    
    Formula from specification:
    x_u^t = P_t · x_u
    x_u^y = P_y · x_u
    x_u = σ(AGG({P_imp · (h_u^(L) ⊕ h_p^(L)) ⊕ M_u,p × P_obs · e_u,p^(L) | p ∈ P_C^ctx}))
    """
    def __init__(self, node_dim, edge_attr_dim, aggr='mean', dropout=0.1, testing=False):
        """
        Initialize the contextual modeling component.
        
        Args:
            hidden_dim: Dimension of the hidden representations
            aggr: Aggregation function ('mean', 'sum', 'max')
            testing: Whether to initialize parameters for testing
        """
        super(AugContextualModelingV2, self).__init__()
        
        # Network layers for projections
        self.proj_imp = nn.Linear(2 * node_dim, node_dim)  # Project concatenated embeddings
        self.proj_obs = nn.Linear(edge_attr_dim, node_dim)               # Project edge attributes 
        self.dropout_layer = nn.Dropout(dropout)
        self.proj_t = nn.Linear(node_dim, node_dim)        # Project to treatment space
        self.proj_y = nn.Linear(node_dim, node_dim)        # Project to outcome space
        
        # Initialize for testing if needed
        init_testing(self.proj_imp, testing)
        init_testing(self.proj_obs, testing)
        init_testing(self.proj_t, testing)
        init_testing(self.proj_y, testing)
        
        # Aggregation function
        self.aggr = aggr
    
    def forward(self, unit_emb, attr_emb, imputed_attrs, data_edge_index, data_edge_attr, n_units, n_attrs):
        """
        Efficient forward pass using batch operations.
        
        Args:
            unit_emb: Unit node embeddings [n_units + n_attrs, hidden_dim]
            attr_emb: Attribute node embeddings [n_attrs, hidden_dim]
            imputed_attrs: Dictionary mapping (unit_idx, attr_idx) to imputed values
            data_edge_index: Edge indices [2, E]
            data_edge_attr: Edge attributes [E, 1]
            n_units: Number of unit nodes
            n_attrs: Number of attribute nodes
            
        Returns:
            Tuple of (x_t, x_y) where:
            - x_t: Treatment-specific representations [n_units, hidden_dim]
            - x_y: Outcome-specific representations [n_units, hidden_dim]
        """
        device = unit_emb.device
        hidden_dim = unit_emb.size(1)
        
        # 1. Create a mapping from (unit, attr) to edge_idx for all observed edges
        edge_dict = {}
        for e in range(data_edge_index.size(1)):
            src, dst = data_edge_index[0, e].item(), data_edge_index[1, e].item()
            # if not observed_mask[e]:        # jason randomly edge drop impact
            #     continue
            if src < n_units and dst >= n_units:
                edge_dict[(src, dst - n_units)] = e
            elif dst < n_units and src >= n_units:
                edge_dict[(dst, src - n_units)] = e
        
        # 2. Pre-compute all unit-attribute pair imputed representations
        # First create tensors to hold all unit-attribute pairs
        all_unit_indices = []
        all_attr_indices = []
        all_edge_indices = []
        
        for u in range(n_units):
            for p in range(n_attrs):
                all_unit_indices.append(u)
                all_attr_indices.append(p)
                all_edge_indices.append(edge_dict.get((u, p), -1))
        
        # Convert to tensors
        all_unit_indices = torch.tensor(all_unit_indices, device=device)
        all_attr_indices = torch.tensor(all_attr_indices, device=device)
        all_edge_indices = torch.tensor(all_edge_indices, device=device)
        
        # 3. Compute imputed representations for all pairs in one batch
        # Get embeddings for all units and attributes
        batch_unit_emb = unit_emb[all_unit_indices]  # [n_units*n_attrs, hidden_dim]
        batch_attr_emb = attr_emb[all_attr_indices]  # [n_units*n_attrs, hidden_dim]
        
        batch_concat = torch.cat([batch_unit_emb, batch_attr_emb], dim=1)
        imp_rep = self.proj_imp(batch_concat)
        
        # 4. Compute observed representations where available
        
        representations = imp_rep.clone()  # Start with imputed representations

        has_edge = all_edge_indices >= 0
        if has_edge.any():
            edge_attr_indices = all_edge_indices[has_edge]
            edge_values = data_edge_attr[edge_attr_indices]  # [num_edges, 1]
            obs_rep = self.proj_obs(edge_values)  # [num_edges, hidden_dim]
            
            # Get indices of pairs with edges
            pair_indices = torch.nonzero(has_edge).squeeze(1)
            
            # Replace imputed representations with observed ones for pairs with edges
            representations[pair_indices] += obs_rep
        
        # 5. Reshape to group by unit for aggregation
        representations_by_unit = torch.zeros(n_units, n_attrs, hidden_dim, device=device)
        for i, (u, p) in enumerate(zip(all_unit_indices, all_attr_indices)):
            representations_by_unit[u, p] = representations[i]
        
        # 6. Perform aggregation for each unit
        if self.aggr == 'mean':
            x_units = torch.mean(representations_by_unit, dim=1)  # [n_units, hidden_dim]
        elif self.aggr == 'sum':
            x_units = torch.sum(representations_by_unit, dim=1)  # [n_units, hidden_dim]
        elif self.aggr == 'max':
            x_units = torch.max(representations_by_unit, dim=1)[0]  # [n_units, hidden_dim]
        
        x_units = F.relu(x_units)
        # self.dropout_layer(x_units)
        x1 = self.dropout_layer(x_units)
        # 7. Project to treatment and outcome spaces
        x_t = self.proj_t(x1)
        x_y = self.proj_y(x1)
        
        return x_units, x_t, x_y

#######################################
# Interference Modeling Modules
#######################################

class RelationalAttention(nn.Module):
    """
    Computes attention weights alpha_uv based on relation type r.
    Uses the *base* unit representations from ACM for computing attention.
    """
    def __init__(self, node_dim: int, n_rel_types: int, leaky_relu_slope: float = 0.2, testing=False):
        """
        Args:
            input_dim: Dimension of base unit representations (x_u_base from ACM).
            n_rel_types: Number of distinct relation types (data.n_rel_types).
            leaky_relu_slope: Slope for LeakyReLU activation.
            testing: Whether to initialize parameters for testing
        """
        super().__init__()
        self.input_dim = node_dim
        self.n_rel_types = n_rel_types
        # Learnable parameter vector a_r per relation type
        self.a_r_params = nn.Parameter(torch.Tensor(n_rel_types, 2 * node_dim))
        
        if testing:
            nn.init.ones_(self.a_r_params)
        else:
            nn.init.xavier_uniform_(self.a_r_params.data)

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x_base: torch.Tensor, rel_edge_index: torch.Tensor, rel_edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_base: Base unit representations from ACM [N_units, input_dim].
            rel_edge_index: Relational edge index [2, N_rel_edges]. Indices are unit indices.
            rel_edge_type: Type of each relational edge [N_rel_edges].

        Returns:
            alpha: Attention weights for each edge [N_rel_edges].
        """
        # n_units = x_base.size(0)
        src, dst = rel_edge_index[0], rel_edge_index[1]
        x_u = x_base[src]  # [N_rel_edges, input_dim]
        x_v = x_base[dst]  # [N_rel_edges, input_dim]
        
        x_uv_concat = torch.cat([x_u, x_v], dim=-1)  # [N_rel_edges, 2 * input_dim]
        a_r = self.a_r_params[rel_edge_type]  # [N_rel_edges, 2 * input_dim]    choose the a parameter for each edge
        
        # sigma( a_r^T * (x_u || x_v) )
        e_activated = self.leaky_relu(torch.sum(a_r * x_uv_concat, dim=-1))  # [N_rel_edges]
        alpha = scatter_softmax(e_activated, src, dim=0)
        return alpha 
    

class InterferenceModelingV1(nn.Module):
    """
    IM Version 1: Interference via Imputed Treatment.
    Aggregates neighbors' *imputed* treatments (hat_t_v) using attention.

    Assumes the main model computes and provides the base unit representations (x_u_base)
    from ACM and the imputed treatments (hat_t).
    """
    def __init__(self, node_dim: int, n_rel_types: int, treatment_dim: int = 1, testing=False):
        """
        Args:
            node_dim: Dimension of base unit representations (x_u_base) used for attention.
            n_rel_types: Number of distinct relation types (data.n_rel_types).
            treatment_dim: Dimension of the (imputed) treatment hat_t. Usually 1.
            testing: Whether to initialize parameters for testing
        """
        super().__init__()
        self.attention = RelationalAttention(node_dim, n_rel_types, testing=testing)
        self.treatment_dim = treatment_dim

    def forward(self, x_unit_base: torch.Tensor, hat_t: torch.Tensor, rel_edge_index: torch.Tensor, rel_edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_unit_base: Base unit representations from ACM [N_units, base_repr_dim]. Used for attention.
            hat_t: Imputed treatment values computed by the main model [N_units].
            rel_edge_index: Relational edge index [2, N_rel_edges].
            rel_edge_type: Type of each relational edge [N_rel_edges].

        Returns:
            g_u: Interference signal for each unit [N_units].
        """
        n_units = x_unit_base.size(0)
        # device = x_unit_base.device
        src, dst = rel_edge_index[0], rel_edge_index[1]

        # 1. Calculate attention weights alpha_uv using x_unit_base
        alpha = self.attention(x_unit_base, rel_edge_index, rel_edge_type) # [N_rel_edges]

        # print("alpha", alpha.shape, '\n', alpha)

        # 2. Get neighbors' imputed treatments hat_t_v
        hat_t_v = hat_t[dst] # [N_rel_edges]

        # 3. Compute weighted treatments: alpha_uv * hat_t_v
        weighted_t = alpha * hat_t_v # [N_rel_edges]
        # Weighted treatments calculated

        # 4. Aggregate weighted treatments for each unit u (target node 'u' is indexed by 'src')
        g_u = scatter(weighted_t, src, dim=0, dim_size=n_units, reduce="sum")

        # g_u = torch.zeros(n_units, self.treatment_dim, device=device)
        # g_u.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.treatment_dim), weighted_t) # Sum messages targeting node 'src'

        return g_u.unsqueeze(-1)    # shape [n_units, 1]


class InterferenceModelingV2(nn.Module):
    """
    IM Version 2: Interference via Representation.
    Aggregates neighbors' *treatment-specific representations* (x_v^t) using attention.

    Assumes the main model computes and provides the base unit representations (x_u_base)
    and treatment-specific representations (x_t) from ACM.
    """
    def __init__(self, node_dim: int, n_rel_types: int, t_repr_dim: int, testing=False):
        """
        Args:
            base_repr_dim: Dimension of base unit representations (x_u_base) used for attention.
            n_rel_types: Number of distinct relation types (data.n_rel_types).
            t_repr_dim: Dimension of the treatment-specific representation x^t from ACM.
            testing: Whether to initialize parameters for testing
        """
        super().__init__()
        self.attention = RelationalAttention(node_dim, n_rel_types, testing=testing)
        self.t_repr_dim = t_repr_dim

    # def forward(self, x_unit_base: torch.Tensor, x_t: torch.Tensor, rel_edge_index: torch.Tensor, rel_edge_type: torch.Tensor) -> torch.Tensor:
    #     n_units = x_unit_base.size(0)
    #     src, dst = rel_edge_index[0], rel_edge_index[1]

    #     print(f"--- Debugging Shapes in InterferenceModelingV2.forward ---")
    #     print(f"x_unit_base.shape: {x_unit_base.shape}")
    #     print(f"x_t.shape: {x_t.shape}")  # Crucial: provides N_units and t_repr_dim
    #     print(f"rel_edge_index.shape: {rel_edge_index.shape}")  # Crucial: provides N_rel_edges
    #     print(f"rel_edge_type.shape: {rel_edge_type.shape}")
    #     print(f"Derived n_units: {n_units}")
    #     print(f"src.shape: {src.shape}") # Should be [N_rel_edges]
    #     print(f"dst.shape: {dst.shape}") # Should be [N_rel_edges]

    #     # 1. Calculate attention weights alpha_uv using x_unit_base
    #     alpha = self.attention(x_unit_base, rel_edge_index, rel_edge_type) # [N_rel_edges]
    #     print(f"alpha.shape: {alpha.shape}") # Should be [N_rel_edges]

    #     # For the operation x_t[dst]
    #     print(f"Shapes for x_t[dst]: x_t.shape={x_t.shape}, dst.shape={dst.shape}")
       
    #     # The failing line:
    #     # weighted_xt = alpha.unsqueeze(-1) * x_t[dst]
    #     print(f"x_t shape: {x_t.shape}\t dst shape: {dst.shape}")
    #     print(f"--- End Debugging Shapes ---")

    #     weighted_xt = alpha.unsqueeze(-1) * x_t[dst]

    #     g_u = scatter(weighted_xt, src, dim=0, dim_size=n_units, reduce="sum")
    #     return g_u

    def forward(self, x_unit_base: torch.Tensor, x_t: torch.Tensor, rel_edge_index: torch.Tensor, rel_edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_unit_base: Base unit representations from ACM [N_units, base_repr_dim]. Used for attention.
            x_t: Treatment-specific representations from ACM [N_units, t_repr_dim].
            rel_edge_index: Relational edge index [2, N_rel_edges].
            rel_edge_type: Type of each relational edge [N_rel_edges].

        Returns:
            g_u: Interference signal for each unit [N_units, t_repr_dim].
        """
        n_units = x_unit_base.size(0)
        src, dst = rel_edge_index[0], rel_edge_index[1]

        # 1. Calculate attention weights alpha_uv using x_unit_base
        alpha = self.attention(x_unit_base, rel_edge_index, rel_edge_type) # [N_rel_edges]

        # 2. Get neighbors' treatment representations x_v^t
        # x_t_v = x_t[dst] # [N_rel_edges, t_repr_dim]

        # 3. Compute weighted representations: alpha_uv * x_v^t
        weighted_xt = alpha.unsqueeze(-1) * x_t[dst] 

        # 4. Aggregate weighted representations for each unit u (target node 'u' is indexed by 'src')
        g_u = scatter(weighted_xt, src, dim=0, dim_size=n_units, reduce="sum")

        # g_u = torch.zeros(n_units, self.t_repr_dim, device=device)
        # g_u.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.t_repr_dim), weighted_xt) # Sum messages targeting node 'src'

        return g_u


#######################################
# Prediction Heads
#######################################

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
        activation='relu',
        dropout=0.0,
        output_activation=None,
        testing=False
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
            if testing:
                init_testing(linear, testing)
            
            self.layers.append(nn.Sequential(
                linear,
                get_activation(activation),
                nn.Dropout(dropout)
            ))
            prev_dim = hidden_dim
        
        # Add output layer
        output_linear = nn.Linear(prev_dim, output_dim)
        if testing:
            init_testing(output_linear, testing)
            
        self.layers.append(nn.Sequential(
            output_linear,
            get_activation(output_activation)
        ))
    
    def forward(self, x):
        """Forward pass through the MLP."""
        input_var = x
        for layer in self.layers:
            input_var = layer(input_var)
        return input_var


#######################################
# Loss Functions
#######################################

def compute_imputation_loss(outputs, data, use_index=None, **params):
    """
    Compute loss for attribute imputation - optimized version.
    
    Args:
        outputs: Model outputs containing imputed_attrs
        data: Data from load_heter_graph
        n_units: Number of unit nodes
        use_index: bool [N*M]
    
    Returns:
        Imputation loss
    """
    if use_index is not None:
        counter = 0
        all_edge_idx = torch.zeros_like(data.observed_mask)
        for i in range(len(data.observed_mask)):
            if data.observed_mask[i]:
                all_edge_idx[i] = counter
                counter += 1
        
        # all_edge_idx = torch.tensor(all_edge_idx, dtype=torch.long)
        all_edge_idx = all_edge_idx.clone().to(torch.long)
        use_observe_mask = data.observed_mask & use_index
        observed_edge_idx = all_edge_idx[use_observe_mask]
        
    else:
        use_observe_mask = data.observed_mask
        observed_edge_idx = torch.ones(len(data.edge_attr)//2).to(torch.bool)
    
    true_vals = data.edge_attr[:len(data.edge_attr)//2][observed_edge_idx]
    pred_vals = outputs['imputed_attrs'][use_observe_mask]
    
    # if not on_observed:
    #     true_data0 = data.unobs_edge_attr
    #     true_vals0 = true_data0[:len(true_data0)//2]
    #     true_vals = torch.cat([true_vals, true_vals0], dim=0)

    #     pred_vals0 = outputs['imputed_attrs'][data.unobserved_mask]
    #     pred_vals = torch.cat([pred_vals, pred_vals0], dim=0)

    if true_vals.numel() == 0 or pred_vals.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=pred_vals.device)
    
    return F.mse_loss(pred_vals, true_vals)
    

def compute_treatment_loss(outputs, data, unit_indexes=None):
    """
    Compute loss for treatment prediction.
    
    Args:
        outputs: Model outputs containing pred_treatments
        data: Data from load_heter_graph
    
    Returns:
        Treatment prediction loss
    """
    device = data.treatment.device
    treatment = data.treatment
    treatment_mask = data.treatment_mask
    if unit_indexes is not None:
        treatment_mask = data.treatment_mask & unit_indexes
    
    if treatment_mask.sum() > 0:
        # print(outputs['pred_treatments'].dtype, treatment.dtype)
        t_loss = F.binary_cross_entropy(
            outputs['pred_treatments'][treatment_mask],
            treatment[treatment_mask].float()
        )
    else:
        t_loss = torch.tensor(0.0, device=device)
    
    return t_loss


def compute_outcome_loss(outputs, data, unit_indexes=None):
    """
    Compute loss for outcome prediction.
    
    Args:
        outputs: Model outputs containing pred_outcomes
        data: Data from load_heter_graph
    
    Returns:
        Outcome prediction loss
    """
    device = data.outcome.device
    outcome = data.outcome
    outcome_mask = data.outcome_mask

    if unit_indexes is not None:       # train, validataion, test
        outcome_mask = data.outcome_mask & unit_indexes
    
    if outcome_mask.sum() > 0:
        y_loss = F.mse_loss(
            outputs['pred_outcomes'][outcome_mask],
            outcome[outcome_mask]
        )
    else:
        y_loss = torch.tensor(0.0, device=device)
    
    return y_loss


def compute_balance_loss(outputs, data, unit_indexes=None):
    """
    Compute representation balancing loss.
    
    Args:
        outputs: Model outputs
        data: Data from load_heter_graph
        hidden_dim: Dimension of hidden representations
        version: BaLu version (affects how balance loss is computed)
    
    Returns:
        Balance loss measuring representation divergence between treatment groups
    """
    device = data.treatment.device
    treatment = data.treatment
    treatment_mask = data.treatment_mask
    
    # Identify treated and control units
    treated_mask = (treatment > 0.5) & treatment_mask
    control_mask = (treatment <= 0.5) & treatment_mask
    
    if unit_indexes is not None:
        treated_mask &= unit_indexes
        control_mask &= unit_indexes
    
    if treated_mask.sum() > 0 and control_mask.sum() > 0:
        # For Version 2, use representation interference
        treated_rep = outputs['unit_embeddings'][treated_mask]
        # torch.cat([
        #     outputs['interference'][treated_mask],
        #     outputs['unit_embeddings'][treated_mask]
        # ], dim=1)
        
        control_rep = outputs['unit_embeddings'][control_mask] 
        # torch.cat([
        #     outputs['interference'][control_mask],
        #     outputs['unit_embeddings'][control_mask]
        # ], dim=1)
        
        b_loss, _ = wasserstein(treated_rep, control_rep)
        
    else:
        b_loss = torch.tensor(0.0, device=device)
    
    return b_loss

def wasserstein1(x,y,p=0.5,lam=10,its=10,sq=False,backpropT=False,cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]
    
    x = x.squeeze()
    y = y.squeeze()
    
#    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x,y) #distance_matrix(x,y,p=2)
    
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M,10.0/(nx*ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta*torch.ones(M[0:1,:].shape)
    col = torch.cat([delta*torch.ones(M[:,0:1].shape),torch.zeros((1,1))],0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1))/nx,(1-p)*torch.ones((1,1))],0)
    b = torch.cat([(1-p)*torch.ones((ny,1))/ny, p*torch.ones((1,1))],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K/a

    u = a

    for i in range(its):
        u = 1.0/(ainvK.matmul(b/torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b/(torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def wasserstein(x, y, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """return W dist between x and y"""
    nx = x.shape[0]
    ny = y.shape[0]
    device = x.device
    
    x = x.squeeze().contiguous()
    y = y.squeeze().contiguous()
    
    # Use torch.cdist for faster distance calculation
    M = torch.cdist(x, y, p=2.0)
    
    # Estimate lambda and delta
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, min(10.0/(nx*ny), 0.5))  # Cap dropout probability
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()
    
    # Pre-allocate augmented distance matrix
    Mt = torch.zeros(nx+1, ny+1, device=device)
    Mt[:nx, :ny] = M
    Mt[nx, :ny] = delta
    Mt[:nx, ny] = delta
    
    # Compute marginals
    a = torch.zeros(nx+1, 1, device=device)
    a[:nx, 0] = p / nx
    a[nx, 0] = 1-p
    
    b = torch.zeros(ny+1, 1, device=device)
    b[:ny, 0] = (1-p) / ny
    b[ny, 0] = p
    
    # Compute kernel
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + 1e-6
    
    # Sinkhorn iterations
    u = a.clone()
    
    for _ in range(its):
        KTu = K.t() @ u
        v = b / KTu.clamp_min(1e-6)
        Kv = K @ v
        u = a / Kv.clamp_min(1e-6)
    
    # Compute transport cost
    P = u @ v.t() * K
    D = 2 * torch.sum(P * Mt)
    
    return D, Mlam


#######################################
# Utility Functions
#######################################  
def apply_edge_dropout(data, train_data, dropout_rate=0.3, random_seed=None):
    """
    Apply edge dropout to data edges for model robustness, entirely on GPU,
    using torch_scatter.scatter to update observed_mask.
    
    Args:
        data:       Data object from load_heter_graph, with attributes:
                      - edge_index (2×E tensor)
                      - edge_attr (E tensor)
                      - observed_mask ((n_units*n_attrs=E,) tensor)
                      - n_units (int)
                      - n_attrs (int)
        train_data: Copy of data that will be modified in-place.
        dropout_rate: Rate of edges to drop (0.0–1.0).
        random_seed: Optional int for reproducibility.
        
    Returns:
        train_data with randomly dropped edges and updated observed_mask.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    device = data.edge_index.device
    double_E = data.edge_index.size(1)
    single_E = double_E // 2  # assume forward pairs  (E = observed data)
    
    # 1) Sample which forward edges to keep
    keep_E = torch.rand(single_E, device=device) > dropout_rate
    keep_2E = torch.cat([keep_E, keep_E], dim=0)
    
    # 3) Mask out dropped edges
    train_data.edge_index = data.edge_index[:, keep_2E]
    train_data.edge_attr = data.edge_attr[keep_2E]
    train_data.observed_mask = data.observed_mask[keep_E]      # for data (length = E)

    return train_data

#########################################################################################
#       GraphConvolution implementation from pygcn (used in Netdeconf and SPnet)
#       copied from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
#########################################################################################

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



# from torch_geometric.nn import RGCNConv
# import numpy as np

# class RGCN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, n_rel_types, dropout=0.1, n_layers=2):
#         super().__init__()
#         self.rgcs = nn.ModuleList()
#         self.rgcs.append(RGCNConv(in_channels, hidden_channels, n_rel_types))
#         for i in range(1, n_layers):
#             self.rgcs.append(RGCNConv(hidden_channels, out_channels if i == n_layers-1 else hidden_channels, n_rel_types))
#         self.dropout = dropout

#     def forward(self, data):
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         if type(data.arr_X) is np.ndarray:
#             x = torch.from_numpy(data.arr_X)
#             x = x.to(device)
#         else:
#             x = data.arr_X
#         rel_edge_index = data.rel_edge_index
#         rel_edge_type  = data.rel_edge_type
#         for rgc in self.rgcs:
#             x = rgc(x, rel_edge_index, rel_edge_type)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         return x


# def apply_edge_dropout(data, train_data, dropout_rate=0.3, random_seed=None):
#     """
#     Apply edge dropout to data edges for model robustness.
    
#     Args:
#         data: Data object from load_heter_graph
#         dropout_rate: Rate of edges to drop (0.0 - 1.0)
#         random_seed: Optional seed for reproducibility
        
#     Returns:
#         Data object with randomly dropped edges
#     """
#     # Set random seed if provided
#     if random_seed is not None:
#         torch.manual_seed(random_seed)
    
#     device = data.edge_index.device
#     # train_data = copy.deepcopy(train_data)

#     # Generate dropout mask
#     half_n_edges = data.edge_index.size(1) // 2  # double direction unit->attr, attr->unit
#     keep_mask = torch.rand(half_n_edges, device=device) > dropout_rate
#     data_edge_mask = torch.cat((keep_mask, keep_mask), dim=0)
#     train_data.edge_index = data.edge_index[:, data_edge_mask]
#     train_data.edge_attr = data.edge_attr[data_edge_mask]
    
#     # Update observed mask 
#     # new_mask = torch.zeros_like(data.observed_mask)
#     new_mask = torch.zeros_like(data.observed_mask, device=device)
#     for i in range(train_data.edge_index.size(1)//2):
#         src, dst = train_data.edge_index[0, i], train_data.edge_index[1, i]
#         if src < data.n_units and dst >= data.n_units:
#             unit_idx, attr_idx = src.item(), dst.item() - data.n_units
#         elif dst < data.n_units and src >= data.n_units:
#             unit_idx, attr_idx = dst.item(), src.item() - data.n_units
#         new_mask[unit_idx * data.n_attrs + attr_idx] = 1

#     train_data.observed_mask = new_mask
    
#     return train_data




# def wasserstein(x, y, p=0.5, lam=10, its=10): #, sq=False, backpropT=False):
#     """return W dist between x with shape [n1, m] and y with shape [n2, m]"""
#     '''distance matrix M'''
#     device = x.device
#     nx = x.shape[0]
#     ny = y.shape[0]

#     x = x.squeeze()
#     y = y.squeeze()

#     #    pdist = torch.nn.PairwiseDistance(p=2)

#     M = pdist(x, y)  # distance_matrix(x,y,p=2)

#     '''estimate lambda and delta'''
#     M_mean = torch.mean(M)
#     M_drop = F.dropout(M, 10.0 / (nx * ny))
#     delta = torch.max(M_drop).detach()
#     eff_lam = (lam / M_mean).detach()

#     '''compute new distance matrix'''
#     Mt = M
#     row = delta * torch.ones(M[0:1, :].shape, device=device)
#     col = torch.cat([delta * torch.ones(M[:, 0:1].shape, device=device), torch.zeros((1, 1), device=device)], 0)
#     # if cuda:
#     #     row = row.cuda()
#     #     col = col.cuda()
#     Mt = torch.cat([M, row], 0)
#     Mt = torch.cat([Mt, col], 1)

#     '''compute marginal'''
#     a = torch.cat([p * torch.ones((nx, 1), device=device) / nx, (1 - p) * torch.ones((1, 1), device=device)], 0)
#     b = torch.cat([(1 - p) * torch.ones((ny, 1), device=device) / ny, p * torch.ones((1, 1), device=device)], 0)

#     '''compute kernel'''
#     Mlam = eff_lam * Mt
#     temp_term = torch.ones(1, device=device) * 1e-6
#     # if cuda:
#     #     temp_term = temp_term.cuda()
#     #     a = a.cuda()
#     #     b = b.cuda()
#     K = torch.exp(-Mlam) + temp_term
#     U = K * Mt
#     ainvK = K / a

#     u = a

#     for i in range(its):
#         u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
#         # if cuda:
#         #     u = u.cuda()
#     v = b / (torch.t(torch.t(u).matmul(K)))
#     # if cuda:
#     #     v = v.cuda()

#     upper_t = u * (torch.t(v) * K).detach()

#     E = upper_t * Mt
#     D = 2 * torch.sum(E)

#     # if cuda:
#     #     D = D.cuda()

#     return D, Mlam


# def pdist(sample_1, sample_2, norm=2, eps=1e-5):
#     """Compute the matrix of all squared pairwise distances.
#     Arguments
#     ---------
#     sample_1 : torch.Tensor or Variable
#         The first sample, should be of shape ``(n_1, d)``.
#     sample_2 : torch.Tensor or Variable
#         The second sample, should be of shape ``(n_2, d)``.
#     norm : float
#         The l_p norm to be used.
#     Returns
#     -------
#     torch.Tensor or Variable
#         Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
#         ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
#     n_1, n_2 = sample_1.size(0), sample_2.size(0)
#     norm = float(norm)
#     if norm == 2.:
#         norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
#         norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
#         norms = (norms_1.expand(n_1, n_2) +
#                  norms_2.transpose(0, 1).expand(n_1, n_2))
#         distances_squared = norms - 2 * sample_1.mm(sample_2.t())
#         return torch.sqrt(eps + torch.abs(distances_squared))
#     else:
#         dim = sample_1.size(1)
#         expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
#         expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
#         differences = torch.abs(expanded_1 - expanded_2) ** norm
#         inner = torch.sum(differences, dim=2, keepdim=False)
#         return (eps + inner) ** (1. / norm)





# def compute_imputation_loss(outputs, data, n_units):
#     """
#     Compute loss for attribute imputation.
    
#     Args:
#         outputs: Model outputs containing imputed_attrs
#         data: Data from load_heter_graph
#         n_units: Number of unit nodes
    
#     Returns:
#         Imputation loss
#     """
#     device = data.edge_index.device
#     imp_loss = torch.tensor(0.0, device=device)
#     obs_count = 0
    
#     # Process each edge in the data
#     for i in range(data.edge_index.size(1)):
#         src, dst = data.edge_index[0, i], data.edge_index[1, i]
        
#         # Handle unit -> attribute edges
#         if src < n_units and dst >= n_units:
#             unit_idx, attr_idx = src.item(), dst.item() - n_units
#             edge_tuple = (unit_idx, attr_idx)
            
#             if edge_tuple in outputs['imputed_attrs']:
#                 pred_val = outputs['imputed_attrs'][edge_tuple]
#                 true_val = data.edge_attr[i]
#                 imp_loss += F.mse_loss(pred_val, true_val)
#                 obs_count += 1
        
#         # # Handle attribute -> unit edges
#         # elif dst < n_units and src >= n_units:
#         #     unit_idx, attr_idx = dst.item(), src.item() - n_units
#         #     edge_tuple = (unit_idx, attr_idx)
            
#         #     if edge_tuple in outputs['imputed_attrs']:
#         #         pred_val = outputs['imputed_attrs'][edge_tuple]
#         #         true_val = data.edge_attr[i]
#         #         imp_loss += F.mse_loss(pred_val, true_val)
#         #         obs_count += 1
    
#     # Average loss over observed edges
#     if obs_count > 0:
#         imp_loss /= obs_count
    
#     return imp_loss

# class AugContextualModelingV2(nn.Module):
#     """
#     Formula:
#     x_u^t = P_t · x_u
#     x_u^y = P_y · x_u
#     x_u = σ(P·(h_u^(L) ⊕ x_u^emb))
    
#     f_u = ⊕_{p ∈ P_C^ctx} {
#         (1-M_u,p)×P_imp · (h_u^(L) ⊕ h_p^(L)) +
#         M_u,p × P_obs · e_u,p^(L)
#     }
#     """
#     def __init__(self, node_dim, edge_dim, n_attrs, dropout=0.1, testing=False):
#         """
#         Initialize the contextual modeling component.
        
#         Args:
#             node_dim: Dimension of the node representations
#             edge_dim: Dimension of the edge representations
#             dropout: Dropout rate
#             testing: Whether to initialize parameters for testing
#         """
#         super(AugContextualModelingV2, self).__init__()
        
#         # Network layers for projections
#         self.proj_imp = nn.Linear(2 * node_dim, node_dim)  # Project concatenated node embeddings
#         self.proj_obs = nn.Linear(edge_dim, node_dim)  # Project edge attributes
        
#         # Projection for final unit representation
#         self.proj_final = nn.Linear(node_dim + n_attrs*node_dim, node_dim)  # +1 for the unit's own embedding
        
#         # Projections for treatment and outcome
#         self.proj_t = nn.Linear(node_dim, node_dim)  # Project to treatment space
#         self.proj_y = nn.Linear(node_dim, node_dim)  # Project to outcome space
        
#         self.dropout_layer = nn.Dropout(dropout)
        
#         # Initialize for testing if needed
#         if testing:
#             init_testing(self.proj_imp, testing)
#             init_testing(self.proj_obs, testing)
#             init_testing(self.proj_final, testing)
#             init_testing(self.proj_t, testing)
#             init_testing(self.proj_y, testing)
        
#         self.node_dim = node_dim
    
#     def forward(self, unit_emb, attr_emb, imputed_attrs, data_edge_index, data_edge_attr, n_units, n_attrs):
#         """
#         Forward pass implementing the concatenation-based aggregation.
        
#         Args:
#             unit_emb: Unit node embeddings [n_units, hidden_dim]
#             attr_emb: Attribute node embeddings [n_attrs, hidden_dim]
#             data_edge_index: Edge indices [2, E]
#             data_edge_attr: Edge embedding [E, edge_dim]
#             observed_mask: Boolean mask indicating observed attributes [n_units, n_attrs]
#             n_units: Number of unit nodes
#             n_attrs: Number of attribute nodes
            
#         Returns:
#             Tuple of (x_u, x_t, x_y) where:
#             - x_u: Unit representations [n_units, hidden_dim]
#             - x_t: Treatment-specific representations [n_units, hidden_dim]
#             - x_y: Outcome-specific representations [n_units, hidden_dim]
#         """
#         device = unit_emb.device
#         hidden_dim = self.node_dim
        
#         # Create edge mapping for efficient lookup
#         edge_dict = {}
#         for e in range(data_edge_index.size(1)):
#             src, dst = data_edge_index[0, e].item(), data_edge_index[1, e].item()
#             if src < n_units and dst >= n_units:
#                 # Unit -> Attribute edge
#                 edge_dict[(src, dst - n_units)] = e
#             elif dst < n_units and src >= n_units:
#                 # Attribute -> Unit edge
#                 edge_dict[(dst, src - n_units)] = e
        
#         # Generate f_u for each unit
#         all_f_u = []
        
#         for u in range(n_units):
#             # For each unit, collect features for all attributes
#             attr_features = []
#             for p in range(n_attrs):
#                 # Check if this attribute is relevant for the unit (either observed or imputed)
#                 if (u, p) in edge_dict:
#                     # Observed attribute: M_u,p = 1
#                     edge_idx = edge_dict.get((u, p), -1)
#                     feature = self.proj_obs(data_edge_attr[edge_idx])
#                     attr_features.append(feature)
#                 else:
#                     # Imputed attribute: (1-M_u,p) = 1
#                     # Calculate: P_imp · (h_u^(L) ⊕ h_p^(L))
#                     concat_emb = torch.cat([unit_emb[u], attr_emb[p]]).unsqueeze(0)
#                     imp_feature = self.proj_imp(concat_emb).squeeze(0)
#                     attr_features.append(imp_feature)
#             # Concatenate all attribute features to form f_u
#             if attr_features:
#                 f_u = torch.cat(attr_features)
#             else:
#                 f_u = torch.zeros(n_attrs * hidden_dim, device=device)
                
#             # Form h_u^(L) ⊕ f_u
#             combined = torch.cat([unit_emb[u], f_u])
#             all_f_u.append(combined)
        
#         # Stack all unit representations
#         all_unit_repr = torch.stack(all_f_u)
        
#         # Apply final projection: x_u = σ(P·(h_u^(L) ⊕ f_u))
#         x_u = F.relu(self.proj_final(all_unit_repr))
#         x1 = self.dropout_layer(x_u)
        
#         # Project to treatment and outcome spaces
#         x_t = self.proj_t(x1)
#         x_y = self.proj_y(x1)
#         return x_u, x_t, x_y
      