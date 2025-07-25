from src.imputer.balu_grape import BaLu_GRAPE_Imp
from src.imputer.balu_igmc import BaLu_IGMC_Imp
from src.imputer.grape import GRAPE_Imp
from src.imputer.igmc import IGMC

# import torch.nn.functional as F
import torch.nn as nn
from src.interference.linear_modules import MLPNet
from src.interference.gnn_interference import GNN_IF
import torch
from argparse import Namespace


class BaLu(nn.Module):
    def __init__(self, data, args):
        super(BaLu, self).__init__()
        if type(args) == dict:
            args = Namespace(**args)
        self.args = args

        self.gconv, self.rconv = args.gconv, args.rconv     # in ['GCN', 'GraphSAGE', 'RGCN']
        
        ########################################  Imputer  ########################################
        self.imputer = self.set_imputer(data, args)     # data.x.shape[1] -> args.imputer_node_dims[-1]
        
        ######################################## X predictor  ########################################
        self.x_pred = self.set_x_net(data, args)         # 2*args.imputer_node_dims[-1] -> 1 (--> data.n_attrs)

        ######################################## T predictor  ########################################
        self.t_net, self.t_pred = self.set_t_net(data, args)   # args.imputer_node_dims[-1] -> args.imputer_node_dims[-1] -> 1
        
        ######################################## Interference ########################################
        self.interference_net = self.set_interference_net(data, args)

        ######################################## Y(t) predictor  ########################################
        self.y0_pred, self.y1_pred = self.set_y_net(data, args)

    def forward(self, data):
        args = self.args

        ######################################## Node updating ########################################
        impute_emb = self.imputer(data)        # [units + attrs, node_dim] 

        ######################################## attribute X impute ########################################
        h_x, attr_emb = impute_emb[:data.n_units], impute_emb[data.n_units:]
        N, M, K = h_x.size(0), attr_emb.size(0), h_x.size(1)
        
        # Use expand for memory efficiency (creates views, not copies)
        node_expanded = h_x.unsqueeze(1).expand(N, M, K)
        attr_expanded = attr_emb.unsqueeze(0).expand(N, M, K)
        combined = torch.cat([node_expanded, attr_expanded], dim=2)

        X_pred = self.x_pred(combined.view(N * M, 2 * K)).squeeze()   # [E,]
        X_star = self.get_enhanced_X(X_pred, data)          # [n_units, n_attrs]

        ######################################## T predictor ########################################
        h_t = self.t_net(torch.cat([h_x, X_star], dim=1))
        T_pred = self.t_pred(h_t).squeeze()

        ######################################## Interference ########################################
        h_r = self.interference_net(h_t, data)

        ######################################## Y(t) predictor  ########################################
        h_y = []
        h_emb = {'h_r': h_r, 
                'X*': X_star,
                'h_t': h_t}
        for component in args.outcome_rep.split("+"):
            if component not in h_emb:
                raise ValueError(f"Unknown component '{component}' in outcome_rep")
            h_y.append(h_emb[component])
        h_y = torch.cat(h_y, dim=1)

        pred_y0, pred_y1 = self.y0_pred(h_y).squeeze(), self.y1_pred(h_y).squeeze()

        ######################################## Outputs  ########################################

        enhanced_T = T_pred.clone().float()
        enhanced_T[data.treatment_mask] = data.treatment[data.treatment_mask].float()  # observed treatment
        pred_outcomes = torch.where(enhanced_T > 0.5, pred_y1, pred_y0)

        treatment_effects = pred_y1 - pred_y0

        return {
            'imputed_attrs': X_pred,
            'unit_embeddings': h_t,                 # for balance
            'pred_treatments': T_pred,
            'pred_outcomes': pred_outcomes,         # impacted by y_norm
            'pred_outcomes_0': pred_y0,             # impacted by y_norm
            'pred_outcomes_1': pred_y1,             # impacted by y_norm
            'estimated_effect': treatment_effects,  # impacted by y_norm
        }

    def get_enhanced_X(self, X_pred, data):
        observed_X = data.edge_attr.squeeze()   # [2E,]
        observed_X = observed_X[:len(observed_X)//2] # [E,]

        enhanced_X = X_pred.clone()
        enhanced_X[data.observed_mask] = observed_X
        return enhanced_X.view(data.n_units, data.n_attrs)
        
    def set_y_net(self, data, args):
        input_dim = 0
        h_x_dim = args.imputer_node_dims[-1]
        h_r_dim = args.interference_node_dims[-1]
        dims = {'h_r': h_r_dim, 
                'X*': data.n_attrs,
                'h_t': h_x_dim}
        for component in args.outcome_rep.split("+"):
            if component not in dims:
                raise ValueError(f"Unknown component '{component}' in outcome_rep")
            input_dim += dims[component]
        y0 = MLPNet(input_dim=input_dim, 
                    hidden_dims=[h_x_dim,],
                    output_dim=1,
                    dropout=args.dropout)
        y1 = MLPNet(input_dim=input_dim, 
                    hidden_dims=[h_x_dim,],
                    output_dim=1,
                    dropout=args.dropout)
        return y0, y1
        

    def set_interference_net(self, data, args):
        h_x_dim = args.imputer_node_dims[-1]        # from h_t (self.t_net)
        if args.interference == 'GNN':
            return GNN_IF(data,
                          in_dim=h_x_dim,
                          gconv=self.gconv,
                          node_dims=args.interference_node_dims,
                          dropout=args.dropout)
        else:
            raise Exception(f"no such interference module: {args.interference}")


    def set_t_net(self, data, args):    
        input_dim = args.imputer_node_dims[-1] + data.n_attrs  # from h_x (self.imputer) and X^* (self.x_pred, i.e., the enhanced X)           
        h_x_dim = args.imputer_node_dims[-1]        
        t_net = MLPNet(input_dim=input_dim, 
                       output_dim=h_x_dim,
                       hidden_dims=[h_x_dim,], 
                       dropout=args.dropout)
        t_pred = MLPNet(input_dim=h_x_dim, 
                        output_dim=1,
                        hidden_dims=[], 
                        out_activation='sigmoid',
                        dropout=args.dropout)
        return t_net, t_pred


    def set_x_net(self, data, args):
        h_x_dim = args.imputer_node_dims[-1]
        input_dim, output_dim = 2 * h_x_dim, 1
        return MLPNet(input_dim=input_dim, 
                      output_dim=output_dim,
                      hidden_dims=[h_x_dim,], 
                      dropout=args.dropout)


    def set_imputer(self, data, args):
        imputer_out_dim = args.imputer_node_dims[-1]        # last imputer dim, each element in [0,1]
        if args.imputer == 'GRAPE':
            return GRAPE_Imp(data, 
                             node_dims=args.imputer_node_dims, 
                             edge_dim=args.edge_dim, 
                             out_dim=imputer_out_dim, 
                             adj_dropout=args.rel_dropout, 
                             dropout=args.dropout)
        elif args.imputer == 'BaLu_GRAPE':
            return BaLu_GRAPE_Imp(data, 
                                  rconv=self.rconv, 
                                  node_dims=args.imputer_node_dims, 
                                  edge_dim=args.edge_dim,
                                  out_dim=imputer_out_dim,
                                  adj_dropout=args.rel_dropout, 
                                  dropout=args.dropout)
        elif args.imputer == 'IGMC':
            return IGMC(data,
                        gconv=self.gconv,
                        node_dims=args.imputer_node_dims,
                        out_dim=imputer_out_dim,
                        adj_dropout=args.rel_dropout, 
                        dropout=args.dropout)
        elif args.imputer == 'BaLu_IGMC':
            return BaLu_IGMC_Imp(data,
                                 gconv=self.gconv,
                                 rconv=self.rconv,
                                 node_dims=args.imputer_node_dims,
                                 out_dim=imputer_out_dim,
                                 adj_dropout=args.rel_dropout, 
                                 dropout=args.dropout)
        else:
            raise Exception(f"has no such imputer module: {args.imputer}")


