
import torch.nn.functional as f
from src.utils import *
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GraphConv, ARMAConv
import torch


''' parser.add_argument('--rep_hidden_dims', type=list, default=[64, 64])
    parser.add_argument('--gnn_dims', type=list, default=[64, 32])
    parser.add_argument("--ind_pol_dims", type=list, default=[64, 32])
'''

class GCN_HSIC(nn.Module):
    def __init__(self, config): # ablation ['no', 'no_rep', '1y_fun']
        super(GCN_HSIC, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rep_net = Rep(config).to(self.device)
        self.ts_net = GraphNewPolicyNetwork(config).to(self.device) 
    
    def forward(self, init_rep, tc, edge_index):
        rep = self.rep_net(init_rep, tc)
        cf_y1, cf_y0, graph_rep = self.ts_net(rep, edge_index)
        # rep = tr.cat((rep, graph_rep), dim=1)
        return cf_y1, cf_y0, rep, graph_rep 


class Rep(nn.Module):

    def __init__(self, config):
        super(Rep, self).__init__()
        self.config = config
        self.device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
        self.rep_hidden_dims = self.config.rep_hidden_dims
        self.features_dim = self.config.features_dim

        t_layers = []
        prev_hidden_size = self.features_dim
        for next_hidden_size in self.rep_hidden_dims:
            t_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.ReLU(),
            ])
            prev_hidden_size = next_hidden_size
        self.t_rep = nn.Sequential(*t_layers)

        c_layers = []
        prev_hidden_size = self.features_dim
        for next_hidden_size in self.rep_hidden_dims:
            c_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.ReLU(),
            ])
            prev_hidden_size = next_hidden_size
        self.c_rep = nn.Sequential(*c_layers)

    def forward(self, init_rep, tc):

        if self.config == '1y_fun':
            rep = self.t_rep(init_rep)
            return rep

        rep = tr.zeros((tc.shape[0], self.rep_hidden_dims[-1])).to(self.device)
        t_idx = (tc == 1).nonzero().flatten()
        c_idx = (tc == 0).nonzero().flatten()
        if len(t_idx) != 0:
            rep[t_idx] = self.t_rep(init_rep[t_idx])
        if len(c_idx) != 0:
            rep[c_idx] = self.c_rep(init_rep[c_idx])

        return rep

class GraphNewPolicyNetwork(nn.Module):

    def __init__(self, config):
        super(GraphNewPolicyNetwork, self).__init__()
        self.config = config
        self.features_dim = self.config.rep_hidden_dims[-1]     #features_dim
        self.ind_pol_dims = self.config.ind_pol_dims
        self.dropout = config.dropout

        self.gnn_fun = self.config.gnn_fun
        self.gnn_dims = self.config.gnn_dims
        self.conv1, self.conv2, self.conv3 = None, None, None
        self.gnn_layers = []
        self.set_gnn()
        self.initialize_gnn_layers()

        # init_dim = self.gnn_dims[-1] + self.config.rep_hidden_dims[-1]  #self.features_dim
        init_dim = self.gnn_dims[-1] if self.config.ablation == 'no_rep' else self.gnn_dims[-1] + self.config.rep_hidden_dims[-1] 

        spill_t_layers = []
        prev_hidden_size = init_dim
        for next_hidden_size in self.ind_pol_dims:
            spill_t_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.ReLU(),
            ])
            prev_hidden_size = next_hidden_size
        spill_t_layers.extend([
            nn.Linear(prev_hidden_size, 1)
        ])
        self.spill_t = nn.Sequential(*spill_t_layers)

        spill_c_layers = []
        prev_hidden_size = init_dim
        for next_hidden_size in self.ind_pol_dims:
            spill_c_layers.extend([
                nn.Linear(prev_hidden_size, next_hidden_size),
                nn.ReLU(),
            ])
            prev_hidden_size = next_hidden_size
        
        spill_c_layers.extend([
            nn.Linear(prev_hidden_size, 1)
        ])
        self.spill_c = nn.Sequential(*spill_c_layers)

        # ind_layers = []
        # prev_hidden_size = init_dim
        # for next_hidden_size in self.ind_pol_dims:
        #     ind_layers.extend([
        #         nn.Linear(prev_hidden_size, next_hidden_size),
        #         nn.ReLU(),
        #     ])
        #     prev_hidden_size = next_hidden_size
        # ind_layers.extend([
        #     nn.Linear(prev_hidden_size, 1) #,
        #     # nn.Sigmoid()
        # ])
        # self.ind_policy = nn.Sequential(*ind_layers)

    def set_gnn(self):
        # Only consider 2 or 3 layers

        if self.gnn_fun == 'GCN':
            self.conv1 = GCNConv(self.features_dim, self.gnn_dims[0], improved=True)
            self.conv2 = GCNConv(self.gnn_dims[0], self.gnn_dims[1], improved=True)
            self.gnn_layers = [self.conv1, self.conv2]
            if len(self.gnn_dims) == 3:
                self.conv3 = GCNConv(self.gnn_dims[1], self.gnn_dims[2], improved=True)
                self.gnn_layers.append(self.conv3)

        elif self.gnn_fun == 'GraphSAGE':
            self.conv1 = SAGEConv(self.features_dim, self.gnn_dims[0], normalize=True)
            self.conv2 = SAGEConv(self.gnn_dims[0], self.gnn_dims[1], normalize=True)
            self.gnn_layers = [self.conv1, self.conv2]
            if len(self.gnn_dims) == 3:
                self.conv3 = SAGEConv(self.gnn_dims[1], self.gnn_dims[2], normalize=True)
                self.gnn_layers.append(self.conv3)

        elif self.gnn_fun == 'GraphConv':
            self.conv1 = GraphConv(self.features_dim, self.gnn_dims[0], aggr='mean')
            self.conv2 = GraphConv(self.gnn_dims[0], self.gnn_dims[1], aggr='mean')
            self.gnn_layers = [self.conv1, self.conv2]
            if len(self.gnn_dims) == 3:
                self.conv3 = GraphConv(self.gnn_dims[1], self.gnn_dims[2], aggr='mean')
                self.gnn_layers.append(self.conv3)

    def initialize_gnn_layers(self):
        for conv in self.gnn_layers:
            for param in conv.parameters():
                tr.nn.init.xavier_uniform_(param, gain=1) if len(param.shape) >= 2 else None

    @staticmethod
    def sample(prob):
        # Use Gumbel trick for Bernoulli sampling, prob is probability for treatment
        logits = tr.stack([1 - prob, prob], dim=1)
        gumbel_sampled = gumbel_softmax(logits, dim=1, hard=True, tau=1)
        sampled = to_float(tr.mm(gumbel_sampled, to_float(np.array([[0], [1]]))))
        return sampled

    def forward(self, rep, edge_index):

        h_rep = f.dropout(f.relu(self.conv1(rep, edge_index)), training=self.training, p=self.dropout)
        h_rep = f.dropout(f.relu(self.conv2(h_rep, edge_index)), training=self.training, p=self.dropout)
        if len(self.gnn_dims) == 3:
            h_rep = f.dropout(f.relu(self.conv3(h_rep, edge_index)), training=self.training, p=self.dropout)
        
        rep = h_rep if self.config.ablation == 'no_rep' else tr.cat((rep, h_rep), dim=1)

        # cf_y = tr.zeros(rep.shape[0], 1).to(self.device)
        # t_idx = (tc == 1).nonzero().flatten()
        # c_idx = (tc == 0).nonzero().flatten()
        # if len(t_idx) != 0:
        #     cf_y[t_idx] = self.spill_t(rep[t_idx])
        # if len(c_idx) != 0:
        #     cf_y[c_idx] = self.spill_c(rep[c_idx])

        cf_y1 = self.spill_t(rep)
        cf_y0 = self.spill_c(rep)

        return tr.squeeze(cf_y1), tr.squeeze(cf_y0), h_rep
    
        # ind_pol_prob = self.ind_policy(rep)
        # ind_pol = self.sample(ind_pol_prob.flatten())

        # return tr.squeeze(ind_pol_prob), tr.squeeze(ind_pol)
