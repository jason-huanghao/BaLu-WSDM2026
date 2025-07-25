import torch
import torch.nn as nn
import torch.nn.functional as F
# from pygcn.layers import GraphConvolution
# import numpy as np


import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
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


class GCN_DECONF(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=False):
        super(GCN_DECONF, self).__init__()

        # self.gc2 = GraphConvolution(nhid, nclass)

        if cuda:
            self.gc = [GraphConvolution(nfeat, nhid).cuda()]
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid).cuda())
            self.gc_t = [GraphConvolution(nfeat, nhid).cuda()]
            for i in range(n_in - 1):
                self.gc_t.append(GraphConvolution(nhid, nhid).cuda())
        else:
            self.gc = [GraphConvolution(nfeat, nhid)]
            for i in range(n_in - 1):
                self.gc.append(GraphConvolution(nhid, nhid))
            self.gc_t = [GraphConvolution(nfeat, nhid)]
            for i in range(n_in - 1):
                self.gc_t.append(GraphConvolution(nhid, nhid))

        self.n_in = n_in
        self.n_out = n_out
        self.nhid = nhid
        if cuda:

            self.out_t00 = [nn.Linear( nhid, nhid).cuda() for i in range(n_out)]
            self.out_t10 = [nn.Linear( nhid, nhid).cuda() for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid, 1).cuda()
            self.out_t11 = nn.Linear(nhid, 1).cuda()

        else:
            self.out_t00 = [nn.Linear(nhid, nhid) for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid, nhid) for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid, 1)
            self.out_t11 = nn.Linear(nhid, 1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, nhid)
        self.pp2 = nn.Linear(nhid, 2)
        self.a = nn.Parameter(torch.empty(size=(4 * nhid, 1)))
        self.leakyrelu = nn.LeakyReLU(0.2)

        if cuda:
            self.pp = self.pp.cuda()
            self.pp2 = self.pp2.cuda()
            self.a = nn.Parameter(torch.empty(size=(4 * nhid, 1)).cuda())
            self.leakyrelu = nn.LeakyReLU(0.2).cuda()
            #self.att_p = nn.Parameter(torch.empty(size=(n, n)).cuda())
        self.pp_act = nn.Sigmoid()
        nn.init.xavier_uniform_(self.a.data, gain=0)
        #nn.init.xavier_uniform_(self.att_p.data, gain=0)

    def forward(self, x, adj, t, cuda=None):
        if cuda is None:
            cuda = torch.cuda.is_available()
        adj_dense = adj.to_dense()
        rep_outcome = F.relu(self.gc[0](x, adj))
        rep_outcome = F.dropout(rep_outcome, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep_outcome = F.relu(self.gc[i](rep_outcome, adj))
            rep_outcome = F.dropout(rep_outcome, self.dropout, training=self.training)

        rep_treatment = F.relu(self.gc_t[0](x, adj))
        rep_treatment = F.dropout(rep_treatment, self.dropout, training=self.training)
        for i in range(1, self.n_in):
            rep_treatment = F.relu(self.gc_t[i](rep_treatment, adj))
            rep_treatment = F.dropout(rep_treatment, self.dropout, training=self.training)
        rep_out_treat = torch.cat((rep_outcome, rep_treatment), 1)
        rep = rep_out_treat

        N = x.shape[0]  # number of units
        if cuda:
            att_final = torch.zeros(N, N).cuda()
        else:
            att_final = torch.zeros(N, N)

        index = adj_dense.nonzero().t()

        # OOM issue
        # att_input = torch.cat((rep[index[0, :], :], rep[index[1, :], :]), dim=1)
        # attention = torch.matmul(att_input, self.a).squeeze()
        # att_final = att_final.index_put(tuple(index), attention) 

        chunk_size = 1000000  # Adjust based on your memory
        n_edges = index.shape[1]

        for start_idx in range(0, n_edges, chunk_size):
            end_idx = min(start_idx + chunk_size, n_edges)
            
            # Get chunk of edge indices
            index_chunk = index[:, start_idx:end_idx]
            
            # Process chunk
            att_input_chunk = torch.cat((rep[index_chunk[0, :], :], rep[index_chunk[1, :], :]), dim=1)
            attention_chunk = torch.matmul(att_input_chunk, self.a).squeeze()
            
            # Update attention matrix
            att_final = att_final.index_put(tuple(index_chunk), attention_chunk)
            
            # Optional: free memory immediately
            del att_input_chunk, attention_chunk
            if cuda:
                torch.cuda.empty_cache()

        att_final = F.softmax(att_final, dim=1)
        att_final = F.dropout(att_final, self.dropout, training=self.training)
        treatment_cur = rep_treatment
        rep_outcome = torch.matmul(att_final, treatment_cur) + rep_outcome 
        treatment_MLP = self.pp(rep_treatment)
        treatment = self.pp_act(self.pp2(treatment_MLP))
        #h_prime= torch.cat((rep_outcome, dim_treat), 1)
        h_prime = F.dropout(rep_outcome, self.dropout, training=self.training)
        rep = h_prime
        rep0 = rep
        rep1 = rep
        for i in range(self.n_out):
            y00 = F.relu(self.out_t00[i](rep0))
            y00 = F.dropout(y00, self.dropout, training=self.training)
            y10 = F.relu(self.out_t10[i](rep1))
            y10 = F.dropout(y10, self.dropout, training=self.training)
            rep0 = y00
            rep1 = y10
        y0 = self.out_t01(y00).view(-1)
        y1 = self.out_t11(y10).view(-1)

        # print(t.shape,y1.shape,y0.shape)
        y = torch.where(t > 0, y1, y0)  # t>0的地方保存y1，否则保存y0

        # p1 = self.pp_act(self.pp(rep)).view(-1)
        # treatment = treatment.view(-1)
        #if self.training != True:
        #   np.savetxt('att.txt',att_final.cpu().detach().numpy())
        return y, rep, treatment

    # OOM issue 
    # def forward(self, x, adj, t, cuda=None):
    #     if cuda is None:
    #         cuda = torch.cuda.is_available()
    #     adj_dense = adj.to_dense()
    #     rep_outcome = F.relu(self.gc[0](x, adj))
    #     rep_outcome = F.dropout(rep_outcome, self.dropout, training=self.training)
    #     for i in range(1, self.n_in):
    #         rep_outcome = F.relu(self.gc[i](rep_outcome, adj))
    #         rep_outcome = F.dropout(rep_outcome, self.dropout, training=self.training)

    #     rep_treatment = F.relu(self.gc_t[0](x, adj))
    #     rep_treatment = F.dropout(rep_treatment, self.dropout, training=self.training)
    #     for i in range(1, self.n_in):
    #         rep_treatment = F.relu(self.gc_t[i](rep_treatment, adj))
    #         rep_treatment = F.dropout(rep_treatment, self.dropout, training=self.training)
    #     rep_out_treat = torch.cat((rep_outcome, rep_treatment), 1)
    #     rep = rep_out_treat

    #     N = x.shape[0]  # number of units
    #     if cuda:
    #         att_final = torch.zeros(N, N).cuda()
    #     else:
    #         att_final = torch.zeros(N, N)

    #     index = adj_dense.nonzero().t()

    #     att_input = torch.cat((rep[index[0, :], :], rep[index[1, :], :]), dim=1)
    #     attention = torch.matmul(att_input, self.a).squeeze()
        
    #     att_final = att_final.index_put(tuple(index), attention) 
    #     att_final = F.softmax(att_final, dim=1)
    #     att_final = F.dropout(att_final, self.dropout, training=self.training)
    #     treatment_cur = rep_treatment
    #     rep_outcome = torch.matmul(att_final, treatment_cur) + rep_outcome 
    #     treatment_MLP = self.pp(rep_treatment)
    #     treatment = self.pp_act(self.pp2(treatment_MLP))
    #     #h_prime= torch.cat((rep_outcome, dim_treat), 1)
    #     h_prime = F.dropout(rep_outcome, self.dropout, training=self.training)
    #     rep = h_prime
    #     rep0 = rep
    #     rep1 = rep
    #     for i in range(self.n_out):
    #         y00 = F.relu(self.out_t00[i](rep0))
    #         y00 = F.dropout(y00, self.dropout, training=self.training)
    #         y10 = F.relu(self.out_t10[i](rep1))
    #         y10 = F.dropout(y10, self.dropout, training=self.training)
    #         rep0 = y00
    #         rep1 = y10
    #     y0 = self.out_t01(y00).view(-1)
    #     y1 = self.out_t11(y10).view(-1)

    #     # print(t.shape,y1.shape,y0.shape)
    #     y = torch.where(t > 0, y1, y0)  # t>0的地方保存y1，否则保存y0

    #     # p1 = self.pp_act(self.pp(rep)).view(-1)
    #     # treatment = treatment.view(-1)
    #     #if self.training != True:
    #     #   np.savetxt('att.txt',att_final.cpu().detach().numpy())
    #     return y, rep, treatment

    def _prepare_attentional_mechanism_input(self, Wh, out_features):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #
        # print("dim of Wh:",Wh.shape)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        # print("dim of Wh_inchunks:",Wh_repeated_in_chunks.shape)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # print("dim of Wh_repeated_alternating:", Wh_repeated_alternating.shape)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # print("dim of combine:",all_combinations_matrix.shape)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * out_features)
