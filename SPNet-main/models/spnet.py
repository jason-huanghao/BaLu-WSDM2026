import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import numpy as np

class GCN_DECONF(nn.Module):
    def __init__(self, nfeat, nhid, dropout, n_in=1, n_out=1, cuda=False, n=100):
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
        self.n = n
        if cuda:

            self.out_t00 = [nn.Linear( nhid, nhid).cuda() for i in range(n_out)]
            self.out_t10 = [nn.Linear( nhid, nhid).cuda() for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid, 1).cuda()
            self.out_t11 = nn.Linear(nhid, 1).cuda()

        else:
            self.out_t00 = [nn.Linear(nhid+1, nhid) for i in range(n_out)]
            self.out_t10 = [nn.Linear(nhid+1, nhid) for i in range(n_out)]
            self.out_t01 = nn.Linear(nhid, 1)
            self.out_t11 = nn.Linear(nhid, 1)

        self.dropout = dropout

        # a linear layer for propensity prediction
        self.pp = nn.Linear(nhid, nhid)
        self.pp2 = nn.Linear(nhid, 2)

        if cuda:
            self.pp = self.pp.cuda()
            self.pp2 = self.pp2.cuda()
            self.a = nn.Parameter(torch.empty(size=(4 * nhid, 1)).cuda())
            self.leakyrelu = nn.LeakyReLU(0.2).cuda()
            #self.att_p = nn.Parameter(torch.empty(size=(n, n)).cuda())
        self.pp_act = nn.Sigmoid()
        nn.init.xavier_uniform_(self.a.data, gain=0)
        #nn.init.xavier_uniform_(self.att_p.data, gain=0)

    def forward(self, x, adj, t, cuda=False):
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
        att_final = torch.zeros(self.n, self.n).cuda()
        index = adj_dense.nonzero().t()
        # print(index.dtype)
        att_input = torch.cat((rep[index[0, :], :], rep[index[1, :], :]), dim=1)
        # print(type(att_input))
        attention = torch.matmul(att_input, self.a).squeeze()
        #print("attention:", attention.dtype)
        att_final = att_final.index_put(tuple(index), attention)
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
