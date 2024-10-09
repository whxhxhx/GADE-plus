import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .LRMNew import *
import numpy as np
from torch.distributions.normal import Normal
import random
import os

VERY_SMALL_NUMBER = 1e-12
INF = 1e20


def compute_pivot_adj(node_pivot_adj, pivot_mask=None):
    '''Can be more memory-efficient'''
    pivot_node_adj = node_pivot_adj.transpose(-1, -2)
    pivot_norm = torch.clamp(pivot_node_adj.sum(
        dim=-2), min=VERY_SMALL_NUMBER) ** -1
    pivot_adj = torch.matmul(
        pivot_node_adj, pivot_norm.unsqueeze(-1) * node_pivot_adj)

    markoff_value = 0
    if pivot_mask is not None:
        pivot_adj = pivot_adj.masked_fill_(
            1 - pivot_mask.byte().unsqueeze(-1), markoff_value)
        pivot_adj = pivot_adj.masked_fill_(
            1 - pivot_mask.byte().unsqueeze(-2), markoff_value)

    return pivot_adj


class AnchorGraphLearner(nn.Module):
    def __init__(self, input_size, anchor_nums, num_pers=16, device=None):
        super(AnchorGraphLearner, self).__init__()
        self.device = device
        self.hidden_size = input_size

        self.anchors = nn.Parameter(torch.randn(anchor_nums, self.hidden_size))
        nn.init.orthogonal_(self.anchors)
        self.anchor_nums = anchor_nums

        self.mhmlp = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        ) for _ in range(num_pers)])


    def forward(self, context):
        N, fdim = context.shape

        concat = torch.cat([context.repeat(1, self.anchor_nums).view(N * self.anchor_nums, -1), self.anchors.repeat(N, 1)], dim=-1).view(N, -1, 2 * fdim)
        attn = []
        # for _ in range(len(self.linear_sims)):
        for _ in range(len(self.mhmlp)):
            a_input = self.mhmlp[_](concat).squeeze(-1)
            a_input = torch.sigmoid(a_input)
            attn.append(a_input)
        attention = torch.mean(torch.stack(attn, 0), 0)
        # attention = torch.sigmoid(attention)
    
        # return attention, anchors
        return attention, self.anchors



class AnchorGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=False):
        super(AnchorGNNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.constant_(self.bias, 0.))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, anchor_mp=True, batch_norm=False):
        support = torch.matmul(input, self.weight)

        if anchor_mp:
            node_anchor_adj = adj
            node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=VERY_SMALL_NUMBER)
            anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))
            
        else:
            node_adj = adj
            output = torch.matmul(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class AnchorGNN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, anchor_nums=10, num_pers=4, device=0, batch_norm=False, aux_ce=True, p2a_reg=True):
        super(AnchorGNN, self).__init__()
        self.dropout = dropout
        self.graph_hops = graph_hops
        self.anchor_nums = anchor_nums
        self.num_pers = num_pers
        self.hidden_dim = nhid
        self.aux_ce = aux_ce
        self.p2a_reg = p2a_reg

        if not isinstance(device, list):
            device = [device]
        self.device = torch.device("cuda:{:d}".format(device[0]))

        self.input_mapper = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )

        self.graphlearner = AnchorGraphLearner(input_size=self.hidden_dim, anchor_nums=self.anchor_nums, num_pers=self.num_pers, device=self.device)

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(AnchorGNNLayer(nhid, nhid, batch_norm=batch_norm))

        self.ln = nn.LayerNorm(nhid)

        if graph_hops > 1:
            for _ in range(graph_hops-2):
                self.graph_encoders.append(AnchorGNNLayer(nhid, nhid, batch_norm=batch_norm))

            self.graph_encoders.append(AnchorGNNLayer(nhid, nhid, batch_norm=batch_norm))

        self.classifier = nn.Sequential(nn.Linear(nhid, nhid),
                                        nn.PReLU(nhid),
                                        nn.Linear(nhid, 2))


    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.input_mapper(x)
        x_loc = x.clone()
        node_anchor_adj, anchor_vec = self.graphlearner(x.view(-1, x.size(-1)))
        node_anchor_adj, anchor_vec = node_anchor_adj.unsqueeze(0), anchor_vec.unsqueeze(0)
        if self.graph_hops > 1:
            for i, encoder in enumerate(self.graph_encoders[:-1]):
                x = F.relu(encoder(x, node_anchor_adj))
                x = F.dropout(x, self.dropout, training=self.training)

            x = F.relu(self.graph_encoders[-1](x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)
        else:
            x = F.relu(self.graph_encoders[0](x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.view(-1, x.size(-1))
        x_loc = x_loc.view(-1, x_loc.size(-1))
        x = self.ln(x+x_loc)

        pred = self.classifier(x)

        if self.aux_ce:
            anchor_logits = self.classifier(anchor_vec.view(-1, anchor_vec.size(-1)))
            if self.p2a_reg:
                return pred, x, anchor_vec, anchor_logits
            else:
                return pred, anchor_logits 
        else:
            if self.p2a_reg:
                return pred, x, anchor_vec
            else:
                return pred

