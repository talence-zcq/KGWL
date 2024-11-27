import math
from copy import deepcopy

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from torch_sparse import SparseTensor
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X
class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if  (normalization.__class__.__name__ != 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.normalizations[i+1](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class MLP_graph(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP_graph, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class PMA(nn.Module):
    """
        PMA part:
        Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
        i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neightbor nodes.
        In GAT, a(x,y) = a^T[x||y]. We use the same logic.
    """
    _alpha: OptTensor

    def __init__(self, in_channels, hid_dim,
                 out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, conv_type='v2e', bias=False ):
        super(PMA, self).__init__()

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.aggr = 'add'
        self.conv_type = conv_type
#         self.input_seed = input_seed

#         This is the encoder part. Where we use 1 layer NN (Theta*x_i in the GATConv description)
#         Now, no seed as input. Directly learn the importance weights alpha_ij.
#         self.lin_O = Linear(heads*self.hidden, self.hidden) # For heads combining
        # For neighbor nodes (source side, key)
        self.lin_K = Linear(in_channels, self.heads*self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = Linear(in_channels, self.heads*self.hidden)
        self.att_r = Parameter(torch.Tensor(
            1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(in_channels=self.heads*self.hidden,
                       hidden_channels=self.heads*self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=.0, Normalization='None',)
        self.ln0 = nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = nn.LayerNorm(self.heads*self.hidden)
#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:

#         Always no bias! (For now)
        self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #         glorot(self.lin_l.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
#         glorot(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
#         zeros(self.bias)

    def forward(self, x, G):
        H, C = self.heads, self.hidden
        alpha_r = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)

        if self.conv_type == 'v2e':
            x_j = x_V[G.v2e_src, :, :]  # [nnz, C]
            alpha_j = alpha_r[G.v2e_src, :]  # [nnz, C]
            alpha_j = F.leaky_relu(alpha_j, self.negative_slope)
            alpha_j = softmax(alpha_j, G.v2e_src, None, G.v2e_src.max() + 1)
            alpha_j = F.dropout(alpha_j, p=self.dropout, training=self.training)
            x_j = x_j * alpha_j.unsqueeze(-1)
            out = scatter(x_j, G.v2e_dst, dim=0, reduce="sum")
        elif self.conv_type == 'e2v':
            x_j = x_V[G.v2e_dst, :, :]  # [nnz, C]
            alpha_j = alpha_r[G.v2e_dst, :]  # [nnz, C]

            alpha_j = F.leaky_relu(alpha_j, self.negative_slope)
            alpha_j = softmax(alpha_j, G.v2e_dst, None, G.v2e_dst.max() + 1)
            alpha_j = F.dropout(alpha_j, p=self.dropout, training=self.training)
            x_j = x_j * alpha_j.unsqueeze(-1)
            out = scatter(x_j, G.v2e_src, dim=0, reduce="sum")


        out += self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out+F.relu(self.rFF(out)))
        return out

class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, X, vertex, edges, alpha, beta, X0):
        N = X.shape[-2]

        Xve = X[..., vertex, :]  # [nnz, C]
        Xe = scatter(Xve, edges, dim=-2, reduce='mean')  # [E, C], reduce is 'mean' here as default

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xv = scatter(Xev, vertex, dim=-2, reduce='mean', dim_size=N)  # [N, C]

        X = Xv


        Xi = (1 - alpha) * X + alpha * X0
        X = (1 - beta) * Xi + beta * self.W(Xi)

        return X

class EquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
                 mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(in_features + out_features, out_features, out_features, mlp2_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                         dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, G, X0):
        N = X.shape[-2]

        Xve = self.W1(X)[..., G.v2e_src, :]  # [nnz, C]
        Xe = scatter(Xve, G.v2e_dst, dim=-2, reduce=self.aggr)  # [E, C], reduce is 'mean' here as default

        Xev = Xe[..., G.v2e_dst, :]  # [nnz, C]
        Xev = self.W2(torch.cat([X[..., G.v2e_src, :], Xev], -1))
        Xv = scatter(Xev, G.v2e_src, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]

        X = Xv

        X = (1 - self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X



class Subgraph_Conv(nn.Module):
    def __init__(self, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
                 mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()
        self.heads = 2
        self.hidden = out_features // self.heads

        self.lin_Q = Linear(in_features, self.heads * self.hidden)
        self.lin_K = Linear(in_features, self.heads*self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = Linear(in_features, self.heads*self.hidden)


        if mlp1_layers > 0:
            self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)
            self.W1_1 = MLP(out_features, out_features, out_features, mlp1_layers,
                            dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(in_features + out_features, out_features, out_features, mlp2_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                         dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout
        self._norm_fact = 1 / math.sqrt(out_features)

    def reset_parameters(self):
        glorot(self.lin_Q.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        # nn.init.xavier_uniform_(self.att_r)

        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
            self.W1_1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, G, X0):
        N = X.shape[-2]
        Xve = self.W1(X)[..., G.v2e_src, :]  # [nnz, C]
        Xe = scatter(Xve, G.v2e_dst, dim=-2, reduce=self.aggr)  # [E, C], reduce is 'mean' here as default

        Xev = Xe[..., G.v2e_dst, :]  # [nnz, C]
        Xev = self.W2(torch.cat([X[..., G.v2e_src, :], Xev], -1))
        Xv = scatter(Xev, G.v2e_src, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]

        X = Xv

        # attention
        X_Q = self.lin_Q(X)
        X_K = self.lin_K(X)
        X_V = self.lin_V(X)

        A = torch.mm(X_Q, X_K.transpose(0,1)) * self._norm_fact

        dist = torch.softmax(A, dim=-1)
        X_V = torch.mm(dist, X_V)

        X = (1 - self.alpha) * X + self.alpha * X0 + 0.1 * X_V
        X = self.W(X)

        return X


class Multi_HyperEdge_Conv(nn.Module):
    def __init__(self, in_features, out_features, heads=1, mlp1_layers=1, mlp2_layers=2,
                 mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(in_features, out_features*heads, out_features*heads, mlp1_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm)

        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.Multi_Head_list = nn.ModuleList()
            for i in range(heads):
                self.Multi_Head_list.append(MLP(in_features + out_features, out_features, out_features, mlp2_layers,
                          dropout=dropout, Normalization=normalization, InputNorm=input_norm))
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                         dropout=dropout, Normalization=normalization, InputNorm=input_norm)
            # self.W = torch.nn.Linear(out_features, out_features)
        else:
            self.W = nn.Identity()

        self.multi_head_decoder = MLP(out_features*heads, out_features, out_features, mlp2_layers,
                         dropout=dropout, Normalization=normalization, InputNorm=input_norm)


        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout
        self.heads = heads
        self.hidden = out_features

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        for i in range(self.heads):
            self.Multi_Head_list[i].reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()
        self.multi_head_decoder.reset_parameters()

    def forward(self, X, G, X0):
        N = X.shape[-2]

        Xve = self.W1(X)[..., G.v2e_src, :]  # [nnz, C]
        Xe = scatter(Xve, G.v2e_dst, dim=-2, reduce=self.aggr).view(-1, self.heads, self.hidden) # [E, C], reduce is 'mean' here as default
        Xe = Xe.transpose(0, 1)
        Xev = Xe[..., G.v2e_dst, :]  # [nnz, C]
        Xev_list = torch.unbind(Xev, dim=0)
        Xev_result_list = []
        for i in range(self.heads):
            Xev_i = torch.cat([X[..., G.v2e_src, :], Xev_list[i]], -1)
            Xev_result_list.append(self.Multi_Head_list[i](Xev_i))
        Xev = torch.cat(Xev_result_list, dim=-1)
        Xev = self.multi_head_decoder(Xev)

        Xv = scatter(Xev, G.v2e_src, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]

        X =Xv
        X = (1 - self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X

