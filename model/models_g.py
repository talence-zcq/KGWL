
import math
from collections import namedtuple

import torch.nn
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter

from model.layers import MLP_graph, UniGCNIIConv, PMA, MLP, EquivSetConv


class CP_HGNN(nn.Module):
    def __init__(
            self,
            args,
            n_layers=3,
            n_mlp_layers=2,
            neighbor_pooling_type="sum",
            readout_type="mean",
            drop_rate=0.5,
    ):
        super(CP_HGNN, self).__init__()
        self.n_layers = n_layers
        self.multi_label = args.multi_label
        self.neighbor_pooling_type = neighbor_pooling_type
        self.readout_type = readout_type
        self.drop_rate = drop_rate
        self.mlps = nn.ModuleList([MLP_graph(n_mlp_layers, args.ft_dim, args.MLP_hidden, args.MLP_hidden)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(args.MLP_hidden)])
        self.concat_pred_head = nn.Linear(args.MLP_hidden * (n_layers - 1), args.n_classes)

        for _ in range(n_layers - 2):
            self.mlps.append(MLP_graph(n_mlp_layers, args.MLP_hidden, args.MLP_hidden, args.MLP_hidden))
            self.bns.append(nn.BatchNorm1d(args.MLP_hidden))

    def norm_row(self, H):
        N = H.shape[0]
        D_neg_1 = 1 / torch.sparse.sum(H, dim=1).to_dense()
        D_neg_1[torch.isinf(D_neg_1)] = 1
        D_neg_1 = torch.sparse_coo_tensor(torch.arange(N).repeat([2, 1]).to(H.device), D_neg_1, size=(N, N))
        return D_neg_1.mm(H)

    def forward(self, inputs):
        X, H, Ns, Ms, sub_X, sub_e_lbl, sub_H, sub_Ns, sub_Ms, sub_batch, sub_k_set, khwl_H, all_khwl_batch, all_batch = inputs
        H_T = H.transpose(0, 1).clone()
        P = H.mm(H_T)

        # grpah pool
        if self.readout_type == "mean":
            P = self.norm_row(P)

        if self.neighbor_pooling_type == "mean":
            H = self.norm_row(H)
            H_T = self.norm_row(H_T)

        # hidden_rep = [X]
        hidden_rep = []
        for idx in range(self.n_layers - 1):
            X = H.mm(H_T.mm(X))
            X = self.mlps[idx](X)
            X = F.relu(self.bns[idx](X))
            hidden_rep.append(X)

        for idx in range(len(hidden_rep)):
            hidden_rep[idx] = P.mm(hidden_rep[idx])
        global_rep = torch.concat(hidden_rep, dim=1)
        out = self.concat_pred_head(global_rep)
        readout = torch.cat([out[all_batch == i].mean(0).unsqueeze(0) for i in torch.unique(all_batch)], dim=0)

        if self.multi_label:
            return readout.sigmoid()
        else:
            return readout

class one_HWL(nn.Module):
    def __init__(
            self,
            args,
            n_layers=3,
            n_mlp_layers=2,
            neighbor_pooling_type="sum",
            readout_type="mean",
            drop_rate=0.5,
    ):
        super(one_HWL, self).__init__()
        self.n_layers = n_layers
        self.multi_label = args.multi_label
        self.neighbor_pooling_type = neighbor_pooling_type
        self.readout_type = readout_type
        self.drop_rate = drop_rate
        self.mlps = nn.ModuleList([MLP_graph(n_mlp_layers, args.ft_dim, args.MLP_hidden, args.MLP_hidden)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(args.MLP_hidden)])
        self.concat_pred_head = nn.Linear(args.MLP_hidden * (n_layers - 1), args.n_classes)

        for _ in range(n_layers - 2):
            self.mlps.append(MLP_graph(n_mlp_layers, args.MLP_hidden, args.MLP_hidden, args.MLP_hidden))
            self.bns.append(nn.BatchNorm1d(args.MLP_hidden))

    def norm_row(self, H):
        N = H.shape[0]
        D_neg_1 = 1 / torch.sparse.sum(H, dim=1).to_dense()
        D_neg_1[torch.isinf(D_neg_1)] = 1
        D_neg_1 = torch.sparse_coo_tensor(torch.arange(N).repeat([2, 1]).to(H.device), D_neg_1, size=(N, N))
        return D_neg_1.mm(H)

    def forward(self, inputs):
        X, H, Ns, Ms, sub_X, sub_e_lbl, sub_H, sub_Ns, sub_Ms, sub_batch, sub_k_set, khwl_H, all_khwl_batch, all_batch = inputs
        H_T = H.transpose(0, 1).clone()

        if self.neighbor_pooling_type == "mean":
            H = self.norm_row(H)
            H_T = self.norm_row(H_T)

        # hidden_rep = [X]
        hidden_rep = []
        for idx in range(self.n_layers - 1):
            X = H.mm(H_T.mm(X))
            X = self.mlps[idx](X)
            X = F.relu(self.bns[idx](X))
            hidden_rep.append(X)

        global_rep = torch.concat(hidden_rep, dim=1)
        out = self.concat_pred_head(global_rep)
        readout = torch.cat([out[all_batch == i].mean(0).unsqueeze(0) for i in torch.unique(all_batch)], dim=0)

        if self.multi_label:
            return readout.sigmoid()
        else:
            return readout

class KHWl_HGNN(nn.Module):
    def __init__(
            self,
            args,
            neighbor_pooling_type="mean",
            readout_type="mean",
    ):
        super(KHWl_HGNN, self).__init__()
        n_mlp_layers = args.MLP_num_layers
        self.K = args.tuple_K
        self.n_layers = args.All_num_layers
        self.khwl_mlp_layers = args.KHWL_num_layers
        self.multi_label = args.multi_label
        self.neighbor_pooling_type = neighbor_pooling_type
        self.readout_type = readout_type
        self.dropout = nn.Dropout(args.dropout)
        self.device = args.cuda
        self.NormLayer = args.normalization

        self.mlps_v2e = nn.ModuleList([MLP(in_channels=args.ft_dim,
                                    hidden_channels=args.MLP_hidden,
                                    out_channels=args.MLP_hidden,
                                    num_layers=n_mlp_layers,
                                    dropout=args.dropout,
                                    Normalization=self.NormLayer,
                                    InputNorm=False)])
        self.mlps_e2v = nn.ModuleList([MLP(in_channels=args.MLP_hidden,
                                    hidden_channels=args.MLP_hidden,
                                    out_channels=args.MLP_hidden,
                                    num_layers=n_mlp_layers,
                                    dropout=args.dropout,
                                    Normalization=self.NormLayer,
                                    InputNorm=False)])

        for _ in range(1, self.n_layers):
            self.mlps_v2e.append(MLP(in_channels=args.MLP_hidden,
                                    hidden_channels=args.MLP_hidden,
                                    out_channels=args.MLP_hidden,
                                    num_layers=n_mlp_layers,
                                    dropout=args.dropout,
                                    Normalization=self.NormLayer,
                                    InputNorm=False))
            self.mlps_e2v.append(MLP(in_channels=args.MLP_hidden,
                                     hidden_channels=args.MLP_hidden,
                                     out_channels=args.MLP_hidden,
                                     num_layers=n_mlp_layers,
                                     dropout=args.dropout,
                                     Normalization=self.NormLayer,
                                     InputNorm=False))

        self.khwl_mlps_v2e = nn.ModuleList([MLP(in_channels=args.ft_dim+ args.MLP_hidden+1,
                                    hidden_channels=args.KHWL_hidden,
                                    out_channels=args.KHWL_hidden,
                                    num_layers=n_mlp_layers,
                                    dropout=args.dropout,
                                    Normalization=self.NormLayer,
                                    InputNorm=False)])

        self.khwl_mlps_e2v = nn.ModuleList([MLP(in_channels=args.KHWL_hidden,
                                    hidden_channels=args.KHWL_hidden,
                                    out_channels=args.KHWL_hidden,
                                    num_layers=self.khwl_mlp_layers,
                                    dropout=args.dropout,
                                    Normalization=self.NormLayer,
                                    InputNorm=False)])

        for _ in range(self.n_layers-1):
            self.khwl_mlps_v2e.append(MLP(in_channels=args.KHWL_hidden,
                                    hidden_channels=args.KHWL_hidden,
                                    out_channels=args.KHWL_hidden,
                                    num_layers=self.khwl_mlp_layers,
                                    dropout=args.dropout,
                                    Normalization=self.NormLayer,
                                    InputNorm=False))
            self.khwl_mlps_e2v.append(MLP(in_channels=args.KHWL_hidden,
                                    hidden_channels=args.KHWL_hidden,
                                    out_channels=args.KHWL_hidden,
                                    num_layers=self.khwl_mlp_layers,
                                    dropout=args.dropout,
                                    Normalization=self.NormLayer,
                                    InputNorm=False))

        # self.x_output_mlp = MLP(in_channels=args.MLP_hidden,
        #                             hidden_channels=int(args.MLP_hidden * 0.5),
        #                             out_channels=args.n_classes,
        #                             num_layers=n_mlp_layers,
        #                             dropout=args.dropout,
        #                             Normalization="None",
        #                             InputNorm=False)

        self.k_hwl_output_mlp = MLP(in_channels=args.KHWL_hidden,
                                    hidden_channels=int(args.KHWL_hidden *0.5),
                                    out_channels=args.n_classes,
                                    num_layers=self.khwl_mlp_layers,
                                    dropout=args.dropout,
                                    Normalization="None",
                                    InputNorm=False)

    def norm_row(self, H):
        N = H.shape[0]
        D_neg_1 = 1 / torch.sparse.sum(H, dim=1).to_dense()
        D_neg_1[torch.isinf(D_neg_1)] = 1
        D_neg_1 = torch.sparse_coo_tensor(torch.arange(N).repeat([2, 1]).to(H.device), D_neg_1, size=(N, N))
        return D_neg_1.mm(H)

    def forward(self, inputs):
        X, H, Ns, Ms, sub_X, sub_e_lbl, sub_H, sub_Ns, sub_Ms, sub_batch, sub_k_set, khwl_H, all_khwl_batch, all_batch = inputs

        G = namedtuple('G', ['v2e_src', 'v2e_dst'])
        G.v2e_src, G.v2e_dst = H.coalesce().indices()[0], H.coalesce().indices()[1]

        H_T = H.transpose(0, 1).clone()
        sub_H_T = sub_H.transpose(0, 1).clone()
        khwl_H_T = khwl_H.transpose(0, 1).clone()

        if self.neighbor_pooling_type == "mean":
            H = self.norm_row(H)
            H_T = self.norm_row(H_T)
            sub_H = self.norm_row(sub_H)
            sub_H_T = self.norm_row(sub_H_T)
            khwl_H = self.norm_row(khwl_H)
            khwl_H_T = self.norm_row(khwl_H_T)

        # initial k_subset_graph
        sub_he = sub_H_T.mm(sub_X)
        sub_he = torch.concat((sub_he, torch.unsqueeze(sub_e_lbl, dim=-1)), dim=-1)
        sub_X = sub_H.mm(sub_he)
        sub_X = scatter(sub_X, sub_batch, dim=-2)  # [N, C]

        X = F.dropout(X, p=0.2)  # Input dropout
        for idx in range(self.n_layers):
            if idx == 0:
                # Xe = F.relu(H_T.mm(self.bns_v2e[idx](self.mlps_v2e[idx](X))))
                Xe = F.relu(H_T.mm(self.mlps_v2e[idx](X)))
                Xe0 = Xe
                Xe = self.dropout(Xe)
                # X = F.relu(H.mm(self.bns_e2v[idx](self.mlps_e2v[idx](Xe))))
                X = F.relu(H.mm(self.mlps_e2v[idx](Xe)))
                X0 = X
            else:
                X = self.dropout(X)
                Xe = F.relu(H_T.mm(self.mlps_v2e[idx](X))+ Xe0)
                Xe0 = Xe
                Xe = self.dropout(Xe)
                X = F.relu(H.mm(self.mlps_e2v[idx](Xe))+ X0)
                X0 = X



        sub_k_indices = sub_k_set.view(-1)
        selected_embeddings = X.index_select(0, sub_k_indices)
        aggregated_embeddings = selected_embeddings.view(sub_k_set.size(0), sub_k_set.size(1), -1).mean(dim=1)
        sub_X = torch.concat([sub_X, aggregated_embeddings], dim=1)

        # K-HWL calculate process
        sub_X = F.dropout(sub_X, p=0.2)  # Input dropout
        for idx in range(self.n_layers):
            if idx == 0:
                sub_Xe = F.relu(khwl_H_T.mm(self.khwl_mlps_v2e[idx](sub_X)))
                # sub_Xe = F.relu(khwl_H_T.mm(self.khwl_bns_v2e[idx](self.khwl_mlps_v2e[idx](sub_X))))
                sub_Xe0 = sub_Xe
                sub_Xe = self.dropout(sub_Xe)
                sub_X =  F.relu(khwl_H.mm(self.khwl_mlps_e2v[idx](sub_Xe)))
                # sub_X =  F.relu(khwl_H.mm(self.khwl_bns_e2v[idx](self.khwl_mlps_e2v[idx](sub_Xe))))
                sub_X0 = sub_X
            else:
                sub_X = self.dropout(sub_X)
                # sub_Xe = F.relu(khwl_H_T.mm(self.khwl_mlps_v2e[idx](sub_X)) + sub_Xe0)
                sub_Xe = F.relu(khwl_H_T.mm(self.khwl_mlps_v2e[idx](sub_X)) + sub_Xe0)
                sub_Xe0 = sub_Xe
                sub_Xe = self.dropout(sub_Xe)
                # sub_X =  F.relu(khwl_H.mm(self.khwl_bns_e2v[idx](self.khwl_mlps_e2v[idx](sub_Xe))) + sub_X0)
                sub_X =  F.relu(khwl_H.mm(self.khwl_mlps_e2v[idx](sub_Xe)) + sub_X0)
                sub_X0 = sub_X

        khwl_out = self.k_hwl_output_mlp(sub_X)
        res = torch.cat([khwl_out[all_khwl_batch == i].mean(0).unsqueeze(0) for i in torch.unique(all_khwl_batch)], dim=0)
        if self.multi_label:
            return res.sigmoid()
        else:
            return res


class UniGCNII_g(nn.Module):
    def __init__(self, args):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super(UniGCNII_g, self).__init__()
        nhid = args.MLP_hidden

        self.nhid = nhid

        nhid = args.MLP_hidden
        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act['relu']  # Default relu
        self.dropout = nn.Dropout(0.2)  # 0.2 is chosen for GCNII
        self.alpha = 0.5

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(args.ft_dim, nhid))
        for _ in range(args.All_num_layers):
            self.convs.append(UniGCNIIConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, args.n_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(0.2)  # 0.2 is chosen for GCNII

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, inputs):
        X, H, Ns, Ms, sub_X, sub_e_lbl, sub_H, sub_Ns, sub_Ms, sub_batch, sub_k_set, khwl_H, all_khwl_batch, all_batch = inputs

        V, E = H.coalesce().indices()[0], H.coalesce().indices()[1]
        lamda, alpha = 0.5, self.alpha
        x = self.dropout(X)
        x = F.relu(self.convs[0](x))
        x0 = x
        for i, con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda / (i + 1) + 1)
            x = F.relu(con(x, V, E, alpha, beta, x0))
        x = self.dropout(x)
        x = self.convs[-1](x)
        readout = torch.cat([x[all_batch == i].mean(0).unsqueeze(0) for i in torch.unique(all_batch)], dim=0)

        return readout

class AllSetTransformer_g(nn.Module):
    def __init__(self, args):
        super(AllSetTransformer_g, self).__init__()
        self.task_kind = args.task_kind
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.NormLayer = args.normalization
        if args.cuda in [0, 1, 2, 3, 4, 5, 6, 7]:
            self.device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()
        self.aphla = 0.5
        self.beta = 0.1
        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.ft_dim,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(PMA(args.ft_dim, args.MLP_hidden, args.MLP_hidden, args.MLP_num_layers, heads=args.heads, conv_type="v2e"))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(PMA(args.MLP_hidden, args.MLP_hidden, args.MLP_hidden, args.MLP_num_layers, heads=args.heads, conv_type="e2v"))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers - 1):
                self.V2EConvs.append(PMA(args.MLP_hidden, args.MLP_hidden, args.MLP_hidden, args.MLP_num_layers, heads=args.heads, conv_type="v2e"))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(PMA(args.MLP_hidden, args.MLP_hidden, args.MLP_hidden, args.MLP_num_layers, heads=args.heads, conv_type="e2v"))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))

            self.classifier = MLP(in_channels=args.MLP_hidden,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.n_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
            self.classifier = MLP(in_channels=args.MLP_hidden,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.n_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, inputs):
        X, H, Ns, Ms, sub_X, sub_e_lbl, sub_H, sub_Ns, sub_Ms, sub_batch, sub_k_set, khwl_H, all_khwl_batch, all_batch = inputs
        G = namedtuple('G', ['v2e_src', 'v2e_dst'])
        G.v2e_src, G.v2e_dst = H.coalesce().indices()[0], H.coalesce().indices()[1]
        # consider the self-loop condition
        x = F.dropout(X, p=0.2, training=self.training)  # Input dropout
        for i, _ in enumerate(self.V2EConvs):
            x = F.relu(self.V2EConvs[i](x, G))
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = F.relu(self.E2VConvs[i](x, G))
            x = F.dropout(x, p=self.dropout, training=self.training)

        readout = torch.cat([x[all_batch == i].mean(0).unsqueeze(0) for i in torch.unique(all_batch)], dim=0)
        readout = self.classifier(readout)

        return readout

class EquivSetGNN_g(nn.Module):
    def __init__(self, args):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.mlp1_layers = args.MLP_num_layers
        self.mlp2_layers = 1
        self.mlp3_layers = 1

        self.act = act['relu']
        self.dropout = nn.Dropout(args.dropout)  # 0.2 is chosen for GCNII

        self.nlayer = args.All_num_layers

        self.lin_in = torch.nn.Linear(args.ft_dim, args.MLP_hidden)
        self.conv = EquivSetConv(args.MLP_hidden, args.MLP_hidden, mlp1_layers=self.mlp1_layers,
                                 mlp2_layers=self.mlp2_layers,
                                 mlp3_layers=self.mlp3_layers, alpha=0.5, aggr="mean",
                                 dropout=args.dropout, normalization=args.normalization,
                                 input_norm=True)

        self.classifier = MLP(in_channels=args.MLP_hidden,
                              hidden_channels=args.Classifier_hidden,
                              out_channels=args.n_classes,
                              num_layers=args.Classifier_num_layers,
                              dropout=args.dropout,
                              Normalization=args.normalization,
                              InputNorm=False)

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.conv.reset_parameters()

        self.classifier.reset_parameters()

    def forward(self, inputs):
        X, H, Ns, Ms, sub_X, sub_e_lbl, sub_H, sub_Ns, sub_Ms, sub_batch, sub_k_set, khwl_H, all_khwl_batch, all_batch = inputs
        G = namedtuple('G', ['v2e_src', 'v2e_dst'])
        G.v2e_src, G.v2e_dst = H.coalesce().indices()[0], H.coalesce().indices()[1]
        x = self.dropout(X)
        x = F.relu(self.lin_in(x))
        x0 = x
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.conv(x, G, x0)
            x = self.act(x)
        x = self.dropout(x)
        x = self.classifier(x)
        readout = torch.cat([x[all_batch == i].mean(0).unsqueeze(0) for i in torch.unique(all_batch)], dim=0)

        return readout

class AllDeepSet_g(nn.Module):
    def __init__(self, args, norm=None):
        super(AllDeepSet_g, self).__init__()
        self.task_kind = args.task_kind
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.NormLayer = args.normalization
        if args.cuda in [0, 1, 2, 3, 4, 5, 6, 7]:
            self.device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.f_V2E_ENCs = nn.ModuleList()
        self.f_E2V_ENCs = nn.ModuleList()
        self.f_V2E_DECs = nn.ModuleList()
        self.f_E2V_DECs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()
        self.aphla = 0.5
        self.beta = 0.1
        self.classifier = MLP_graph(args.Classifier_num_layers, args.MLP_hidden, args.Classifier_hidden, args.n_classes)

        # if self.All_num_layers == 0:
        #     self.classifier = MLP_graph(args.Classifier_num_layers, args.ft_dim, args.Classifier_hidden, args.n_classes)
        # else:
        #     self.f_V2E_ENCs.append(MLP_graph(args.MLP_num_layers, args.ft_dim, args.MLP_hidden, args.MLP_hidden))
        #     self.f_E2V_ENCs.append(MLP_graph(args.MLP_num_layers, args.MLP_hidden, args.MLP_hidden, args.MLP_hidden))
        #     self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
        #     self.f_V2E_DECs.append(MLP_graph(args.MLP_num_layers, args.MLP_hidden, args.MLP_hidden, args.MLP_hidden))
        #     self.f_E2V_DECs.append(MLP_graph(args.MLP_num_layers, args.MLP_hidden, args.MLP_hidden, args.MLP_hidden))
        #     self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
        #     for _ in range(self.All_num_layers - 1):
        #         self.f_V2E_ENCs.append(MLP_graph(args.MLP_num_layers, args.MLP_hidden, args.MLP_hidden, args.MLP_hidden))
        #         self.f_E2V_ENCs.append(MLP_graph(args.MLP_num_layers, args.MLP_hidden, args.MLP_hidden, args.MLP_hidden))
        #         self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
        #         self.f_V2E_DECs.append(MLP_graph(args.MLP_num_layers, args.MLP_hidden, args.MLP_hidden, args.MLP_hidden))
        #         self.f_E2V_DECs.append(MLP_graph(args.MLP_num_layers, args.MLP_hidden, args.MLP_hidden, args.MLP_hidden))
        #         self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.ft_dim,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.n_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.f_V2E_ENCs.append(MLP(in_channels=args.ft_dim,
                                  hidden_channels=args.MLP_hidden,
                                  out_channels=args.MLP_hidden,
                                  num_layers=args.MLP_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False))
            self.f_E2V_ENCs.append(MLP(in_channels=args.MLP_hidden,
                                  hidden_channels=args.MLP_hidden,
                                  out_channels=args.MLP_hidden,
                                  num_layers=args.MLP_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.f_V2E_DECs.append(MLP(in_channels=args.MLP_hidden,
                                  hidden_channels=args.MLP_hidden,
                                  out_channels=args.MLP_hidden,
                                  num_layers=args.MLP_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False))
            self.f_E2V_DECs.append(MLP(in_channels=args.MLP_hidden,
                                  hidden_channels=args.MLP_hidden,
                                  out_channels=args.MLP_hidden,
                                  num_layers=args.MLP_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers - 1):
                self.f_V2E_ENCs.append(MLP(in_channels=args.MLP_hidden,
                                  hidden_channels=args.MLP_hidden,
                                  out_channels=args.MLP_hidden,
                                  num_layers=args.MLP_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False))
                self.f_E2V_ENCs.append(MLP(in_channels=args.MLP_hidden,
                                  hidden_channels=args.MLP_hidden,
                                  out_channels=args.MLP_hidden,
                                  num_layers=args.MLP_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.f_V2E_DECs.append(MLP(in_channels=args.MLP_hidden,
                                  hidden_channels=args.MLP_hidden,
                                  out_channels=args.MLP_hidden,
                                  num_layers=args.MLP_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False))
                self.f_E2V_DECs.append(MLP(in_channels=args.MLP_hidden,
                                  hidden_channels=args.MLP_hidden,
                                  out_channels=args.MLP_hidden,
                                  num_layers=args.MLP_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))

    def reset_parameters(self):
        for layer in self.f_V2E_ENCs:
            layer.reset_parameters()
        for layer in self.f_E2V_ENCs:
            layer.reset_parameters()
        for layer in self.f_V2E_DECs:
            layer.reset_parameters()
        for layer in self.f_E2V_DECs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, inputs):
        X, H, Ns, Ms, sub_X, sub_e_lbl, sub_H, sub_Ns, sub_Ms, sub_batch, sub_k_set, khwl_H, all_khwl_batch, all_batch = inputs
        G = namedtuple('G', ['v2e_src', 'v2e_dst'])
        G.v2e_src, G.v2e_dst = H.coalesce().indices()[0], H.coalesce().indices()[1]
        # consider the self-loop condition
        x = F.dropout(X, p=0.2, training=self.training)  # Input dropout
        for i, _ in enumerate(self.f_V2E_ENCs):
            x = F.relu(self.f_V2E_ENCs[i](x))[..., G.v2e_src, :]
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = scatter(x, G.v2e_dst, dim=0, reduce="sum")
            x = F.relu(self.f_V2E_DECs[i](x))

            x = F.relu(self.f_E2V_ENCs[i](x))[..., G.v2e_dst, :]
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = scatter(x, G.v2e_src, dim=0, reduce="sum")
            x = F.relu(self.f_E2V_DECs[i](x))

        readout = torch.cat([x[all_batch == i].mean(0).unsqueeze(0) for i in torch.unique(all_batch)], dim=0)
        readout = self.classifier(readout)
        return readout


class MLP_model_g(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, args, InputNorm=False):
        super(MLP_model_g, self).__init__()
        in_channels = args.ft_dim
        hidden_channels = args.MLP_hidden
        out_channels = args.n_classes
        num_layers = args.All_num_layers
        dropout = args.dropout
        Normalization = args.normalization

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
            if (normalization.__class__.__name__ != 'Identity'):
                normalization.reset_parameters()

    def forward(self, inputs):
        X, H, Ns, Ms, sub_X, sub_e_lbl, sub_H, sub_Ns, sub_Ms, sub_batch, sub_k_set, khwl_H, all_khwl_batch, all_batch = inputs

        x = self.normalizations[0](X)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        readout = torch.cat([x[all_batch == i].mean(0).unsqueeze(0) for i in torch.unique(all_batch)], dim=0)

        return readout