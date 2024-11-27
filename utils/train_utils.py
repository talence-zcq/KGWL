import numpy as np
import scipy
import torch
from dhg import Graph, Hypergraph
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader


def performance(preds: np.ndarray, targets: np.ndarray, multi_label: bool):
    if multi_label:
        if isinstance(preds, scipy.sparse.csc_matrix):
            preds = preds.todense()
        else:
            preds = (preds > 0.5).astype(int)
        # multi-label classification metric:
        # https://medium.datadriveninvestor.com/a-survey-of-evaluation-metrics-for-multilabel-classification-bb16e8cd41cd
        # acc = (preds==lbls).mean()
        # Exact Match Ratio (EMR)
        EMR = (preds == targets).all(1).mean()
        # Example-based Accuracy
        EB_acc = (
            np.logical_and(preds, targets).sum(1) / np.logical_or(preds, targets).sum(1)
        ).mean()
        # Example-based Precision
        EB_pre = np.logical_and(preds, targets).sum(1) / preds.sum(1)
        EB_pre[np.isnan(EB_pre)] = 0
        EB_pre = EB_pre.mean()
        res = {"EMR": EMR, "EB_acc": EB_acc, "EB_pre": EB_pre}
        return EMR, res
    else:
        if len(preds.shape) == 2:
            preds = np.argmax(preds, axis=1)
        acc = accuracy_score(targets, preds)
        f1_micro = f1_score(targets, preds, average="micro")
        f1_macro = f1_score(targets, preds, average="macro")
        f1_weighted = f1_score(targets, preds, average="weighted")
        res = {
            "acc": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }
        return acc, res


class HypergraphLoader:
    def __init__(self, x_list, y_list, batch_size, num_worker, shuffle, transform_func=None):
        if isinstance(x_list[0]["dhg"], Graph):
            assert transform_func is not None
            for x in x_list:
                x["dhg"] = transform_func(x["dhg"])
        self.dataset = StructureDataset(x_list, y_list)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_worker,
            collate_fn=hypergraph_collate_fn,
            drop_last=True

        )

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        batches = iter(self.dataloader)
        for data, target in batches:
            data = create_batch_hypergraph(*data)
            if len(target.shape) > 1:
                yield data, target.float()
            else:
                yield data, target.long()

class StructureDataset(Dataset):
    def __init__(self, x_list, y_list):
        self.x_list = x_list
        self.y_list = y_list
        if isinstance(x_list[0]["dhg"], Hypergraph):
            self.data_type = "hypergraph"
        else:
            self.data_type = "graph"

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return self.x_list[idx], self.y_list[idx]


def create_batch_hypergraph(X, H_idx, N, M, sub_X, all_sub_e_lbl, sub_H_idx, sub_N, sub_M, sub_batch, sub_k_set, khwl_H_idx, khwl_N, khwl_M, all_khwl_batch, all_batch):
    num_v, num_e = N.sum(), M.sum()
    H = torch.sparse_coo_tensor(H_idx, torch.ones(H_idx.shape[1]), size=(num_v, num_e)).float()
    sub_num_v, sub_num_e = sub_N.sum(), sub_M.sum()
    sub_H = torch.sparse_coo_tensor(sub_H_idx, torch.ones(sub_H_idx.shape[1]), size=(sub_num_v, sub_num_e)).float()
    khwl_num_v, khwl_num_e = khwl_N.sum(), khwl_M.sum()
    khwl_H = torch.sparse_coo_tensor(khwl_H_idx, torch.ones(khwl_H_idx.shape[1]), size=(khwl_num_v, khwl_num_e)).float()
    return X, H, N, M, sub_X, all_sub_e_lbl,sub_H, sub_N, sub_M, sub_batch, sub_k_set, khwl_H, all_khwl_batch, all_batch


def hypergraph_collate_fn(batch):
    all_X = []
    all_Y = []
    all_N = []
    all_M = []
    all_H_idx = []
    all_batch = []
    bias_batch = 0
    all_sub_X = []
    all_sub_e_lbl = []
    all_sub_N = []
    all_sub_M = []
    all_sub_H_idx = []
    all_khwl_N = []
    all_khwl_M = []
    all_khwl_H_idx = []
    bias_row, bias_col = 0, 0
    sub_bias_row, sub_bias_col = 0, 0
    khwl_bias_row, khwl_bias_col = 0, 0
    all_sub_batch = []
    sub_bias_batch = 0
    all_sub_k_set = []
    sub_bias_k_set = 0
    all_khwl_batch = []
    khwl_bias_batch = 0

    for idx, (x, y) in enumerate(batch):
        N, M = x["dhg"].num_v, x["dhg"].num_e
        H_idx = x["dhg"].H.clone()._indices()
        H_idx[0] += bias_row
        H_idx[1] += bias_col
        bias_row += N
        bias_col += M
        all_X.append(torch.tensor(x["v_ft"]))
        all_Y.append(y)
        all_N.append(N)
        all_M.append(M)
        all_H_idx.append(H_idx)
        the_batch = [bias_batch] * N
        bias_batch += 1

        # KHWL
        sub_N, sub_M = x["sub_dhg"].num_v, x["sub_dhg"].num_e
        sub_H_idx = x["sub_dhg"].H.clone()._indices()
        sub_H_idx[0] += sub_bias_row
        sub_H_idx[1] += sub_bias_col
        sub_bias_row += sub_N
        sub_bias_col += sub_M

        khwl_N, khwl_M = x["khwl_hypergraph"].num_v, x["khwl_hypergraph"].num_e
        khwl_H_idx = x["khwl_hypergraph"].H.clone()._indices()
        khwl_H_idx[0] += khwl_bias_row
        khwl_H_idx[1] += khwl_bias_col
        khwl_bias_row += khwl_N
        khwl_bias_col += khwl_M

        sub_batch = x['sub_batch'] + sub_bias_batch
        sub_bias_batch += x['K_pair_list'].shape[0]
        sub_k_set = x['K_pair_list'] + sub_bias_k_set
        sub_bias_k_set += N
        khwl_batch = [khwl_bias_batch] * x['K_pair_list'].shape[0]
        khwl_bias_batch += 1

        all_sub_X.append(torch.tensor(x["sub_v_ft"]))
        all_sub_e_lbl.append(torch.tensor(x["sub_e_lbl"]))
        all_sub_N.append(sub_N)
        all_sub_M.append(sub_M)
        all_sub_H_idx.append(sub_H_idx)
        all_batch.append(torch.LongTensor(the_batch))

        all_khwl_N.append(khwl_N)
        all_khwl_M.append(khwl_M)
        all_khwl_H_idx.append(khwl_H_idx)

        all_sub_batch.append(torch.LongTensor(sub_batch))
        all_khwl_batch.append(torch.LongTensor(khwl_batch))
        all_sub_k_set.append(torch.tensor(sub_k_set))

    all_X = torch.cat(all_X).float()
    all_Y = torch.tensor(all_Y)
    all_N = torch.tensor(all_N)
    all_M = torch.tensor(all_M)
    all_H_idx = torch.cat(all_H_idx, dim=1)
    all_batch = torch.cat(all_batch)

    # KHWL
    all_sub_X = torch.cat(all_sub_X).float()
    all_sub_e_lbl = torch.cat(all_sub_e_lbl).float()
    all_sub_N = torch.tensor(all_sub_N)
    all_sub_M = torch.tensor(all_sub_M)
    all_sub_H_idx = torch.cat(all_sub_H_idx, dim=1)
    all_khwl_N = torch.tensor(all_khwl_N)
    all_khwl_M = torch.tensor(all_khwl_M)
    all_khwl_H_idx = torch.cat(all_khwl_H_idx, dim=1)
    all_sub_batch = torch.cat(all_sub_batch)
    all_khwl_batch = torch.cat(all_khwl_batch)
    all_sub_k_set = torch.cat(all_sub_k_set)

    return (all_X, all_H_idx, all_N, all_M, all_sub_X, all_sub_e_lbl, all_sub_H_idx, all_sub_N, all_sub_M, all_sub_batch, all_sub_k_set, all_khwl_H_idx, all_khwl_N, all_khwl_M, all_khwl_batch, all_batch), all_Y
