import os
import pickle
import random

import dhg
import numpy as np
import scipy
from dhg import Graph, Hypergraph
import time
import torch
from itertools import combinations
from itertools import permutations
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import multiprocessing

from model.hypergraph_wl_subtree_kernel import HypergraphSubtreeKernel

g2hg_func = dhg.Hypergraph.from_graph
hg2g_func = dhg.Graph.from_hypergraph_clique

def load_data(name, degree_as_tag, model_type, K = 2, multi_threads=False, kwl_type = 'owl', sample_method = 'top'):
    start_time = time.time()

    # graph dataset
    if name in ["RG_macro", "RG_sub"]:
        data_type = "graph"
        folder = "RG"
        multi_label = False
    elif name in ["MUTAG", "NCI1", "PROTEINS", "IMDBMULTI", "IMDBBINARY"]:
        data_type = "graph"
        folder = name
        multi_label = False
    elif name in ["RHG_3", "RHG_10", "RHG_table", "RHG_pyramid"]:
        data_type = "hypergraph"
        folder = "RHG"
        multi_label = False
    elif name in ["steam_player"]:
        data_type = "hypergraph"
        folder = "STEAM"
        multi_label = False
    elif name in ["IMDB_dir_genre_m"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = True
    elif name in ["IMDB_dir_form", "IMDB_dir_genre"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = False
    elif name in ["IMDB_wri_genre_m"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = True
    elif name in ["IMDB_wri_form", "IMDB_wri_genre"]:
        data_type = "hypergraph"
        folder = "IMDB"
        multi_label = False
    elif name in ["twitter_friend"]:
        data_type = "hypergraph"
        folder = "TWITTER"
        multi_label = False
    else:
        raise NotImplementedError
    if data_type == "graph" and model_type == "hypergraph":
        trans_func = g2hg_func
    elif data_type == "hypergraph" and model_type == "graph":
        trans_func = hg2g_func
    else:
        trans_func = lambda x: x
    print("KHWL type:", kwl_type)
    if sample_method == "none":
        if os.path.exists(f"./data/{data_type}/{folder}/{name}_{K}{kwl_type}.pkl"):
            with open(f"./data/{data_type}/{folder}/{name}_{K}{kwl_type}.pkl", 'rb') as f:
                g_list, y_list, meta = pickle.load(f)
            return g_list, y_list, meta
    else:
        if os.path.exists(f"./data/{data_type}/{folder}/{name}_{K}{kwl_type}_{sample_method}.pkl"):
            with open(f"./data/{data_type}/{folder}/{name}_{K}{kwl_type}_{sample_method}.pkl", 'rb') as f:
                g_list, y_list, meta = pickle.load(f)
            return g_list, y_list, meta



    # read data
    x_list = []
    with open(f"./data/{data_type}/{folder}/{name}.txt", "r") as f:
        n_g = int(f.readline().strip())
        for _ in range(n_g):
            row = f.readline().strip().split()
            num_v, num_e = int(row[0]), int(row[1])
            g_lbl = [int(x) for x in row[2:]]
            v_lbl = f.readline().strip().split()
            v_lbl = [[int(x) for x in s.split("/")] for s in v_lbl]
            e_list = []
            for _ in range(num_e):
                row = f.readline().strip().split()
                e_list.append([int(x) for x in row])
            if data_type == "graph":
                d = Graph(num_v, e_list)
            else:
                d = Hypergraph(num_v, e_list)
            d = trans_func(d)
            x_list.append(
                {
                    "num_v": num_v,
                    "num_e": d.num_e,
                    "v_lbl": v_lbl,
                    "g_lbl": g_lbl,
                    "e_list": d.e[0],
                    "e_lbl": [int(x) for x in d.e[1]],
                    "dhg": d,
                }
            )
    if degree_as_tag:
        for x in x_list:
            x["v_lbl"] = [int(v) for v in x["dhg"].deg_v]
            if isinstance(x["dhg"], Graph):
                x["e_lbl"] = [2] * x["num_e"]
            else:
                x["e_lbl"] = [int(e) for e in x["dhg"].deg_e]


    # # initialization the k_pair value, save in k_hwl_matrix
    # if multi_threads:
    #     # use multi-threads
    #     num_threads = 32
    #     with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #         futures = [executor.submit(Initialize_k_tuple, x, K) for x in x_list]
    #         for future in tqdm(as_completed(futures), total=len(x_list), desc=" multi-threads Processing"):
    #             result = future.result()
    # else:
    # single threads
    # print(f"single-thread Processing")
    # for x in tqdm(x_list):
    #     khwl_list.append()
    # filename = f"./data/{data_type}/{folder}/{name}_{K}hwl.pkl"
    # with open(filename, 'wb') as f:
    #     pickle.dump(khwl_list, f)


    v_lbl_set, e_lbl_set, g_lbl_set = set(), set(), set()
    for x in x_list:
        # one node maybe have multi labels
        if isinstance(x["v_lbl"][0], list):
            for v_lbl in x["v_lbl"]:
                v_lbl_set.update(v_lbl)
        else:
            v_lbl_set.update(x["v_lbl"])
        e_lbl_set.update(x["e_lbl"])

        g_lbl_set.update(x["g_lbl"])
    # re-map labels
    v_lbl_map = {x: i for i, x in enumerate(sorted(v_lbl_set))}
    e_lbl_map = {x: i for i, x in enumerate(sorted(e_lbl_set))}
    g_lbl_map = {x: i for i, x in enumerate(sorted(g_lbl_set))}
    ft_dim, n_classes = len(v_lbl_set), len(g_lbl_set)
    print(f'KHWL TYPE: {K}{kwl_type}, SAMPLE METHOD: {sample_method}')
    for x in tqdm(x_list):
        x["g_lbl"] = [g_lbl_map[c] for c in x["g_lbl"]]
        if isinstance(x["v_lbl"][0], list):
            # because of the multi-label of nodes, so the tuple is used
            x["v_lbl"] = [tuple(sorted([v_lbl_map[c] for c in s])) for s in x["v_lbl"]]
        else:
            x["v_lbl"] = [v_lbl_map[c] for c in x["v_lbl"]]
        x["e_lbl"] = [e_lbl_map[c] for c in x["e_lbl"]]
        x["v_ft"] = np.zeros((x["num_v"], ft_dim))
        row_idx, col_idx = [], []
        for v_idx, v_lbls in enumerate(x["v_lbl"]):
            if isinstance(v_lbls, list) or isinstance(v_lbls, tuple):
                for v_lbl in v_lbls:
                    row_idx.append(v_idx)
                    col_idx.append(v_lbl)
            else:
                row_idx.append(v_idx)
                col_idx.append(v_lbls)
        x["v_ft"][row_idx, col_idx] = 1
        # khwl initialize
        if kwl_type == 'owl':
            x.update(Initialize_khwl_owl(x, K, sample_method))
        elif kwl_type == 'fwl':
            x.update(Initialize_khwl_fwl(x, K, sample_method))

    y_list = []
    if multi_label:
        for x in x_list:
            tmp = np.zeros(n_classes).astype(int)
            tmp[x["g_lbl"]] = 1
            y_list.append(tmp.tolist())
    else:
        y_list = [g["g_lbl"][0] for g in x_list]
    meta = {
        "multi_label": multi_label,
        "data_type": data_type,
        "ft_dim": ft_dim,
        "n_classes": len(g_lbl_set),
    }
    end_time = time.time()

    data_to_save = (x_list, y_list, meta)
    if sample_method == "none":
        with open(f'./data/{data_type}/{folder}/{name}_{K}{kwl_type}.pkl', 'wb') as f:
            pickle.dump(data_to_save, f)
    else:
        with open(f'./data/{data_type}/{folder}/{name}_{K}{kwl_type}_{sample_method}.pkl', 'wb') as f:
            pickle.dump(data_to_save, f)
    return x_list, y_list, meta

# def sub_graph(K_tuple_node, hypergraph, v_lbl, e_lbl):
#     if isinstance(v_lbl[0], list):
#         v_label = [ "".join(str(one_lbl) for one_lbl in v_lbl[K_index]) for K_index in K_tuple_node]
#     else:
#         v_label = [str(v_lbl[K_index]) for K_index in K_tuple_node]
#     # get the hyperedge of the node group
#     Ne_list = [hypergraph.N_e(K_index).tolist() for K_index in K_tuple_node]
#     Eid_map_tupleE = {}
#     tupleE_map_Elbl = {}
#     for i, Ne in enumerate(Ne_list):
#         for e in Ne:
#             if e in Eid_map_tupleE:
#                 Eid_map_tupleE[e].append(i)
#             else:
#                 Eid_map_tupleE[e] = [i]
#     # e_dict{e_id:node_list}, result and e_lbl need to align
#     for e_id, node_list in Eid_map_tupleE.items():
#         if tuple(node_list) in tupleE_map_Elbl:
#             tupleE_map_Elbl[tuple(node_list)].append(e_lbl[e_id])
#         else:
#             tupleE_map_Elbl[tuple(node_list)] = [e_lbl[e_id]]
#
#     hypergraph = dhg.Hypergraph(len(K_tuple_node), list(tupleE_map_Elbl.keys()))
#     return {
#         "v_lbl": v_label,
#         "dhg": hypergraph,
#         "e_lbl": [sorted(e_lbl) for e_lbl in tupleE_map_Elbl.values()]
#     }

def sub_graph_v2(K_tuple_node, h, v_lbl, e_lbl, H):
    if isinstance(v_lbl[0], list):
        v_label = [ "".join(str(one_lbl) for one_lbl in v_lbl[K_index]) for K_index in K_tuple_node]
    else:
        v_label = [str(v_lbl[K_index]) for K_index in K_tuple_node]
    tupleE_map_Eindex = {}

    indices = H.coalesce().indices()
    # Iterate over each hyperedge (column in H)
    for hyperedge_index in range(h.H.size(1)):
        # Find nodes that are part of the hyperedge
        nodes_in_hyperedge = indices[0][indices[1] == hyperedge_index]
        # Find the intersection between nodes_in_hyperedge and node_tuple
        intersecting_nodes = [node for node in K_tuple_node if node in nodes_in_hyperedge]
        # If the intersecting nodes are a subset of node_tuple and there are at least two nodes
        if len(intersecting_nodes) > 0:
            tupleE_map_Eindex[tuple(intersecting_nodes)] = hyperedge_index
    hypergraph = dhg.Hypergraph(len(K_tuple_node), [[K_tuple_node.index(element) for element in edge_tuple] for edge_tuple in tupleE_map_Eindex.keys()])
    return {
        "v_lbl": v_label,
        "dhg": hypergraph,
        "e_lbl": [e_lbl[e_index] for e_index in tupleE_map_Eindex.values()]
    }


def sub_graph_with_hyper_edge(K_tuple_node, h, v_lbl, H):
    v_label = [v_lbl[K_index] for K_index in K_tuple_node]
    tupleE_map_Eindex = {}

    indices = H.coalesce().indices()
    # Iterate over each hyperedge (column in H)
    for hyperedge_index in range(h.H.size(1)):
        # Find nodes that are part of the hyperedge
        nodes_in_hyperedge = indices[0][indices[1] == hyperedge_index]
        # Find the intersection between nodes_in_hyperedge and node_tuple
        intersecting_nodes = [node for node in K_tuple_node if node in nodes_in_hyperedge]
        # If the intersecting nodes are a subset of node_tuple and there are at least two nodes

        # Consider self-loop in sub hypergraph
        if len(intersecting_nodes) > 0:
            if tuple(intersecting_nodes) in tupleE_map_Eindex:
                tupleE_map_Eindex[tuple(intersecting_nodes)] = tupleE_map_Eindex[tuple(intersecting_nodes)] + 1
            else:
                tupleE_map_Eindex[tuple(intersecting_nodes)] = 1

        # not Consider self-loop in sub hypergraph
        # if len(intersecting_nodes) > 1:
        #     tupleE_map_Eindex[tuple(intersecting_nodes)] = hyperedge_index
    hyperedge = [[K_tuple_node.index(element) for element in edge_tuple] for edge_tuple in tupleE_map_Eindex.keys()]
    return {
        "v_ft": v_label,
        "hyperedge": hyperedge,
        "e_lbl": list(tupleE_map_Eindex.values())
    }

def Initialize_khwl_owl(x, K, sample_method):
    random_num = 10
    sample_nodes = []
    if sample_method == "none" or x["num_v"] <= random_num:
        sample_nodes = range(x["num_v"])
        K_pair_list = list(combinations(sample_nodes, K))
    elif sample_method == "random":
        sample_nodes = random.sample(range(x["num_v"], random_num))
        K_pair_list = list(combinations(sample_nodes, K))
    elif sample_method == "top":
        deg_v = x["dhg"].D_v
        deg_v_values = deg_v.values()
        deg_v_indices = deg_v.indices()
        top_degv_values, top_degv_indices = torch.topk(deg_v_values, random_num)
        sample_nodes = sorted(deg_v_indices[0, top_degv_indices].tolist())
        K_pair_list = list(combinations(sample_nodes, K))

    H = x["dhg"].H
    khwl_hyperedge = set()
    index = 0
    batch = []
    sub_v_ft = []
    sub_e_lbl = []
    sub_hyperedge_list = []
    for K_pair in K_pair_list:
        sub_graph_info = sub_graph_with_hyper_edge(K_pair, x["dhg"], x["v_ft"], H)
        # initialize single subset structure
        sub_hyperedge_list += [[c+len(sub_v_ft) for c in hyper_edge] for hyper_edge in sub_graph_info['hyperedge']]
        sub_v_ft += sub_graph_info["v_ft"]
        sub_e_lbl += sub_graph_info["e_lbl"]
        batch += [index] * K
        index +=1

        # initialize total subset structure
        for i in range(len(K_pair)):
            current_point = K_pair[i]
            hyperedges = [e for e in range(H.shape[1]) if H[current_point, e] == 1]
            other_points = set()
            for e in hyperedges:
                other_points.update([v for v in sample_nodes if H[v, e] == 1 and v not in K_pair])
            one_hyperedge = []
            one_hyperedge.append(K_pair_list.index(K_pair))
            # if not other_points:
            #     continue
            for other_point in other_points:
                new_K_pair = list(K_pair)
                new_K_pair[i] = other_point
                one_hyperedge.append(K_pair_list.index(tuple(sorted(new_K_pair))))
            khwl_hyperedge.add(tuple(sorted(one_hyperedge)))

    sub_dhg = Hypergraph(len(sub_v_ft), sub_hyperedge_list)

    return {
        "sub_dhg": sub_dhg,
        "sub_v_ft": np.array(sub_v_ft),
        "sub_e_lbl": np.array(sub_e_lbl),
        "sub_batch": np.array(batch),
        "khwl_hypergraph": dhg.Hypergraph(len(K_pair_list), list(khwl_hyperedge)),
        "K_pair_list": np.array(K_pair_list)
    }
def Initialize_khwl_fwl(x, K, sample_method):
    random_num = 10
    if sample_method == "none" or x["num_v"] <= random_num:
        sample_nodes = range(x["num_v"])
        K_pair_list = list(combinations(sample_nodes, K))
    elif sample_method == "random":
        sample_nodes = random.sample(range(x["num_v"], random_num))
        K_pair_list = list(combinations(sample_nodes, K))
    elif sample_method == "top":
        deg_v = x["dhg"].D_v
        deg_v_values = deg_v.values()
        deg_v_indices = deg_v.indices()
        top_degv_values, top_degv_indices = torch.topk(deg_v_values, random_num)
        sample_nodes = sorted(deg_v_indices[0, top_degv_indices].tolist())
        K_pair_list = list(combinations(sample_nodes, K))

    H = x["dhg"].H
    khwl_hyperedge = set()
    index = 0
    batch = []
    sub_v_ft = []
    sub_e_lbl = []
    sub_hyperedge_list = []
    for K_pair in K_pair_list:
        sub_graph_info = sub_graph_with_hyper_edge(K_pair, x["dhg"], x["v_ft"], H)
        # initialize single subset structure
        sub_hyperedge_list += [[c+len(sub_v_ft) for c in hyper_edge] for hyper_edge in sub_graph_info['hyperedge']]
        sub_v_ft += sub_graph_info["v_ft"]
        sub_e_lbl += sub_graph_info["e_lbl"]
        batch += [index] * K
        index += 1

        # initialize total khwl structure
        local_nodes = []
        for i in range(len(K_pair)):
            current_point = K_pair[i]
            hyperedges = [e for e in range(H.shape[1]) if H[current_point, e] == 1]
            other_points = set()
            for e in hyperedges:
                other_points.update([v for v in sample_nodes if H[v, e] == 1 and v not in K_pair])
            local_nodes.append(other_points)

        all_local_nodes = set().union(*local_nodes)
        for node in all_local_nodes:
            one_hyperedge = []
            one_hyperedge.append(K_pair_list.index(K_pair))
            for j in range(len(local_nodes)):
                if node in local_nodes[j] and node not in K_pair:
                    new_K_pair = list(K_pair)
                    new_K_pair[j] = node
                    one_hyperedge.append(K_pair_list.index(tuple(sorted(new_K_pair))))
            khwl_hyperedge.add(tuple(sorted(one_hyperedge)))

    sub_dhg = Hypergraph(len(sub_v_ft), sub_hyperedge_list)

    return {
        "sub_dhg": sub_dhg,
        "sub_v_ft": np.array(sub_v_ft),
        "sub_e_lbl": np.array(sub_e_lbl),
        "sub_batch": np.array(batch),
        "khwl_hypergraph": dhg.Hypergraph(len(K_pair_list), list(khwl_hyperedge)),
        "K_pair_list": np.array(K_pair_list)
    }

# def Initialize_khwl_gwl(x, K):
#     K_pair_list = list(combinations(range(x["num_v"]), K))
#     H = x["dhg"].H
#     khwl_hyperedge = set()
#     index = 0
#     batch = []
#     sub_v_ft = []
#     sub_hyperedge_list = []
#     for K_pair in K_pair_list:
#         sub_graph_info = sub_graph_with_hyper_edge(K_pair, x["dhg"], x["v_ft"], H)
#         # initialize single subset structure
#         sub_hyperedge_list += [[c+len(sub_v_ft) for c in hyper_edge] for hyper_edge in sub_graph_info['hyperedge']]
#         sub_v_ft += sub_graph_info["v_ft"]
#         batch += [index] * K
#         index +=1
#         one_hyperedge = []
#         # initialize total subset structure
#         for i in range(len(K_pair)):
#             current_point = K_pair[i]
#             hyperedges = [e for e in range(H.shape[1]) if H[current_point, e] == 1]
#             other_points = set()
#             for e in hyperedges:
#                 other_points.update([v for v in range(H.shape[0]) if H[v, e] == 1 and v not in K_pair])
#             one_hyperedge.append(K_pair_list.index(K_pair))
#
#             for other_point in other_points:
#                 new_K_pair = list(K_pair)
#                 new_K_pair[i] = other_point
#                 one_hyperedge.append(K_pair_list.index(tuple(sorted(new_K_pair))))
#         khwl_hyperedge.add(tuple(sorted(one_hyperedge)))
#
#     sub_dhg = Hypergraph(len(sub_v_ft), sub_hyperedge_list)
#
#     return {
#         "sub_dhg": sub_dhg,
#         "sub_v_ft": np.array(sub_v_ft),
#         "sub_batch": np.array(batch),
#         "khwl_hypergraph": dhg.Hypergraph(len(K_pair_list), list(khwl_hyperedge)),
#         "K_pair_list": np.array(K_pair_list)
#     }


def separate_data(x_list, y_list, n_fold):
    kf = KFold(n_splits=n_fold, shuffle=True)
    n_fold_idx = []
    for train_idx, test_idx in kf.split(x_list, y_list):
        n_fold_idx.append((train_idx, test_idx))
    return n_fold_idx


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

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices.long()]
        valid_idx = labeled_nodes[val_indices.long()]
        test_idx = labeled_nodes[test_indices.long()]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        #         ipdb.set_trace()
        # indices = []
        # for i in range(label.max()+1):
        #     index = torch.where((label == i))[0].view(-1)
        #     index = index[torch.randperm(index.size(0))]
        #     indices.append(index)
        #
        # percls_trn = int(train_prop/(label.max()+1)*len(label))
        # val_lb = int(valid_prop*len(label))
        # train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        # rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        # rest_index = rest_index[torch.randperm(rest_index.size(0))]
        # valid_idx = rest_index[:val_lb]
        # test_idx = rest_index[val_lb:]
        # split_idx = {'train': train_idx,
        #              'valid': valid_idx,
        #              'test': test_idx}

        indices = []
        for i in range(int(label.max()) + 1):
            index = torch.where(label == i)[0].tolist()
            indices.append(index)

        valid_size = int(valid_prop * min([len(sublist) for sublist in indices]))
        test_size = int(valid_size)

        valid_indices = []
        test_indices = []
        indices = [np.random.permutation(x) for x in indices]
        for idx_list in indices:
            valid_indices.extend(idx_list[:valid_size])
            test_indices.extend(idx_list[valid_size:valid_size + test_size])

        train_indices = []
        for idx_list in indices:
            train_indices.extend(idx_list[valid_size + test_size:])

        split_idx = {
            'train': torch.tensor(train_indices),
            'valid': torch.tensor(valid_indices),
            'test': torch.tensor(test_indices)
        }
    return split_idx

if __name__ == "__main__":
    g_list, y_list, meta = load_data("steam_player", False, "hypergraph", 3)
    print(g_list[0]["k_hwl_matrix"])


# def one_HWL(hg1, hg2):
#     one_HWL_map = {}
#     n_iter = 4
#     remap_v(hg1, one_HWL_map)
#     remap_v(hg2, one_HWL_map)
#     remap_e(hg1, one_HWL_map)
#     remap_e(hg2, one_HWL_map)
#     for _ in range(n_iter):
#         # hg1 process
#         tmp = []
#         for e_idx in range(hg1["dhg"].num_e):
#             cur_lbl = hg1["e_lbl"][e_idx]
#             nbr_lbl = sorted(
#                 hg1["v_lbl"][v_idx] for v_idx in hg1["dhg"].nbr_v(e_idx)
#             )
#             tmp.append(f"{cur_lbl},{nbr_lbl}")
#         hg1["e_lbl"] = tmp
#         remap_e(hg1, one_HWL_map)
#         tmp = []
#         for v_idx in range(hg1["dhg"].num_v):
#             cur_lbl = hg1["v_lbl"][v_idx]
#             nbr_lbl = sorted(
#                 hg1["e_lbl"][e_idx] for e_idx in hg1["dhg"].nbr_e(v_idx)
#             )
#             tmp.append(f"{cur_lbl},{nbr_lbl}")
#         hg1["v_lbl"] = tmp
#         remap_v(hg1, one_HWL_map)
#         # hg2 process
#         tmp = []
#         for e_idx in range(hg2["dhg"].num_e):
#             cur_lbl = hg2["e_lbl"][e_idx]
#             nbr_lbl = sorted(
#                 hg2["v_lbl"][v_idx] for v_idx in hg2["dhg"].nbr_v(e_idx)
#             )
#             tmp.append(f"{cur_lbl},{nbr_lbl}")
#         hg2["e_lbl"] = tmp
#         remap_e(hg2, one_HWL_map)
#         tmp = []
#         for v_idx in range(hg2["dhg"].num_v):
#             cur_lbl = hg2["v_lbl"][v_idx]
#             nbr_lbl = sorted(
#                 hg2["e_lbl"][e_idx] for e_idx in hg2["dhg"].nbr_e(v_idx)
#             )
#             tmp.append(f"{cur_lbl},{nbr_lbl}")
#         hg2["v_lbl"] = tmp
#         remap_v(hg2, one_HWL_map)
#         if not sorted(hg1["v_lbl"]) == sorted(hg2["v_lbl"]):
#             return 0
#     return 1
#
# def remap_e(hg, one_HWL_map):
#     for e_idx in range(hg["dhg"].num_e):
#         if isinstance(hg["e_lbl"][e_idx], list):
#             cur_lbl = "e" + ''.join(str(e_id) for e_id in hg["e_lbl"][e_idx])
#         else:
#             cur_lbl = hg["e_lbl"][e_idx]
#             cur_lbl = "e" + str(cur_lbl)
#         if cur_lbl not in one_HWL_map:
#             one_HWL_map[cur_lbl] = len(one_HWL_map)
#         hg["e_lbl"][e_idx] = one_HWL_map[cur_lbl]
#
#
# def remap_v(hg, one_HWL_map):
#     for v_idx in range(hg["dhg"].num_v):
#         cur_lbl = hg["v_lbl"][v_idx]
#         cur_lbl = "v" + str(cur_lbl)
#         if cur_lbl not in one_HWL_map:
#             one_HWL_map[cur_lbl] = len(one_HWL_map)
#         hg["v_lbl"][v_idx] = one_HWL_map[cur_lbl]
#
#
# def Initialize_k_tuple(x, K):
#     node_id = 1
#     K_tuple_dict = {}
#     shape = (x["num_v"],) * K
#     x["k_hwl_matrix"] = np.zeros(shape, dtype=object)
#     K_pair_list = list(combinations(range(x["num_v"]), K))
#     for K_pair in K_pair_list:
#         if isinstance(x["v_lbl"][0], list):
#             node_label = [''.join(str(v_lbl) for v_lbl in sorted(x["v_lbl"][K_index])) for K_index in K_pair]
#             node_multiset_string = "".join(sorted(node_label))
#         else:
#             node_label = [str(x["v_lbl"][K_index]) for K_index in K_pair]
#             node_multiset_string = "".join(sorted(node_label))
#
#         Ne_list = [x["dhg"].N_e(K_index).tolist() for K_index in K_pair]
#         edge_label = [''.join(str(x["e_lbl"][edge]) for edge in Ne_sub_list) for Ne_sub_list in Ne_list]
#         edge_multiset_string = "".join(sorted(edge_label))
#         if (node_multiset_string, edge_multiset_string) in K_tuple_dict.keys():
#             found = False
#             # try to use 1HWL
#             for sub_graph_node in K_tuple_dict[(node_multiset_string, edge_multiset_string)]:
#                 if one_HWL(sub_graph(sub_graph_node, x["dhg"], x["v_lbl"], x["e_lbl"]),
#                            sub_graph(K_pair, x["dhg"], x["v_lbl"], x["e_lbl"])):
#                     x["k_hwl_matrix"][tuple(zip(*list(permutations(K_pair))))] = x["k_hwl_matrix"][sub_graph_node]
#                     found = True
#                     break
#             if not found:
#                 K_tuple_dict[(node_multiset_string, edge_multiset_string)].append(K_pair)
#                 x["k_hwl_matrix"][tuple(zip(*list(permutations(K_pair))))] = node_id
#                 node_id += 1
#         else:
#             x["k_hwl_matrix"][tuple(zip(*list(permutations(K_pair))))] = node_id
#             node_id += 1
#             K_tuple_dict[(node_multiset_string, edge_multiset_string)] = [K_pair]
#     return x
#
