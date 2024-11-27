from collections import defaultdict
from copy import deepcopy


import time

import dhg
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

from data.data_utils import load_data, separate_data
from model.hypergraph_wl_hyedge_kernel import HypergraphHyedgeKernel
from model.hypergraph_wl_subtree_kernel import HypergraphSubtreeKernel
from model.load_model import parse_method
from utils.evaluate import count_parameters, eval_acc
from utils.logging import Logger
from utils.train_utils import HypergraphLoader, performance
from utils.config import parse_config
from dhg.experiments import HypergraphVertexClassificationTask as Task

import torch
g2hg_func = dhg.Hypergraph.from_graph

def train(train_data, model, optimizer, epoch, device):
    model.train()

    for batch_idx, (data, target) in enumerate(train_data):
        data = [d.to(device) for d in data]
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 10 == 0:
            # print(f"Epoch {epoch} [{batch_idx}/{len(train_data)}] -> loss: {loss.item():.6f}")

def test(test_data, model, device, multi_label):
    model.eval()
    outputs, targets = [], []
    for data, target in test_data:
        data = [d.to(device) for d in data]
        target = target.to(device)
        output = model(data)
        outputs.append(output.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets)
    val, res = performance(outputs, targets, multi_label)
    torch.cuda.empty_cache()
    # print(f"----> test res: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])} \n")
    return val, res


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

if __name__ == '__main__':
    args = parse_config()

    torch.manual_seed(42) # CPU
    torch.cuda.manual_seed(42) # GPU
    torch.cuda.manual_seed_all(42) # All GPU

    if args.cuda in [0, 1, 2, 3, 4 ,5, 6, 7]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    work_root = "./log"

    # load data
    # print("="*20, f'load {args.dataset}',"="*20)
    x_list, y_list, meta = load_data(args.dataset, args.degree_as_tag, args.model_type, args.tuple_K, args.multi_threads, args.kwl_type, args.sample_method)
    args.multi_label = meta["multi_label"]
    args.ft_dim = meta["ft_dim"]
    args.n_classes = meta["n_classes"]
    # # spilite dataset to train and test
    n_fold_idx = separate_data(x_list, y_list, args.nfold)

    if args.multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    ### Training loop ###
    test_res, test_all_res = [], defaultdict(list)
    for fold_idx, (train_idx, test_idx) in tqdm(enumerate(n_fold_idx)):

        train_x_list, train_y_list, test_x_list, test_y_list = [], [], [], []
        for idx in train_idx:
            train_x_list.append(x_list[idx])
            train_y_list.append(y_list[idx])
        for idx in test_idx:
            test_x_list.append(x_list[idx])
            test_y_list.append(y_list[idx])

        train_data = HypergraphLoader(
            train_x_list, train_y_list, args.batch_size, 0, True, g2hg_func
        )
        test_data = HypergraphLoader(
            test_x_list, test_y_list, args.batch_size, 0, False, g2hg_func
        )

        best_test_val = 0
        model = parse_method(args)
        model = model.to(device)
        # getModelSize(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        epoch = 0
        tolerate = 0
        while True:
            epoch += 1
            tolerate += 1
            train(train_data, model, optimizer, epoch, device)
            test_val, res = test(test_data, model, device, args.multi_label)
            if test_val > best_test_val:
                print(f"best test val:{test_val}, epoch:{epoch}")
                best_test_val = test_val
                tolerate = 0
                best_res = res
            scheduler.step()
            if tolerate > args.epochs:
                break
        print(f"[{fold_idx+1}/{len(n_fold_idx)}] test results: {best_test_val:.4f}\n\n")
        test_res.append(best_test_val)
        for k, v in best_res.items():
            test_all_res[k].append(v)

    res = {k: sum(v) / len(v) for k, v in test_all_res.items()}
    print(f"mean test results: {' | '.join([f'{k}:{v:.5f}' for k, v in res.items()])}")
    print("--------------------------------------------------")
    print(f"mean test acc: {np.mean(test_res)*100:.2f}±{np.std(test_res)*100:.2f}")


