import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred)
        acc_list.append(f1)

    return sum(acc_list) / len(acc_list)

def eval_precision(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        pre = precision_score(y_true, y_pred)
        acc_list.append(pre)

    return sum(acc_list) / len(acc_list)

def eval_recall(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        recall = recall_score(y_true, y_pred)
        acc_list.append(recall)
    
    return sum(acc_list) / len(acc_list)

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    torch.save(y_true,'y_true.pt')
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    pd.DataFrame(y_pred).to_csv("results/y_pred.csv")
    pd.DataFrame(y_true).to_csv("results/y_true.csv")
    return sum(acc_list) / len(acc_list)

@torch.no_grad()
def evaluate_gpu(model, dataset, split_idx, eval_func, criterion, args, device, result=None):
    model.eval()

    model.to(device)
    dataset.label = dataset.label.to(device)
    adjs_, x = dataset.graph['adjs'], dataset.graph['node_feat']
    if args.dataset in ['crosssite','weibo','douban','crosssite_en']:
        x = []
        x.append(dataset.graph['node_post'].to(device))
        x.append(dataset.graph['topic_post'].to(device))
        x.append(dataset.graph['node_part'].to(device))
        x.append(dataset.graph['topic'].to(device))

    adjs = []
    adjs.append(adjs_[0].to(device))
    for k in range(args.rb_order - 1):
        adjs.append(adjs_[k + 1].to(device))
    if args.dataset in ['crosssite','weibo','douban','crosssite_en']:
        out, _ = model(x[0],x[1],x[2],x[3], adjs)
    else:
        out, _ = model(x, adjs)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out