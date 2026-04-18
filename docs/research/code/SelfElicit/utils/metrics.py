import copy
import torch
import functools
import numpy as np
from typing import List, Union
from collections import defaultdict, OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, precision_recall_curve

softmax = lambda x: torch.softmax(torch.from_numpy(x), dim=-1).float()


def aggregate_prob(prob, method):
    if method == 'max':
        agg_prob = prob.max()
    elif method == 'prod':
        agg_prob = 1 - (1 - prob).prod()
    elif method == 'mean':
        agg_prob = prob.mean()
    else:
        raise ValueError
    return agg_prob


def overlength_penalty(prob, length, gamma=None):
    if gamma is None or length > gamma:
        penalty = 1.0
    else:
        penalty = np.exp(1 - gamma / length)

    return prob * penalty


def thresh_max_metric(y_true, y_prob, metric_fn):
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()
    if len(set(y_true)) > 2:
        raise NotImplementedError
    assert (0 <= y_prob).all() and (y_prob <= 1).all()

    valid_thresholds, metrics_thresholds = [], []
    _, _, thresholds = precision_recall_curve(y_true, y_prob)
    for idx, thres in enumerate(thresholds):
        y_pred = (y_prob >= thres).astype(int)
        cur_metric = metric_fn(y_true, y_pred)
        if not np.isnan(cur_metric):
            metrics_thresholds.append(cur_metric)
            valid_thresholds.append(thres)

    # locate the index of the largest metric
    idx = np.argmax(metrics_thresholds)
    best_thresh = valid_thresholds[idx]
    best_metric = metrics_thresholds[idx]
    return best_thresh, best_metric


def get_results(probs: List[Union[np.ndarray, List]],
                labels: List[Union[np.ndarray, List]],
                sample_labels: List[bool],
                strategies: List[str],
                aggregate: str,
                gamma: int = None):
    sentence_prob, sample_prob = defaultdict(list), defaultdict(list)
    sentence_label, sample_label = defaultdict(list), defaultdict(list)

    for strategy in strategies:
        for sid in range(len(probs)):
            prob = copy.deepcopy(probs[sid])
            if strategy == 'ignore':
                prob = softmax(prob[:, :2])
                prob[torch.isnan(prob).any(dim=-1)] = torch.tensor([1., 0])
                prob = prob[:, 1].float().numpy()
            elif strategy == 'neg':
                prob = softmax(prob)
                prob = prob[:, 1].float().numpy()
            elif strategy == 'pos':
                prob[:, 1] += prob[:, 2]
                prob = prob[:, [0, 1]]
                prob = softmax(prob)
                prob = prob[:, 1].float().numpy()
                prob[prob > 1] = 1.0

            sentence_prob[strategy].append(prob)
            sentence_label[strategy].append(1 - np.array(labels[sid]))
            agg_prob = aggregate_prob(prob, aggregate)
            agg_prob = overlength_penalty(agg_prob, len(prob), gamma=gamma)
            sample_prob[strategy].append(agg_prob)
            sample_label[strategy].append(1 - np.array(sample_labels[sid]))

    return sentence_prob, sentence_label, sample_prob, sample_label


def get_metric(prob: np.ndarray, label: np.ndarray):
    f1_thres, best_f1 = thresh_max_metric(label, prob, metric_fn=f1_score)
    f2_thres, best_f2 = thresh_max_metric(label, prob, metric_fn=functools.partial(fbeta_score, beta=2))
    auc = roc_auc_score(label, prob)

    # accu/prec/recall with f1 threshold
    pred = (prob > f1_thres).astype(int)
    accu = accuracy_score(label, pred)
    prec = precision_score(label, pred)
    recall = recall_score(label, pred)

    metric = OrderedDict({
        'accu': accu,
        'prec': prec,
        'recall': recall,
        'f1': best_f1,
        'f2': best_f2,
        'auc': auc,
        'f1_thres': f1_thres,
        'f2_thres': f2_thres
    })
    return metric


def get_metrics(probs: List[Union[np.ndarray, List]],
                labels: List[Union[np.ndarray, List]],
                sample_labels: List[bool],
                strategies: List[str],
                aggregate: str,
                penalty: int = None,
                verbose: bool = True):
    results = get_results(probs,
                          labels,
                          sample_labels,
                          strategies=strategies,
                          aggregate=aggregate,
                          gamma=penalty)
    sentence_prob, sentence_label, sample_prob, sample_label = results

    metrics = {}
    for strategy in strategies:
        # sentence-level
        _sentence_prob = np.stack([ele for li in sentence_prob[strategy] for ele in li])
        _sentence_label = np.stack([ele for li in sentence_label[strategy] for ele in li])
        metric = get_metric(_sentence_prob, _sentence_label)
        for k, v in metric.items():
            metrics[f"{strategy}-{k}"] = float(v)
        if verbose:
            print(f"{strategy}\tsentence")
            print(f"{metric['accu']:.4f}\t{metric['prec']:.4f}\t{metric['recall']:.4f}\t", end="")
            print(f"{metric['f1_thres']:.4f}\t{metric['f2_thres']:.4f}\t", end="")
            print(f"{metric['f1']:.4f}\t{metric['f2']:.4f}\t{metric['auc']:.4f}")

        # query-level
        _sample_prob = np.stack(sample_prob[strategy])
        _sample_label = np.stack(sample_label[strategy])
        metric = get_metric(_sample_prob, _sample_label)
        for k, v in metric.items():
            metrics[f"{strategy}-{k}-agg"] = float(v)
        if verbose:
            print(f"query\t{aggregate}")
            print(f"{metric['accu']:.4f}\t{metric['prec']:.4f}\t{metric['recall']:.4f}\t", end="")
            print(f"{metric['f1_thres']:.4f}\t{metric['f2_thres']:.4f}\t", end="")
            print(f"{metric['f1']:.4f}\t{metric['f2']:.4f}\t{metric['auc']:.4f}")

    return metrics
