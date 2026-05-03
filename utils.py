import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _aggregate(vals, aggregation: str, topk: int):
    if aggregation == "mean":
        return float(np.mean(vals))
    elif aggregation == "max":
        return float(np.max(vals))
    elif aggregation == "topk":
        k = min(topk, len(vals))
        return float(np.mean(np.sort(vals)[-k:]))
    else:
        raise ValueError(f"Unknown AGGREGATION={aggregation}")


def apply_stress(x, blk, mode=None):
    if mode is None:
        return x

    x_mod = x.clone()
    B = x.size(0)

    if mode == "zero_nonparents":
        if blk.idx_other.any():
            x_mod[:, :, blk.idx_other] = 0.0
        return x_mod

    def _permute(mask):
        if not mask.any():
            return
        inds = torch.where(mask)[0].tolist()
        for pidx in inds:
            perm = torch.randperm(B, device=x.device)
            x_mod[:, :, pidx] = x[perm, :, pidx]

    if mode == "permute_nonparents":
        _permute(blk.idx_other)
        return x_mod
    if mode == "permute_parents":
        _permute(blk.idx_causal)
        return x_mod
    return x_mod


def adjust_predicts(score, label, threshold, calc_latency=False):
    score = np.asarray(score)
    label = (np.asarray(label) > 0.1).astype(bool)
    thr = np.asarray(threshold, dtype=float)
    if thr.ndim == 0:
        thr = np.full_like(score, float(thr))
    assert thr.shape[0] == score.shape[0], "Threshold length mismatch."

    predict = score > thr
    latency = 0
    state = False
    cnt = 0
    for i in range(len(score)):
        if label[i] and predict[i] and not state:
            state = True
            cnt += 1
            for j in range(i, -1, -1):
                if not label[j]:
                    break
                if not predict[j]:
                    predict[j] = True
                    latency += 1
        elif not label[i]:
            state = False
        if state:
            predict[i] = True
    return (predict, latency / (cnt + 1e-4)) if calc_latency else predict


def calc_point2point(pred, actual):
    actual = actual.astype(bool)
    pred = pred.astype(bool)
    TP = np.sum(pred & actual)
    FP = np.sum(pred & ~actual)
    FN = np.sum(~pred & actual)
    prec = TP / (TP + FP + 1e-5)
    rec = TP / (TP + FN + 1e-5)
    f1 = 2 * prec * rec / (prec + rec + 1e-5)
    auc = roc_auc_score(actual, pred) if actual.any() else 0.0
    return f1, prec, rec, TP, FP, FN, auc