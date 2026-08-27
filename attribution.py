import numpy as np
import torch
from sklearn.metrics import ndcg_score

from model import gaussian_nll
from utils import _aggregate

def parse_external_labels(filepath, D, one_based=True):
    labels = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            interval, dims_str = line.split(":")
            start, end = map(int, interval.split("-"))
            raw_dims = [int(x.strip()) for x in dims_str.split(",") if x.strip() != ""]
            if one_based:
                raw_dims = [d - 1 for d in raw_dims]
            dims = {d for d in raw_dims if 0 <= d < D}
            labels.append({"start": start, "end": end, "dims": dims})
    return labels

def get_ground_truth_dims(time_index, external_labels):
    for label in external_labels:
        if label["start"] <= time_index <= label["end"]:
            return label["dims"]
    return set()

def hit_att(ascore, labels, ps=(100, 150)):
    out = {}
    for p in ps:
        hits = []
        for a, l in zip(ascore, labels):
            gt = set(np.where(l == 1)[0])
            if len(gt) == 0:
                continue
            k = max(1, round(p * len(gt) / 100))
            top_k = set(np.argsort(a)[::-1][:k])
            hits.append(len(top_k & gt) / len(gt))
        out[f"Hit@{p}%"] = float(np.mean(hits)) if hits else float("nan")
    return out

def ndcg_att(ascore, labels, ps=(100, 150)):
    out = {}
    for p in ps:
        vals = []
        for a, l in zip(ascore, labels):
            g = int(l.sum())
            if g == 0:
                continue
            k = max(1, round(p * g / 100))
            vals.append(ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k=k))
        out[f"NDCG@{p}%"] = float(np.mean(vals)) if vals else float("nan")
    return out

