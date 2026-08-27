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

