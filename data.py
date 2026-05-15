import pickle
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


class TimeWindowDataset(Dataset):
    """Build samples of shape (WINDOW, P) for each target variable and time."""

    def __init__(self, data: np.ndarray, num_vars: int, tau_max: int, window: int):
        self.data = data
        self.num_vars = num_vars
        self.tau_max = tau_max
        self.window = window

        T = len(data)
        self.index = [
            (i, t)
            for i in range(self.num_vars)
            for t in range(self.tau_max + self.window, T)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, t = self.index[idx]
        cols = [
            self.data[t - self.window - lag : t - lag, j]
            for j in range(self.num_vars)
            for lag in range(1, self.tau_max + 1)
        ]
        x = np.stack(cols, axis=1)
        y = self.data[t, i]
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            i,
            t,
        )


class TargetBatchSampler(Sampler):
    """Group samples by target variable so each batch uses one model block."""

    def __init__(self, ds, bs: int):
        self.bs = int(bs)
        bucket = defaultdict(list)

        if hasattr(ds, "index"):
            for local_idx, (var_i, _t) in enumerate(ds.index):
                bucket[int(var_i)].append(local_idx)
        elif isinstance(ds, Subset) and hasattr(ds.dataset, "index"):
            base = ds.dataset
            for local_idx, base_idx in enumerate(ds.indices):
                var_i, _t = base.index[base_idx]
                bucket[int(var_i)].append(local_idx)
        else:
            for local_idx in range(len(ds)):
                var_i = int(ds[local_idx][2])
                bucket[var_i].append(local_idx)

        self.batches = []
        for idxs in bucket.values():
            random.shuffle(idxs)
            for i in range(0, len(idxs), self.bs):
                self.batches.append(idxs[i : i + self.bs])
        random.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def coll(batch):
    """Collate a batch that shares the same target variable index."""
    var = batch[0][2]
    x = torch.stack([s[0] for s in batch])
    y = torch.stack([s[1] for s in batch])
    ts = [s[3] for s in batch]
    return x, y, var, ts


@dataclass
class DataBundle:
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    scaler: MinMaxScaler
    train_arr: np.ndarray
    test_arr: np.ndarray
    D: int
    P: int
    graph_fdr: object
    pcmci_edges: set
    edge_list: list
    sensor_ids: list
    train_ds: Dataset
    test_ds: Dataset
    train_dl: DataLoader
    test_dl: DataLoader
    x_meds: np.ndarray


def make_loader(ds, batch_size: int):
    return DataLoader(
        ds,
        batch_sampler=TargetBatchSampler(ds, batch_size),
        collate_fn=coll,
    )


def load_data_bundle(cfg) -> DataBundle:
    df_train = pd.read_csv(cfg.dataset.train_csv)
    df_test = pd.read_csv(cfg.dataset.test_csv)

    scaler = MinMaxScaler().fit(df_train.values)
    train_arr = scaler.transform(df_train.values).astype(np.float32)
    test_arr = scaler.transform(df_test.values).astype(np.float32)

    D = df_train.shape[1]
    P = D * cfg.tau_max

    with open(cfg.dataset.pcmci_pkl, "rb") as f:
        graph_fdr = pickle.load(f)["graph_fdr"]

    pcmci_edges = {
        (i, j, lag)
        for i in range(D)
        for j in range(D)
        for lag in range(1, cfg.tau_max + 1)
        if graph_fdr[i, j, lag] == "-->"
    }

    edge_list = [(j, lag) for j in range(D) for lag in range(1, cfg.tau_max + 1)]
    sensor_ids = [j for j in range(D) for _ in range(cfg.tau_max)]

    train_ds = TimeWindowDataset(train_arr, num_vars=D, tau_max=cfg.tau_max, window=cfg.window)
    test_ds = TimeWindowDataset(test_arr, num_vars=D, tau_max=cfg.tau_max, window=cfg.window)

    train_dl = make_loader(train_ds, cfg.batch_size)
    test_dl = make_loader(test_ds, cfg.batch_size)

    return DataBundle(
        df_train=df_train,
        df_test=df_test,
        scaler=scaler,
        train_arr=train_arr,
        test_arr=test_arr,
        D=D,
        P=P,
        graph_fdr=graph_fdr,
        pcmci_edges=pcmci_edges,
        edge_list=edge_list,
        sensor_ids=sensor_ids,
        train_ds=train_ds,
        test_ds=test_ds,
        train_dl=train_dl,
        test_dl=test_dl,
        x_meds=np.median(train_arr, axis=0),
    )