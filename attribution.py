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

@torch.no_grad()
def score_with_matrix_shadow(model, loader, T_len, D, gamma, cfg, device):
    from collections import defaultdict

    s = defaultdict(list)
    dim_mat = np.zeros((T_len, D), dtype=np.float32)
    model.eval()

    for x, y, var, ts in loader:
        x = x.to(device)
        y = y.to(device)
        (mu_c, logv_c), (mu_o, logv_o), *_ = model(
            var,
            x,
            y=None,
            mc_samples=cfg.mc_samples,
            use_prior_at_eval=True,
        )

        def _nll(mu, logv):
            nll = gaussian_nll(mu, logv, y)
            return nll.mean(dim=-1) if nll.dim() == 2 else nll

        nll_c = _nll(mu_c, logv_c).detach().cpu().numpy()
        nll_o = _nll(mu_o, logv_o).detach().cpu().numpy()
        blend = (1 - gamma) * nll_c + gamma * nll_o

        for idx, t in enumerate(ts):
            t = int(t)
            dim_mat[t, int(var)] = float(blend[idx])
            s[t].append(float(blend[idx]))

    out = np.zeros(T_len, dtype=np.float32)
    for t in range(out.shape[0]):
        out[t] = _aggregate(s[t], cfg.aggregation, cfg.topk) if t in s else 0.0
    return out, dim_mat

@torch.no_grad()
def _build_feature_window(arr, t, D, tau_max, window):
    cols = [
        arr[t - window - lag : t - lag, j]
        for j in range(D)
        for lag in range(1, tau_max + 1)
    ]
    return np.stack(cols, axis=1).astype(np.float32)


@torch.no_grad()
def score_single_t(model, arr, t, gamma, data_bundle, cfg, device):
    assert cfg.tau_max + cfg.window <= t < arr.shape[0], "t outside valid windowed range"
    model.eval()
    x = torch.from_numpy(
        _build_feature_window(arr, t, data_bundle.D, cfg.tau_max, cfg.window)
    ).unsqueeze(0).to(device)

    vals = []
    with torch.no_grad():
        for var in range(data_bundle.D):
            y = torch.tensor(arr[t, var], dtype=torch.float32, device=device).unsqueeze(0)
            (mu_c, logv_c), (mu_o, logv_o), *_ = model(
                var,
                x,
                y=None,
                mc_samples=cfg.mc_samples,
                use_prior_at_eval=True,
            )
            nll_c = gaussian_nll(mu_c, logv_c, y)
            nll_o = gaussian_nll(mu_o, logv_o, y)
            nllc = nll_c.mean(dim=-1).item() if nll_c.dim() == 2 else nll_c.item()
            nllo = nll_o.mean(dim=-1).item() if nll_o.dim() == 2 else nll_o.item()
            vals.append((1 - gamma) * nllc + gamma * nllo)

    return _aggregate(vals, cfg.aggregation, cfg.topk)


def local_cf_delta(model, t, sensors, gamma, data_bundle, cfg, device, also_current=False):
    lo = max(0, t - (cfg.window + cfg.tau_max))
    hi = t
    arr = data_bundle.test_arr.copy()
    for s in sensors:
        arr[lo:hi, s] = data_bundle.x_meds[s]
        if also_current:
            arr[t, s] = data_bundle.x_meds[s]

    new_score = score_single_t(model, arr, t, gamma, data_bundle, cfg, device)
    orig_score = score_single_t(model, data_bundle.test_arr, t, gamma, data_bundle, cfg, device)
    return new_score - orig_score