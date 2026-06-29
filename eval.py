import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from attribution import (
    _build_feature_window,
    get_ground_truth_dims,
    hit_att,
    ndcg_att,
    parse_external_labels,
    score_with_matrix_shadow,
)
from data import make_loader
from model import gaussian_nll
from spot_methods import SPOT
from utils import _aggregate, adjust_predicts, apply_stress, calc_point2point


@torch.no_grad()
def score_param(model, loader, cfg, device, gamma, mc_samples=None, stress_mode=None):
    from collections import defaultdict

    if mc_samples is None:
        mc_samples = cfg.mc_samples

    s = defaultdict(list)
    model.eval()

    for x, y, var, ts in loader:
        x = x.to(device)
        y = y.to(device)
        blk = model.blocks[str(var)]
        x = apply_stress(x, blk, stress_mode)

        (mu_c, logv_c), (mu_o, logv_o), *_ = model(
            var,
            x,
            y=None,
            mc_samples=mc_samples,
            use_prior_at_eval=True,
        )

        def nll_avg(mu, logv):
            nll = gaussian_nll(mu, logv, y)
            return nll.mean(dim=-1) if nll.dim() == 2 else nll

        vals = (1 - gamma) * nll_avg(mu_c, logv_c) + gamma * nll_avg(mu_o, logv_o)
        vals = vals.detach().cpu().numpy()
        for t, v in zip(ts, vals):
            s[t].append(float(v))

    if not s:
        return np.zeros(0, dtype=np.float32)

    out = np.zeros(max(s) + 1, dtype=np.float32)
    for t in range(out.shape[0]):
        out[t] = _aggregate(s[t], cfg.aggregation, cfg.topk) if t in s else 0.0
    return out


@torch.no_grad()
def score(model, loader, cfg, device, active_gamma, mc_samples=None, gamma=None):
    g = active_gamma if gamma is None else float(gamma)
    return score_param(model, loader, cfg, device, gamma=g, mc_samples=mc_samples, stress_mode=None)


@torch.no_grad()
def average_gate_margin(model):
    margins = []
    for _, blk in model.blocks.items():
        a = torch.sigmoid(blk.logits).detach().cpu()
        if blk.idx_causal.any() and blk.idx_other.any():
            margins.append(float(a[blk.idx_causal].mean() - a[blk.idx_other].mean()))
    return float(np.mean(margins)) if len(margins) else 0.0


@torch.no_grad()
def safe_scores(model, train_loader, test_loader, cfg, device, base_gamma):
    s_mix_test = score_param(model, test_loader, cfg, device, gamma=base_gamma, mc_samples=cfg.mc_samples, stress_mode=None)
    s_perm_np_test = score_param(
        model,
        test_loader,
        cfg,
        device,
        gamma=base_gamma,
        mc_samples=cfg.mc_samples,
        stress_mode="permute_nonparents",
    )
    _ = score_param(model, test_loader, cfg, device, gamma=0.0, mc_samples=cfg.mc_samples, stress_mode=None)

    eps = 1e-6
    R = float(np.mean(np.abs(s_mix_test - s_perm_np_test)) / (np.mean(np.abs(s_mix_test)) + eps))
    M = average_gate_margin(model)

    gamma_used = base_gamma
    fallback = False
    if (R > cfg.safe_tau_rel) or (M < cfg.safe_tau_alpha):
        if cfg.safe_use_soft_gamma:
            scale = max(0.0, min(1.0, (M - cfg.safe_tau_alpha) / max(1e-6, 1.0 - cfg.safe_tau_alpha)))
            gamma_used = base_gamma * scale
        else:
            gamma_used = 0.0
            fallback = True

    train_scores = score_param(model, train_loader, cfg, device, gamma=gamma_used, mc_samples=cfg.mc_samples, stress_mode=None)
    test_scores = score_param(model, test_loader, cfg, device, gamma=gamma_used, mc_samples=cfg.mc_samples, stress_mode=None)

    decision = {
        "R_rel": R,
        "M_margin": M,
        "gamma_base": float(base_gamma),
        "gamma_used": float(gamma_used),
        "fallback": bool(fallback),
    }
    logging.info(f"[SAFE] R_rel={R:.4f}  M={M:.4f}  gamma_used={gamma_used:.4f}  fallback={fallback}")
    return train_scores, test_scores, decision


def compute_spot_split_idx(test_scores, cfg):
    offset = cfg.tau_max + cfg.window
    scores_full = test_scores[offset:]
    n_full = len(scores_full)
    init_len = min(max(cfg.burn_min, int(cfg.burn_frac * n_full)), max(100, n_full // 2))
    split_idx = offset + init_len
    return offset, init_len, split_idx


def run_shadow_eval_and_spot(model, data_bundle, cfg, device):
    active_gamma = cfg.gamma

    train_dl_eval = make_loader(data_bundle.train_ds, cfg.batch_size)
    test_dl_eval = make_loader(data_bundle.test_ds, cfg.batch_size)

    train_scores, test_scores, decision = safe_scores(
        model,
        train_dl_eval,
        test_dl_eval,
        cfg,
        device,
        base_gamma=cfg.gamma,
    )
    active_gamma = decision["gamma_used"]
    print(f"[SAFE] decision: {decision}")

    shift = (-float(train_scores.min()) + 1e-6) if len(train_scores) and float(train_scores.min()) < 0 else 0.0
    train_scores_spot = train_scores + shift
    test_scores_spot = test_scores + shift

    labels_test = pd.read_csv(cfg.dataset.label_csv)["label"].values.astype(bool)
    offset = cfg.tau_max + cfg.window
    assert len(test_scores) == len(labels_test), f"Label length {len(labels_test)} != score length {len(test_scores)}"

    scores_full = test_scores_spot[offset:]
    labels_full = labels_test[offset:]

    n = len(scores_full)
    init_len = min(max(cfg.burn_min, int(cfg.burn_frac * n)), max(100, n // 2))
    init_score = scores_full[:init_len]
    run_score = scores_full[init_len:]
    run_labels = labels_full[init_len:]
    if len(run_score) == 0:
        raise ValueError("Not enough test points after burn-in to evaluate.")

    s = SPOT(cfg.q_spot)
    s.fit(init_score, run_score)
    s.initialize(level=cfg.level_spot, min_extrema=False, verbose=False)
    ret = s.run(dynamic=True)
    thr_seq = np.asarray(ret["thresholds"]).reshape(-1) * cfg.lm_spot[1]

    pred_dyn, latency_dyn = adjust_predicts(run_score, run_labels, thr_seq, calc_latency=True)
    f1_d, p_d, r_d, TP_d, FP_d, FN_d, auc_d = calc_point2point(pred_dyn, run_labels)

    try:
        auc_raw = roc_auc_score(labels_full.astype(int), scores_full)
        ap_raw = average_precision_score(labels_full.astype(int), scores_full)
        print(f"[Sanity] Raw-score ROC AUC (post-window slice): {auc_raw:.3f}")
        print(f"[Sanity] Raw-score PR  AUC (post-window slice): {ap_raw:.3f}")
    except Exception as e:
        print(f"[Sanity] AUC failed: {e}")

    print(
        f"[SPOT-burnin] burn_in={init_len}  F1={f1_d:.3f}  "
        f"P/R={p_d:.3f}/{r_d:.3f}  AUC={auc_d:.3f}  latency={latency_dyn:.2f}"
    )

    np.save("train_scores.npy", train_scores)
    np.save("test_scores.npy", test_scores)
    with open("shadow_decision.txt", "w") as f:
        f.write(str(decision))
    print("[Saved] train_scores.npy, test_scores.npy, shadow_decision.txt")

    T_test = data_bundle.test_arr.shape[0]
    _, test_dim_mat = score_with_matrix_shadow(
        model,
        test_dl_eval,
        T_len=T_test,
        D=data_bundle.D,
        gamma=active_gamma,
        cfg=cfg,
        device=device,
    )

    offset, init_len_check, split_idx = compute_spot_split_idx(test_scores, cfg)
    baseline_slice = test_dim_mat[max(0, offset):split_idx]

    if baseline_slice.size == 0:
        baseline_mean = test_dim_mat.mean(axis=0)
        baseline_std = test_dim_mat.std(axis=0) + 1e-6
    else:
        baseline_mean = baseline_slice.mean(axis=0)
        baseline_std = baseline_slice.std(axis=0) + 1e-6

    dim_scores_z = (test_dim_mat - baseline_mean) / baseline_std

    CF_MC = 1
    ALPHA_EPS = 1e-3
    USE_SEGMENT_REPS = True

    def prepare_cf_structures(model_):
        cols_by_sensor = {s: [p for p, (j, lag) in enumerate(data_bundle.edge_list) if j == s] for s in range(data_bundle.D)}
        affected_vars = {s: [] for s in range(data_bundle.D)}
        for i in range(data_bundle.D):
            blk = model_.blocks[str(i)]
            alpha = torch.sigmoid(blk.logits).detach().cpu().numpy()
            causal = blk.idx_causal.detach().cpu().numpy()
            influ_mask = (alpha > ALPHA_EPS) | causal
            for s_idx in range(data_bundle.D):
                if np.any(influ_mask[cols_by_sensor[s_idx]]):
                    affected_vars[s_idx].append(i)
        return cols_by_sensor, affected_vars

    @torch.no_grad()
    def build_window_for_t(t: int):
        x = torch.from_numpy(
            _build_feature_window(data_bundle.test_arr, int(t), data_bundle.D, cfg.tau_max, cfg.window)
        ).unsqueeze(0).to(device)
        return x

    @torch.no_grad()
    def clamp_sensor_in_window_(xwin: torch.Tensor, s_idx: int, cols_by_sensor):
        cols = cols_by_sensor[s_idx]
        xwin[:, :, cols] = float(data_bundle.x_meds[s_idx])

    @torch.no_grad()
    def score_vars_at_t_from_window(xwin, t, vars_idx, gamma, mc=CF_MC):
        vals = np.zeros(len(vars_idx), dtype=np.float32)
        use_amp = device.type == "cuda"
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
            for k, i in enumerate(vars_idx):
                y = torch.tensor(data_bundle.test_arr[t, i], dtype=torch.float32, device=device).unsqueeze(0)
                (mu_c, lv_c), (mu_o, lv_o), *_ = model(
                    i,
                    xwin,
                    y=None,
                    mc_samples=mc,
                    use_prior_at_eval=True,
                )
                nllc = gaussian_nll(mu_c, lv_c, y)
                nllo = gaussian_nll(mu_o, lv_o, y)
                v = (
                    (1 - gamma) * (nllc.mean(dim=-1) if nllc.dim() == 2 else nllc)
                    + gamma * (nllo.mean(dim=-1) if nllo.dim() == 2 else nllo)
                )
                vals[k] = float(v)
        return vals

    @torch.no_grad()
    def fast_cf_deltas_for_t(t: int, gamma: float, cols_by_sensor, affected_vars):
        x0 = build_window_for_t(t)
        base_vals = score_vars_at_t_from_window(x0, t, range(data_bundle.D), gamma, mc=CF_MC)
        base_agg = _aggregate(base_vals, cfg.aggregation, cfg.topk)
        deltas = np.zeros(data_bundle.D, dtype=np.float32)

        for s_idx in range(data_bundle.D):
            vars_s = affected_vars[s_idx]
            if not vars_s:
                continue
            x = x0.clone()
            clamp_sensor_in_window_(x, s_idx, cols_by_sensor)
            new_vals_s = score_vars_at_t_from_window(x, t, vars_s, gamma, mc=CF_MC)
            updated = base_vals.copy()
            updated[vars_s] = new_vals_s
            deltas[s_idx] = _aggregate(updated, cfg.aggregation, cfg.topk) - base_agg
        return deltas

    def segment_reprs(pred_bool: np.ndarray):
        idx = np.where(pred_bool)[0]
        if not len(idx):
            return np.array([], dtype=int)
        reps = []
        start = idx[0]
        prev = start
        for t in idx[1:]:
            if t != prev + 1:
                reps.append((start + prev) // 2)
                start = t
            prev = t
        reps.append((start + prev) // 2)
        return np.array(reps, dtype=int)

    cols_by_sensor, affected_vars = prepare_cf_structures(model)

    predictions_full = np.zeros_like(test_scores, dtype=bool)
    predictions_full[split_idx:split_idx + len(pred_dyn)] = pred_dyn

    if USE_SEGMENT_REPS:
        ts_to_explain = segment_reprs(predictions_full)
    else:
        ts_to_explain = np.where(predictions_full)[0]
    ts_to_explain = [t for t in ts_to_explain if t >= (cfg.tau_max + cfg.window)]

    external_labels = parse_external_labels(
        cfg.dataset.ext_labels_txt,
        data_bundle.D,
        one_based=cfg.dataset.external_labels_one_based,
    )

    ascore_rows_cf, ascore_rows_z, label_rows = [], [], []
    for t in ts_to_explain:
        t_int = int(t)
        deltas = fast_cf_deltas_for_t(t_int, active_gamma, cols_by_sensor, affected_vars)
        ascore_rows_cf.append(-deltas)
        ascore_rows_z.append(dim_scores_z[t_int])

        gt_vec = np.zeros(data_bundle.D, dtype=int)
        for d in get_ground_truth_dims(t_int, external_labels):
            if 0 <= d < data_bundle.D:
                gt_vec[d] = 1
        label_rows.append(gt_vec)

    if label_rows:
        labels_mat = np.stack(label_rows)

        ascore_cf = np.stack(ascore_rows_cf)
        print("\n=== Dimension-level attribution (FAST counterfactual Δ) ===")
        print(hit_att(ascore_cf, labels_mat, ps=(100, 150)))
        print(ndcg_att(ascore_cf, labels_mat, ps=(100, 150)))

        ascore_z = np.stack(ascore_rows_z)
        print("\n=== Dimension-level attribution (blended NLL z-score) ===")
        print(hit_att(ascore_z, labels_mat, ps=(100, 150)))
        print(ndcg_att(ascore_z, labels_mat, ps=(100, 150)))
    else:
        print("\n[Attribution] No candidate timesteps to explain.")

    return train_scores, test_scores, decision, active_gamma