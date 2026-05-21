import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import gaussian_nll

def create_optimizer(model, cfg):
    return optim.Adam(model.parameters(), lr=cfg.lr)

def train_loop(model, opt, data_bundle, cfg, device):
    model.train(True)
    
    for ep in range(cfg.epochs):
        tot_loss, n_seen = 0.0, 0
        gamma_t = min(cfg.gamma, (ep + 1) / max(1, cfg.warmup_eps) * cfg.gamma)
        beta_t = cfg.beta_kl * min(1.0, (ep + 1) / max(1, cfg.warmup_eps))
        for x, y, var, ts in data_bundle.train_dl:
            x = x.to(device)
            y = y.to(device)
            
            (mu_c, logv_c), (mu_o, logv_o), alpha, grp, dmu, dlogv, kl_z = model(
                var,
                x,
                y=y,
                mc_samples=1,
                use_prior_at_eval=False,
            )
            blk = model.blocks[str(var)]

            nll_c = gaussian_nll(mu_c, logv_c, y).mean()

            mu_o_loss = mu_c.detach() + dmu
            logv_o_loss = logv_c.detach() + dlogv
            nll_o = gaussian_nll(mu_o_loss, logv_o_loss, y).mean()

            res_pen = dmu.pow(2).mean() + dlogv.pow(2).mean()
