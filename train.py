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
