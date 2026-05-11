import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import gaussian_nll

def create_optimizer(model, cfg):
    return optim.Adam(model.parameters(), lr=cfg.lr)