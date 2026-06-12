import math
from collections import defaultdict

import torch
import torch.nn as nn


def gaussian_nll(mu, logv, y):
    """Gaussian negative log likelihood, supporting optional MC dimension."""
    if mu.dim() == 2:
        y = y.unsqueeze(1).expand_as(mu)
    var = torch.exp(logv)
    return 0.5 * (math.log(2 * math.pi) + logv) + (y - mu) ** 2 / (2 * var)
