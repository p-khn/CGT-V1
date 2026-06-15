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
