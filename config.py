import logging
import random
import warnings
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    machine: str
    train_csv: str
    test_csv: str
    label_csv: str
    ext_labels_txt: str
    pcmci_pkl: str
    external_labels_one_based: bool = True


DATASET_CONFIGS = {
    "ASD": DatasetConfig(
        name="ASD",
        machine="",
        train_csv="path/to/data",
        test_csv="path/to/data",
        label_csv="path/to/data",
        ext_labels_txt="path/to/data",
        pcmci_pkl="path/to/data",
        external_labels_one_based=True,
    ),
    "SMD": DatasetConfig(
        name="SMD",
        machine="",
        train_csv="path/to/data",
        test_csv="path/to/data",
        label_csv="path/to/data",
        ext_labels_txt="path/to/data",
        pcmci_pkl="path/to/data",
        external_labels_one_based=True,
    ),
}


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    run_train: bool = True
    run_eval: bool = True
    seed: int = 42

    window: int = 30
    tau_max: int = 7
    d_model: int = 64
    n_head: int = 2
    d_ff: int = 128
    n_layers: int = 2
    epochs: int = 1
    batch_size: int = 16
    lr: float = 0.0003898458749305196
    grad_clip: float = 0.822586395204725

    lambda_grp: float = 5.940525755701932e-06
    lambda_prior: float = 0.008431736905057536
    lambda_causal_l1: float = 0.0030943771470831397
    lambda_other_l1: float = 0.03724146189492819
    warmup_eps: int = 7
    lambda_resid: float = 0.001096569987677643

    gamma: float = 0.020628141621898986

    q_spot: float = 0.0015348567636032401
    level_spot: float = 0.047122383283896056
    lm_spot: tuple = (0.98, 1.0)

    aggregation: str = "mean"
    topk: int = 3

    burn_frac: float = 0.10238143619189914
    burn_min: int = 100

    z_dim: int = 8
    beta_kl: float = 0.24541875150782977
    mc_samples: int = 4
    use_prior_at_eval: bool = False

    safe_tau_rel: float = 0.08627784363506777
    safe_tau_alpha: float = 0.03651719374641294
    safe_use_soft_gamma: bool = True


ACTIVE_DATASET = "ASD"


def get_config(active_dataset: str = ACTIVE_DATASET) -> AppConfig:
    if active_dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{active_dataset}'. Available: {list(DATASET_CONFIGS)}")
    return AppConfig(dataset=DATASET_CONFIGS[active_dataset])


def setup_environment(seed: int):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warnings.filterwarnings("ignore")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")