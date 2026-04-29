from config import get_config, setup_environment
from data import load_data_bundle
from eval import run_shadow_eval_and_spot
from model import TwoHeadForecaster
from train import create_optimizer, train_loop


# switch datasets.
ACTIVE_DATASET = "ASD"  # or "SMD"


def main():
    cfg = get_config(ACTIVE_DATASET)
    device = setup_environment(cfg.seed)

    data_bundle = load_data_bundle(cfg)
    model = TwoHeadForecaster(cfg, data_bundle).to(device)
    opt = create_optimizer(model, cfg)

    if cfg.run_train:
        train_loop(model, opt, data_bundle, cfg, device)
    if cfg.run_eval:
        run_shadow_eval_and_spot(model, data_bundle, cfg, device)


if __name__ == "__main__":
    main()