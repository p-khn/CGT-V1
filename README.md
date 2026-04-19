# Notice
**This repository will be updated soon.**

# Causally-Constrained Probabilistic Forecasting for Time-Series Anomaly Detection

This repository contains the codebase accompanying the paper **"Causally-Constrained Probabilistic Forecasting for Time-Series Anomaly Detection"**.

The implementation combines causal prior information with probabilistic forecasting for multivariate time-series anomaly detection. At a high level, the code:
- builds windowed multivariate forecasting samples for each target variable,
- injects a precomputed causal graph as a structural prior,
- trains a two-head causal transformer forecaster,
- computes anomaly scores from predictive negative log-likelihood,
- applies SPOT-based thresholding for detection,
- and evaluates dimension-level attribution using external annotations and counterfactual analysis.

---

## Repository status
This repository is an active research codebase and **will be updated soon** with additional cleanup, documentation, and dataset-specific instructions.

## File structure

```text
.
├── main.py            # Entry point
├── config.py          # Dataset definitions and global hyperparameters
├── data.py            # Data loading, scaling, windowing, data loaders
├── model.py           # Two-head causal transformer model
├── train.py           # Training loop and checkpoint saving
├── eval.py            # Scoring, SPOT evaluation, attribution pipeline
├── attribution.py     # External-label parsing and counterfactual attribution
└── utils.py           # Aggregation, stress tests, prediction adjustment, metrics
