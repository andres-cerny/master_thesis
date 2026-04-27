# `src/models_implementation_testing/other_experiments/` — Earlier / Discarded Approaches

This directory holds experiments that were tried during development but did **not** end up as one of the five models compared in the thesis (the four UC variants and the GRU predictor). They are kept for transparency.

---

## Contents

```
other_experiments/
├── compare_A_estimates.ipynb           # Compare custom KF parameter estimators on a single sensor
├── compare_A_fourier_seasonal.py       # Multiprocessed batch version of the above
├── compare_gru_dynamic_trend.ipynb     # GRU vs. UC trend extraction comparison
│
├── kf/                                 # Custom Kalman filter implementation (WRLS-based)
│   ├── kf.py                              # WaterMeterKalmanFilter (Fourier seasonal)
│   ├── kf_seasonal_lags.py                # Variant with seasonal lags
│   ├── kf_wrls_estimator.py               # WRLS estimator for A, Q, R (Fourier)
│   ├── kf_wrls_seasonal_lags.py           # WRLS estimator with seasonal lags
│   ├── kf_metrics.py                      # KF-specific evaluation metrics
│   ├── generate_artificial_water_data.py  # Synthetic data generator for KF debugging
│   ├── kalman_filter.ipynb                # Walkthrough notebook
│   ├── interpolation.ipynb                # Pre-KF interpolation experiments
│   └── sensor_test_df_{A,P,Q,R}.npy       # Cached estimator outputs from a sample sensor
│
├── gru_autoencoder/                    # Reconstruction-based GRU (instead of prediction-based)
│   ├── run_gru.py                         # Batch runner
│   └── gru_test.ipynb
│
└── isolation_forrest/                  # Isolation Forest baseline with Fourier features
    └── isolation_forrest.ipynb
```

---

## What's Here and Why It Was Set Aside

### Custom Kalman Filter — `kf/`

A from-scratch Kalman filter implementation tailored to water-meter data, with Fourier seasonal components (`kf.py`) and a seasonal-lag variant (`kf_seasonal_lags.py`). The state transition matrix `A`, process noise `Q`, and observation noise `R` were estimated using a **Weighted Recursive Least Squares (WRLS)** estimator with a forgetting factor (`kf_wrls_estimator.py`, `kf_wrls_seasonal_lags.py`). The notebook `kalman_filter.ipynb` walks through the derivation and a single-sensor demonstration.

This implementation worked but was ultimately superseded by `statsmodels`' `UnobservedComponents` model in the main pipeline (`state_space_models/`), which provides:
- A more standard MLE fitting procedure.
- Better-tested numerical stability.
- Simpler, more directly comparable model variants.

The cached `.npy` files (`sensor_test_df_A.npy`, etc.) contain the `A`, `P`, `Q`, `R` estimates from running the WRLS estimator on a single test sensor and are loaded by the comparison notebook below.

### Estimator Comparison — `compare_A_estimates.ipynb` & `compare_A_fourier_seasonal.py`

Compares the two custom KF estimators (Fourier vs. seasonal-lags) on individual sensors and across the fleet. The `.py` version parallelises the comparison with `multiprocessing.Pool` over a year-or-longer subset of sensors (sensors with less than 365 days of data or > 12-hour periodicity are skipped). The findings here informed the decision to use frequency-domain seasonal terms in the final UC variant.

### GRU Trend Comparison — `compare_gru_dynamic_trend.ipynb`

A side-by-side look at the latent trend extracted by the UC model and the implicit trend learned by the GRU, intended as a sanity-check that both families are capturing similar low-frequency dynamics.

### GRU Autoencoder — `gru_autoencoder/`

A **reconstruction-based** GRU autoencoder, where anomalies are flagged from reconstruction error rather than one-step-ahead prediction error. This was an alternative anomaly-detection paradigm explored during development. It was set aside in favour of the simpler one-step prediction GRU in `gru_predictor/`, which is directly comparable to the UC family (both flag anomalies from the same kind of one-step residuals) and avoids added complexity.

### Isolation Forest — `isolation_forrest/`

`sklearn.ensemble.IsolationForest` applied to features `[Diff, sin_daily, cos_daily, sin_weekly, cos_weekly]` per sensor. A simple non-parametric baseline. Useful as a sanity check, but not a fit for the comparison study because:
- It requires an expected contamination rate, which is unknown in the dataset (no labels).
- It does not produce one-step predictions, so it cannot be evaluated on the same metrics as the UC and GRU models.

---

## Status

Nothing in this directory is part of the final evaluation. These files are kept so that:

1. The decisions to use `statsmodels` UC and a prediction-based GRU are reproducible and traceable.
2. Future extensions of the work can resume from the custom KF or GRU-AE branches without starting from scratch.

