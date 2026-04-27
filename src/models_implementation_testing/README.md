# `src/models_implementation_testing/` — Models Used in the Thesis

This directory implements and evaluates the two model families compared in the thesis: **Unobserved Components (UC) state-space models** and a **GRU neural network**, both used in a one-step-ahead-prediction-residual paradigm for unsupervised anomaly detection.

Earlier or unused approaches live in `other_experiments/` and are documented separately in [`other_experiments/README.md`](other_experiments/README.md).

---

## Contents

```
models_implementation_testing/
├── compare_models_final.ipynb          # Aggregates all results, produces thesis figures and tables
│
├── state_space_models/                 # UC family (statsmodels)
│   ├── local_level_unobserved_components.py            # Batch runner — all 4 UC variants
│   ├── local_level_unobserved_components_next_one_step_pred.py # Used for injected anomaly testing - does not do batch predict like variant above 
│   ├── resample.py                                     # Adaptive gap-aware resampling
│   ├── compare_models.ipynb
│   └── results/                                        # Output CSVs from runs
│
├── gru_predictor/                      # GRU family (PyTorch)
│   ├── run_gru.py                                      # Batch runner — sliding-window GRU
│   ├── run_gru_one_step_predict.py                     # Used for injected anomaly testing otherwise same structure as variant above
│   └── results/                                        # Output CSVs from runs
│
├── helper_scripts/                     # Shared utilities used by both families
│   ├── calculate_metrics.py                            # Regression + anomaly classification metrics
│   ├── create_anomalies.py                             # Synthetic spike injection on Diff
│   ├── train_test_split.py                             # 80/20 sensor-level split → pickles
│   └── visulize_anomaly_prediction.ipynb               # Per-sensor visualisation of predictions
│
└── other_experiments/                  # See own README
```

---

## Common Evaluation Protocol

Both families use the **same 6-week sliding-window protocol** so their results are directly comparable:

| Segment | Weeks | Purpose |
|---|---|---|
| **Train 1** | 1–4 | Initial parameter fit |
| **Train 2** | 2–5 | Warm-start refit; residuals here are used to compute thresholding statistics |
| **Predict** | 6 | One-step-ahead predictions; residuals here are scored for anomalies |

The window start is randomly sampled per sensor with `numpy.random.default_rng(seed = int(filename))` for reproducibility. Parallel execution uses `ProcessPoolExecutor` with a per-file timeout so that pathological sensors do not block the rest of the batch.

---

## State-Space Models — `state_space_models/`

Built on top of `statsmodels.tsa.statespace.structural.UnobservedComponents`. Four variants are evaluated, all with a stochastic local level (no slope) and differing in the seasonal component:

| Variant | Daily seasonal | Weekly seasonal | Representation |
|---|:---:|:---:|---|
| **Local level only** | — | — | — |
| **+ Daily** | ✅ | — | Time-domain dummies |
| **+ Daily + Weekly** | ✅ | ✅ | Time-domain dummies |
| **+ Daily + Weekly (frequency)** | ✅ | ✅ | Trigonometric / Fourier |

Hyperparameters are estimated by maximum likelihood (BFGS / L-BFGS) inside `statsmodels`, with a per-fit timeout (default 120 s) to bound runtime.

### Resampling — `resample.py`

State-space models in `statsmodels` require equally spaced observations. `fill_gaps_with_periodicity_adaptive` walks the time axis using the most-common inter-reading interval as the step, attaches each real reading to its closest expected slot within a 10% tolerance, and inserts NaN-valued slots when no real reading falls within tolerance. Sensors with sub-3-minute periodicities are upsampled (×5, ×3, ×2 depending on the cadence) to keep grid sizes manageable. NaN actuals propagate naturally through the Kalman filter (no update step) and are masked out of metric computations.

### Run

```bash
cd src/models_implementation_testing/state_space_models/
python local_level_unobserved_components.py --workers 7 --verbose
```

Outputs land in `state_space_models/results/` — one CSV per run, with run parameters baked into the filename (sensor count, seed, daily/weekly/fourier flags, timeout, …).

---

## GRU Predictor — `gru_predictor/`

A standard GRU regressor implemented in PyTorch. Designed to operate **directly on irregular data** without resampling.

### Input features (6 per step)

- Normalised meter reading
- Normalised Δt (time since previous reading) — this is what lets the model handle irregular sampling
- `sin(2π · hour/24)`, `cos(2π · hour/24)` — time-of-day harmonics
- `sin(2π · dow/7)`, `cos(2π · dow/7)` — day-of-week harmonics

### Architecture

`GRUNet`: GRU layer(s) → linear head, predicting one step ahead. Defaults: `hidden_size=16`, `num_layers=1`, `dropout=0.1`, optimised with Adam (MSE loss) on a sliding window. The window length matches one seasonal period (one day at the sensor's periodicity).

### Run

```bash
cd src/models_implementation_testing/gru_predictor/
python run_gru.py --workers 7 --verbose
```

Outputs land in `gru_predictor/results/` with the same naming convention as the UC results.

---

## Anomaly Scoring (shared between both families)

Implemented inside the run scripts as `compute_z_scores(residuals, ref_residuals)`:

- **z-score**: `(r - mean(ref)) / std(ref)` — classical, sensitive to extreme values in the reference.
- **MAD-based score**: `0.6745 · (r - median(ref)) / MAD(ref)` — robust to outliers in the reference.

The reference (`ref_residuals`) is **the residuals of the second training segment**, not the prediction segment. A residual in the prediction segment is flagged as an anomaly if its score exceeds a configurable threshold (default 3 σ / MAD-units). Both flags are computed and stored side-by-side, allowing the comparison study in Chapter 7.

---

## Helper Scripts — `helper_scripts/`

### `calculate_metrics.py`

Computes both regression and anomaly-detection metrics from a results dataframe with columns `actual`, `predicted`, `residual`, `is_anomaly_actual`, `is_anomaly_predicted`, `is_anomaly_robust_predicted`. Returns RMSE, MAE, R², MAPE, WAPE, normalised RMSE, median APE, prediction bias, MPE, direction accuracy, MedAE, percentile absolute errors (75th, 90th), and TP/TN/FP/FN + accuracy/precision/recall/F1 for both the z-score and MAD flags. NaNs are masked out before computation.

### `create_anomalies.py`

`inject_spike_anomalies_diff` adds 1–10 synthetic spike anomalies to the `Diff` column at random positions. Spike magnitudes are drawn from `Uniform(3·σ, 10·σ)` where σ is the standard deviation of the (clean) `Diff` column; if σ ≈ 0 the script falls back to the median of positive diffs to avoid trivial spikes. The original values are preserved in `Diff_original`, and an `is_anomaly` column stores the injected delta (zero for non-anomalous rows).

### `train_test_split.py`

A small one-shot script that reads all CSVs in `src/data/sensor_data`, shuffles them with `random.seed(42)`, splits 80/20 by sensor, and pickles the two file lists to `pickles/train_set.pkl` and `pickles/test_set.pkl`. The same pickles are loaded by the UC and GRU run scripts so the split is stable across runs.

### `visulize_anomaly_prediction.ipynb`

Interactive per-sensor visualisation of `actual` vs `predicted`, residuals, and the predicted/actual anomaly flags from any results CSV.

---

## Final Comparison — `compare_models_final.ipynb`

Aggregates the result CSVs of all four UC variants and the GRU model from their respective `results/` folders, computes summary tables (mean/median across the test sensor subset of every metric in `calculate_metrics.py`), and produces the comparison figures used in Chapter 7. This is the notebook to run if you want to reproduce the headline results table from the thesis.

---

## Reproducing the Headline Results

```bash
# from the project root
python src/models_implementation_testing/helper_scripts/train_test_split.py

cd src/models_implementation_testing/state_space_models
python local_level_unobserved_components.py --workers 7

cd ../gru_predictor
python run_gru.py --workers 7

# then open compare_models_final.ipynb and run all cells
```

Adjust `--workers` to your machine. All runs are CPU-only — no GPU is required.
