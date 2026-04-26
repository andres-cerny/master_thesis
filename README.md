# Anomaly Detection in Water Consumption Time Series

**Master's Thesis — Czech Technical University in Prague, Faculty of Information Technology**

| | |
|---|---|
| **Author** | Bc. Ondřej Černý |
| **Supervisor** | doc. Ing. Kamil Dedecius, Ph.D. |
| **Programme** | Informatics — Knowledge Engineering |
| **Department** | Department of Applied Mathematics |
| **Year** | 2026 |

---

## Overview

This repository contains all code and source materials for the master's thesis *Anomaly Detection in Water Consumption Time Series*. The work addresses anomaly detection in a large-scale industrial water metering dataset provided by [Softlink s.r.o.](https://www.softlink.cz/), comprising over **618 million readings** from more than **27,000 sensors** across industrial, municipal, and commercial sites in the Czech Republic.

Two families of models are developed and compared:

- **Unobserved Components (UC) state-space models** — Implemented via `statsmodels`, fitted by maximum likelihood estimation, with anomaly detection driven by one-step-ahead Kalman-filter residuals.
- **GRU neural network** — A Gated Recurrent Unit operating on irregularly sampled data using delta-time and calendar (hour-of-day, day-of-week) harmonics as inputs; anomalies are detected from one-step-ahead prediction residuals.

Residuals are thresholded using either a classical **z-score** or a robust **MAD (Median Absolute Deviation)** score, computed from a held-out reference segment.

---

## Repository Structure

```
diplomka/
├── src/
│   ├── data_examination/                # Dataset preprocessing, exploration, metadata
│   │   ├── data_exploration.ipynb
│   │   └── helper_scripts/
│   │       ├── split_and_concat_dfs.py
│   │       ├── split_and_save_df.py
│   │       ├── convert_timestamp_utc.py
│   │       ├── create_diff.py
│   │       ├── create_metadata.py
│   │       └── period_resample_examination.ipynb
│   │
│   └── models_implementation_testing/   # Model implementations and experiments
│       ├── compare_models_final.ipynb
│       ├── state_space_models/          # UC model (statsmodels)
│       │   ├── local_level_unobserved_components.py
│       │   ├── local_level_unobserved_components_next_one_step_pred.py
│       │   ├── resample.py
│       │   ├── compare_models.ipynb
│       │   └── results/                 # Output CSVs from runs
│       ├── gru_predictor/               # GRU model (PyTorch)
│       │   ├── run_gru.py
│       │   ├── run_gru_one_step_predict.py
│       │   └── results/                 # Output CSVs from runs
│       ├── helper_scripts/              # Shared utilities
│       │   ├── calculate_metrics.py
│       │   ├── create_anomalies.py
│       │   ├── train_test_split.py
│       │   └── visulize_anomaly_prediction.ipynb
│       └── other_experiments/           # Earlier / discarded approaches
│           ├── kf/                      # Custom Kalman filter (WRLS-based)
│           ├── gru_autoencoder/         # GRU autoencoder variant
│           ├── isolation_forrest/       # Isolation Forest baseline
│           ├── compare_A_estimates.ipynb
│           ├── compare_A_fourier_seasonal.py
│           └── compare_gru_dynamic_trend.ipynb
│
└── thesis_text/                         # LaTeX source for the written thesis
    ├── ctufit-thesis.tex
    ├── ctufit-thesis.cls
    ├── changelog.md
    ├── LICENSE
    ├── .gitlab-ci.yml
    └── text/
        ├── text.tex
        ├── appendix.tex
        ├── medium.tex
        └── bib-database.bib
```

---

## End-to-End Pipeline

The following is the rough order in which the code is intended to be run:

1. **Raw data preparation** (`src/data_examination/helper_scripts/`)  
   Split monthly delivery files by sensor `id` (`split_and_concat_dfs.py`), add UTC timestamps with DST handling (`convert_timestamp_utc.py`), compute consumption increments (`create_diff.py`), and generate per-sensor metadata JSONs (`create_metadata.py`).

2. **Exploratory analysis** (`src/data_examination/data_exploration.ipynb`)  
   Inspection of sensor counts, sampling periodicity, gap distribution, and missing-value patterns.

3. **Train/test split** (`src/models_implementation_testing/helper_scripts/train_test_split.py`)  
   Random 80/20 sensor-level split, persisted as pickles for reproducibility.

4. **Model runs**
   - `state_space_models/local_level_unobserved_components.py` — runs all four UC variants (local level, +daily, +daily+weekly time-domain, +daily+weekly trigonometric) in batch with per-file process timeouts.
   - `gru_predictor/run_gru.py` — runs the GRU predictor on the same train/test split using the same sliding-window protocol so results are directly comparable.

5. **Comparison and visualisation**  
   `compare_models_final.ipynb` aggregates result CSVs from `state_space_models/results/` and `gru_predictor/results/` and produces the tables and plots used in the thesis.

---

## Requirements

The code is Python 3.10+. Main dependencies:

```
numpy
pandas
scipy
scikit-learn
statsmodels
torch
matplotlib
seaborn
tqdm
upsetplot
pytz
```

The dataset itself is **not** included in this repository — it is the property of Softlink s.r.o. Paths in the scripts assume a local `data_w_diff_001/` folder of per-sensor CSVs and a `metadata_001/` folder of per-sensor JSON metadata.

---

## Sub-directory READMEs

- [`thesis_text/README.md`](thesis_text/README.md) — Building the LaTeX thesis.
- [`src/data_examination/README.md`](src/data_examination/README.md) — Dataset preprocessing pipeline.
- [`src/models_implementation_testing/README.md`](src/models_implementation_testing/README.md) — UC and GRU implementations evaluated in the thesis.
- [`src/models_implementation_testing/other_experiments/README.md`](src/models_implementation_testing/other_experiments/README.md) — Earlier approaches that were ultimately not used as the main models.

---

## Contact

**Ondřej Černý** — cernyo14@fit.cvut.cz  
Supervisor: **doc. Ing. Kamil Dedecius, Ph.D.**
