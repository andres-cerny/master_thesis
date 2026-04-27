# `src/data_examination/` — Dataset Preprocessing & Exploration

This directory contains everything needed to turn the raw monthly data dumps from Softlink into the per-sensor CSV format consumed by the models, plus the exploratory analysis used in Chapter 5 of the thesis.

> ⚠️ The dataset itself is **not** included in this repository because of its size (>10GB), but can be found on [Zenodo](https://zenodo.org/records/19735021).

---

## Contents

```
data_examination/
├── data_exploration.ipynb              # Main EDA notebook — produces figures used in the thesis
└── helper_scripts/
    ├── split_and_concat_dfs.py         # Step 1: split monthly dumps by sensor id and concatenate per id
    ├── split_and_save_df.py            # (Subset of the above — only the splitting step)
    ├── convert_timestamp_utc.py        # Step 2: add `timestamp_utc` column with DST handling
    ├── create_diff.py                  # Step 3: sort by timestamp and compute consumption increments (`Diff`)
    ├── create_metadata.py              # Step 4: per-sensor metadata JSONs (statistics, periodicity, ACF)
    └── period_resample_examination.ipynb  # Notebook — investigation of sampling cadences and resampling strategies
```

---

## Preprocessing Pipeline

All scripts are designed to be run from this directory. They produce intermediate folders in the working directory.

### 1. Split & concatenate — `split_and_concat_dfs.py`

The raw delivery (`src/data/original_data`) is a set of monthly CSVs each containing readings from many sensors mixed together. This script:

- Walks the delivery folder.
- Splits each monthly file by sensor `id` into `split_data/{id}-{year}.csv`.
- Concatenates all per-id year shards into a single `data/{id}.csv` per sensor.


### 2. UTC timestamps — `convert_timestamp_utc.py`

Adds a `timestamp_utc` column to each per-sensor CSV. Originals are stored in local Czech time (`Europe/Prague`), which means autumn DST transitions create ambiguous timestamps. The script:

- Localises with `ambiguous='infer'` where possible.
- Falls back to per-row resolution when the bulk operation raises `AmbiguousTimeError`.
- Sets unresolvable rows to `NaT`.
- Uses a `ThreadPoolExecutor` for parallel processing.

### 3. Diffs — `create_diff.py`

Two stages:

- **Sorting** (`sort_dfs_timestamp`): repairs files with non-monotonic timestamps by swapping adjacent rows where appropriate.
- **Diffs** (`calculate_diff_multithreaded`): computes `Diff = hodnota.diff()`. Negative diffs below `-0.002` (likely meter rollovers / faults) are set to NaN; small negative diffs in `(-0.002, 0)` are clamped to 0 to suppress numerical noise.

Output folder: `src/data/sensor_data`.

### 4. Metadata — `create_metadata.py`

For every per-sensor CSV, generates a JSON in `src/data/metadata` containing:

- Record count, time range
- Value statistics: min/max/mean/std of `hodnota` and `Diff`
- Missing-value counts: `nan_value_count`, `nan_diff_count`, `na_timestamp_utc_count`
- **Periodicity**: most common sampling interval (`common_periodicity_seconds`) plus the top-10 modes
- **Gap counts**: number of inter-reading intervals deviating from the mode by more than 5 minutes (`num_gaps_5min`) or by more than 10% (`num_gaps_10percent`), plus their proportions

Uses `multiprocessing.Pool` over CPU cores for speed.

---

## Exploratory Analysis — `data_exploration.ipynb`

The main EDA notebook. Operates on the outputs of the pipeline above (`src/data/sensor_data` + `src/data/metadata`) and produces:

- Total sensor and reading counts.
- Distributions of sampling periodicity across the fleet.
- Missing-value and gap-frequency distributions.
- Investigations of unsorted-timestamp cases and how they are handled.
- Sensor-level value ranges and consumption profiles.
- UpSet plots (via the `upsetplot` package) summarising data-quality criteria across sensors.

The figures included in Chapter 5 of the thesis are exported from this notebook.

---

## Resampling Investigation — `helper_scripts/period_resample_examination.ipynb`

Examines how to deal with irregular sampling. This notebook explores the periodicity statistics across the fleet and the trade-offs between resampling strategies (forward fill, interpolation, gap-aware adaptive resampling), which motivates the `fill_gaps_with_periodicity_adaptive` function used by the UC models in `state_space_models/resample.py`.


