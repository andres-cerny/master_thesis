import os
import random
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def _safe_typical_increment(diff_series, fallback=0.001):
    """
    Compute a robust 'typical increment' from a Diff series.
    Avoids empty-slice warnings by checking for any valid values.
    """
    s = pd.to_numeric(diff_series, errors="coerce")

    # Prefer positive increments (for leaks, spikes up, etc.)
    s_pos = s.clip(lower=0)
    s_pos = s_pos.replace(0, np.nan)

    if s_pos.notna().any():
        val = s_pos.median()
    else:
        val = fallback

    if pd.isna(val) or val == 0:
        val = fallback

    return float(val)


def _try_apply_local_anomaly(df, idx, delta, anomaly_column_name='Diff'):
    """
    Try to apply a local anomaly at idx by adding 'delta' to hodnota.
    If this would break monotonic non-decreasing behavior, do nothing.
    In that case, the point is NOT marked as an outlier.
    """
    cur_val = df.loc[df.index[idx], anomaly_column_name]
    proposed = cur_val + delta

    if idx == 0:
        prev_val = None
    else:
        prev_val = df.loc[df.index[idx - 1], anomaly_column_name]

    # Enforce monotone non-decreasing anomaly_column_name
    if prev_val is not None and proposed < prev_val:
        # Reject anomaly: keep original value, no is_anomaly mark
        return

    # Accept anomaly
    df.loc[df.index[idx], anomaly_column_name] = proposed
    df.loc[df.index[idx], "is_anomaly"] = delta


def inject_synthetic_anomalies(
    df,
    leak_prob=0.01,
    spike_prob=0.01,
    flatline_prob=0.005,
    leak_factor_range=(2.0, 5.0),
    spike_factor_range=(5.0, 20.0),
    flatline_duration_range=(3, 10),
    random_state=None,
    anomaly_column_name = "Diff",
):
    rng = np.random.default_rng(random_state)

    n = len(df)
    if n == 0:
        return df
    
    df = df.sort_values("timestamp_utc").copy()
    df[f"{anomaly_column_name}_original"] = df[anomaly_column_name]
    df["is_anomaly"] = 0.0

    # ------------------------------------------------------------------
    # 1) Leak-like anomalies: sustained increase in consumption
    # ------------------------------------------------------------------
    n_leaks = rng.binomial(n=1, p=leak_prob)

    if n_leaks > 0 and n >= 5:
        # Pick random start and duration
        start_idx = rng.integers(low=0, high=n - 3)
        duration = rng.integers(low=5, high=min(50, n - start_idx))
        end_idx = start_idx + duration  # exclusive upper bound

        # Robust typical increment
        typical_inc = _safe_typical_increment(df["Diff"])

        leak_factor = rng.uniform(*leak_factor_range)
        extra_inc = typical_inc * leak_factor

        # Total extra offset at the *end* of the leak window
        # step goes from 1 to duration  => total = extra_inc * (1+...+duration)
        total_offset = extra_inc * (duration * (duration + 1) / 2.0)

        # 1a) Add leak effect cumulatively within the window
        for i in range(start_idx, min(end_idx, n)):
            step = i - start_idx + 1
            delta = extra_inc * step
            df.loc[df.index[i], anomaly_column_name] += delta
            df.loc[df.index[i], "is_anomaly"] = delta

        # 1b) Propagate the final leak offset to all *future* points
        if end_idx < n:
            df.loc[df.index[end_idx: ], anomaly_column_name] += total_offset

    # ------------------------------------------------------------------
    # 2) Spike anomalies: single large jump (up or down)
    #    but reject if it would violate non-decreasing hodnota
    # ------------------------------------------------------------------
    if spike_prob > 0:
        n_spikes = rng.binomial(n=n, p=spike_prob)
    else:
        n_spikes = 0

    if n_spikes > 0:
        spike_indices = rng.integers(low=0, high=n, size=n_spikes)

        typical_inc_spike = _safe_typical_increment(df["Diff"])

        for idx in np.unique(spike_indices):
            spike_factor = rng.uniform(*spike_factor_range)
            # allow both upward and downward spikes
            direction = rng.choice([-1, 1])
            delta = direction * typical_inc_spike * spike_factor

            _try_apply_local_anomaly(df, idx, delta)

    # ------------------------------------------------------------------
    # 3) Flatline anomalies: sensor stuck at a constant value
    # ------------------------------------------------------------------
    n_flat = rng.binomial(n=1, p=flatline_prob)

    if n_flat > 0 and n >= 5:
        start_idx = rng.integers(low=0, high=n - 3)
        duration = rng.integers(
            low=flatline_duration_range[0],
            high=min(flatline_duration_range[1], n - start_idx),
        )
        end_idx = start_idx + duration

        flat_value = df.loc[df.index[start_idx], anomaly_column_name]
        for i in range(start_idx + 1, min(end_idx, n)):
            old_val = df.loc[df.index[i], anomaly_column_name]
            delta = flat_value - old_val
            if delta != 0:
                # flat_value is taken from an earlier point in time, so
                # by construction hodnota[i] >= hodnota[i-1] still holds
                df.loc[df.index[i], anomaly_column_name] = flat_value
                df.loc[df.index[i], "is_anomaly"] = delta

    # Final safety check (optional): clamp any residual decreases
    hod = df[anomaly_column_name].to_numpy()
    for i in range(1, len(hod)):
        if hod[i] < hod[i - 1]:
            hod[i] = hod[i - 1]
    df[anomaly_column_name] = hod

    # Return only relevant columns
    return df.copy()


# ------------------------------
# File-level worker
# ------------------------------


def process_one_file(
    filename,
    src_dir,
    dst_dir,
    inject_kwargs=None,
):
    """
    Read one CSV, inject anomalies, and write to dst_dir with same filename.
    """
    if inject_kwargs is None:
        inject_kwargs = {}

    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    try:
        df = pd.read_csv(src_path)

        if "timestamp_utc" not in df.columns:
            raise ValueError(
                f"No timestamp_utc column found. Available: {df.columns.tolist()}"
            )
        if "hodnota" not in df.columns:
            raise ValueError(
                f"No 'hodnota' column found. Available: {df.columns.tolist()}"
            )
        if len(df) < 2:
            raise ValueError("DataFrame passed is too short len < 2")

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

        min_ts = df["timestamp_utc"].min()
        max_ts = df["timestamp_utc"].max()

        span = max_ts - min_ts
        if span < pd.Timedelta(days=100):
            raise ValueError(
                "Not enough data. Do not have at least 100 days of data."
            )

        base_name = os.path.splitext(filename)[0]
        file_seed = int(base_name)

        df_out = inject_synthetic_anomalies(
            df,
            random_state=file_seed,
            **inject_kwargs,
        )
        
        # Recompute Diff
        df_out['Diff'] = df_out['hodnota'].diff()
        df_out.loc[df_out['Diff'] <= -0.002, 'Diff'] = np.nan
        df_out.loc[(df_out['Diff'] > -0.002) & (df_out['Diff'] < 0), 'Diff'] = 0

        os.makedirs(dst_dir, exist_ok=True)
        df_out.to_csv(dst_path, index=False)

        return filename, True, None
    except Exception as e:
        return filename, False, str(e)


def _clear_directory_files_only(directory):
    if not os.path.isdir(directory):
        return  # nothing to clear
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            os.remove(path)


# ------------------------------
# Main function: sample 1000 files and run in parallel
# ------------------------------


def generate_anomalies_for_random_files(
    src_dir,
    dst_dir,
    n_files=1000,
    inject_kwargs=None,
    n_processes=None,
    seed=42,
):
    """
    Randomly chooses up to n_files CSVs from src_dir, runs anomaly injection
    per file in parallel, and writes outputs to dst_dir with same filenames.
    """
    # List all files (filter to CSV if needed)
    all_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".csv")]
    if not all_files:
        print("No CSV files found in source directory.")
        return

    _clear_directory_files_only(dst_dir)

    random.seed(seed)

    # Randomly sample
    n_sample = min(n_files, len(all_files))
    sampled_files = random.sample(all_files, n_sample)

    # Prepare multiprocessing
    if n_processes is None:
        n_processes = max(cpu_count() - 1, 1)

    worker = partial(
        process_one_file,
        src_dir=src_dir,
        dst_dir=dst_dir,
        inject_kwargs=inject_kwargs,
    )

    results = []
    with Pool(processes=n_processes) as pool:
        for res in tqdm(
            pool.imap_unordered(worker, sampled_files),
            total=len(sampled_files),
            desc="Processing files",
        ):
            results.append(res)

    # Optional: simple logging of failures
    failed = [r for r in results if not r[1]]
    if failed:
        print(f"Number of failed files: {len(failed)}.")
        print("Files that failed:")
        for fname, ok, err in failed:
            print(f"  {fname}: {err}")

    print(
        f"Successfully processed {len(sampled_files) - len(failed)} / {len(sampled_files)} files."
    )


# ------------------------------
# Example usage
# ------------------------------

if __name__ == "__main__":
    SRC_DIR = "./data_w_diff_001"
    DST_DIR = "./data_w_anomalies"

    inject_params = {
        "leak_prob": 0.0075,
        "spike_prob": 0.02,
        "flatline_prob": 0.0075,
    }

    generate_anomalies_for_random_files(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
        n_files=1100,
        inject_kwargs=inject_params,
        n_processes=None,  # auto: cpu_count()-1
    )
