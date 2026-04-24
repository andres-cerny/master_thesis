import numpy as np
import pandas as pd


def inject_spike_anomalies_diff(
    df,
    min_spikes=1,
    max_spikes=10,
    spike_factor_range=(3.0, 10.0),
    random_state=None,
    diff_col="Diff",
):
    rng = np.random.default_rng(random_state)
    n = len(df)
    if n == 0:
        return df

    df = df.copy()
    df[f"{diff_col}_original"] = df[diff_col]
    df["is_anomaly"] = 0.0

    diff_vals = pd.to_numeric(df[diff_col], errors="coerce")
    sigma = float(diff_vals.std())

    # Fall back to median-based scale if std is zero or NaN
    if pd.isna(sigma) or sigma == 0:
        pos = diff_vals[diff_vals > 0]
        sigma = float(pos.median()) if pos.notna().any() else 0.001

    n_spikes = int(rng.integers(low=min_spikes, high=max_spikes + 1))
    candidate_indices = rng.choice(n, size=min(n_spikes * 3, n), replace=False)

    used = 0
    for idx in candidate_indices:
        if used >= n_spikes:
            break

        spike_factor = rng.uniform(*spike_factor_range)
        delta = spike_factor * sigma  # 3–10 std above normal

        original = df.iloc[idx][diff_col]
        if pd.isna(original):
            continue

        df.iloc[idx, df.columns.get_loc(diff_col)] = original + delta
        df.iloc[idx, df.columns.get_loc("is_anomaly")] = delta
        used += 1

    return df