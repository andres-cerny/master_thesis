import numpy as np
import pandas as pd

def _safe_typical_increment_from_hodnota(hod_series, fallback=0.001):
    """
    Compute a robust 'typical increment' from the hodnota series
    by looking at positive diffs. This makes spike sizes scale
    with typical consumption for that sensor.
    """
    hod = pd.to_numeric(hod_series, errors="coerce")
    diff = hod.diff()
    diff = pd.to_numeric(diff, errors="coerce")

    # Positive increments only
    diff_pos = diff.clip(lower=0)
    diff_pos = diff_pos.replace(0, np.nan)

    if diff_pos.notna().any():
        val = diff_pos.median()
    else:
        val = fallback

    if pd.isna(val) or val == 0:
        val = fallback

    return float(val)


def _try_apply_hodnota_spike(df, idx, delta, column_name="hodnota"):
    """
    Try to apply a spike at idx by adding 'delta' to hodnota.
    Enforce monotone non-decreasing hodnota: if it would break,
    do nothing and do NOT mark an anomaly.
    """
    cur_val = df.loc[df.index[idx], column_name]
    proposed = cur_val + delta

    if idx == 0:
        prev_val = None
    else:
        prev_val = df.loc[df.index[idx - 1], column_name]

    # Enforce monotone non-decreasing hodnota
    if prev_val is not None and proposed < prev_val:
        return  # reject

    # Also ensure we do not cause later values to become < current
    # (only needed for downward spikes)
    if idx < len(df) - 1:
        next_val = df.loc[df.index[idx + 1], column_name]
        if proposed > next_val:
            return  # reject

    # Accept anomaly
    df.loc[df.index[idx], column_name] = proposed
    df.loc[df.index[idx], "is_anomaly"] = delta


def inject_spike_anomalies_hodnota(
    df,
    min_spikes=1,
    max_spikes=10,
    spike_factor_range=(5.0, 20.0),
    random_state=None,
    column_name="hodnota",
):
    """
    Inject only spike anomalies into the cumulative 'hodnota' column.

    - Number of spikes is uniform between min_spikes and max_spikes.
    - Spike size is typical_increment * spike_factor, with random sign,
      but spikes that would break monotonicity are skipped.
    """
    rng = np.random.default_rng(random_state)

    n = len(df)
    if n == 0:
        return df

    df[f"{column_name}_original"] = df[column_name]
    df["is_anomaly"] = 0.0

    # Decide how many spikes to inject in this df
    if max_spikes < min_spikes:
        max_spikes = min_spikes

    n_spikes = rng.integers(low=min_spikes, high=max_spikes + 1)

    # Typical increment from hodnota
    typical_inc = _safe_typical_increment_from_hodnota(df[column_name])

    # Draw candidate positions (may be fewer effective spikes if some are rejected)
    candidate_indices = rng.integers(low=0, high=n, size=n_spikes * 3)
    used = 0

    for idx in np.unique(candidate_indices):
        if used >= n_spikes:
            break

        spike_factor = rng.uniform(*spike_factor_range)
        direction = rng.choice([-1, 1])
        delta = direction * typical_inc * spike_factor

        before_val = df.loc[df.index[idx], column_name]
        _try_apply_hodnota_spike(df, idx, delta, column_name=column_name)
        after_val = df.loc[df.index[idx], column_name]

        if after_val != before_val:
            used += 1

    # If we somehow ended with zero effective spikes (e.g., all rejected),
    # we do not force anything else; you can optionally relax constraints
    # if you want to guarantee >= 1 spike no matter what.

    return df.copy()