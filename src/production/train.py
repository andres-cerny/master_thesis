"""
train.py — Offline (scheduled) training for the streaming UC water-meter
anomaly detector.

Run this on a 4-week segment of raw meter data. It:
  1. resamples / gap-fills via resample.fill_gaps_with_periodicity_adaptive
     (this also fixes the effective periodicity and the daily/weekly geometry);
  2. (optionally) warm-starts the Kalman state from the previous fit, but only
     if the structural shape is unchanged;
  3. fits the local-level + daily/weekly Fourier UC model by MLE;
  4. computes reference residuals (out-of-sample on a holdout tail by default)
     and the per-position + global residual statistics for z-scoring;
  5. extracts the state-space system matrices (T, Z, H, Q_full);
  6. persists model.npz + config.json + the initial state.npz, atomically.

The expensive work (MLE) lives here. predict.py never fits anything.

Usage:
    python train.py --meter-id 137402 \
        --csv ../../../data/sensor_data/137402.csv \
        --state-dir ./state \
        [--holdout-days 7] [--z-threshold 3.5] \
        [--harmonics-daily 2] [--harmonics-weekly 2] \
        [--k-hours 2.0] [--min-k-positions 2] [--tolerance-percentage 10]

Requires resample.py (the project's resampler) to be importable.
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

# resample.py ships with the project; make sure it is importable.
from resample import fill_gaps_with_periodicity_adaptive

import shared

import warnings
from statsmodels.tools.sm_exceptions import (
    SpecificationWarning, ConvergenceWarning,
)
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SpecificationWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _build_model(endog, freq_seasonal, stochastic_freq_seasonal):
    return UnobservedComponents(
        endog=np.asarray(endog, dtype=float),
        level="local level",   # local level only, no slope
        seasonal=None,         # daily/weekly handled via freq_seasonal (Fourier)
        freq_seasonal=freq_seasonal,
        stochastic_level=True,
        stochastic_freq_seasonal=stochastic_freq_seasonal,
    )


def _extract_matrices(theta, freq_seasonal, stochastic_freq_seasonal, weekly_steps):
    """Pull the time-invariant state-space matrices for fixed parameters theta.
    Built once here so the prediction path never touches statsmodels."""
    dummy = _build_model(np.zeros(int(weekly_steps) + 2), freq_seasonal, stochastic_freq_seasonal)
    dummy.update(theta)

    def get_mat(name):
        m = dummy.ssm[name]
        return m[:, :, 0] if m.ndim == 3 else m

    T = np.array(get_mat("transition"), dtype=float)
    Z = np.array(get_mat("design"), dtype=float)
    R = np.array(get_mat("selection"), dtype=float)
    Q = np.array(get_mat("state_cov"), dtype=float)
    H = np.array(get_mat("obs_cov"), dtype=float)
    Q_full = R @ Q @ R.T
    return T, Z, H, Q_full


def train(meter_id, csv_path, state_dir,
          holdout_days=7,
          harmonics_daily=2, harmonics_weekly=2,
          k_hours=2.0, min_k_positions=2,
          z_threshold=3.5, tolerance_percentage=10.0,
          verbose=False):

    meter_dir = os.path.join(state_dir, str(meter_id))

    # ------------------------------------------------------------------
    # Load + resample
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    for col in ("timestamp_utc", "hodnota"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Have: {df.columns.tolist()}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Input series too short (len < 2).")

    df_res, diag = fill_gaps_with_periodicity_adaptive(df, timestamp_col="timestamp_utc")
    periodicity = float(diag.get("periodicity_used_seconds", 0) or 0)
    if periodicity <= 0:
        raise ValueError("Resampling returned a non-positive periodicity.")

    # The resampler emits hodnota (+ is_anomaly); Diff is the modelled signal.
    # Recompute it here if the resampler did not carry it through.
    if "Diff" not in df_res.columns:
        df_res["Diff"] = df_res["hodnota"].diff()

    daily_steps = int(round(24 * 3600 / periodicity))
    if daily_steps < 2:
        raise ValueError(f"Periodicity too coarse: daily_period_steps={daily_steps} (<2).")
    weekly_steps = 7 * daily_steps

    freq_seasonal = [
        {"period": weekly_steps, "harmonics": int(harmonics_weekly)},  # weekly
        {"period": daily_steps,  "harmonics": int(harmonics_daily)},   # daily
    ]
    stochastic_freq_seasonal = [True, True]
    k_positions = shared.compute_k_positions(daily_steps, k_hours=k_hours, min_k=min_k_positions)

    # ------------------------------------------------------------------
    # Fit / reference split
    # ------------------------------------------------------------------
    last_ts = df_res["timestamp_utc"].max()
    if holdout_days and holdout_days > 0:
        cutoff = last_ts - pd.Timedelta(days=holdout_days)
        df_fit = df_res[df_res["timestamp_utc"] < cutoff].reset_index(drop=True)
        df_ref = df_res[df_res["timestamp_utc"] >= cutoff].reset_index(drop=True)
    else:
        df_fit = df_res
        df_ref = df_res  # in-sample fallback (slightly optimistic band)
    if len(df_fit) < 2 or len(df_ref) < 2:
        raise ValueError("Fit or reference segment too short after holdout split.")

    # ------------------------------------------------------------------
    # Warm-start (only if structurally identical to the previous fit)
    # ------------------------------------------------------------------
    sig = shared.structural_signature(periodicity, daily_steps, weekly_steps, freq_seasonal)
    init_mean, init_cov = shared.warmstart_state(meter_dir, sig)

    model_fit = _build_model(df_fit["Diff"].values, freq_seasonal, stochastic_freq_seasonal)
    if init_mean is not None and init_cov is not None and init_mean.shape[0] == model_fit.k_states:
        model_fit.initialize_known(initial_state=init_mean, initial_state_cov=init_cov)
        warm = True
    else:
        warm = False
    res_fit = model_fit.fit(disp=False)
    theta = res_fit.params
    converged = bool(res_fit.mle_retvals.get("converged", False))

    fs = res_fit.filter_results.filtered_state
    fc = res_fit.filter_results.filtered_state_cov
    end_fit_mean = fs[:, -1].copy()
    end_fit_cov = fc[:, :, -1].copy()

    # ------------------------------------------------------------------
    # Reference residuals + carry-forward state
    # ------------------------------------------------------------------
    if holdout_days and holdout_days > 0:
        # Filter the holdout tail from the end-of-fit state -> genuine
        # out-of-sample one-step residuals, and a carry-forward state aligned
        # to the true last timestamp of the 4-week window.
        model_ref = _build_model(df_ref["Diff"].values, freq_seasonal, stochastic_freq_seasonal)
        model_ref.initialize_known(initial_state=end_fit_mean, initial_state_cov=end_fit_cov)
        res_ref = model_ref.filter(theta)
        ref_pred = np.clip(np.asarray(res_ref.get_prediction().predicted_mean, dtype=float), 0.0, None)
        fs2 = res_ref.filter_results.filtered_state
        fc2 = res_ref.filter_results.filtered_state_cov
        x0 = fs2[:, -1].copy()
        P0 = fc2[:, :, -1].copy()
        ref_df = df_ref
    else:
        # In-sample: reuse the fit's own one-step predictions + end state.
        ref_pred = np.clip(np.asarray(res_fit.get_prediction().predicted_mean, dtype=float), 0.0, None)
        x0 = end_fit_mean
        P0 = end_fit_cov
        ref_df = df_fit

    ref_actual = ref_df["Diff"].values.astype(float)
    ref_resid = np.abs(ref_actual - ref_pred)

    # ------------------------------------------------------------------
    # z-score statistics (global + per position on the daily cycle)
    # ------------------------------------------------------------------
    ref_positions = shared.compute_position_in_period(
        ref_df["timestamp_utc"].values, periodicity, daily_steps
    )
    means_per_pos, stds_per_pos = shared.compute_per_position_stats(
        ref_resid, ref_positions, daily_steps, k_positions
    )
    valid = ref_resid[~np.isnan(ref_resid)]
    global_mean = float(np.nanmean(valid)) if valid.size > 0 else 0.0
    global_std = float(np.nanstd(valid, ddof=1)) if valid.size > 1 else 0.0

    # ------------------------------------------------------------------
    # System matrices + carry-forward bookkeeping
    # ------------------------------------------------------------------
    T, Z, H, Q_full = _extract_matrices(theta, freq_seasonal, stochastic_freq_seasonal, weekly_steps)

    # state_time = timestamp x0/P0 are aligned to (end of ref segment).
    # last_real_hodnota = last real cumulative reading (skip synthetic NaN rows).
    state_time = ref_df["timestamp_utc"].iloc[-1]
    real_rows = ref_df.dropna(subset=["hodnota"])
    if len(real_rows) == 0:
        raise ValueError("No real hodnota readings in reference segment.")
    last_real_hodnota = float(real_rows["hodnota"].iloc[-1])

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    config = {
        "schema_version": shared.SCHEMA_VERSION,
        "meter_id": str(meter_id),
        "periodicity_seconds": periodicity,
        "tolerance_percentage": float(tolerance_percentage),
        "daily_period_steps": int(daily_steps),
        "weekly_period_steps": int(weekly_steps),
        "freq_seasonal": freq_seasonal,
        "k_positions": int(k_positions),
        "k_hours": float(k_hours),
        "z_threshold": float(z_threshold),
        "global_mean": global_mean,
        "global_std": global_std,
        "structural_signature": sig,
        "outage_cap_steps": int(weekly_steps),
        "theta": [float(v) for v in np.asarray(theta).ravel()],
        "warm_started": bool(warm),
        "converged": converged,
        "holdout_days": float(holdout_days),
        "n_fit": int(len(df_fit)),
        "n_ref": int(len(ref_df)),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }

    shared.save_model(
        meter_dir,
        T=T, Z=Z, H=H, Q_full=Q_full, P0_train=P0,
        means_per_pos=means_per_pos, stds_per_pos=stds_per_pos,
        config=config, x0=x0, P0=P0,
        state_time=state_time, last_real_hodnota=last_real_hodnota,
    )

    if verbose:
        print(json.dumps({
            "meter_id": str(meter_id),
            "periodicity_seconds": periodicity,
            "daily_period_steps": daily_steps,
            "k_positions": k_positions,
            "warm_started": warm,
            "converged": converged,
            "global_mean": round(global_mean, 4),
            "global_std": round(global_std, 4),
            "state_time": pd.Timestamp(state_time).isoformat(),
            "last_real_hodnota": last_real_hodnota,
            "saved_to": meter_dir,
        }, indent=2))

    return config


def main():
    p = argparse.ArgumentParser(description="Train the streaming UC anomaly detector for one meter.")
    p.add_argument("--meter-id", required=True)
    p.add_argument("--csv", required=True, help="Path to the meter CSV (timestamp_utc, hodnota[, Diff]).")
    p.add_argument("--state-dir", default="./state", help="Root dir for per-meter state.")
    p.add_argument("--holdout-days", type=float, default=7.0,
                   help="Tail days held out of the fit for out-of-sample reference residuals "
                        "(0 = in-sample, slightly tighter band).")
    p.add_argument("--harmonics-daily", type=int, default=2)
    p.add_argument("--harmonics-weekly", type=int, default=2)
    p.add_argument("--k-hours", type=float, default=2.0)
    p.add_argument("--min-k-positions", type=int, default=2)
    p.add_argument("--z-threshold", type=float, default=3.5)
    p.add_argument("--tolerance-percentage", type=float, default=10.0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    train(
        meter_id=args.meter_id,
        csv_path=args.csv,
        state_dir=args.state_dir,
        holdout_days=args.holdout_days,
        harmonics_daily=args.harmonics_daily,
        harmonics_weekly=args.harmonics_weekly,
        k_hours=args.k_hours,
        min_k_positions=args.min_k_positions,
        z_threshold=args.z_threshold,
        tolerance_percentage=args.tolerance_percentage,
        verbose=True,
    )


if __name__ == "__main__":
    main()
