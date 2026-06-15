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
from gap_resample import fill_gaps_with_periodicity_adaptive
# series_diagnostics.py ships with the project (seasonal strength + MASE).
from series_diagnostics import compute_seasonal_diagnostics, compute_mase

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
          timestamp_col="timestamp", hodnota_col="hodnota",
          timezone="Europe/Prague",
          apply_quality_gate=True, mase_metric="mase_seasonal",
          quality_overrides=None,
          negative_diff_tol=0.002,
          verbose=False):

    meter_dir = os.path.join(state_dir, str(meter_id))

    # ------------------------------------------------------------------
    # Load, standardize column names, convert local time -> UTC
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found. Have: {df.columns.tolist()}")
    if hodnota_col not in df.columns:
        raise ValueError(f"Value column '{hodnota_col}' not found. Have: {df.columns.tolist()}")

    # Rename to the canonical internal names used throughout the pipeline.
    # Drop any pre-existing canonical column FIRST, so the rename can't create a
    # duplicate. Sensor files often still carry a 'timestamp_utc' left over from
    # earlier preprocessing; renaming 'timestamp' -> 'timestamp_utc' on top of it
    # would make df['timestamp_utc'] a DataFrame and break the conversion.
    rename = {}
    drop_cols = []
    if hodnota_col != "hodnota":
        rename[hodnota_col] = "hodnota"
        if "hodnota" in df.columns:
            drop_cols.append("hodnota")
    if timestamp_col != "timestamp_utc":
        rename[timestamp_col] = "timestamp_utc"
        if "timestamp_utc" in df.columns:
            drop_cols.append("timestamp_utc")
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.rename(columns=rename)

    # DST-aware conversion (mirrors the batch add_utc_timestamp). tz-aware input
    # passes through; ambiguous/nonexistent local times become NaT and are dropped.
    df["timestamp_utc"] = shared.localize_to_utc(df["timestamp_utc"], timezone=timezone)
    n_before = len(df)
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    # Enforce strictly increasing timestamps. Sorting fixes out-of-order arrivals;
    # dropping exact duplicates removes the zero/negative gaps that otherwise make
    # the resampler try to insert a negative number of rows ("'shape' elements
    # cannot be negative").
    df = df.drop_duplicates(subset="timestamp_utc", keep="first").reset_index(drop=True)
    n_dropped = n_before - len(df)
    if len(df) < 2:
        raise ValueError("Input series too short (len < 2) after timestamp conversion.")

    df_res, diag = fill_gaps_with_periodicity_adaptive(df, timestamp_col="timestamp_utc")
    periodicity = float(diag.get("periodicity_used_seconds", 0) or 0)
    if periodicity <= 0:
        raise ValueError("Resampling returned a non-positive periodicity.")

    # The resampler emits hodnota (+ is_anomaly); Diff is the modelled signal.
    # Recompute it here if the resampler did not carry it through.
    if "Diff" not in df_res.columns:
        df_res["Diff"] = df_res["hodnota"].diff()

    # Clean negative diffs (meter rollback / reading noise), matching the batch
    # convention: |neg| < tol -> 0 (rounding), larger neg -> NaN (dropped, so the
    # Kalman filter skips it just like a gap).
    _d = df_res["Diff"]
    df_res["Diff"] = _d.mask((_d < 0) & (_d > -negative_diff_tol), 0.0)
    _d = df_res["Diff"]
    df_res["Diff"] = _d.mask(_d <= -negative_diff_tol, np.nan)

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
    is_retrain = init_mean is not None  # a structurally-compatible prior exists
    thr = shared.resolve_quality_thresholds(is_retrain, quality_overrides)

    # ------------------------------------------------------------------
    # Quality gate — PRE-TRAIN (cheap: rejects before the expensive MLE)
    #   fill_pct : share of resampled rows that are synthetic gap-fills
    #   f_s      : daily seasonal strength (Wang-Smith-Hyndman) on clean Diff
    # ------------------------------------------------------------------
    fill_pct = 100.0 * float(pd.isna(df_res["hodnota"]).mean())

    # Seasonal strength needs enough *real* points to support the periods it
    # decomposes on. A very sparse/flat meter (e.g. 103459, 103475) can have a
    # 4-week slice that, after gap-fill + negative-diff cleaning, holds fewer
    # valid points than ~2 weekly cycles; the decomposition then does a
    # length-minus-period computation that goes negative ("'shape' elements
    # cannot be negative"). Treat that as "no usable seasonality" -> NaN f_s,
    # which the quality gate already rejects, rather than crashing the meter.
    diff_vals = df_res["Diff"].values
    n_valid = int(np.count_nonzero(~np.isnan(diff_vals)))
    if n_valid < 2 * daily_steps:
        seas = {"f_s_daily": float("nan"), "f_s_weekly": None}
    else:
        try:
            seas = compute_seasonal_diagnostics(diff_vals, daily_steps, weekly_steps)
        except Exception as e:
            if verbose:
                print(f"[{meter_id}] seasonal diagnostics failed ({e!r}); "
                      f"treating f_s as NaN (n_valid={n_valid}).")
            seas = {"f_s_daily": float("nan"), "f_s_weekly": None}
    f_s_daily = float(seas.get("f_s_daily")) if seas.get("f_s_daily") is not None else float("nan")
    f_s_weekly = seas.get("f_s_weekly")

    pre_reasons = shared.quality_gate_failures(thr, fill_pct=fill_pct, f_s=f_s_daily)
    if apply_quality_gate and pre_reasons:
        result = {
            "meter_id": str(meter_id), "status": "rejected", "accepted": False,
            "stage": "pre_train", "reasons": pre_reasons, "is_retrain": is_retrain,
            "thresholds": thr,
            "metrics": {"fill_pct": fill_pct, "f_s_daily": f_s_daily,
                        "f_s_weekly": f_s_weekly, "mase": None, "mase_seasonal": None},
        }
        if verbose:
            print(json.dumps(result, indent=2, default=str))
        return result

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
    # Quality gate — POST-TRAIN (needs the fitted model): MASE skill score.
    #   model MAE on the reference (out-of-sample if holdout) one-step errors,
    #   scaled by the in-sample naive MAE on the fit Diff.
    # ------------------------------------------------------------------
    mase_d = compute_mase(ref_actual, ref_pred, df_fit["Diff"].values, m_seasonal=daily_steps)
    mase_val = mase_d.get(mase_metric)
    post_reasons = shared.quality_gate_failures(thr, mase=mase_val)
    if apply_quality_gate and post_reasons:
        result = {
            "meter_id": str(meter_id), "status": "rejected", "accepted": False,
            "stage": "post_train", "reasons": post_reasons, "is_retrain": is_retrain,
            "thresholds": thr,
            "metrics": {"fill_pct": fill_pct, "f_s_daily": f_s_daily,
                        "f_s_weekly": f_s_weekly,
                        "mase": mase_d.get("mase"), "mase_seasonal": mase_d.get("mase_seasonal"),
                        "mase_metric": mase_metric},
        }
        if verbose:
            print(json.dumps(result, indent=2, default=str))
        return result  # fitted but NOT persisted -> sensor not put into service

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
        "timezone": timezone,
        "negative_diff_tol": float(negative_diff_tol),
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
        "n_dropped_bad_timestamp": int(n_dropped),
        "is_retrain": bool(is_retrain),
        "quality_gate_applied": bool(apply_quality_gate),
        "quality_thresholds": thr,
        "fill_pct": fill_pct,
        "f_s_daily": f_s_daily,
        "f_s_weekly": (None if f_s_weekly is None else float(f_s_weekly)),
        "mase": mase_d.get("mase"),
        "mase_seasonal": mase_d.get("mase_seasonal"),
        "mase_metric": mase_metric,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }

    shared.save_model(
        meter_dir,
        T=T, Z=Z, H=H, Q_full=Q_full, P0_train=P0,
        means_per_pos=means_per_pos, stds_per_pos=stds_per_pos,
        config=config, x0=x0, P0=P0,
        state_time=state_time, last_real_hodnota=last_real_hodnota,
    )

    result = {
        "meter_id": str(meter_id), "status": "success", "accepted": True,
        "stage": None, "reasons": [], "is_retrain": is_retrain, "thresholds": thr,
        "metrics": {"fill_pct": fill_pct, "f_s_daily": f_s_daily,
                    "f_s_weekly": (None if f_s_weekly is None else float(f_s_weekly)),
                    "mase": mase_d.get("mase"), "mase_seasonal": mase_d.get("mase_seasonal"),
                    "mase_metric": mase_metric},
        "config": config,
    }

    if verbose:
        print(json.dumps({
            "meter_id": str(meter_id),
            "status": "success",
            "is_retrain": is_retrain,
            "periodicity_seconds": periodicity,
            "daily_period_steps": daily_steps,
            "k_positions": k_positions,
            "warm_started": warm,
            "converged": converged,
            "fill_pct": round(fill_pct, 2),
            "f_s_daily": (None if not np.isfinite(f_s_daily) else round(f_s_daily, 3)),
            "mase": (None if mase_d.get("mase") is None else round(mase_d["mase"], 3)),
            "mase_seasonal": (None if mase_d.get("mase_seasonal") is None
                              else round(mase_d["mase_seasonal"], 3)),
            "global_mean": round(global_mean, 4),
            "global_std": round(global_std, 4),
            "state_time": pd.Timestamp(state_time).isoformat(),
            "last_real_hodnota": last_real_hodnota,
            "saved_to": meter_dir,
        }, indent=2))

    return result


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
    p.add_argument("--timestamp-col", default="timestamp", help="Name of the timestamp column in the CSV.")
    p.add_argument("--hodnota-col", default="hodnota", help="Name of the meter-value column in the CSV.")
    p.add_argument("--timezone", default="Europe/Prague", help="Local timezone of the raw timestamps.")
    # ---- quality gate ----
    p.add_argument("--no-quality-gate", action="store_true",
                   help="Compute the quality metrics but do not reject any sensor.")
    p.add_argument("--mase-metric", default="mase_seasonal", choices=["mase_seasonal", "mase"],
                   help="Which MASE to gate on (default: seasonal-naive scaled).")
    p.add_argument("--negative-diff-tol", type=float, default=0.002,
                   help="Negative Diff magnitude below which it's treated as 0 (rounding); "
                        "at or above it the reading is dropped (NaN/skip).")
    p.add_argument("--fs-min-initial", type=float, default=None)
    p.add_argument("--fs-min-retrain", type=float, default=None)
    p.add_argument("--mase-max-initial", type=float, default=None)
    p.add_argument("--mase-max-retrain", type=float, default=None)
    p.add_argument("--fill-max-pct-initial", type=float, default=None)
    p.add_argument("--fill-max-pct-retrain", type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    quality_overrides = {
        "fs_min_initial": args.fs_min_initial,
        "fs_min_retrain": args.fs_min_retrain,
        "mase_max_initial": args.mase_max_initial,
        "mase_max_retrain": args.mase_max_retrain,
        "fill_max_pct_initial": args.fill_max_pct_initial,
        "fill_max_pct_retrain": args.fill_max_pct_retrain,
    }

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
        timestamp_col=args.timestamp_col,
        hodnota_col=args.hodnota_col,
        timezone=args.timezone,
        apply_quality_gate=not args.no_quality_gate,
        mase_metric=args.mase_metric,
        quality_overrides=quality_overrides,
        negative_diff_tol=args.negative_diff_tol,
        verbose=True,
    )


if __name__ == "__main__":
    main()