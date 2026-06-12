"""
Unobserved Components (Local Level + Daily/Weekly Fourier) Batch Processor
for Water Meter Anomaly Detection
=================================================================================================

Processes multiple water meter CSV files with:
- Multithreaded execution
- Resampling & gap filling using fill_gaps_with_periodicity_adaptive
- UnobservedComponents model with local level (no slope) + daily and weekly
  seasonality encoded via Fourier harmonics
- Initial training on weeks 1–4 of a random 6-week window
- Second training on weeks 2–5, initialized from previous state
- Prediction on week 6, initialized from previous state
- Position-conditional z-score thresholding: residual statistics are estimated
  per time-of-day position (using a ±k-position neighborhood on the circular
  daily axis), so nights with small prediction errors get a tighter detection
  band and daytime hours with larger errors get a wider band.

Usage:
    python seasonal_batch_processor.py --workers 7 --verbose

Notes on NaN handling:
- NaN values in Diff are preserved and handled by the Kalman filter
  (skips update step)
- All parameter initialization uses np.nanstd/np.nanmean to avoid NaN propagation
- Residuals with NaN actuals are masked before metrics computation
"""


import os
import json
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
import logging
import traceback
from typing import Dict, Optional, List, Tuple
import warnings
import random
from tqdm import tqdm
import time
import sys
import pickle
import signal
from contextlib import contextmanager
from statsmodels.tsa.statespace.structural import UnobservedComponents
from pandas.tseries.offsets import DateOffset
from numpy.random import default_rng


from resample import fill_gaps_with_periodicity_adaptive, get_periodicity

parent_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(parent_path)

from helper_scripts.calculate_metrics import calculate_metrics, calculate_metrics_unresampled
from helper_scripts.create_anomalies import inject_spike_anomalies_diff


import warnings
from statsmodels.tools.sm_exceptions import SpecificationWarning
from scipy.sparse import SparseEfficiencyWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=SpecificationWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(log_file="seasonal_batch_processor.log", verbose: bool = False):
    """Configure logging to file and console"""
    level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging(verbose=False)


# ============================================================================
# HELPERS: POSITION-CONDITIONAL THRESHOLDING
# ============================================================================


def compute_position_in_period(timestamps, periodicity_seconds, daily_period_steps):
    """
    Map each timestamp to its integer position within the daily cycle,
    in [0, daily_period_steps). Position is derived from time-of-day (UTC),
    so two timestamps separated by exactly one day get the same position
    regardless of where the resampled grid starts.

    Parameters:
    -----------
    timestamps : array-like of datetimes
        UTC timestamps to convert.
    periodicity_seconds : float
        Sampling interval in seconds (e.g., 1800 for 30 min).
    daily_period_steps : int
        Number of measurements that fit into 24 h at this periodicity.

    Returns:
    --------
    np.ndarray of int : Position in [0, daily_period_steps) for each timestamp.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    seconds_since_midnight = (
        ts.hour.values.astype(float) * 3600.0
        + ts.minute.values.astype(float) * 60.0
        + ts.second.values.astype(float)
    )
    positions = np.round(seconds_since_midnight / periodicity_seconds).astype(int)
    return positions % daily_period_steps


def compute_per_position_stats(ref_residuals, ref_positions,
                               daily_period_steps, k_positions):
    """
    For each position p in [0, daily_period_steps), pool reference residuals
    from positions in [p - k, p + k] modulo the period and compute mean, std,
    median, and MAD.

    Returns four arrays of length `daily_period_steps`; positions whose
    neighborhood contains no reference data hold NaN (the caller should
    fall back to global statistics in that case).
    """
    ref_residuals = np.asarray(ref_residuals, dtype=float)
    ref_positions = np.asarray(ref_positions, dtype=int)

    valid = ~np.isnan(ref_residuals)
    ref_residuals = ref_residuals[valid]
    ref_positions = ref_positions[valid]

    period = daily_period_steps
    by_pos = [[] for _ in range(period)]
    for r, p in zip(ref_residuals, ref_positions):
        by_pos[p].append(r)
    by_pos = [np.asarray(lst, dtype=float) for lst in by_pos]

    means = np.full(period, np.nan)
    stds = np.full(period, np.nan)
    medians = np.full(period, np.nan)
    mads = np.full(period, np.nan)

    for p in range(period):
        pooled_chunks = [
            by_pos[(p + off) % period]
            for off in range(-k_positions, k_positions + 1)
            if by_pos[(p + off) % period].size > 0
        ]
        if not pooled_chunks:
            continue
        pooled = np.concatenate(pooled_chunks)
        means[p] = pooled.mean()
        stds[p] = pooled.std(ddof=1) if pooled.size > 1 else 0.0
        medians[p] = np.median(pooled)
        mads[p] = np.median(np.abs(pooled - medians[p]))

    return means, stds, medians, mads


def compute_k_positions(daily_period_steps, k_hours=2.0, min_k=2):
    """
    Convert a desired neighborhood width in real hours to a number of
    positions on the daily cycle, with a floor of `min_k` so coarse-frequency
    sensors still pool enough samples.
    """
    return max(min_k, int(round(k_hours * daily_period_steps / 24)))


# ============================================================================
# HELPER: RESIDUAL-BASED ANOMALIES
# ============================================================================


def compute_z_scores(residuals, result, ref_residuals=None,
                     pred_positions=None, ref_positions=None,
                     daily_period_steps=None, k_positions=None):
    """
    Compute z-score and robust MAD-based score for each residual.

    Two modes:

    1. Global (default): a single mean/std and median/MAD are estimated from
       ref_residuals and applied to every residual.

    2. Position-conditional: if `pred_positions`, `ref_positions`,
       `daily_period_steps`, and `k_positions` are all supplied, statistics
       are estimated per position on the daily cycle. For each position p,
       mean/std/median/MAD are computed from reference residuals at
       positions in [p - k, p + k] modulo the period. Each prediction-segment
       residual is then scored using the statistics for its own position.
       If a position's neighborhood is empty, the global statistics are used
       as a fallback.

    Parameters:
    -----------
    residuals : array-like
        Residuals to be scored (prediction segment).
    result : dict
        Dict to store reference statistics for diagnostics.
    ref_residuals : array-like or None
        Reference residuals (second training segment). If None, falls back
        to `residuals`.
    pred_positions : array-like of int or None
        Position-in-period for each entry of `residuals`.
    ref_positions : array-like of int or None
        Position-in-period for each entry of `ref_residuals`.
    daily_period_steps : int or None
        Length of the daily cycle in measurements.
    k_positions : int or None
        Half-width of the neighborhood used to pool reference residuals
        around each position.

    Returns:
    --------
    z_score : np.ndarray
        Classical z-scores for each element of residuals.
    z_score_robust : np.ndarray
        Robust MAD-based scores for each element of residuals.
    mu_used : np.ndarray
        Mean of the reference distribution that was applied at each step
        (broadcast to a per-element array, even in non-conditional mode).
    sd_used : np.ndarray
        Std of the reference distribution that was applied at each step.
    med_used : np.ndarray
        Median of the reference distribution that was applied at each step.
    sigma_used : np.ndarray
        Robust sigma (1.4826 · MAD) that was applied at each step.
    """
    residuals = np.asarray(residuals, dtype=float)
    ref = np.asarray(ref_residuals, dtype=float) if ref_residuals is not None else residuals

    valid_ref = ref[~np.isnan(ref)]
    valid = ~np.isnan(residuals)

    z_score = np.full_like(residuals, np.nan)
    z_score_robust = np.full_like(residuals, np.nan)
    mu_used = np.full_like(residuals, np.nan)
    sd_used = np.full_like(residuals, np.nan)
    med_used = np.full_like(residuals, np.nan)
    sigma_used = np.full_like(residuals, np.nan)

    if valid_ref.size == 0:
        return z_score, z_score_robust, mu_used, sd_used, med_used, sigma_used

    conditional = (
        pred_positions is not None
        and ref_positions is not None
        and daily_period_steps is not None
        and k_positions is not None
    )

    # Global statistics — used as fallback in conditional mode, and as the
    # only statistics in non-conditional mode.
    global_mean = float(np.nanmean(valid_ref))
    global_std = float(np.nanstd(valid_ref, ddof=1)) if valid_ref.size > 1 else 0.0
    global_median = float(np.nanmedian(valid_ref))
    global_mad = float(np.nanmedian(np.abs(valid_ref - global_median)))
    global_sigma_robust = 1.4826 * global_mad if global_mad > 0 else global_std

    if conditional:
        means, stds, medians, mads = compute_per_position_stats(
            ref, ref_positions, daily_period_steps, k_positions
        )
        pred_positions = np.asarray(pred_positions, dtype=int)

        mu = means[pred_positions]
        sd = stds[pred_positions]
        med = medians[pred_positions]
        mad_arr = mads[pred_positions]

        # Where the position-neighborhood has no data or no variability,
        # fall back to global statistics.
        bad_classic = np.isnan(mu) | np.isnan(sd) | (sd == 0)
        mu = np.where(bad_classic, global_mean, mu)
        sd = np.where(bad_classic, global_std, sd)

        bad_robust = np.isnan(med) | np.isnan(mad_arr) | (mad_arr == 0)
        med = np.where(bad_robust, global_median, med)
        sigma = np.where(bad_robust, global_sigma_robust, 1.4826 * mad_arr)

        sd_safe = np.where(sd == 0, 1.0, sd)
        sigma_safe = np.where(sigma == 0, 1.0, sigma)

        z_classic = np.where(sd == 0, 0.0, (residuals - mu) / sd_safe)
        z_robust = np.where(sigma == 0, 0.0, (residuals - med) / sigma_safe)

        z_score[valid] = z_classic[valid]
        z_score_robust[valid] = z_robust[valid]

        # Record the per-step statistics that were actually applied,
        # for detection-band plotting downstream.
        mu_used[:] = mu
        sd_used[:] = sd
        med_used[:] = med
        sigma_used[:] = sigma

        result["ref_mean_per_position"] = means.tolist()
        result["ref_std_per_position"] = stds.tolist()
        result["ref_median_per_position"] = medians.tolist()
        result["ref_mad_per_position"] = mads.tolist()
        result["k_positions"] = k_positions
        # Keep a global summary too, for diagnostics.
        result["ref_mean"] = global_mean
        result["ref_std"] = global_std
        return z_score, z_score_robust, mu_used, sd_used, med_used, sigma_used

    # ---------- Non-conditional (global) path ----------
    result["ref_mean"] = global_mean
    result["ref_std"] = global_std

    if global_std > 0:
        z_score[valid] = (residuals[valid] - global_mean) / global_std
    else:
        z_score[valid] = 0.0

    if global_sigma_robust > 0:
        z_score_robust[valid] = (residuals[valid] - global_median) / global_sigma_robust
    else:
        z_score_robust[valid] = 0.0

    # Broadcast global stats to per-element arrays so the return shape is
    # the same in both modes.
    mu_used[:] = global_mean
    sd_used[:] = global_std
    med_used[:] = global_median
    sigma_used[:] = global_sigma_robust

    return z_score, z_score_robust, mu_used, sd_used, med_used, sigma_used


def build_results_df(df_predict, predictions, result, ref_residuals=None,
                     ref_timestamps=None, periodicity_seconds=None,
                     daily_period_steps=None, k_positions=None,
                     threshold_z_score=3, threshold_z_score_robust=3,
                     save_detection_bands=False):
    """
    Build a result DataFrame with anomaly scores based on residuals.

    Handles NaN values properly by masking them before rolling calculations.

    If `ref_timestamps`, `periodicity_seconds`, `daily_period_steps`, and
    `k_positions` are supplied (in addition to `ref_residuals`), the z-scores
    are computed using position-conditional statistics on the daily cycle.
    Otherwise the original global statistics are used.

    Parameters:
    -----------
    df_predict : pd.DataFrame
        Prediction segment dataframe with 'Diff' and 'timestamp_utc' columns.
    predictions : array-like
        Predicted Diff values (may contain NaN).
    result : dict
        Result dictionary to store summary counts.
    ref_residuals : array-like or None
        Residuals from the second training segment used to estimate
        thresholding statistics. If None, the prediction segment residuals
        are used instead (old behaviour).
    ref_timestamps : array-like or None
        Timestamps for each entry of `ref_residuals`. Required (together
        with the four arguments below) to enable position-conditional scoring.
    periodicity_seconds : float or None
        Sampling interval in seconds.
    daily_period_steps : int or None
        Number of measurements in 24 h.
    k_positions : int or None
        Half-width (in positions) of the neighborhood used to pool
        reference residuals around each position.
    threshold_z_score : int
        Threshold for the classical z-score (default: 3).
    threshold_z_score_robust : int
        Threshold for the robust MAD-based score (default: 3).
    save_detection_bands : bool
        If True, add `upper_band`, `lower_band`, `upper_band_robust`,
        `lower_band_robust` columns to the output DataFrame. The bands are
        expressed in actual-value (Diff) units; an actual value outside
        [lower_band, upper_band] would correspond to |z_score| >
        threshold_z_score (analogously for the robust variant).
        Lower bands are clipped at 0 since Diff is non-negative.

    Returns:
    --------
    pd.DataFrame : Result dataframe with anomaly scores and flags
    """
    actuals = df_predict["Diff"].values.astype(float)
    timestamps = df_predict["timestamp_utc"].values

    if 'is_anomaly' in df_predict.columns:
        anomalies_floats = df_predict["is_anomaly"].values
        anomalies_floats = np.asarray(anomalies_floats)
        anomalies = (anomalies_floats != 0.0).astype(int)
    else:
        anomalies = np.zeros(len(df_predict), dtype=int)

    timestamps = pd.to_datetime(timestamps, utc=True)
    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)

    residuals = np.abs(actuals - predictions)

    result_df = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "actual": actuals,
            "predicted": predictions,
            "residual": residuals,
        }
    )

    # Decide whether we can run position-conditional scoring.
    use_conditional = (
        ref_residuals is not None
        and ref_timestamps is not None
        and periodicity_seconds is not None
        and daily_period_steps is not None
        and k_positions is not None
    )

    if use_conditional:
        pred_positions = compute_position_in_period(
            timestamps, periodicity_seconds, daily_period_steps
        )
        ref_positions = compute_position_in_period(
            ref_timestamps, periodicity_seconds, daily_period_steps
        )
        result_df["position_in_period"] = pred_positions
    else:
        pred_positions = None
        ref_positions = None

    z_score, z_score_robust, mu_used, sd_used, med_used, sigma_used = compute_z_scores(
        residuals,
        result,
        ref_residuals=ref_residuals,
        pred_positions=pred_positions,
        ref_positions=ref_positions,
        daily_period_steps=daily_period_steps,
        k_positions=k_positions,
    )

    result_df["z_score"] = z_score
    result_df["z_score_robust"] = z_score_robust
    result_df["is_anomaly_actual"] = anomalies
    result_df["is_anomaly_predicted"] = (
        np.abs(result_df["z_score"]) > threshold_z_score
    ).astype(int)
    result_df["is_anomaly_robust_predicted"] = (
        np.abs(result_df["z_score_robust"]) > threshold_z_score_robust
    ).astype(int)

    if save_detection_bands:
        # Translate the z-score / robust-z thresholds back into the actual-
        # value (Diff) domain. The residual scored is |actual - predicted|,
        # so the per-step residual threshold is (mu + threshold * sd) for
        # the classical band and (med + threshold * sigma) for the robust
        # band. Lower bands are clipped at 0 since Diff is non-negative.
        threshold_resid = mu_used + threshold_z_score * sd_used
        threshold_resid_robust = med_used + threshold_z_score_robust * sigma_used

        result_df["upper_band"] = predictions + threshold_resid
        result_df["lower_band"] = np.clip(predictions - threshold_resid, 0.0, None)
        result_df["upper_band_robust"] = predictions + threshold_resid_robust
        result_df["lower_band_robust"] = np.clip(
            predictions - threshold_resid_robust, 0.0, None
        )

    result['number_of_anomalies_actual'] = int(result_df["is_anomaly_actual"].sum())
    result['number_of_anomalies'] = int(result_df["is_anomaly_predicted"].sum())
    result['number_of_anomalies_robust'] = int(result_df["is_anomaly_robust_predicted"].sum())

    result['anomaly_indices'] = result_df.index[
        result_df["is_anomaly_predicted"] == 1
    ].tolist()
    result['anomaly_indices_robust'] = result_df.index[
        result_df["is_anomaly_robust_predicted"] == 1
    ].tolist()

    return result_df


# ===================================================================
# REMOVE METERS THAT COULD TAKE TOO LONG TO TRAIN
# ===================================================================
def can_long_run_length(train_samples, train_samples_filled, periodicity_seconds):
    '''
    Raises an error if current meter is expected to run too long
    This check is done by pretrained model
    '''
    with open("./train_time_tree.pkl", "rb") as f:
        clf = pickle.load(f)

    number_of_gaps = train_samples_filled - train_samples
    sgn_of_gaps = np.sign(number_of_gaps)
    percentage_of_gaps = np.abs(number_of_gaps / train_samples_filled)

    x_tree = pd.DataFrame(
        [{
            "percentage_of_gaps": percentage_of_gaps,
            "sgn_of_gaps": sgn_of_gaps,
            "periodicity_seconds": periodicity_seconds,
        }]
    )
    y_pred_tree = clf.predict(x_tree)[0]

    if y_pred_tree == 1:
        return True

    return False


# ============================================================================
# TRAIN MODEL
# ============================================================================
def train_model(df_train, result, freq_seasonal, stochastic_freq_seasonal,
                initial_train=True, init_state_mean=None, init_state_cov=None):
    """
    Fit a local-level + daily/weekly Fourier UC model to df_train.

    The UC `seasonal` argument is always None — both daily and weekly
    seasonalities are encoded via `freq_seasonal` (Fourier harmonics).
    """
    y_train = df_train["Diff"].values.astype(float)

    train_period = 'train' if initial_train else 'second'

    t_start_train = time.time()
    try:
        model_train = UnobservedComponents(
            endog=y_train,
            level="local level",  # local level only, no slope
            seasonal=None,        # daily/weekly handled via freq_seasonal
            freq_seasonal=freq_seasonal,
            stochastic_level=True,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
        )

        if init_state_mean is not None and init_state_cov is not None:
            model_train.initialize_known(
                initial_state=init_state_mean, initial_state_cov=init_state_cov
            )

        res_train = model_train.fit(disp=False)

    except Exception as e:
        raise ValueError(f"Failed to fit {train_period} training model: {e}")

    result[f"converged_{train_period}"] = res_train.mle_retvals['converged']
    if not res_train.mle_retvals['converged']:
        result[f'warnflag_{train_period}'] = res_train.mle_retvals['warnflag']
        result[f"gopt_{train_period}"] = res_train.mle_retvals['gopt']
    else:
        result[f'warnflag_{train_period}'] = ''
        result[f"gopt_train_{train_period}"] = ''

    t_end_train = time.time()
    result[f"{train_period}_train_time_seconds"] = t_end_train - t_start_train

    # Get filtered states at end of training segment
    filtered_state = res_train.filter_results.filtered_state
    filtered_cov = res_train.filter_results.filtered_state_cov
    last_state_mean = filtered_state[:, -1].copy()
    last_state_cov = filtered_cov[:, :, -1].copy()

    theta_train = res_train.params
    result[f"{train_period}_train_loglike"] = res_train.llf

    return last_state_mean, last_state_cov, theta_train


# ============================================================================
# PREDICT MODEL — BATCH (vectorized, used for reference residuals on df_second)
# ============================================================================
def predict_model_batch(df_predict, result, freq_seasonal, stochastic_freq_seasonal,
                        init_state_mean, init_state_cov, theta):
    """
    Vectorized Kalman filter pass over df_predict using fixed parameters theta.
    Used to obtain reference residuals from the clean second training segment.
    NaN observations are handled by statsmodels (measurement update skipped).
    """
    y_pred_segment = df_predict["Diff"].values.astype(float)

    t_start_pred = time.time()
    try:
        model_pred = UnobservedComponents(
            endog=y_pred_segment,
            level="local level",
            seasonal=None,
            freq_seasonal=freq_seasonal,
            stochastic_level=True,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
        )

        if init_state_mean is not None and init_state_cov is not None:
            model_pred.initialize_known(
                initial_state=init_state_mean, initial_state_cov=init_state_cov
            )

        res_pred = model_pred.filter(theta)
        pred_obj = res_pred.get_prediction()
        pred_mean = pred_obj.predicted_mean
        pred_mean = np.asarray(pred_mean, dtype=float)
        pred_mean = np.clip(pred_mean, 0.0, None)

    except Exception as e:
        raise ValueError(f"Failed to generate batch predictions: {e}")

    t_end_pred = time.time()
    result["second_prediction_time_seconds"] = t_end_pred - t_start_pred

    return pred_mean


# ============================================================================
# PREDICT MODEL — ONLINE (step-by-step, used for prediction segment)
# ============================================================================
def predict_model_online(df_predict, result, freq_seasonal, stochastic_freq_seasonal,
                         init_state_mean, init_state_cov, theta,
                         ref_residuals=None, ref_timestamps=None,
                         periodicity_seconds=None, daily_period_steps=None,
                         k_positions=None, z_threshold=3):
    """
    One-step-ahead prediction with online anomaly masking.

    At each time step the Kalman filter:
      1. Produces the prior predicted observation y_hat from the current state.
      2. Computes the residual |y_t - y_hat| and its z-score.
         - If reference data + position info is given, the z-score uses
           position-conditional statistics for this step's position on the
           daily cycle.
         - Otherwise it uses a single global mean/std from ref_residuals.
      3. If |z| > z_threshold OR y_t is NaN: skips the measurement update so
         the anomalous value does not corrupt the state carried forward.
      4. Otherwise: performs the standard Kalman measurement update.

    Parameters:
    -----------
    ref_residuals : array-like or None
        Residuals from the second training segment used to compute the
        reference mean and std (global, and optionally per-position).
    ref_timestamps : array-like or None
        Timestamps aligned with `ref_residuals`. Required (with
        `periodicity_seconds`, `daily_period_steps`, `k_positions`) for
        position-conditional thresholding.
    periodicity_seconds : float or None
        Sampling interval in seconds. Used to map timestamps to positions.
    daily_period_steps : int or None
        Number of measurements in 24 h. Used as the daily cycle length for
        position-conditional scoring.
    k_positions : int or None
        Half-width (in positions) of the neighborhood used to pool
        reference residuals around each position.
    z_threshold : float
        Z-score threshold above which an observation is treated as anomalous
        and excluded from the measurement update (default: 3).
    """
    y_pred_segment = df_predict["Diff"].values.astype(float)
    n = len(y_pred_segment)

    # ------------------------------------------------------------------
    # Pre-compute reference statistics. Two regimes:
    #   - global: single (ref_mean, ref_std) used for every step
    #   - position-conditional: per-position arrays, used per step
    # In both cases we also keep the globals as a fallback.
    # ------------------------------------------------------------------
    use_conditional = (
        ref_residuals is not None
        and ref_timestamps is not None
        and periodicity_seconds is not None
        and daily_period_steps is not None
        and k_positions is not None
    )

    means_per_pos = None
    stds_per_pos = None
    pred_positions = None
    global_mean = None
    global_std = None

    if ref_residuals is not None:
        ref_arr = np.asarray(ref_residuals, dtype=float)
        valid_ref = ref_arr[~np.isnan(ref_arr)]
        global_mean = float(np.nanmean(valid_ref)) if valid_ref.size > 0 else 0.0
        global_std = float(np.nanstd(valid_ref, ddof=1)) if valid_ref.size > 1 else None

        if use_conditional:
            ref_positions = compute_position_in_period(
                ref_timestamps, periodicity_seconds, daily_period_steps
            )
            pred_positions = compute_position_in_period(
                df_predict["timestamp_utc"].values,
                periodicity_seconds, daily_period_steps,
            )
            means_per_pos, stds_per_pos, _, _ = compute_per_position_stats(
                ref_arr, ref_positions, daily_period_steps, k_positions
            )
            result["ref_mean_per_position"] = means_per_pos.tolist()
            result["ref_std_per_position"] = stds_per_pos.tolist()
            result["k_positions"] = k_positions
            result["ref_mean"] = global_mean
            result["ref_std"] = global_std
        else:
            result['ref_mean'] = global_mean
            result['ref_std'] = global_std
    else:
        result['ref_mean'] = None
        result['ref_std'] = None

    t_start_pred = time.time()
    try:
        # Build the model with a dummy endog to extract system matrices.
        # We only need the matrices, not a fit — update(theta) populates them.
        model_pred = UnobservedComponents(
            endog=np.zeros(n),
            level="local level",
            seasonal=None,
            freq_seasonal=freq_seasonal,
            stochastic_level=True,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
        )
        model_pred.update(theta)

        # Statsmodels stores time-invariant matrices as either 2D (k, k)
        # or 3D (k, k, nobs). This helper handles both cases.
        def get_mat(name):
            m = model_pred.ssm[name]
            return m[:, :, 0] if m.ndim == 3 else m

        # T : state transition          (k_states, k_states)
        # Z : observation/design        (1, k_states)
        # R : selection                 (k_states, k_posdef)
        # Q : state noise cov           (k_posdef, k_posdef)
        # H : observation noise cov     (1, 1)
        T = get_mat('transition')
        Z = get_mat('design')
        R = get_mat('selection')
        Q = get_mat('state_cov')
        H = get_mat('obs_cov')

        # Full process noise covariance: R @ Q @ R.T
        Q_full = R @ Q @ R.T
        k = T.shape[0]
        I = np.eye(k)

        # Initialise state from the end of the second training segment
        x = init_state_mean.copy().reshape(-1)
        P = init_state_cov.copy()

        predictions = np.full(n, np.nan)

        for t in range(n):
            # ── Prediction step ──────────────────────────────────────────
            x_prior = T @ x
            P_prior = T @ P @ T.T + Q_full

            y_hat = float(Z @ x_prior)
            predictions[t] = max(y_hat, 0.0)   # clip negatives

            y_t = y_pred_segment[t]

            # ── Decide whether to update or skip ─────────────────────────
            skip_update = bool(np.isnan(y_t))

            if not skip_update:
                # Pick mean/std for this step. In conditional mode, use the
                # position-specific stats; fall back to globals if the
                # position has no data or zero variability.
                if use_conditional and pred_positions is not None:
                    p = pred_positions[t]
                    mu = means_per_pos[p]
                    sd = stds_per_pos[p]
                    if np.isnan(mu) or np.isnan(sd) or sd == 0:
                        mu, sd = global_mean, global_std
                else:
                    mu, sd = global_mean, global_std

                if sd is not None and sd > 0:
                    residual = abs(y_t - y_hat)
                    z = (residual - mu) / sd
                    if abs(z) > z_threshold:
                        skip_update = True

            # ── Measurement update ────────────────────────────────────────
            if skip_update:
                # Anomalous or missing: propagate state without updating
                x = x_prior
                P = P_prior
            else:
                S = float(Z @ P_prior @ Z.T + H)        # innovation variance
                K = (P_prior @ Z.T) / S                 # Kalman gain (k, 1)
                innovation = y_t - y_hat
                x = x_prior + K.flatten() * innovation
                # Joseph form for numerical stability
                IKZ = I - K @ Z
                P = IKZ @ P_prior @ IKZ.T + K * H[0, 0] @ K.T

    except Exception as e:
        raise ValueError(f"Failed to generate online predictions: {e}")

    t_end_pred = time.time()
    result["prediction_time_seconds"] = t_end_pred - t_start_pred

    return predictions


def split_df_sliding_weeks(
    df_raw,
    seed,
    result,
    total_weeks=6,
    min_days_per_week=7,
    min_days_with_data_per_day=1,
    min_periodicity_month_train=20
):
    """
    Split df into 3 overlapping chunks on a random 6-week window:
    - Weeks 1–4: initial training
    - Weeks 2–5: second training
    - Week 6: prediction (and anomaly injection)

    Constraints:
    - The chosen 6-week window must have at least one measurement per day.

    Also computes and stores `daily_period_steps` in `result` — the number
    of measurements that fit into 24 h at the inferred periodicity. This
    is used both for the Fourier daily/weekly periods and for the
    position-conditional z-score.
    """

    df = df_raw.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    min_ts = df["timestamp_utc"].min()
    max_ts = df["timestamp_utc"].max()

    total_days_required = 7 * total_weeks
    if max_ts - min_ts < pd.Timedelta(days=total_days_required):
        raise ValueError(
            f"Not enough data. Need at least {total_days_required} days "
            f"of data to form a {total_weeks}-week window."
        )

    # Pre-compute per-day counts to help check gaps quickly
    df["date"] = df["timestamp_utc"].dt.floor("D")
    daily_counts = df.groupby("date").size().rename("count").reset_index()

    # Helper to check if a [start, end) window has at least one measurement per day
    def window_has_no_big_gaps(start_ts, end_ts):
        window_days = pd.date_range(
            start=start_ts.floor("D"),
            end=(end_ts - pd.Timedelta(seconds=1)).floor("D"),
            freq="D",
        )
        df_counts_window = daily_counts[
            (daily_counts["date"] >= window_days.min())
            & (daily_counts["date"] <= window_days.max())
        ]

        df_counts_full = (
            pd.DataFrame({"date": window_days})
            .merge(df_counts_window, on="date", how="left")
            .fillna({"count": 0})
        )

        return (df_counts_full["count"] >= min_days_with_data_per_day).all()

    rng = default_rng(seed)

    max_start = max_ts - pd.Timedelta(days=total_days_required)
    max_tries = 30
    chosen_start = None

    for _ in range(max_tries):
        u = rng.random()
        rand_start = min_ts + (max_start - min_ts) * u
        rand_start = pd.to_datetime(rand_start)
        rand_start = rand_start.floor("D")

        rand_end = rand_start + pd.Timedelta(days=total_days_required)

        if rand_end > max_ts:
            continue

        if window_has_no_big_gaps(rand_start, rand_end):
            chosen_start = rand_start
            break

    if chosen_start is None:
        raise ValueError(
            f"Could not find a {total_weeks}-week window with at least one "
            f"measurement per day after {max_tries} tries."
        )

    chosen_end = chosen_start + pd.Timedelta(days=total_days_required)

    df_6w = df[
        (df["timestamp_utc"] >= chosen_start)
        & (df["timestamp_utc"] < chosen_end)
    ].copy().reset_index(drop=True)

    df_predict_unresampled = df_6w[
        (df_6w["timestamp_utc"] >= (chosen_end - pd.Timedelta(weeks=1)))
        & (df_6w["timestamp_utc"] < chosen_end)
    ].copy().reset_index(drop=True)

    # Get periodicity
    df_6w, diag = fill_gaps_with_periodicity_adaptive(
        df_6w, timestamp_col="timestamp_utc")

    periodicity_seconds = diag["periodicity_used_seconds"]
    result["df_samples_filled"] = len(df_6w)
    result["periodicity_seconds"] = periodicity_seconds

    if periodicity_seconds == 0:
        raise ValueError("Periodicity is equal to 0")

    # Daily cycle length in measurements — used for Fourier periods and for
    # the position-conditional z-score.
    daily_period_steps = int(round(24 * 3600 / periodicity_seconds))
    result["daily_period_steps"] = daily_period_steps

    weeks_train = 4
    if periodicity_seconds < min_periodicity_month_train * 60:
        weeks_train = 2

    # Week for prediction
    start_pred = chosen_end - pd.Timedelta(weeks=1)
    end_pred = chosen_end

    # Weeks for second training
    start_second = start_pred - pd.Timedelta(weeks=weeks_train)
    end_second = start_pred

    # Weeks for initial training
    start_train = start_second - pd.Timedelta(weeks=1)
    end_train = end_second - pd.Timedelta(weeks=1)

    if start_train < chosen_start:
        raise ValueError(
            f"Choosen 6 weeks segment starts on {chosen_start} "
            f"but initial start want to start earlier then that at {start_train}"
        )

    # Slice dataframes
    df_train = df_6w[
        (df_6w["timestamp_utc"] >= start_train)
        & (df_6w["timestamp_utc"] < end_train)
    ].copy().reset_index(drop=True)

    df_second = df_6w[
        (df_6w["timestamp_utc"] >= start_second)
        & (df_6w["timestamp_utc"] < end_second)
    ].copy().reset_index(drop=True)

    df_predict = df_6w[
        (df_6w["timestamp_utc"] >= start_pred)
        & (df_6w["timestamp_utc"] < end_pred)
    ].copy().reset_index(drop=True)

    result["train_samples"] = len(df_train)
    result["second_train_samples"] = len(df_second)
    result["predict_samples"] = len(df_predict)
    result["train_predict_window_days"] = total_days_required

    if len(df_train) < 2 or len(df_second) < 2 or len(df_predict) < 2:
        raise ValueError(
            "Insufficient data in one of the 6-week subsegments "
            "(train/second/predict) for modeling."
        )

    return df_train, df_second, df_predict, df_predict_unresampled


# ============================================================================
# MAIN PROCESSING FUNCTION FOR ONE METER
# ============================================================================


def process_single_meter(
    csv_filepath: str,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    predictions_output_csv: Optional[str] = None,
    threshold_z_score: int = 3,
    threshold_z_score_robust: int = 3,
    k_hours: float = 2.0,
    min_k_positions: int = 2,
    save_detection_bands: bool = False,
) -> Dict:
    """
    Process a single water meter CSV file with UnobservedComponents
    (local level + daily/weekly Fourier seasonality).

    Splitting logic (time-based, sliding window):
    - Choose a random 6-week window with at least one measurement per day.
    - Weeks 1–4 of that window: initial training.
    - Weeks 2–5 of that window: second training (initialized from previous state).
    - Week 6 of that window: prediction (initialized from previous state,
      anomalies injected only here).

    Anomaly scoring:
    - Thresholding statistics are estimated from the second training segment
      residuals and applied to score the prediction segment residuals.
    - Statistics are computed *per position on the daily cycle* using a
      ±k_positions neighborhood (k_positions derived from `k_hours` of real
      time, with a floor of `min_k_positions`). Nights where the model
      predicts well get a tighter band; daytime hours with larger errors
      get a wider band.

    Parameters:
    -----------
    csv_filepath : str
        Path to meter CSV file.
    device : torch.device, optional
        Device (kept for interface compatibility, not used).
    verbose : bool
        Enable verbose logging.
    predictions_output_csv : str, optional
        Path to save prediction results.
    threshold_z_score : int
        Threshold for the classical z-score (default: 3).
    threshold_z_score_robust : int
        Threshold for the robust MAD-based score (default: 3).
    k_hours : float
        Half-width of the position-neighborhood, in real hours (default: 2.0).
    min_k_positions : int
        Minimum half-width in positions, used as a floor for coarse-frequency
        sensors (default: 2).
    save_detection_bands : bool
        If True, the prediction DataFrame gains `upper_band`, `lower_band`,
        `upper_band_robust`, `lower_band_robust` columns (in Diff units),
        useful for plotting the detection envelope around the prediction.

    Returns:
    --------
    dict : Result dictionary with metadata and metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {
        "filename": Path(csv_filepath).stem,
        "filepath": csv_filepath,
        "status": "processing",
        "error": None,
    }

    try:
        if verbose:
            logger.info(f"Processing {result['filename']}...")

        df_raw = pd.read_csv(csv_filepath)

        if "timestamp_utc" not in df_raw.columns:
            raise ValueError(
                f"No timestamp_utc column found. Available: {df_raw.columns.tolist()}"
            )

        if "hodnota" not in df_raw.columns:
            raise ValueError(
                f"No 'hodnota' column found. Available: {df_raw.columns.tolist()}"
            )

        if len(df_raw) < 2:
            raise ValueError("DataFrame passed is too short len < 2")

        df_raw["timestamp_utc"] = pd.to_datetime(df_raw["timestamp_utc"], utc=True)
        df_raw = df_raw.copy()
        df_raw.dropna(subset=["timestamp_utc"], inplace=True)

        if verbose:
            logger.info(f"Processing {result['filename']} done, starting resampling...")

        # ===================================================================
        # RESAMPLE and SPLIT into 3 CHUNKS
        # ===================================================================
        seed = int(result['filename'])
        df_train, df_second, df_predict, df_predict_unresampled = split_df_sliding_weeks(
            df_raw=df_raw, seed=seed, result=result
        )
        periodicity_seconds = result["periodicity_seconds"]
        daily_period_steps = result["daily_period_steps"]

        if daily_period_steps < 2:
            raise ValueError(
                f"Daily period in measurements is < 2 (periodicity too coarse)."
            )

        # ---------------------------------------------------------------
        # Daily + weekly Fourier seasonality (UC `seasonal` is always None).
        # ---------------------------------------------------------------
        weekly_period_steps = 7 * daily_period_steps
        freq_seasonal = [
            {"period": weekly_period_steps, "harmonics": 2},  # weekly
            {"period": daily_period_steps,  "harmonics": 2},  # daily
        ]
        stochastic_freq_seasonal = [True, True]

        # Half-width of the position-neighborhood used by the conditional
        # z-score. ±k_hours of real time, converted to positions, floored
        # at `min_k_positions` so very coarse sensors still pool enough data.
        k_positions = compute_k_positions(
            daily_period_steps, k_hours=k_hours, min_k=min_k_positions
        )
        result["k_positions"] = k_positions
        result["k_hours"] = k_hours

        if verbose:
            logger.info(f"daily_period_steps={daily_period_steps}, k_positions={k_positions}")
            logger.info(f"Resampling {result['filename']} done, starting init training...")

        # ===================================================================
        # INITIAL TRAINING
        # ===================================================================
        last_state_mean, last_state_cov, _ = train_model(
            df_train=df_train,
            result=result,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            initial_train=True,
        )

        if verbose:
            logger.info(f"Init training done {result['filename']}, starting second training...")

        # ===================================================================
        # SECOND TRAINING
        # ===================================================================
        init_second_mean = last_state_mean
        init_second_cov = last_state_cov

        last_state_mean_2, last_state_cov_2, theta_second = train_model(
            df_train=df_second,
            result=result,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            initial_train=False,
            init_state_mean=init_second_mean,
            init_state_cov=init_second_cov,
        )

        if verbose:
            logger.info(f"Second training done {result['filename']}, computing reference residuals...")

        # ===================================================================
        # COMPUTE REFERENCE RESIDUALS FROM SECOND TRAINING SEGMENT
        # These residuals are used to estimate thresholding statistics
        # (mean, std, median, MAD) for anomaly scoring — both global and
        # per-position — ensuring the calibration window is anomaly-free.
        # ===================================================================
        second_predictions = predict_model_batch(
            df_predict=df_second,
            result=result,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            init_state_mean=init_second_mean,   # state from end of initial training
            init_state_cov=init_second_cov,
            theta=theta_second,
        )
        second_actuals = df_second["Diff"].values.astype(float)
        second_residuals = np.abs(second_actuals - second_predictions)

        if verbose:
            logger.info(f"Reference residuals computed for {result['filename']}, injecting anomalies...")

        # ===================================================================
        # Inject anomalies to prediction df
        # ===================================================================
        df_predict = inject_spike_anomalies_diff(df_predict, random_state=int(result['filename']))

        if verbose:
            logger.info(f"Injection done {result['filename']}, predicting...")

        # ===================================================================
        # PREDICTION
        # ===================================================================
        init_predict_mean = last_state_mean_2
        init_predict_cov = last_state_cov_2

        predictions_mean = predict_model_online(
            df_predict=df_predict,
            result=result,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            init_state_mean=init_predict_mean,
            init_state_cov=init_predict_cov,
            theta=theta_second,
            ref_residuals=second_residuals,
            ref_timestamps=df_second["timestamp_utc"].values,
            periodicity_seconds=periodicity_seconds,
            daily_period_steps=daily_period_steps,
            k_positions=k_positions,
            z_threshold=threshold_z_score,
        )

        # Build results dataframe: scores applied to prediction residuals,
        # but statistics estimated from second training segment residuals,
        # per position on the daily cycle.
        predictions_df = build_results_df(
            df_predict,
            predictions_mean,
            result,
            ref_residuals=second_residuals,
            ref_timestamps=df_second["timestamp_utc"].values,
            periodicity_seconds=periodicity_seconds,
            daily_period_steps=daily_period_steps,
            k_positions=k_positions,
            threshold_z_score=threshold_z_score,
            threshold_z_score_robust=threshold_z_score_robust,
            save_detection_bands=save_detection_bands,
        )

        result["z_scores"] = predictions_df["z_score"].tolist()

        if predictions_output_csv is not None:
            predictions_df.to_csv(predictions_output_csv, index=False)

        if verbose:
            logger.info(f"Prediction done {result['filename']}, calculating metrics...")

        # ===================================================================
        # METRICS
        # ===================================================================
        if len(predictions_df) < 2 or len(df_predict_unresampled) < 2:
            raise ValueError(
                "Less than 2 data points in one of the prediction dfs for metric calculation."
            )

        metrics = calculate_metrics(predictions_df)

        for key, value in metrics.items():
            result[f"metric_{key}"] = value

        result["status"] = "success"

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Failed: {result['filename']} - Error: {e}")
        if verbose:
            logger.debug(traceback.format_exc())
        return result


# ============================================================================
# BATCH PROCESSING WITH MULTITHREADING
# ============================================================================


def process_batch(
    csv_filepaths: List[str],
    output_csv: str,
    num_workers: int = 4,
    verbose: bool = False,
    per_file_timeout: int = 120,  # seconds; tune as needed (e.g. 300, 900)
):
    """
    Process multiple water meter CSV files in parallel with UnobservedComponents.
    Uses ProcessPoolExecutor so that hung meters can be truly killed via timeout.

    Parameters:
    -----------
    csv_filepaths : list of str
        Paths to meter CSV files
    output_csv : str
        Path to save aggregated results
    num_workers : int
        Number of parallel workers
    verbose : bool
        Enable verbose logging
    per_file_timeout : int
        Per-file timeout in seconds
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device (for info only): {device}")
        logger.info(
            f"Processing {len(csv_filepaths)} files with {num_workers} workers"
        )
        logger.info(f"Per-file timeout: {per_file_timeout} seconds")

    all_results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        for filepath in csv_filepaths:
            filename = Path(filepath).stem

            future = executor.submit(
                process_single_meter,
                filepath,
                device=None,
                verbose=verbose,
            )
            futures[future] = filename

        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Processing meters"),
            1,
        ):
            filename = futures[future]
            try:
                result = future.result(timeout=per_file_timeout)
                all_results.append(result)
                if verbose:
                    logger.info(f"[{i}/{len(futures)}] Completed {filename}")

            except TimeoutError:
                logger.error(
                    f"[{i}/{len(futures)}] Timeout when processing {filename} "
                    f"(>{per_file_timeout}s). Marking as failed and cancelling."
                )
                future.cancel()
                all_results.append(
                    {
                        "filename": filename,
                        "status": "failed",
                        "error": f"Timeout after {per_file_timeout}s",
                    }
                )
            except Exception as e:
                logger.error(f"[{i}/{len(futures)}] Failed to process {filename}: {e}")
                all_results.append(
                    {
                        "filename": filename,
                        "status": "failed",
                        "error": str(e),
                    }
                )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)

    # save flat z-score file
    zscore_rows = []
    for r in all_results:
        if r.get("status") == "success" and "z_scores" in r:
            for z in r["z_scores"]:
                zscore_rows.append({"filename": r["filename"], "z_score": z})

    pd.DataFrame(zscore_rows).to_csv(output_csv.replace(".csv", "_zscores.csv"), index=False)

    logger.info("=" * 70)
    logger.info("BATCH PROCESSING COMPLETE (Daily/Weekly Fourier UC)")
    logger.info("=" * 70)
    logger.info(f"Total processed: {len(results_df)}")
    logger.info(f"Successful: {(results_df['status'] == 'success').sum()}")
    logger.info(f"Failed: {(results_df['status'] == 'failed').sum()}")
    logger.info(f"Metrics saved to: {output_csv}")

    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Daily/Weekly Fourier UnobservedComponents Batch Processor for Water Meter Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--workers", type=int, default=7, help="Number of parallel workers (default: 7)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (default: False)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of random meters to process (default: 1000)",
    )

    args = parser.parse_args()

    global logger
    logger = setup_logging(verbose=args.verbose)

    # Data directory
    directory = "../../../data/sensor_data"
    seed_value = 42

    #all_files = [
    #    os.path.join(directory, f)
    #    for f in os.listdir(directory)
    #    if os.path.isfile(os.path.join(directory, f))
    #]
    #random.seed(seed_value)
    #
    #csv_filepaths = random.sample(all_files, min(args.samples, len(all_files)))

    with open("../../pickles/test_set.pkl", "rb") as f:
        csv_filenames = pickle.load(f)

    #with open("../pickles/common_sensors.pkl", "rb") as f:
    #    csv_filepaths = pickle.load(f)
    
    csv_filepaths = [os.path.join(directory, name) for name in csv_filenames]

    logger.info(f"Loaded {len(csv_filepaths)} CSV filepaths")

    output_csv = f"./results/6_weeks_results_seasonal_uc_{args.samples}_seed_42_clipped_tree_timeout_120_daily_weekly_fourier_z_score_adaptive_test_anomalies.csv"

    _ = process_batch(
        csv_filepaths,
        output_csv,
        num_workers=args.workers,
        verbose=args.verbose,
    )

    logger.info(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()
