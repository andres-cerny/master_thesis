"""
Unobserved Components (Local Level + Seasonal) Batch Processor for Water Meter Anomaly Detection
=================================================================================================

Processes multiple water meter CSV files with:
- Multithreaded execution
- Configurable periodicity
- Resampling & gap filling using fill_gaps_with_periodicity_adaptive
- UnobservedComponents model with local level (no slope) + seasonal component
- Initial training on 3–2 months ago data
- Second training on 2–1 months ago data, initialized from previous month state
- Prediction on last month of data, initialized from previous month state
- Comprehensive metrics calculation
- Proper handling of NaN values in Diff


Usage:
    python seasonal_batch_processor.py --workers 7 --verbose


Notes on NaN handling:
- NaN values in Diff are preserved and handled by the Kalman filter (skips update step)
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

from calculate_metrics import calculate_metrics, calculate_metrics_unresampled
from create_anomalies import inject_spike_anomalies_diff


import warnings
from statsmodels.tools.sm_exceptions import SpecificationWarning
from scipy.sparse import SparseEfficiencyWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=SpecificationWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================================
# TIMEOUT ERROR SETUP
# ============================================================================
#class FitTimeoutError(TimeoutError):
#    pass
#
#@contextmanager
#def time_limit(seconds: int):
#    def handler(signum, frame):
#        raise FitTimeoutError(f"Model fitting exceeded {seconds} seconds")
#
#    old_handler = signal.signal(signal.SIGALRM, handler)
#    signal.alarm(seconds)
#    try:
#        yield
#    finally:
#        signal.alarm(0)
#        signal.signal(signal.SIGALRM, old_handler)


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
# HELPER: DETECT SEASONAL PERIOD FROM PERIODICITY
# ============================================================================


def calculate_seasonal_period(
    periodicity_seconds: float,
    seasonal_cycle: str = "daily"
) -> int:
    """
    Convert a seasonal cycle (e.g., 'daily', 'weekly') to number of steps
    based on the periodicity of the resampled data.

    Parameters:
    -----------
    periodicity_seconds : float
        Base interval in seconds (e.g., 1800 for 30 minutes)
    seasonal_cycle : str
        One of 'daily', 'weekly', or a number of seconds (parsed as float)

    Returns:
    --------
    int : Number of time steps per seasonal cycle
    """
    if isinstance(seasonal_cycle, str):
        if seasonal_cycle.lower() == "daily":
            cycle_seconds = 24 * 3600
        elif seasonal_cycle.lower() == "weekly":
            cycle_seconds = 7 * 24 * 3600
        else:
            try:
                cycle_seconds = float(seasonal_cycle)
            except ValueError:
                logger.warning(
                    f"Unknown seasonal_cycle '{seasonal_cycle}', defaulting to daily"
                )
                cycle_seconds = 24 * 3600
    else:
        cycle_seconds = float(seasonal_cycle)

    seasonal_period = int(np.round(cycle_seconds / periodicity_seconds))
    
    return seasonal_period


# ============================================================================
# HELPER: RESIDUAL-BASED ANOMALIES
# ============================================================================

import numpy as np

def compute_z_scores(residuals, result, ref_residuals=None):
    """
    Compute z-score and robust MAD-based score for each residual.

    Statistics (mean, std, median, MAD) are estimated from ref_residuals
    (intended to be the second training segment residuals) and then applied
    to score residuals (intended to be the prediction segment residuals).
    If ref_residuals is None, residuals itself is used as the reference,
    which matches the old behaviour.

    Parameters:
    -----------
    residuals : array-like
        Residuals to be scored (prediction segment).
    ref_residuals : array-like or None
        Reference residuals used to estimate thresholding statistics
        (second training segment). If None, falls back to residuals.

    Returns:
    --------
    z_score : np.ndarray
        Classical z-scores for each element of residuals.
    z_score_robust : np.ndarray
        Robust MAD-based scores for each element of residuals.
    """
    # Ensure float numpy arrays
    residuals = np.asarray(residuals, dtype=float)

    # Use ref_residuals for statistics if provided, otherwise fall back to residuals
    if ref_residuals is not None:
        ref = np.asarray(ref_residuals, dtype=float)
    else:
        ref = residuals

    # Mask valid (non-NaN) values
    valid_ref = ref[~np.isnan(ref)]
    valid = ~np.isnan(residuals)

    # Initialize outputs
    z_score = np.full_like(residuals, np.nan)
    z_score_robust = np.full_like(residuals, np.nan)

    # If no usable reference residuals -> all NaN
    if valid_ref.size == 0:
        return z_score, z_score_robust

    # ---------- Classic mean/std z-score ----------
    # Statistics estimated from the reference (second training) segment
    res_mean = np.nanmean(valid_ref)
    res_std = np.nanstd(valid_ref, ddof=1)
    
    result["ref_mean"] = res_mean
    result["ref_std"] = res_std

    if not (np.isnan(res_std) or res_std == 0):
        z_score[valid] = (residuals[valid] - res_mean) / res_std
    else:
        # No variability: treat all valid points as typical
        z_score[valid] = 0.0

    # ---------- Robust median/MAD z-score ----------
    # Statistics estimated from the reference (second training) segment
    median_resid = np.nanmedian(valid_ref)
    mad = np.nanmedian(np.abs(valid_ref - median_resid))

    # Fallback to classic std if MAD unusable
    if np.isnan(mad) or mad == 0:
        sigma_robust = res_std if not (np.isnan(res_std) or res_std == 0) else np.nan
    else:
        sigma_robust = 1.4826 * mad

    if np.isnan(sigma_robust) or sigma_robust == 0:
        # No variability: all valid z_robust = 0
        z_score_robust[valid] = 0.0
    else:
        z_score_robust[valid] = (residuals[valid] - median_resid) / sigma_robust

    return z_score, z_score_robust



def build_results_df(df_predict, predictions, result, ref_residuals=None, threshold_z_score=3, threshold_z_score_robust=3):
    """
    Build a result DataFrame with anomaly scores based on residuals.

    Handles NaN values properly by masking them before rolling calculations.

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
        thresholding statistics (mean, std, median, MAD). If None,
        the prediction segment residuals are used instead (old behaviour).
    threshold_z_score : int
        Threshold for the classical z-score (default: 3).
    threshold_z_score_robust : int
        Threshold for the robust MAD-based score (default: 3).

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
            "residual": residuals
        }
    )
    
    if anomalies is not None:
        # Scores are computed on prediction residuals, but thresholding statistics
        # (mean, std, median, MAD) are estimated from ref_residuals (second training segment)
        z_score, z_score_robust = compute_z_scores(residuals, result, ref_residuals=ref_residuals)
        result_df["z_score"] = z_score
        result_df["z_score_robust"] = z_score_robust
        result_df["is_anomaly_actual"] = anomalies
            
        result_df["is_anomaly_predicted"] = (np.abs(result_df["z_score"]) > threshold_z_score).astype(int)
        result_df["is_anomaly_robust_predicted"] = (np.abs(result_df["z_score_robust"]) > threshold_z_score_robust).astype(int)

        result['number_of_anomalies_actual'] = result_df["is_anomaly_actual"].sum()
        result['number_of_anomalies'] = result_df["is_anomaly_predicted"].sum()
        result['number_of_anomalies_robust'] = result_df["is_anomaly_robust_predicted"].sum()

        result['anomaly_indices'] = result_df.index[result_df["is_anomaly_predicted"] == 1].tolist()
        result['anomaly_indices_robust'] = result_df.index[result_df["is_anomaly_robust_predicted"] == 1].tolist()
    
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
def train_model(df_train, seasonal_period_steps, result, initial_train=True, init_state_mean=None, init_state_cov=None, freq_seasonal=None, stochastic_freq_seasonal=None):
    
    y_train = df_train["Diff"].values.astype(float)
    
    train_period = 'train'
    if not initial_train:
        train_period = 'second'
        
    t_start_train = time.time()
    try:
        model_train = UnobservedComponents(
            endog=y_train,
            level="local level",  # local level only, no slope
            seasonal=seasonal_period_steps,
            freq_seasonal=freq_seasonal,
            stochastic_level=True,
            stochastic_seasonal=True,
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
def predict_model_batch(df_predict, seasonal_period_steps, result, init_state_mean, init_state_cov, theta, freq_seasonal=None, stochastic_freq_seasonal=None):
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
            seasonal=seasonal_period_steps,
            freq_seasonal=freq_seasonal,
            stochastic_level=True,
            stochastic_seasonal=True,
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
def predict_model_online(df_predict, seasonal_period_steps, result, init_state_mean, init_state_cov, theta,
                         freq_seasonal=None, stochastic_freq_seasonal=None,
                         ref_residuals=None, z_threshold=3):
    """
    One-step-ahead prediction with online anomaly masking.

    At each time step the Kalman filter:
      1. Produces the prior predicted observation y_hat from the current state.
      2. Computes the residual |y_t - y_hat| and its z-score using statistics
         estimated from ref_residuals (second training segment).
      3. If |z| > z_threshold OR y_t is NaN: skips the measurement update so
         the anomalous value does not corrupt the state carried forward.
      4. Otherwise: performs the standard Kalman measurement update.

    This mirrors production behaviour where anomalous observations are not
    incorporated into the state, preventing a single spike from degrading
    predictions at subsequent time steps.

    Parameters:
    -----------
    ref_residuals : array-like or None
        Residuals from the second training segment used to compute the
        reference mean and std for online z-score thresholding.
        If None, masking is only applied to NaN observations.
    z_threshold : float
        Z-score threshold above which an observation is treated as anomalous
        and excluded from the measurement update (default: 3).
    """
    y_pred_segment = df_predict["Diff"].values.astype(float)
    n = len(y_pred_segment)

    # Pre-compute reference statistics from second training residuals
    if ref_residuals is not None:
        ref = np.asarray(ref_residuals, dtype=float)
        valid_ref = ref[~np.isnan(ref)]
        ref_mean = float(np.nanmean(valid_ref)) if valid_ref.size > 0 else 0.0
        ref_std  = float(np.nanstd(valid_ref, ddof=1)) if valid_ref.size > 1 else None
    else:
        ref_mean = None
        ref_std  = None
        
    result['ref_mean'] = ref_mean
    result['ref_std'] = ref_std

    t_start_pred = time.time()
    try:
        # Build the model with a dummy endog to extract system matrices.
        # We only need the matrices, not a fit — update(theta) populates them.
        model_pred = UnobservedComponents(
            endog=np.zeros(n),
            level="local level",
            seasonal=seasonal_period_steps,
            freq_seasonal=freq_seasonal,
            stochastic_level=True,
            stochastic_seasonal=True,
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

            if not skip_update and ref_std is not None and ref_std > 0:
                residual = abs(y_t - y_hat)
                z = (residual - ref_mean) / ref_std
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
    
    seasonal_period_steps = calculate_seasonal_period(periodicity_seconds)
    result["seasonal_period_steps"] = seasonal_period_steps
    
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
    seasonal_cycle: str = "daily",
    device: Optional[torch.device] = None,
    verbose: bool = False,
    predictions_output_csv: Optional[str] = None,
    weekly_seasonality: bool = True,
    daily_steps: bool = False,
    threshold_z_score: int = 3,
    threshold_z_score_robust: int = 3
) -> Dict:
    """
    Process a single water meter CSV file with UnobservedComponents
    (local level + seasonal model, no slope).

    Splitting logic (time-based, sliding window):
    - Choose a random 6-week window with at least one measurement per day.
    - Weeks 1–4 of that window: initial training.
    - Weeks 2–5 of that window: second training (initialized from previous state).
    - Week 6 of that window: prediction (initialized from previous state, anomalies injected only here).

    Anomaly scoring:
    - Thresholding statistics (mean, std, median, MAD) are estimated from the
      second training segment residuals and then applied to score the prediction
      segment residuals, so that the calibration window is anomaly-free.

    Parameters:
    -----------
    csv_filepath : str
        Path to meter CSV file
    seasonal_cycle : str
        One of 'daily', 'weekly', or float seconds (default: 'daily')
    device : torch.device, optional
        Device (kept for interface compatibility, not used)
    verbose : bool
        Enable verbose logging
    predictions_output_csv : str, optional
        Path to save prediction results

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
        df_train, df_second, df_predict, df_predict_unresampled = split_df_sliding_weeks(df_raw=df_raw, seed=seed, result=result)
        seasonal_period_steps = result["seasonal_period_steps"]

        if seasonal_period_steps < 2:
            raise ValueError(f'Number of steps in a season is smaller then 2.')

        result["seasonal_period_steps"] = seasonal_period_steps

        if weekly_seasonality:
            weekly_period_steps = 7 * seasonal_period_steps
            freq_seasonal = [
                {
                    "period": weekly_period_steps,
                    "harmonics": 2,
                },
            ]
            stochastic_freq_seasonal = [True]
            
            if not daily_steps:
                freq_seasonal.append({
                    "period": seasonal_period_steps,
                    "harmonics": 2,
                })
                stochastic_freq_seasonal.append(True)
                seasonal_period_steps = None
        else:
            freq_seasonal = None
            stochastic_freq_seasonal = None
            
        if verbose:
            logger.info(f"{result['seasonal_period_steps']} and {len(df_train)}")
            logger.info(f"Resampling {result['filename']} done, starting init training...")
            
        # ===================================================================
        # INITIAL TRAINING
        # ===================================================================
        last_state_mean, last_state_cov, _ = train_model(df_train=df_train,
                                                         seasonal_period_steps=seasonal_period_steps,
                                                         result=result,
                                                         initial_train=True,
                                                         freq_seasonal=freq_seasonal,
                                                         stochastic_freq_seasonal=stochastic_freq_seasonal,
                                                         )
        
        if verbose:
            logger.info(f"Init training done {result['filename']} done, starting init second train...")

        # ===================================================================
        # SECOND TRAINING
        # ===================================================================
        init_second_mean = last_state_mean
        init_second_cov = last_state_cov
        
        last_state_mean_2, last_state_cov_2, theta_second = train_model(df_train=df_second,
                                                                        seasonal_period_steps=seasonal_period_steps,
                                                                        result=result,
                                                                        initial_train=False,
                                                                        init_state_mean=init_second_mean,
                                                                        init_state_cov=init_second_cov,
                                                                        freq_seasonal=freq_seasonal,
                                                                        stochastic_freq_seasonal=stochastic_freq_seasonal,
                                                                        )
        if verbose:
            logger.info(f"Second training done {result['filename']}, computing reference residuals...")

        # ===================================================================
        # COMPUTE REFERENCE RESIDUALS FROM SECOND TRAINING SEGMENT
        # These residuals are used to estimate thresholding statistics
        # (mean, std, median, MAD) for anomaly scoring, ensuring the
        # calibration window is anomaly-free.
        # ===================================================================
        second_predictions = predict_model_batch(
            df_predict=df_second,
            seasonal_period_steps=seasonal_period_steps,
            result=result,
            init_state_mean=init_second_mean,   # state from end of initial training
            init_state_cov=init_second_cov,
            theta=theta_second,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
        )
        second_actuals = df_second["Diff"].values.astype(float)
        second_residuals = np.abs(second_actuals - second_predictions)

        if verbose:
            logger.info(f"Reference residuals computed for {result['filename']}, injecting anomalies...")

        # ===================================================================
        # Inject anomalies to prediction df 
        # ===================================================================
        #df_predict = inject_spike_anomalies_diff(df_predict, random_state=int(result['filename']))
        
        if verbose:
            logger.info(f"Injection done {result['filename']} done, predicting...")    

        # ===================================================================
        # PREDICTION 
        # ===================================================================
        init_predict_mean = last_state_mean_2
        init_predict_cov = last_state_cov_2
            
        predictions_mean = predict_model_online(
            df_predict=df_predict,
            seasonal_period_steps=seasonal_period_steps,
            result=result,
            init_state_mean=init_predict_mean,
            init_state_cov=init_predict_cov,
            theta=theta_second,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            ref_residuals=second_residuals,
            z_threshold=threshold_z_score,
        )

        # Build results dataframe: scores applied to prediction residuals,
        # but statistics estimated from second training segment residuals.
        predictions_df = build_results_df(
            df_predict,
            predictions_mean,
            result,
            ref_residuals=second_residuals,
            threshold_z_score=threshold_z_score,
            threshold_z_score_robust=threshold_z_score_robust,
        )
        
        result["z_scores"] = predictions_df["z_score"].tolist()

        if predictions_output_csv is not None:
            predictions_df.to_csv(predictions_output_csv, index=False)

        if verbose:
            logger.info(f"Prediction done {result['filename']} done, calculating metrics...") 

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
    seasonal_cycle: str = "daily",
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
    seasonal_cycle : str
        Seasonal period specification ('daily', 'weekly', or seconds)
    num_workers : int
        Number of parallel workers
    verbose : bool
        Enable verbose logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device (for info only): {device}")
        logger.info(
            f"Processing {len(csv_filepaths)} files with {num_workers} workers"
        )
        logger.info(f"Seasonal cycle: {seasonal_cycle}")
        logger.info(f"Per-file timeout: {per_file_timeout} seconds")

    all_results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        for filepath in csv_filepaths:
            filename = Path(filepath).stem

            future = executor.submit(
                process_single_meter,
                filepath,
                seasonal_cycle=seasonal_cycle,
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
    logger.info("BATCH PROCESSING COMPLETE (Seasonal UnobservedComponents)")
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
        description="Seasonal UnobservedComponents Batch Processor for Water Meter Anomaly Detection",
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
        "--seasonal",
        type=str,
        default="daily",
        help="Seasonal cycle ('daily', 'weekly', or seconds) (default: 'daily')",
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
    directory = "../data_w_diff_001"
    seed_value = 42

    #all_files = [
    #    os.path.join(directory, f)
    #    for f in os.listdir(directory)
    #    if os.path.isfile(os.path.join(directory, f))
    #]
    #random.seed(seed_value)
#
    #csv_filepaths = random.sample(all_files, min(args.samples, len(all_files)))
    
    #with open("../pickles/test_set.pkl", "rb") as f:
    #    csv_filepaths = pickle.load(f)
      
    with open("../pickles/common_sensors.pkl", "rb") as f:
        csv_filepaths = pickle.load(f)
    
    logger.info(f"Loaded {len(csv_filepaths)} CSV filepaths")

    output_csv = f"./6_weeks_results_seasonal_uc_{args.samples}_seed_42_{args.seasonal}_clipped_tree_timeout_120_reworked_daily_weekly_fourier_anomalies_common_z_score.csv"
    
    _ = process_batch(
        csv_filepaths,
        output_csv,
        seasonal_cycle=args.seasonal,
        num_workers=args.workers,
        verbose=args.verbose,
    )

    logger.info(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()
