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
from calculate_metrics import calculate_metrics, calculate_metrics_unresampled
from create_anomalies_dfs import inject_synthetic_anomalies

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

def compute_z_scores(residuals):
    # Ensure float numpy array
    residuals = np.asarray(residuals, dtype=float)

    # Mask valid (non-NaN) residuals
    valid = ~np.isnan(residuals)
    res_valid = residuals[valid]

    # Initialize outputs
    z_score = np.full_like(residuals, np.nan)
    z_score_robust = np.full_like(residuals, np.nan)

    # If no usable residuals -> all NaN
    if res_valid.size == 0:
        return z_score, z_score_robust

    # ---------- Classic mean/std z-score ----------
    res_mean = np.nanmean(res_valid)
    res_std = np.nanstd(res_valid, ddof=1)

    if not (np.isnan(res_std) or res_std == 0):
        z_score[valid] = (res_valid - res_mean) / res_std
    else:
        # No variability: treat all valid points as typical
        z_score[valid] = 0.0

    # ---------- Robust median/MAD z-score ----------
    median_resid = np.nanmedian(res_valid)
    mad = np.nanmedian(np.abs(res_valid - median_resid))

    # Fallback to classic std if MAD unusable
    if np.isnan(mad) or mad == 0:
        sigma_robust = res_std if not (np.isnan(res_std) or res_std == 0) else np.nan
    else:
        sigma_robust = 1.4826 * mad

    if np.isnan(sigma_robust) or sigma_robust == 0:
        # No variability: all valid z_robust = 0
        z_score_robust[valid] = 0.0
    else:
        z_score_robust[valid] = (res_valid - median_resid) / sigma_robust

    return z_score, z_score_robust



def build_results_df(df_predict, predictions, result, threshold_z_score=3, threshold_z_score_robust=3):
    """
    Build a result DataFrame with anomaly scores based on residuals.

    Handles NaN values properly by masking them before rolling calculations.

    Parameters:
    -----------
    timestamps : array-like
        Timestamp array (will be converted to datetime UTC)
    actuals : array-like
        Actual Diff values (may contain NaN)
    predictions : array-like
        Predicted Diff values (may contain NaN)

    Returns:
    --------
    pd.DataFrame : Result dataframe with anomaly scores and flags
    """
    actuals = df_predict["Diff"].values.astype(float)
    timestamps = df_predict["timestamp_utc"].values
    
    if 'is_anomaly' in df_predict.columns:
        anomalies = df_predict["is_anomaly"].values
        anomalies = np.asarray(anomalies)
    else:
        anomalies = np.zeros(len(df_predict), dtype=float)
    
    timestamps = pd.to_datetime(timestamps, utc=True)
    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)

    residuals = np.abs(actuals - predictions)

    z_score, z_score_robust = compute_z_scores(residuals)

    # build result df
    result_df = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "actual": actuals,
            "predicted": predictions,
            "residual": residuals,
            "z_score": z_score,
            "z_score_robust": z_score_robust,
            "is_anomaly_actual": anomalies
        }
    )
    result_df["is_anomaly_predicted"] = (np.abs(result_df["z_score"]) > 3).astype(int)
    result_df["is_anomaly_robust_predicted"] = (np.abs(result_df["z_score_robust"]) > 3).astype(int)
    
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
        #logger.info(f"Did not converge because of {res_train.mle_retvals['warnflag']} at value {res_train.mle_retvals['gopt']}")
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
# PREDICT MODEL
# ============================================================================
def predict_model(df_predict, seasonal_period_steps, result, init_state_mean, init_state_cov, theta, freq_seasonal=None, stochastic_freq_seasonal=None):    
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
        
        # Initialize from last state of second training segment
        if init_state_mean is not None and init_state_cov is not None:
            model_pred.initialize_known(
                initial_state=init_state_mean, initial_state_cov=init_state_cov
            )
            
        # Filter with parameters from second training
        res_pred = model_pred.filter(theta)
        # Get one-step-ahead predictions
        pred_obj = res_pred.get_prediction()
        pred_mean = pred_obj.predicted_mean
        pred_mean = np.asarray(pred_mean, dtype=float)
        pred_mean = np.clip(pred_mean, 0.0, None)
        
    except Exception as e:
        raise ValueError(f"Failed to generate predictions: {e}")
    
    t_end_pred = time.time()
    result["prediction_time_seconds"] = t_end_pred - t_start_pred

    return pred_mean

def split_df(df_raw, seed, result, test_window_days=100, min_periodicity_month_train=20):
    '''
    Split df into 3 chunks for pre-training, training and prediction
    '''
    df_raw["timestamp_utc"] = pd.to_datetime(df_raw["timestamp_utc"], utc=True)
    
    min_ts = df_raw["timestamp_utc"].min()
    max_ts = df_raw["timestamp_utc"].max()

    span = max_ts - min_ts
    if span < pd.Timedelta(days=test_window_days):
        raise ValueError(
            f"Not enough data. Do not have at least {test_window_days} days of data."
        )
        
    min_window = pd.Timedelta(days=test_window_days)
    latest_start = max_ts - min_window
    
    # single random draw between min_ts and latest_start for this meter (reproducible for this seed)
    rng = default_rng(seed)
    u = rng.random()
    
    rand_start = min_ts + (latest_start - min_ts) * u
    rand_start = pd.to_datetime(rand_start)
    rand_end = rand_start + min_window

    df_3m = df_raw[(df_raw["timestamp_utc"] >= rand_start) &
                   (df_raw["timestamp_utc"] < rand_end)].copy().reset_index(drop=True)
    
    periodicity_sec_df_3m = get_periodicity(df_3m)
    
    window_days = 30
    if periodicity_sec_df_3m < min_periodicity_month_train * 60:
        window_days = 14
        
    result['train_predict_window_days'] = window_days
    
    end = df_3m.loc[df_3m.index[-1], "timestamp_utc"]
    start_pred = end - DateOffset(days=window_days)
    start_second = start_pred - DateOffset(days=window_days)
    start_train = start_second - DateOffset(days=window_days)
    
    #if 3*window_size did not fit into df_3m choose the last 3 window_size from df_raw
    if start_train < df_3m.loc[0, "timestamp_utc"]:
        end = df_raw.loc[df_raw.index[-1], "timestamp_utc"]
        start_pred = end - DateOffset(days=window_days)
        start_second = start_pred - DateOffset(days=window_days)
        start_train = start_second - DateOffset(days=window_days)
    
        if start_train < df_raw.loc[0, "timestamp_utc"]:
            raise ValueError(
                f"Not enough data. Start time needed for train {start_train}, "
                f"but earliest possible is {df_3m['timestamp_utc'].iloc[0]}"
            )
        
    mask = (df_3m["timestamp_utc"] >= start_pred) & (df_3m["timestamp_utc"] < end)
    df_predict = df_3m.loc[mask].copy().reset_index(drop=True)
    
    mask = (df_3m["timestamp_utc"] >= start_second) & (
        df_3m["timestamp_utc"] < start_pred
    )
    df_second = df_3m.loc[mask].copy().reset_index(drop=True)
    
    mask = (df_3m["timestamp_utc"] >= start_train) & (
        df_3m["timestamp_utc"] < start_second
    )
    df_train = df_3m.loc[mask].copy().reset_index(drop=True)
    
    result["train_samples"] = len(df_train)
    result["second_train_samples"] = len(df_second)
    result["predict_samples"] = len(df_predict)
    
    return df_train, df_second, df_predict

def resample_dfs(df_train, df_second, df_predict, result, verbose, leniency = 0.333):
    '''
    leniency = max percentage time difference (in terms of percentage of a period) between last two steps in dfs with different periods
    '''
    
    df_predict_unresampled = df_predict.copy()
    
    # ===== RESAMPLE / FILL GAPS =====
    df_train, diag_train = fill_gaps_with_periodicity_adaptive(
        df_train, timestamp_col="timestamp_utc"
    )
    periodicity_seconds_train = diag_train["periodicity_used_seconds"]
    df_second, diag_second = fill_gaps_with_periodicity_adaptive(
        df_second, timestamp_col="timestamp_utc"
    )
    periodicity_seconds_second = diag_second["periodicity_used_seconds"]
    df_predict, diag_predict = fill_gaps_with_periodicity_adaptive(
        df_predict, timestamp_col="timestamp_utc"
    )
    periodicity_seconds_predict = diag_predict["periodicity_used_seconds"]
    result["train_samples_filled"] = len(df_train)
    result["second_train_samples_filled"] = len(df_second)
    result["predict_samples_filled"] = len(df_predict)
    result["periodicity_seconds_train"] = periodicity_seconds_train
    result["periodicity_seconds_second"] = periodicity_seconds_second
    result["periodicity_seconds_predict"] = periodicity_seconds_predict
    periods = [
        periodicity_seconds_train,
        periodicity_seconds_second,
        periodicity_seconds_predict,
    ]
    p_min = min(periods)
    if p_min == 0:
        raise ValueError(
                        "One of periodicities is equal to 0"
                        f"Periodicity Train: {periodicity_seconds_train}, "
                        f"Periodicity Second: {periodicity_seconds_second}, "
                        f"Periodicity Predict: {periodicity_seconds_predict}."
                        )
    
    leniency_seconds = p_min * leniency
    seasonal_period_steps = calculate_seasonal_period(p_min)
    
    all_within_pct = all(abs(p*seasonal_period_steps - p_min*seasonal_period_steps) <= leniency_seconds for p in periods)
    if not all_within_pct:
        raise ValueError(
            f"All three periodicities are not within {leniency*100}% range. "
            f"Since if we apply daily steps the last steps of two seasons differ by more then {leniency_seconds} seconds. "
            f"Periodicity Train: {periodicity_seconds_train}, "
            f"Periodicity Second: {periodicity_seconds_second}, "
            f"Periodicity Predict: {periodicity_seconds_predict}."
        )
    if verbose:
        logger.info(
            f"  Train: {len(df_train)}, Second: {len(df_second)}, Predict: {len(df_predict)}"
        )
    if len(df_train) < 10 or len(df_second) < 10 or len(df_predict) < 10:
        raise ValueError("Insufficient data in one of the segments for modeling.")
    
    return df_train, df_second, df_predict, seasonal_period_steps, df_predict_unresampled

# ============================================================================
# MAIN PROCESSING FUNCTION FOR ONE METER
# ============================================================================


def process_single_meter(
    csv_filepath: str,
    seasonal_cycle: str = "daily",
    device: Optional[torch.device] = None,
    verbose: bool = False,
    predictions_output_csv: Optional[str] = None,
    weekly_seasonality: bool = False,
    daily_steps: bool = True,
    threshold_z_score: int = 3,
    threshold_z_score_robust: int = 3
) -> Dict:
    """
    Process a single water meter CSV file with UnobservedComponents
    (local level + seasonal model, no slope).

    Splitting logic (in terms of number of readings):
    - 3–2 months ago: initial training
    - 2–1 months ago: second training (initialized from previous state)
    - last month: prediction (initialized from previous state)

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
        
        # ===== SPLIT INTO 3 CHUNKS =====
        seed = int(result['filename'])
        df_train, df_second, df_predict = split_df(df_raw=df_raw, seed=seed, result=result)

        # ===== RESAMPLE / FILL GAPS =====
        df_train, df_second, df_predict, seasonal_period_steps, df_predict_unresampled = resample_dfs(df_train=df_train,
                                                       df_second=df_second,
                                                       df_predict=df_predict,
                                                       result=result,
                                                       verbose=verbose)

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
        # ===================================================================
        # INITIAL TRAINING
        # ===================================================================
        
        #If meter runs potentially too long do not run the model
        #can_long_run_length(train_samples=result['train_samples'],
        #                    train_samples_filled=result['train_samples_filled'],
        #                    periodicity_seconds=result['periodicity_seconds_train'])
        
        last_state_mean, last_state_cov, _ = train_model(df_train=df_train,
                                                         seasonal_period_steps=seasonal_period_steps,
                                                         result=result,
                                                         initial_train=True,
                                                         freq_seasonal=freq_seasonal,
                                                         stochastic_freq_seasonal=stochastic_freq_seasonal,
                                                         )
        
        # ===================================================================
        # SECOND TRAINING
        # ===================================================================
        
        #If meter runs potentially too long do not run the model
        #can_long_run_length(train_samples=result['second_train_samples'],
        #                    train_samples_filled=result['second_train_samples_filled'],
        #                    periodicity_seconds=result['periodicity_seconds_second'])
        
        init_second_mean = None
        init_second_cov = None
        if result['periodicity_seconds_train'] == result['periodicity_seconds_second']:
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
        # ===================================================================
        # Inject anomalies to prediction df 
        # ===================================================================
        df_predict = inject_synthetic_anomalies(df_predict, random_state=int(result['filename']))
                
        # ===================================================================
        # PREDICTION 
        # ===================================================================
        init_predict_mean = None
        init_predict_cov = None
        if result['periodicity_seconds_second'] == result['periodicity_seconds_predict']:
            init_predict_mean = last_state_mean_2
            init_predict_cov = last_state_cov_2
            
        
        predictions_mean = predict_model(df_predict=df_predict,
                                        seasonal_period_steps=seasonal_period_steps,
                                        result=result,
                                        init_state_mean=init_predict_mean,
                                        init_state_cov=init_predict_cov,
                                        theta=theta_second,
                                        freq_seasonal=freq_seasonal,
                                        )
        # Build results dataframe (keeps NaN values)
        predictions_df = build_results_df(df_predict, predictions_mean, result, threshold_z_score, threshold_z_score_robust)

        if predictions_output_csv is not None:
            predictions_df.to_csv(predictions_output_csv, index=False)
       
        # ===================================================================
        # METRICS 
        # ===================================================================
        metrics = calculate_metrics(predictions_df)
        
        metrics_unresampled = calculate_metrics_unresampled(df_predict_unresampled, predictions_df)
        
        for key, value in metrics.items():
            result[f"metric_{key}"] = value
            
        for key, value in metrics_unresampled.items():
            result[f"unresampled_metric_{key}"] = value

        result["status"] = "success"

        if verbose:
            logger.info(
                f"Processed successfully (Seasonal UC): {result['filename']} - "
                f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
                f"R2: {metrics['r2']:.4f}, Seasonal Period: {seasonal_period_steps}"
            )

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
    per_file_timeout: int = 600,  # seconds; tune as needed (e.g. 300, 900)
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

    logger.info("=" * 70)
    logger.info("BATCH PROCESSING COMPLETE (Seasonal UnobservedComponents)")
    logger.info("=" * 70)
    logger.info(f"Total processed: {len(results_df)}")
    logger.info(f"Successful: {(results_df['status'] == 'success').sum()}")
    logger.info(f"Failed: {(results_df['status'] == 'failed').sum()}")
    logger.info(f"Metrics saved to: {output_csv}")

    successful = results_df[results_df["status"] == "success"]
    if len(successful) > 0:
        logger.info("Metrics Summary (successful runs only):")
        logger.info(
            f"  RMSE: {successful['metric_rmse'].mean():.4f} +/- {successful['metric_rmse'].std():.4f}"
        )
        logger.info(
            f"  MAE:  {successful['metric_mae'].mean():.4f} +/- {successful['metric_mae'].std():.4f}"
        )
        logger.info(
            f"  R2:   {successful['metric_r2'].mean():.4f} +/- {successful['metric_r2'].std():.4f}"
        )
        logger.info(
            f"  MAPE: {successful['metric_mape'].mean():.4f} +/- {successful['metric_mape'].std():.4f}"
        )
        logger.info(
            f"  Valid data %: {successful['metric_valid_percentage'].mean():.2f}%"
        )
        logger.info(
            f"  NaN samples: {successful['metric_nan_count'].mean():.1f} avg per meter"
        )

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

    all_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    random.seed(seed_value)
    
    csv_filepaths = random.sample(all_files, min(args.samples, len(all_files)))
    directory = "../data_w_anomalies"
    csv_filepaths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    logger.info(f"Loaded {len(csv_filepaths)} CSV filepaths")

    output_csv = f"./results_seasonal_uc_{args.samples}_seed_42_{args.seasonal}_clipped_tree_timeout_600_reworked_w_anomalies.csv"
    
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
