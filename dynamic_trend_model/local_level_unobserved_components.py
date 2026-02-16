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
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
import traceback
from typing import Dict, Optional, List, Tuple
import warnings
import random
from tqdm import tqdm
import time
import sys
from statsmodels.tsa.statespace.structural import UnobservedComponents
from pandas.tseries.offsets import DateOffset


from resample import fill_gaps_with_periodicity_adaptive

import warnings
from statsmodels.tools.sm_exceptions import SpecificationWarning
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SpecificationWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


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
    seasonal_period = max(2, seasonal_period)  # Ensure at least 2 steps
    return seasonal_period


# ============================================================================
# METRICS CALCULATION
# ============================================================================


def calculate_metrics(actuals, predictions, residuals):
    """
    Calculate comprehensive metrics for predictions, including both all-data
    and non-zero-only metrics.

    Properly handles NaN values in actuals and predictions.

    Parameters:
    -----------
    actuals : array-like
        Actual values (may contain NaN)
    predictions : array-like
        Predicted values (may contain NaN)
    residuals : array-like
        Residuals (may contain NaN)

    Returns:
    --------
    dict : Dictionary containing all metrics with suffixes for non-zero variants
    """
    metrics = {}

    # Convert to numpy arrays and create mask for valid (non-NaN) values
    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)
    residuals = np.asarray(residuals)

    valid_mask = ~(np.isnan(actuals) | np.isnan(predictions) | np.isnan(residuals))
    valid_actuals = actuals[valid_mask]
    valid_predictions = predictions[valid_mask]
    valid_residuals = residuals[valid_mask]

    # Store counts
    metrics["total_count"] = len(actuals)
    metrics["valid_count"] = np.sum(valid_mask)
    metrics["nan_count"] = np.sum(~valid_mask)
    metrics["valid_percentage"] = (
        (np.sum(valid_mask) / len(actuals)) * 100 if len(actuals) > 0 else 0
    )

    if len(valid_actuals) == 0:
        logger.warning("No valid (non-NaN) data points for metrics calculation")
        # Return all NaN metrics
        metrics.update({
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "mean_residual": np.nan,
            "std_residual": np.nan,
            "max_residual": np.nan,
            "min_residual": np.nan,
            "normalized_rmse": np.nan,
            "median_ape": np.nan,
            "prediction_bias": np.nan,
            "direction_accuracy": np.nan,
        })
        return metrics

    # ====================
    # ALL VALID DATA METRICS
    # ====================

    # Basic metrics
    metrics["rmse"] = np.sqrt(mean_squared_error(valid_actuals, valid_predictions))
    metrics["mae"] = mean_absolute_error(valid_actuals, valid_predictions)
    metrics["r2"] = r2_score(valid_actuals, valid_predictions)

    # MAPE
    try:
        metrics["mape"] = mean_absolute_percentage_error(
            valid_actuals, valid_predictions
        )
    except Exception:
        metrics["mape"] = np.nan

    # Residual statistics
    metrics["mean_residual"] = np.nanmean(valid_residuals)
    metrics["std_residual"] = np.nanstd(valid_residuals)
    metrics["max_residual"] = np.nanmax(valid_residuals)
    metrics["min_residual"] = np.nanmin(valid_residuals)

    # RMSE normalized by actual variance
    actual_var = np.nanvar(valid_actuals)
    if actual_var > 0:
        metrics["normalized_rmse"] = metrics["rmse"] / np.sqrt(actual_var)
    else:
        metrics["normalized_rmse"] = np.nan

    # Median Absolute Percentage Error (robust to outliers)
    try:
        mape_values = np.abs(
            (valid_actuals - valid_predictions) / (np.abs(valid_actuals) + 1e-8)
        )
        metrics["median_ape"] = np.nanmedian(mape_values)
    except Exception:
        metrics["median_ape"] = np.nan

    # Prediction bias
    metrics["prediction_bias"] = np.nanmean(valid_predictions - valid_actuals)

    # Direction accuracy
    if len(valid_actuals) > 1:
        actual_diff = np.diff(valid_actuals)
        pred_diff = np.diff(valid_predictions)
        if len(actual_diff) > 0:
            direction_matches = np.sum((actual_diff > 0) == (pred_diff > 0))
            metrics["direction_accuracy"] = direction_matches / len(actual_diff)
        else:
            metrics["direction_accuracy"] = np.nan
    else:
        metrics["direction_accuracy"] = np.nan

    # ====================
    # NON-ZERO ONLY METRICS
    # ====================

    non_zero_mask = valid_actuals != 0
    non_zero_count = np.sum(non_zero_mask)

    metrics["non_zero_count"] = non_zero_count
    metrics["zero_count"] = len(valid_actuals) - non_zero_count
    metrics["non_zero_percentage"] = (
        (non_zero_count / len(valid_actuals)) * 100 if len(valid_actuals) > 0 else 0
    )

    if non_zero_count > 0:
        actuals_nz = valid_actuals[non_zero_mask]
        predictions_nz = valid_predictions[non_zero_mask]
        residuals_nz = valid_residuals[non_zero_mask]

        metrics["rmse_nz"] = np.sqrt(mean_squared_error(actuals_nz, predictions_nz))
        metrics["mae_nz"] = mean_absolute_error(actuals_nz, predictions_nz)

        try:
            metrics["r2_nz"] = r2_score(actuals_nz, predictions_nz)
        except Exception:
            metrics["r2_nz"] = np.nan

        try:
            metrics["mape_nz"] = mean_absolute_percentage_error(
                actuals_nz, predictions_nz
            )
        except Exception:
            metrics["mape_nz"] = np.nan

        metrics["mean_residual_nz"] = np.nanmean(residuals_nz)
        metrics["std_residual_nz"] = np.nanstd(residuals_nz)
        metrics["max_residual_nz"] = np.nanmax(residuals_nz)
        metrics["min_residual_nz"] = np.nanmin(residuals_nz)

        actual_var_nz = np.nanvar(actuals_nz)
        if actual_var_nz > 0:
            metrics["normalized_rmse_nz"] = metrics["rmse_nz"] / np.sqrt(actual_var_nz)
        else:
            metrics["normalized_rmse_nz"] = np.nan

        try:
            mape_values_nz = np.abs((actuals_nz - predictions_nz) / np.abs(actuals_nz))
            metrics["median_ape_nz"] = np.nanmedian(mape_values_nz)
        except Exception:
            metrics["median_ape_nz"] = np.nan

        metrics["prediction_bias_nz"] = np.nanmean(predictions_nz - actuals_nz)

        if len(actuals_nz) > 1:
            actual_diff_nz = np.diff(actuals_nz)
            pred_diff_nz = np.diff(predictions_nz)
            if len(actual_diff_nz) > 0:
                direction_matches_nz = np.sum((actual_diff_nz > 0) == (pred_diff_nz > 0))
                metrics["direction_accuracy_nz"] = direction_matches_nz / len(
                    actual_diff_nz
                )
            else:
                metrics["direction_accuracy_nz"] = np.nan
        else:
            metrics["direction_accuracy_nz"] = np.nan
    else:
        metrics["rmse_nz"] = np.nan
        metrics["mae_nz"] = np.nan
        metrics["r2_nz"] = np.nan
        metrics["mape_nz"] = np.nan
        metrics["mean_residual_nz"] = np.nan
        metrics["std_residual_nz"] = np.nan
        metrics["max_residual_nz"] = np.nan
        metrics["min_residual_nz"] = np.nan
        metrics["normalized_rmse_nz"] = np.nan
        metrics["median_ape_nz"] = np.nan
        metrics["prediction_bias_nz"] = np.nan
        metrics["direction_accuracy_nz"] = np.nan

    return metrics


# ============================================================================
# HELPER: RESIDUAL-BASED ANOMALIES
# ============================================================================


def build_results_df(timestamps, actuals, predictions):
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
    timestamps = pd.to_datetime(timestamps, utc=True)
    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)

    residuals = np.abs(actuals - predictions)
    residual_series = pd.Series(residuals)

    # Rolling statistics, which naturally handle NaN
    rolling_mean = residual_series.rolling(window=24, center=True, min_periods=1).mean()
    rolling_std = residual_series.rolling(window=24, center=True, min_periods=1).std()

    # Avoid division by zero
    rolling_std = rolling_std.fillna(1e-6)
    rolling_std = rolling_std.replace(0, 1e-6)

    anomaly_score = (residual_series - rolling_mean) / (rolling_std + 1e-6)

    result_df = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "actual": actuals,
            "predicted": predictions,
            "residual": residuals,
            "anomaly_score": anomaly_score.values,
        }
    )
    result_df["is_anomaly"] = (np.abs(result_df["anomaly_score"]) > 4).astype(int)
    return result_df


# ============================================================================
# MAIN PROCESSING FUNCTION FOR ONE METER
# ============================================================================


def process_single_meter(
    csv_filepath: str,
    seasonal_cycle: str = "daily",
    device: Optional[torch.device] = None,
    verbose: bool = False,
    predictions_output_csv: Optional[str] = None,
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
        if "Diff" not in df_raw.columns:
            raise ValueError(
                f"No 'Diff' column found. Available: {df_raw.columns.tolist()}"
            )
        if "hodnota" not in df_raw.columns:
            raise ValueError(
                f"No 'hodnota' column found. Available: {df_raw.columns.tolist()}"
            )

        df_raw["timestamp_utc"] = pd.to_datetime(df_raw["timestamp_utc"], utc=True)
        df_raw = df_raw[["timestamp_utc", "hodnota", "Diff"]].copy()
        df_raw.dropna(subset=["timestamp_utc"], inplace=True)

        # ===== SPLIT INTO 3 CHUNKS (3–2m, 2–1m, 1m) =====
        df_raw["timestamp_utc"] = pd.to_datetime(df_raw["timestamp_utc"], utc=True)
        end = df_raw.loc[df_raw.index[-1], "timestamp_utc"]

        start_pred = end - DateOffset(months=1)
        start_second = start_pred - DateOffset(months=1)
        start_train = start_second - DateOffset(months=1)

        if start_train < df_raw.loc[0, "timestamp_utc"]:
            raise ValueError(
                f"Not enough data. Start time needed for train {start_train}, "
                f"but earliest possible is {df_raw['timestamp_utc'].iloc[0]}"
            )

        mask = (df_raw["timestamp_utc"] >= start_pred) & (df_raw["timestamp_utc"] < end)
        df_predict = df_raw.loc[mask].copy().reset_index(drop=True)
        mask = (df_raw["timestamp_utc"] >= start_second) & (
            df_raw["timestamp_utc"] < start_pred
        )
        df_second = df_raw.loc[mask].copy().reset_index(drop=True)
        mask = (df_raw["timestamp_utc"] >= start_train) & (
            df_raw["timestamp_utc"] < start_second
        )
        df_train = df_raw.loc[mask].copy().reset_index(drop=True)

        result["train_samples"] = len(df_train)
        result["second_train_samples"] = len(df_second)
        result["predict_samples"] = len(df_predict)

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
            all_within_10pct = all(p == 0 for p in periods)
        else:
            all_within_10pct = all(abs(p - p_min) / p_min <= 0.10 for p in periods)

        if not all_within_10pct:
            raise ValueError(
                f"All three periodicities are not within 10% range. "
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

        # Calculate seasonal period
        seasonal_period_steps = calculate_seasonal_period(
            periodicity_seconds_train, seasonal_cycle
        )
        result["seasonal_period_steps"] = seasonal_period_steps

        # ===================================================================
        # INITIAL TRAINING (3–2 months ago)
        # ===================================================================
        y_train = df_train["Diff"].values.astype(float)

        t_start_train = time.time()
        try:
            model_train = UnobservedComponents(
                endog=y_train,
                level="local level",  # local level only, no slope
                seasonal=seasonal_period_steps,
                stochastic_level=True,
                stochastic_seasonal=True,
            )
            res_train = model_train.fit(disp=False)
        except Exception as e:
            raise ValueError(f"Failed to fit initial training model: {e}")
        
        result["converged_train"] = res_train.mle_retvals['converged']

        if not res_train.mle_retvals['converged']:
            logger.info(f"Did not converge because of {res_train.mle_retvals['warnflag']} at value {res_train.mle_retvals['gopt']}")
            result['warnflag_train'] = res_train.mle_retvals['warnflag']
            result["gopt_train"] = res_train.mle_retvals['gopt']
        else:
            result['warnflag_train'] = ''
            result["gopt_train"] = ''
            
        t_end_train = time.time()
        result["train_time_seconds"] = t_end_train - t_start_train

        # Get filtered states at end of training segment
        filtered_state = res_train.filter_results.filtered_state
        filtered_cov = res_train.filter_results.filtered_state_cov

        last_state_mean = filtered_state[:, -1].copy()
        last_state_cov = filtered_cov[:, :, -1].copy()

        theta_train = res_train.params
        result["train_loglike"] = res_train.llf

        # ===================================================================
        # SECOND TRAINING (2–1 months ago) with fixed initial state
        # ===================================================================
        y_second = df_second["Diff"].values.astype(float)

        t_start_second = time.time()
        try:
            model_second = UnobservedComponents(
                endog=y_second,
                level="local level",
                seasonal=seasonal_period_steps,
                stochastic_level=True,
                stochastic_seasonal=True,
            )

            # Initialize from last state of training segment
            if periodicity_seconds_second == periodicity_seconds_train:
                model_second.initialize_known(
                    initial_state=last_state_mean, initial_state_cov=last_state_cov
                )

            res_second = model_second.fit(disp=False)
        except Exception as e:
            raise ValueError(f"Failed to fit second training model: {e}")
        
        result["converged_second"] = res_second.mle_retvals['converged']
        
        if not res_second.mle_retvals['converged']:
            logger.info(f"Did not converge because of {res_second.mle_retvals['warnflag']} at value {res_second.mle_retvals['gopt']}")
            result['warnflag_second'] = res_second.mle_retvals['warnflag']
            result["gopt_second"] = res_second.mle_retvals['gopt']
        else:
            result['warnflag_second'] = ''
            result["gopt_second"] = ''

        t_end_second = time.time()
        result["second_train_time_seconds"] = t_end_second - t_start_second

        filtered_state_2 = res_second.filter_results.filtered_state
        filtered_cov_2 = res_second.filter_results.filtered_state_cov

        last_state_mean_2 = filtered_state_2[:, -1].copy()
        last_state_cov_2 = filtered_cov_2[:, :, -1].copy()

        theta_second = res_second.params
        result["second_train_loglike"] = res_second.llf

        # ===================================================================
        # PREDICTION (last month) starting from last state of second segment
        # ===================================================================
        y_pred_segment = df_predict["Diff"].values.astype(float)

        t_start_pred = time.time()
        try:
            model_pred = UnobservedComponents(
                endog=y_pred_segment,
                level="local level",
                seasonal=seasonal_period_steps,
                stochastic_level=True,
                stochastic_seasonal=True,
            )

            # Initialize from last state of second training segment
            if periodicity_seconds_predict == periodicity_seconds_second:
                model_pred.initialize_known(
                    initial_state=last_state_mean_2, initial_state_cov=last_state_cov_2
                )

            # Filter with parameters from second training
            res_pred = model_pred.filter(theta_second)

            # Get one-step-ahead predictions
            pred_obj = res_pred.get_prediction()
            pred_mean = pred_obj.predicted_mean
            pred_mean = np.asarray(pred_mean, dtype=float)
            pred_mean = np.clip(pred_mean, 0.0, None)
        except Exception as e:
            raise ValueError(f"Failed to generate predictions: {e}")

        t_end_pred = time.time()
        result["prediction_time_seconds"] = t_end_pred - t_start_pred

        timestamps_pred = df_predict["timestamp_utc"].values
        actuals = y_pred_segment
        predictions_used = pred_mean

        # Build results dataframe (keeps NaN values)
        predictions_df = build_results_df(timestamps_pred, actuals, predictions_used)

        if predictions_output_csv is not None:
            predictions_df.to_csv(predictions_output_csv, index=False)

        # ===== METRICS (handles NaN properly) =====
        residuals = predictions_df["residual"].values
        metrics = calculate_metrics(actuals, predictions_used, residuals)

        for key, value in metrics.items():
            result[f"metric_{key}"] = value

        result["anomalies_detected"] = int(predictions_df["is_anomaly"].sum())
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
):
    """
    Process multiple water meter CSV files in parallel with UnobservedComponents.

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

    all_results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        for filepath in csv_filepaths:
            filename = Path(filepath).stem

            future = executor.submit(
                process_single_meter,
                filepath,
                seasonal_cycle=seasonal_cycle,
                device=device,
                verbose=verbose,
            )
            futures[future] = filename

        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Processing meters"),
            1,
        ):
            filename = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                if verbose:
                    logger.info(f"[{i}/{len(futures)}] Completed {filename}")
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
    logger.info(f"Loaded {len(csv_filepaths)} CSV filepaths")

    output_csv = f"./results_seasonal_uc_{args.samples}_seed_42_{args.seasonal}_clipped.csv"

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
