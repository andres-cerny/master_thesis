"""
Local Trend (Local Level with Fixed Slope) Batch Processor for Water Meter Anomaly Detection
===========================================================================================

Processes multiple water meter CSV files with:
- Multithreaded execution
- Configurable periodicity (window_size via metadata)
- Resampling & gap filling using fill_gaps_with_periodicity_adaptive
- Initial training on 3–2 months ago data
- Second training on 2–1 months ago data, initialized from previous month state
- Prediction on last month of data, initialized from previous month state
- Comprehensive metrics calculation, analogous to the GRU script

Usage:
    python local_trend_batch_processor.py
"""

import os
import json
import pandas as pd
import numpy as np
import torch  # only for device detection / parity, not for model itself
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
from typing import Dict, Optional, List
import warnings
import random
from tqdm import tqdm
import time
import sys
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset

from resample import fill_gaps_with_periodicity_adaptive

#warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file="local_trend_batch_processor.log", verbose: bool = False):
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
# LOCAL LEVEL WITH FIXED SLOPE MODEL
# ============================================================================

class LocalLevelWithFixedSlope(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        k_states = 2  # level and slope
        k_posdef = 2  # process noise for level and slope

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization="approximate_diffuse",
            loglikelihood_burn=48 * 7,
        )
        # Observation equation: y_t = [1, 0] * [level_t, slope_t] + eps_t
        self.ssm["design", 0, 0] = 1

        # State transition:
        # level_t = level_{t-1} + slope_{t-1} + eta_level
        # slope_t = slope_{t-1} + eta_slope (here slope noise will be fixed to 0)
        self.ssm["transition", 0, 0] = 1
        self.ssm["transition", 0, 1] = 1
        self.ssm["transition", 1, 1] = 1  # slope is deterministic

        # Process noise selection
        self.ssm["selection", 0, 0] = 1
        self.ssm["selection", 1, 1] = 1

    def update(self, params, *args, **kwargs):
        params = super().update(params, *args, **kwargs)
        # Only estimate observation and level noise
        self.ssm["obs_cov", 0, 0] = params[0]
        self.ssm["state_cov", 0, 0] = params[1]
        # Slope noise is fixed to zero
        self.ssm["state_cov", 1, 1] = 0

    @property
    def param_names(self):
        return ["sigma2.obs", "sigma2.level"]

    @property
    def start_params(self):
        s = np.nanstd(self.endog)
        return [s, s]

    def transform_params(self, params):
        return np.square(params)

    def untransform_params(self, params):
        return np.sqrt(params)


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(actuals, predictions, residuals):
    """
    Calculate comprehensive metrics for predictions
    """
    metrics = {}

    metrics["rmse"] = np.sqrt(mean_squared_error(actuals, predictions))
    metrics["mae"] = mean_absolute_error(actuals, predictions)
    metrics["r2"] = r2_score(actuals, predictions)

    try:
        metrics["mape"] = mean_absolute_percentage_error(actuals, predictions)
    except Exception:
        metrics["mape"] = np.nan

    metrics["mean_residual"] = np.mean(residuals)
    metrics["std_residual"] = np.std(residuals)
    metrics["max_residual"] = np.max(residuals)
    metrics["min_residual"] = np.min(residuals)

    actual_var = np.var(actuals)
    if actual_var > 0:
        metrics["normalized_rmse"] = metrics["rmse"] / np.sqrt(actual_var)
    else:
        metrics["normalized_rmse"] = np.nan

    try:
        mape_values = np.abs((actuals - predictions) / (np.abs(actuals) + 1e-8))
        metrics["median_ape"] = np.median(mape_values)
    except Exception:
        metrics["median_ape"] = np.nan

    metrics["prediction_bias"] = np.mean(predictions - actuals)

    actual_diff = np.diff(actuals)
    pred_diff = np.diff(predictions)
    if len(actual_diff) > 0:
        direction_matches = np.sum((actual_diff > 0) == (pred_diff > 0))
        metrics["direction_accuracy"] = direction_matches / len(actual_diff)
    else:
        metrics["direction_accuracy"] = np.nan

    return metrics


# ============================================================================
# HELPER: RESIDUAL-BASED ANOMALIES
# ============================================================================

def build_results_df(timestamps, actuals, predictions):
    """
    Build a result DataFrame analogous to GRU prediction output:
    - timestamp_utc
    - actual
    - predicted
    - residual
    - anomaly_score (z-score on rolling residuals)
    - is_anomaly (abs z-score > 4)
    """
    timestamps = pd.to_datetime(timestamps, utc=True)
    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)

    residuals = np.abs(actuals - predictions)
    residual_series = pd.Series(residuals)

    rolling_mean = residual_series.rolling(window=24, center=True).mean()
    rolling_std = residual_series.rolling(window=24, center=True).std()

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
    window_size: int,
    metadata: dict,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict:
    """
    Process a single water meter CSV file with the LocalLevelWithFixedSlope model.

    Splitting logic (in terms of number of readings, analogous to GRU script):
    - 3–2 months ago: initial training
    - 2–1 months ago: second training (initialized from previous state cov/mean)
    - last month: prediction (initialized from previous state cov/mean)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {
        "filename": Path(csv_filepath).stem,
        "filepath": csv_filepath,
        "window_size": window_size,
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
        df_raw.dropna(subset=['timestamp_utc'], inplace=True)
        
        # ===== SPLIT INTO 3 CHUNKS (3–2m, 2–1m, 1m) =====
        df_raw['timestamp_utc'] = pd.to_datetime(df_raw['timestamp_utc'], utc=True)
        end = df_raw.loc[df_raw.index[-1], 'timestamp_utc']
        
        start_pred = end - DateOffset(months=1)
        start_second = start_pred - DateOffset(months=1)
        start_train = start_second - DateOffset(months=1)

        if start_train < df_raw.loc[0, 'timestamp_utc']:
            raise ValueError(
                f"Not enough data. Start time needed for train {start_train}, "
                f"but earliest possible is {df_raw['timestamp_utc'].iloc[0]}"
            )
        
        mask = (df_raw["timestamp_utc"] >= start_pred) & (df_raw["timestamp_utc"] < end)
        df_predict = df_raw.loc[mask].copy().reset_index()
        mask = (df_raw["timestamp_utc"] >= start_second) & (df_raw["timestamp_utc"] < start_pred)
        df_second = df_raw.loc[mask].copy().reset_index()
        mask = (df_raw["timestamp_utc"] >= start_train) & (df_raw["timestamp_utc"] < start_second)
        df_train = df_raw.loc[mask].copy().reset_index()
        
        result["train_samples"] = len(df_train)
        result["second_train_samples"] = len(df_second)
        result["predict_samples"] = len(df_predict)
        
        # ===== RESAMPLE / FILL GAPS =====
        # We assume fill_gaps_with_periodicity_adaptive returns a DataFrame with
        # a regular timestamp index/column and a filled value column (here 'Diff').
        df_train, diag = fill_gaps_with_periodicity_adaptive(df_train, timestamp_col="timestamp_utc")
        periodicity_seconds_train = diag['periodicity_used_seconds']
        df_second, diag = fill_gaps_with_periodicity_adaptive(df_second, timestamp_col="timestamp_utc")
        periodicity_seconds_second = diag['periodicity_used_seconds']
        df_predict, diag = fill_gaps_with_periodicity_adaptive(df_predict, timestamp_col="timestamp_utc")
        periodicity_seconds_predict = diag['periodicity_used_seconds']
                
        result["train_samples_filled"] = len(df_train)
        result["second_train_samples_filled"] = len(df_second)
        result["predict_samples_filled"] = len(df_predict)

        if verbose:
            logger.info(
                f"  Train: {len(df_train)}, Second: {len(df_second)}, Predict: {len(df_predict)}"
            )

        if len(df_train) < 10 or len(df_second) < 10 or len(df_predict) < 10:
            raise ValueError("Insufficient data in one of the segments for modeling.")

        # ----------------------------------------------------------------------
        # INITIAL TRAINING (3–2 months ago)
        # ----------------------------------------------------------------------
        y_train = df_train["Diff"].values.astype(float)

        t_start_train = time.time()
        model_train = LocalLevelWithFixedSlope(y_train)
        res_train = model_train.fit(disp=False)
        t_end_train = time.time()
        result["train_time_seconds"] = t_end_train - t_start_train

        # Get filtered states at the end of training segment
        # filtered_state: shape (k_states, nobs)
        filtered_state = res_train.filter_results.filtered_state
        filtered_cov = res_train.filter_results.filtered_state_cov

        last_state_mean = filtered_state[:, -1].copy()
        last_state_cov = filtered_cov[:, :, -1].copy()

        result["train_loglike"] = res_train.llf

        # ----------------------------------------------------------------------
        # SECOND TRAINING (2–1 months ago) with fixed initial state / covariance
        # ----------------------------------------------------------------------
        y_second = df_second["Diff"].values.astype(float)

        t_start_second = time.time()
        model_second = LocalLevelWithFixedSlope(y_second)
        # Initialize from last_state_mean/cov
        if periodicity_seconds_second == periodicity_seconds_train:
            model_second.initialize_known(
                initial_state=last_state_mean, initial_state_cov=last_state_cov
            )
        res_second = model_second.fit(disp=False)
        t_end_second = time.time()
        result["second_train_time_seconds"] = t_end_second - t_start_second

        filtered_state_2 = res_second.filter_results.filtered_state
        filtered_cov_2 = res_second.filter_results.filtered_state_cov

        last_state_mean_2 = filtered_state_2[:, -1].copy()
        last_state_cov_2 = filtered_cov_2[:, :, -1].copy()

        result["second_train_loglike"] = res_second.llf

        # ----------------------------------------------------------------------
        # PREDICTION (last month) starting from last state of second segment
        # ----------------------------------------------------------------------
        y_pred_segment = df_predict["Diff"].values.astype(float)

        t_start_pred = time.time()
        model_pred = LocalLevelWithFixedSlope(y_pred_segment)
        if periodicity_seconds_predict == periodicity_seconds_second:
            model_pred.initialize_known(
                initial_state=last_state_mean_2, initial_state_cov=last_state_cov_2
            )

        # Filter to get one-step-ahead predictions for this segment
        res_pred = model_pred.filter(model_pred.start_params)
        # get_prediction with dynamic=True gives one-step-ahead in-sample predictions
        pred_obj = res_pred.get_prediction()
        pred_mean = pred_obj.predicted_mean
        t_end_pred = time.time()
        result["prediction_time_seconds"] = t_end_pred - t_start_pred

        timestamps_pred = df_predict["timestamp_utc"].values
        actuals = y_pred_segment
        mask = ~np.isnan(actuals)
        timestamps_used = timestamps_pred[mask]
        actuals_used = actuals[mask]
        predictions_used = pred_mean[mask]

        predictions_df = build_results_df(
            timestamps_used, actuals_used, predictions_used
        )
        print("getting metrics")
        # ===== METRICS =====
        residuals = predictions_df["residual"].values
        metrics = calculate_metrics(actuals_used, predictions_used, residuals)
        print("calculated metrics")
        for key, value in metrics.items():
            result[f"metric_{key}"] = value

        result["anomalies_detected"] = int(predictions_df["is_anomaly"].sum())
        result["status"] = "success"

        if verbose:
            logger.info(
                f"Processed successfully (LocalTrend): {result['filename']} - "
                f"RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, R2: {metrics['r2']}"
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
    metadata_dir: str,
    output_csv: str,
    num_workers: int = 4,
    verbose: bool = False,
):
    """
    Process multiple water meter CSV files in parallel with LocalLevelWithFixedSlope.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device (for info only): {device}")
        logger.info(f"Processing {len(csv_filepaths)} files with {num_workers} workers")

    all_results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}

        for filepath in csv_filepaths:
            filename = Path(filepath).stem

            try:
                metadata_filepath = os.path.join(
                    metadata_dir, filename.split(".")[0] + ".json"
                )
                with open(metadata_filepath, "r") as f:
                    metadata = json.load(f)

                window_size = int(
                    round(24 * 60 * 60 / metadata["common_periodicity_seconds"])
                )
            except Exception as e:
                logger.error(f"Failed to prepare metadata for {filename}: {e}")
                all_results.append(
                    {
                        "filename": filename,
                        "status": "failed",
                        "error": str(e),
                    }
                )
                continue

            future = executor.submit(
                process_single_meter,
                filepath,
                window_size,
                metadata,
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
                logger.error(
                    f"[{i}/{len(futures)}] Failed to process {filename}: {e}"
                )
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
    logger.info("BATCH PROCESSING COMPLETE (LocalTrend)")
    logger.info("=" * 70)
    logger.info(f"Total processed: {len(results_df)}")
    logger.info(f"Successful: {(results_df['status'] == 'success').sum()}")
    logger.info(f"Failed: {(results_df['status'] == 'failed').sum()}")
    logger.info(f"Metrics saved to: {output_csv}")

    successful = results_df[results_df["status"] == "success"]
    if len(successful) > 0:
        logger.info("Metrics Summary (successful runs only):")
        logger.info(
            f"  RMSE: {successful['metric_rmse'].mean()} +/- {successful['metric_rmse'].std()}"
        )
        logger.info(
            f"  MAE:  {successful['metric_mae'].mean()} +/- {successful['metric_mae'].std()}"
        )
        logger.info(
            f"  R2:   {successful['metric_r2'].mean()} +/- {successful['metric_r2'].std()}"
        )
        logger.info(
            f"  MAPE: {successful['metric_mape'].mean()} +/- {successful['metric_mape'].std()}"
        )

    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Local Trend Batch Processor for Water Meter Anomaly Detection",
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

    args = parser.parse_args()

    global logger
    logger = setup_logging(verbose=args.verbose)

    # Data directory (same structure as GRU script)
    directory = "../data_w_diff_001"
    seed_value = 42

    all_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    random.seed(seed_value)
    csv_filepaths = random.sample(all_files, min(10, len(all_files)))
    #csv_filepaths = ['../data_w_diff_001/103458.csv']
    logger.info(f"Loaded {len(csv_filepaths)} CSV filepaths")

    metadata_dir = "../metadata_001"
    output_csv = "./results_local_trend_test.csv"

    _ = process_batch(
        csv_filepaths,
        metadata_dir,
        output_csv,
        num_workers=args.workers,
        verbose=args.verbose,
    )

    logger.info(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()
