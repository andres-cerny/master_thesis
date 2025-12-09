"""
GRU Batch Processor for Water Meter Anomaly Detection
=====================================================

Processes multiple water meter CSV files with:
- Multithreaded execution
- Configurable periodicity (window_size)
- Training on 3-2 months ago data (50 epochs)
- Warm-start retraining on 2-1 months ago data
- Prediction on last month of data
- Comprehensive metrics calculation

Usage:
    python gru_batch_processor.py
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging
from datetime import datetime, timedelta
import traceback
from typing import Dict, Tuple, Optional, List
import warnings
import random
import time
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(log_file="gru_batch_processor.log", verbose: bool = False):
    """Configure logging to file and console"""
    level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Clear existing handlers (important when re-running in notebooks)
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


# Default logger; will be reconfigured in main() with verbose flag
logger = setup_logging(verbose=False)

# ============================================================================
# GRU MODEL DEFINITION
# ============================================================================


class GRUNet(nn.Module):
    """
    Standard GRU Network for Time Series Forecasting with Irregular Data

    Input features:
    - Meter reading (normalized)
    - Delta time since last reading (normalized)
    - Time of day harmonics: sin(2π*hour/24), cos(2π*hour/24)
    - Day of week harmonics: sin(2π*day/7), cos(2π*day/7)

    Total input size: 6 features
    """

    def __init__(self, input_size=6, hidden_size=32, num_layers=1, dropout=0.1):
        super(GRUNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, 6)

        Returns:
            output: Predictions (batch_size, seq_len, 1)
        """
        gru_out, _ = self.gru(x)
        output = self.fc_out(gru_out)
        return output


# ============================================================================
# DATASET CLASS
# ============================================================================


class MeterDataset(Dataset):
    """
    PyTorch Dataset for water meter time series with harmonic features
    """

    def __init__(self, df, window_size=48, train=True, scaler=None):
        """
        Args:
            df: DataFrame with columns ['timestamp_utc', 'Diff']
            window_size: Number of consecutive readings per window
            train: If True, compute scaler; if False, use provided scaler
            scaler: StandardScaler object (required if train=False)
        """
        self.df = df.copy()
        self.window_size = window_size

        # Sort by timestamp
        self.df["timestamp_utc"] = pd.to_datetime(self.df["timestamp_utc"], utc=True)
        self.df = self.df.sort_values("timestamp_utc").reset_index(drop=True)

        # Compute time deltas (in hours)
        self.df["time_delta"] = (
            self.df["timestamp_utc"].diff().dt.total_seconds() / 3600.0
        )
        self.df["time_delta"].fillna(0, inplace=True)

        # Handle missing values
        self.df["is_missing"] = self.df["Diff"].isna().astype(float)
        self.df["Diff"].fillna(method="ffill", inplace=True)
        self.df["Diff"].fillna(0, inplace=True)

        # Normalize meter readings
        if train:
            self.scaler = StandardScaler()
            self.df["Diff_norm"] = self.scaler.fit_transform(self.df[["Diff"]])
        else:
            assert scaler is not None, "Must provide scaler if train=False"
            self.scaler = scaler
            self.df["Diff_norm"] = self.scaler.transform(self.df[["Diff"]])

        # Normalize time delta: cap at 24 hours
        self.df["time_delta_capped"] = np.clip(self.df["time_delta"], 0, 24)
        self.df["time_delta_norm"] = self.df["time_delta_capped"] / 24.0

        # Compute harmonic features
        hours = (
            self.df["timestamp_utc"].dt.hour
            + self.df["timestamp_utc"].dt.minute / 60.0
        )
        self.df["sin_tod"] = np.sin(2 * np.pi * hours / 24.0)
        self.df["cos_tod"] = np.cos(2 * np.pi * hours / 24.0)

        dow = self.df["timestamp_utc"].dt.dayofweek
        self.df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
        self.df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

        self.max_idx = len(self.df) - self.window_size

    def __len__(self):
        return max(1, self.max_idx)

    def __getitem__(self, idx):
        """Returns a window of data as tensors"""
        idx = min(idx, self.max_idx)
        start_idx = idx
        end_idx = idx + self.window_size

        window = self.df.iloc[start_idx:end_idx]

        x = torch.stack(
            [
                torch.tensor(window["Diff_norm"].values, dtype=torch.float32),
                torch.tensor(window["time_delta_norm"].values, dtype=torch.float32),
                torch.tensor(window["sin_tod"].values, dtype=torch.float32),
                torch.tensor(window["cos_tod"].values, dtype=torch.float32),
                torch.tensor(window["sin_dow"].values, dtype=torch.float32),
                torch.tensor(window["cos_dow"].values, dtype=torch.float32),
            ],
            dim=1,
        )

        actual_values = torch.tensor(window["Diff_norm"].values, dtype=torch.float32)

        return x, actual_values


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def train_gru_model(
    df,
    window_size=48,
    model_path=None,
    epochs=50,
    batch_size=32,
    hidden_size=32,
    learning_rate=1e-3,
    loss_function="mae",
    device=None,
    pretrained_model_path=None,
    verbose=False,
):
    """
    Train a GRU model on a single meter's data with harmonic features
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_warmstart = pretrained_model_path is not None

    if verbose:
        logger.info(
            f"Device: {device}, Dataset size: {len(df)}, Window size: {window_size}"
        )

    # Create dataset and dataloader
    dataset = MeterDataset(df, window_size=window_size, train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scaler = dataset.scaler

    # Initialize model
    model = GRUNet(input_size=6, hidden_size=hidden_size, num_layers=1, dropout=0.1)
    model = model.to(device)

    # Load pretrained weights if provided
    if is_warmstart:
        try:
            checkpoint = torch.load(
                pretrained_model_path, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(pretrained_model_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        if verbose:
            logger.info(f"Loaded pretrained weights from {pretrained_model_path}")

    # Loss function selection
    if loss_function.lower() == "mae":
        criterion = nn.L1Loss()
    elif loss_function.lower() == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(
            f"loss_function must be 'mae' or 'mse', got '{loss_function}'"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if is_warmstart:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(2, epochs // 2), gamma=0.7
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for x, actual_values in dataloader:
            x = x.to(device)
            actual_values = actual_values.to(device)

            output = model(x)
            output = output.squeeze(-1)

            loss = criterion(output, actual_values)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        scheduler.step()

        if verbose and ((epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0):
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Save model if path provided
    if model_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "hidden_size": hidden_size,
                "window_size": window_size,
                "is_warmstart": is_warmstart,
            },
            model_path,
        )

    return model, scaler, losses


def load_gru_model(model_path, device=None):
    """Load a previously trained GRU model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    hidden_size = checkpoint["hidden_size"]
    scaler = checkpoint["scaler"]
    window_size = checkpoint.get("window_size", 48)

    model = GRUNet(input_size=6, hidden_size=hidden_size, num_layers=1, dropout=0.1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, scaler, window_size


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================


def prepare_data_for_prediction(df, scaler):
    """Prepare data with all features for prediction"""
    test_df = df.copy()
    test_df["timestamp_utc"] = pd.to_datetime(test_df["timestamp_utc"], utc=True)
    test_df = test_df.dropna()

    # Time delta
    test_df["time_delta"] = (
        test_df["timestamp_utc"].diff().dt.total_seconds() / 3600.0
    )
    print(f"Number of na deltas: {test_df['time_delta'].isna().sum()}")
    test_df = test_df.dropna()
    test_df["time_delta_capped"] = np.clip(test_df["time_delta"], 0, 24)
    test_df["time_delta_norm"] = test_df["time_delta_capped"] / 24.0

    # Normalize readings
    test_df["Diff_norm"] = scaler.transform(test_df[["Diff"]])

    # Harmonic features
    hours = test_df["timestamp_utc"].dt.hour + test_df["timestamp_utc"].dt.minute / 60.0
    test_df["sin_tod"] = np.sin(2 * np.pi * hours / 24.0)
    test_df["cos_tod"] = np.cos(2 * np.pi * hours / 24.0)

    dow = test_df["timestamp_utc"].dt.dayofweek
    test_df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    test_df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    return test_df


def predict_with_gru(model, df, scaler, window_size, device=None):
    """Generate predictions for a meter using trained GRU"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = prepare_data_for_prediction(df, scaler)

    # Build feature matrix
    feature_list = [
        test_df["Diff_norm"].values,
        test_df["time_delta_norm"].values,
        test_df["sin_tod"].values,
        test_df["cos_tod"].values,
        test_df["sin_dow"].values,
        test_df["cos_dow"].values,
    ]

    X_full = np.stack(feature_list, axis=1)
    X_tensor = torch.tensor(X_full, dtype=torch.float32).unsqueeze(0)
    X_tensor = X_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(X_tensor)

    predictions_norm = output.squeeze().cpu().numpy()
    actuals_norm = test_df["Diff_norm"].values

    # Denormalize
    predictions = scaler.inverse_transform(predictions_norm.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(actuals_norm.reshape(-1, 1)).flatten()

    # Compute residuals
    residuals = np.abs(actuals - predictions)

    # Anomaly score: z-score of residuals
    rolling_mean = pd.Series(residuals).rolling(window=24, center=True).mean()
    rolling_std = pd.Series(residuals).rolling(window=24, center=True).std()

    anomaly_score = (residuals - rolling_mean) / (rolling_std + 1e-6)

    result_df = test_df[["timestamp_utc", "Diff"]].copy()
    result_df.columns = ["timestamp_utc", "actual"]
    result_df["predicted"] = predictions
    result_df["residual"] = residuals
    result_df["anomaly_score"] = anomaly_score
    result_df["is_anomaly"] = (np.abs(anomaly_score) > 4).astype(int)

    return result_df


# ============================================================================
# METRICS CALCULATION
# ============================================================================


def calculate_metrics(actuals, predictions, residuals):
    """
    Calculate comprehensive metrics for predictions
    """
    metrics = {}

    # Basic metrics
    metrics["rmse"] = np.sqrt(mean_squared_error(actuals, predictions))
    metrics["mae"] = mean_absolute_error(actuals, predictions)
    metrics["r2"] = r2_score(actuals, predictions)

    # MAPE (handle division by zero)
    try:
        metrics["mape"] = mean_absolute_percentage_error(actuals, predictions)
    except Exception:
        metrics["mape"] = np.nan

    # Additional metrics
    metrics["mean_residual"] = np.mean(residuals)
    metrics["std_residual"] = np.std(residuals)
    metrics["max_residual"] = np.max(residuals)
    metrics["min_residual"] = np.min(residuals)

    # RMSE normalized by actual variance
    actual_var = np.var(actuals)
    if actual_var > 0:
        metrics["normalized_rmse"] = metrics["rmse"] / np.sqrt(actual_var)
    else:
        metrics["normalized_rmse"] = np.nan

    # Median Absolute Percentage Error (robust to outliers)
    try:
        mape_values = np.abs((actuals - predictions) / (np.abs(actuals) + 1e-8))
        metrics["median_ape"] = np.median(mape_values)
    except Exception:
        metrics["median_ape"] = np.nan

    # Prediction bias
    metrics["prediction_bias"] = np.mean(predictions - actuals)

    # Direction accuracy (percentage of correct sign predictions)
    actual_diff = np.diff(actuals)
    pred_diff = np.diff(predictions)
    if len(actual_diff) > 0:
        direction_matches = np.sum((actual_diff > 0) == (pred_diff > 0))
        metrics["direction_accuracy"] = direction_matches / len(actual_diff)
    else:
        metrics["direction_accuracy"] = np.nan

    return metrics


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================


def process_single_meter(
    csv_filepath: str,
    window_size: int,
    temp_dir: str = "./temp_models",
    device: Optional[torch.device] = None,
    epochs_train: int = 20,
    epochs_warmstart: int = 5,
    verbose: bool = False,
) -> Dict:
    """
    Process a single water meter CSV file through training and prediction pipeline
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

        df = pd.read_csv(csv_filepath)

        # Handle column name variations
        if "timestamp_utc" not in df.columns:
            raise ValueError(
                f"No timestamp_utc column found. Available: {df.columns.tolist()}"
            )

        if "Diff" not in df.columns:
            raise ValueError(f"No 'Diff' column found. Available: {df.columns.tolist()}")
        
        if window_size < 1:
            raise ValueError(f"Periodicity smaller then one measurement per day. Cannot use daily harmonics.")

        df = df[["timestamp_utc", "Diff"]].copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.dropna()

        # ===== SPLIT DATA INTO PERIODS =====
        readings_per_day = window_size
        days_3_to_2_months = 30
        days_2_to_1_months = 30
        days_last_month = 30

        readings_3_to_2m = readings_per_day * days_3_to_2_months
        readings_2_to_1m = readings_per_day * days_2_to_1_months
        readings_last_m = readings_per_day * days_last_month

        total_readings_needed = readings_3_to_2m + readings_2_to_1m + readings_last_m

        if len(df) < total_readings_needed:
            raise ValueError(f"Insufficient data: only {len(df)} readings provided but needed {total_readings_needed}.")

        idx_start = max(
            0, len(df) - readings_3_to_2m - readings_2_to_1m - readings_last_m
        )
        
        idx_train_start = idx_start
        idx_train_end = idx_train_start + readings_3_to_2m
        idx_warmstart_end = idx_train_end + readings_2_to_1m
        idx_pred_end = len(df)

        df_train = df.iloc[idx_train_start:idx_train_end].copy()
        df_warmstart = df.iloc[idx_train_end:idx_warmstart_end].copy()
        df_predict = df.iloc[idx_warmstart_end:idx_pred_end].copy()

        result["train_samples"] = len(df_train)
        result["warmstart_samples"] = len(df_warmstart)
        result["predict_samples"] = len(df_predict)

        if verbose:
            logger.info(
                f"  Train: {len(df_train)}, Warmstart: {len(df_warmstart)}, Predict: {len(df_predict)}"
            )

        if len(df_train) < window_size or len(df_warmstart) < window_size:
            raise ValueError("Insufficient training data.")

        # ===== TRAINING PHASE =====
        temp_model_path = os.path.join(temp_dir, f"{result['filename']}_initial.pt")
        os.makedirs(temp_dir, exist_ok=True)

        if verbose:
            logger.info(f"  Training on {len(df_train)} samples ({epochs_train} epochs)...")

        # --- measure initial training time ---
        t_start_train = time.time()
        model, scaler, losses_train = train_gru_model(
            df_train,
            window_size=window_size,
            model_path=temp_model_path,
            epochs=epochs_train,
            batch_size=32,
            hidden_size=32,
            learning_rate=1e-3,
            loss_function="mae",
            device=device,
            verbose=verbose,
        )
        t_end_train = time.time()
        result["train_time_seconds"] = t_end_train - t_start_train
        # --------------------------------------

        result["train_final_loss"] = losses_train[-1] if losses_train else None

        # ===== WARM-START PHASE =====
        warmstart_model_path = os.path.join(
            temp_dir, f"{result['filename']}_warmstart.pt"
        )

        if verbose:
            logger.info(
                f"  Warm-start retraining on {len(df_warmstart)} samples ({epochs_warmstart} epochs)..."
            )

        # --- measure warm-start training time ---
        t_start_warm = time.time()
        model, scaler, losses_warmstart = train_gru_model(
            df_warmstart,
            window_size=window_size,
            model_path=warmstart_model_path,
            epochs=epochs_warmstart,
            batch_size=32,
            hidden_size=32,
            learning_rate=5e-4,
            loss_function="mae",
            device=device,
            pretrained_model_path=temp_model_path,
            verbose=verbose,
        )
        t_end_warm = time.time()
        result["warmstart_time_seconds"] = t_end_warm - t_start_warm
        # -----------------------------------------

        result["warmstart_final_loss"] = losses_warmstart[-1] if losses_warmstart else None

        # ===== PREDICTION PHASE =====
        if verbose:
            logger.info(f"  Generating predictions on {len(df_predict)} samples...")

        # --- measure prediction time ---
        t_start_pred = time.time()
        predictions_df = predict_with_gru(
            model, df_predict, scaler, window_size=window_size, device=device
        )
        t_end_pred = time.time()
        result["prediction_time_seconds"] = t_end_pred - t_start_pred
        # --------------------------------

        # ===== CALCULATE METRICS =====
        actuals = predictions_df["actual"].values
        preds = predictions_df["predicted"].values
        residuals = predictions_df["residual"].values

        metrics = calculate_metrics(actuals, preds, residuals)

        # Add metrics to result
        for key, value in metrics.items():
            result[f"metric_{key}"] = value

        result["anomalies_detected"] = int(predictions_df["is_anomaly"].sum())
        result["status"] = "success"

        # Always show per-file success summary line
        if verbose:
            logger.info(
                f"Processed successfully: {result['filename']} - RMSE: {metrics['rmse']}, "
                f"MAE: {metrics['mae']}, R2: {metrics['r2']}"
            )

        # ===== CLEANUP =====
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        if os.path.exists(warmstart_model_path):
            os.remove(warmstart_model_path)

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        # Always show per-file failure line
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
    epochs_train: int = 50,
    epochs_warmstart: int = 5,
    verbose: bool = False,
):
    """
    Process multiple water meter CSV files in parallel
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
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

            # Submit task
            future = executor.submit(
                process_single_meter,
                filepath,
                window_size,
                device=device,
                epochs_train=epochs_train,
                epochs_warmstart=epochs_warmstart,
                verbose=verbose,
            )
            futures[future] = filename

        # Progress bar over completed futures
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

    # ===== SAVE RESULTS =====
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)

    logger.info("=" * 70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total processed: {len(results_df)}")
    logger.info(f"Successful: {(results_df['status'] == 'success').sum()}")
    logger.info(f"Failed: {(results_df['status'] == 'failed').sum()}")
    logger.info(f"Metrics saved to: {output_csv}")

    # Print summary statistics
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
        description="GRU Batch Processor for Water Meter Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--workers", type=int, default=7, help="Number of parallel workers (default: 7)"
    )
    parser.add_argument(
        "--epochs-train",
        type=int,
        default=50,
        help="Epochs for initial training (default: 50)",
    )
    parser.add_argument(
        "--epochs-warmstart",
        type=int,
        default=5,
        help="Epochs for warm-start (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (default: False)",
    )

    args = parser.parse_args()

    # Reconfigure logger with verbosity
    global logger
    logger = setup_logging(verbose=args.verbose)

    # ===== LOAD FILEPATHS =====
    directory = "../data_w_diff_001"
    seed_value = 42

    all_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    random.seed(seed_value)
    csv_filepaths = random.sample(all_files, min(10, len(all_files)))

    logger.info(f"Loaded {len(csv_filepaths)} CSV filepaths")

    # ===== LOAD METADATA =====
    metadata_dir = "../metadata_001"

    # ===== OUTPUT FILE =====
    output_csv = "./results_gru_test.csv"

    # ===== RUN BATCH PROCESSING =====
    _ = process_batch(
        csv_filepaths,
        metadata_dir,
        output_csv,
        num_workers=args.workers,
        epochs_train=args.epochs_train,
        epochs_warmstart=args.epochs_warmstart,
        verbose=args.verbose,
    )

    logger.info(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()
