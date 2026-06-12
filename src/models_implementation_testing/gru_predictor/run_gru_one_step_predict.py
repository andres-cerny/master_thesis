"""
GRU Batch Processor for Water Meter Anomaly Detection (SLIDING WINDOW)
=======================================================================

Reworked to match the UnobservedComponents model exactly so results are
directly comparable:

  Splitting  – identical sliding-window logic from split_df_sliding_weeks()
               (random 6-week window, seed = int(filename), numpy default_rng)
               • Weeks 1–4  → initial training
               • Weeks 2–5  → warm-start retraining
               • Week  6    → prediction
  Resampling – fill_gaps_with_periodicity_adaptive(), same as UC model
  Window     – one seasonal period (= one day at the meter's periodicity)
  Metrics    – calculate_metrics() / calculate_metrics_unresampled(), same
               imports as UC model; local calculate_metrics() removed
  z-score    – same compute_z_scores() / build_results_df() helpers
  Batch      – ProcessPoolExecutor with per-file timeout (mirrors UC)

Usage:
    python run_gru_sliding_window.py --workers 7 --verbose
"""

import os
import sys
import pickle
import random
import time
import logging
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.random import default_rng
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

parent_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(parent_path)

from helper_scripts.calculate_metrics import calculate_metrics
from helper_scripts.create_anomalies import inject_spike_anomalies_diff

warnings.filterwarnings("ignore")


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file="gru_sliding_window.log", verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logging(verbose=False)

def get_periodicity(df: pd.DataFrame, timestamp_col: str = 'timestamp_utc') -> int:
    """
    Returns:
    --------
    int
        Most common periodicity in DataFrame
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        
    time_diffs = df[timestamp_col].diff().dropna().dt.total_seconds()
    
    if time_diffs.empty:
        raise ValueError("No time difference calculated. No two valid neighboring values found.")

    common_periodicity_mode = time_diffs.mode()
    return common_periodicity_mode.iloc[0]


# ============================================================================
# HELPERS: z-scores and result dataframe  (identical to UC model)
# ============================================================================

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
    residuals = np.asarray(residuals, dtype=float)

    # Use ref_residuals for statistics if provided, otherwise fall back to residuals
    if ref_residuals is not None:
        ref = np.asarray(ref_residuals, dtype=float)
    else:
        ref = residuals

    # Mask valid (non-NaN) values
    valid_ref = ref[~np.isnan(ref)]
    valid = ~np.isnan(residuals)

    z_score = np.full_like(residuals, np.nan)
    z_score_robust = np.full_like(residuals, np.nan)

    if valid_ref.size == 0:
        return z_score, z_score_robust

    # ---------- Classic mean/std z-score ----------
    # Statistics estimated from the reference (second training) segment
    res_mean = np.nanmean(valid_ref)
    res_std  = np.nanstd(valid_ref, ddof=1)
    
    result["ref_mean"] = res_mean
    result["ref_std"] = res_std
    
    if not (np.isnan(res_std) or res_std == 0):
        z_score[valid] = (residuals[valid] - res_mean) / res_std
    else:
        z_score[valid] = 0.0

    # ---------- Robust median/MAD z-score ----------
    # Statistics estimated from the reference (second training) segment
    median_resid = np.nanmedian(valid_ref)
    mad = np.nanmedian(np.abs(valid_ref - median_resid))

    if np.isnan(mad) or mad == 0:
        sigma_robust = res_std if not (np.isnan(res_std) or res_std == 0) else np.nan
    else:
        sigma_robust = 1.4826 * mad

    if np.isnan(sigma_robust) or sigma_robust == 0:
        z_score_robust[valid] = 0.0
    else:
        z_score_robust[valid] = (residuals[valid] - median_resid) / sigma_robust

    return z_score, z_score_robust


def build_results_df(df_predict, predictions, result,
                     ref_residuals=None,
                     threshold_z_score=3, threshold_z_score_robust=3):
    """
    Build a result DataFrame with anomaly scores (identical signature to UC model).

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
    """
    actuals    = df_predict["Diff"].values.astype(float)
    timestamps = df_predict["timestamp_utc"].values

    if "is_anomaly" in df_predict.columns:
        anomalies = np.asarray(df_predict["is_anomaly"].values)
    else:
        anomalies = np.zeros(len(df_predict), dtype=int)

    timestamps  = pd.to_datetime(timestamps, utc=True)
    actuals     = np.asarray(actuals)
    predictions = np.asarray(predictions)
    residuals   = np.abs(actuals - predictions)

    result_df = pd.DataFrame({
        "timestamp_utc": timestamps,
        "actual":        actuals,
        "predicted":     predictions,
        "residual":      residuals,
    })

    if anomalies is not None:
        # Scores are computed on prediction residuals, but thresholding statistics
        # (mean, std, median, MAD) are estimated from ref_residuals (second training segment)
        z_score, z_score_robust = compute_z_scores(residuals, result, ref_residuals=ref_residuals)
        result_df["z_score"]        = z_score
        result_df["z_score_robust"] = z_score_robust
        result_df["is_anomaly_actual"] = anomalies

        result_df["is_anomaly_predicted"] = (
            np.abs(result_df["z_score"]) > threshold_z_score
        ).astype(int)
        result_df["is_anomaly_robust_predicted"] = (
            np.abs(result_df["z_score_robust"]) > threshold_z_score_robust
        ).astype(int)

        result["number_of_anomalies"]        = int(result_df["is_anomaly_predicted"].sum())
        result["number_of_anomalies_robust"] = int(result_df["is_anomaly_robust_predicted"].sum())
        result["anomaly_indices"]        = result_df.index[result_df["is_anomaly_predicted"] == 1].tolist()
        result["anomaly_indices_robust"] = result_df.index[result_df["is_anomaly_robust_predicted"] == 1].tolist()

    return result_df


# ============================================================================
# SEASONAL PERIOD HELPER  (identical to UC model)
# ============================================================================

def calculate_seasonal_period(periodicity_seconds: float,
                              seasonal_cycle: str = "daily") -> int:
    if isinstance(seasonal_cycle, str):
        if seasonal_cycle.lower() == "daily":
            cycle_seconds = 24 * 3600
        elif seasonal_cycle.lower() == "weekly":
            cycle_seconds = 7 * 24 * 3600
        else:
            try:
                cycle_seconds = float(seasonal_cycle)
            except ValueError:
                cycle_seconds = 24 * 3600
    else:
        cycle_seconds = float(seasonal_cycle)
    return int(np.round(cycle_seconds / periodicity_seconds))


# ============================================================================
# SLIDING-WINDOW SPLIT  (identical logic to UC model)
# ============================================================================

def split_df_sliding_weeks(
    df_raw,
    seed,
    result,
    total_weeks: int = 6,
    min_days_with_data_per_day: int = 1,
    min_periodicity_month_train: int = 20,
):
    """
    Identical sliding-window logic to the UC model:
      - Random 6-week window (seeded by meter filename integer)
      - Weeks 1–4  → df_train
      - Weeks 2–5  → df_second  (warm-start)
      - Week 6     → df_predict
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

    df["date"] = df["timestamp_utc"].dt.floor("D")
    daily_counts = df.groupby("date").size().rename("count").reset_index()

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
    max_start  = max_ts - pd.Timedelta(days=total_days_required)
    max_tries  = 30
    chosen_start = None

    for _ in range(max_tries):
        u = rng.random()
        rand_start = min_ts + (max_start - min_ts) * u
        rand_start = pd.to_datetime(rand_start).floor("D")
        rand_end   = rand_start + pd.Timedelta(days=total_days_required)

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

    periodicity_seconds = get_periodicity(df_6w)
    result["df_samples"] = len(df_6w)
    result["periodicity_seconds"] = periodicity_seconds

    if periodicity_seconds == 0:
        raise ValueError("Periodicity is equal to 0")

    seasonal_period_steps = calculate_seasonal_period(periodicity_seconds)
    result["seasonal_period_steps"] = seasonal_period_steps

    # Adjust training length for high-frequency meters (same rule as UC)
    weeks_train = 4
    if periodicity_seconds < min_periodicity_month_train * 60:
        weeks_train = 2

    # Slice boundaries  (same as UC model)
    start_pred   = chosen_end - pd.Timedelta(weeks=1)
    end_pred     = chosen_end

    start_second = start_pred   - pd.Timedelta(weeks=weeks_train)
    end_second   = start_pred

    start_train  = start_second - pd.Timedelta(weeks=1)
    end_train    = end_second   - pd.Timedelta(weeks=1)

    if start_train < chosen_start:
        raise ValueError(
            f"Chosen 6-week segment starts on {chosen_start} but initial "
            f"train wants to start earlier at {start_train}"
        )

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

    result["train_samples"]          = len(df_train)
    result["second_train_samples"]   = len(df_second)
    result["predict_samples"]        = len(df_predict)
    result["train_predict_window_days"] = total_days_required

    if len(df_train) < 2 or len(df_second) < 2 or len(df_predict) < 2:
        raise ValueError(
            "Insufficient data in one of the 6-week subsegments "
            "(train/second/predict) for modeling."
        )

    return df_train, df_second, df_predict


# ============================================================================
# GRU MODEL
# ============================================================================

class GRUNet(nn.Module):
    """
    GRU sequence-to-point model.
    Input features per timestep (6 total):
      Diff_norm, time_delta_norm, sin_tod, cos_tod, sin_dow, cos_dow
    """

    def __init__(self, input_size=6, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc_out(gru_out[:, -1, :])


# ============================================================================
# DATASET
# ============================================================================

class MeterDataset(Dataset):
    """
    Sliding-window dataset.
    Input : features for positions [t-window_size : t]
    Target: scalar Diff_norm at position t
    """

    def __init__(self, df: pd.DataFrame, window_size: int,
                 train: bool = True, scaler: Optional[StandardScaler] = None):
        self.window_size = window_size
        df = df.copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        # Fill NaN Diff with forward-fill then 0 (GRU can't handle NaN inputs)
        df["Diff"] = df["Diff"].fillna(method="ffill").fillna(0.0)

        # Time delta (hours)
        df["time_delta"] = df["timestamp_utc"].diff().dt.total_seconds() / 3600.0
        df["time_delta"].fillna(0, inplace=True)
        df["time_delta_norm"] = np.clip(df["time_delta"], 0, 24) / 24.0

        # Normalise consumption
        if train:
            self.scaler = StandardScaler()
            df["Diff_norm"] = self.scaler.fit_transform(df[["Diff"]])
        else:
            assert scaler is not None, "Must provide scaler when train=False"
            self.scaler = scaler
            df["Diff_norm"] = self.scaler.transform(df[["Diff"]])

        # Harmonic time features
        hours = df["timestamp_utc"].dt.hour + df["timestamp_utc"].dt.minute / 60.0
        df["sin_tod"] = np.sin(2 * np.pi * hours / 24.0)
        df["cos_tod"] = np.cos(2 * np.pi * hours / 24.0)
        dow = df["timestamp_utc"].dt.dayofweek
        df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
        df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

        self.df      = df
        self.max_idx = len(df) - window_size

    def __len__(self):
        return max(1, self.max_idx)

    def __getitem__(self, idx):
        idx = min(idx, self.max_idx)
        w = self.df.iloc[idx : idx + self.window_size]
        x = torch.stack([
            torch.tensor(w["Diff_norm"].values,       dtype=torch.float32),
            torch.tensor(w["time_delta_norm"].values, dtype=torch.float32),
            torch.tensor(w["sin_tod"].values,         dtype=torch.float32),
            torch.tensor(w["cos_tod"].values,         dtype=torch.float32),
            torch.tensor(w["sin_dow"].values,         dtype=torch.float32),
            torch.tensor(w["cos_dow"].values,         dtype=torch.float32),
        ], dim=1)
        target = torch.tensor(w["Diff_norm"].values[-1], dtype=torch.float32)
        return x, target


# ============================================================================
# TRAINING
# ============================================================================

def train_gru(
    df: pd.DataFrame,
    window_size: int,
    model_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    hidden_size: int = 32,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    pretrained_model_path: Optional[str] = None,
    scaler: Optional[StandardScaler] = None,
    verbose: bool = False,
):
    """
    Train (or warm-start) a GRU model.

    Parameters
    ----------
    pretrained_model_path : str, optional
        If given, load weights from this checkpoint before training (warm-start).
    scaler : StandardScaler, optional
        Provide the scaler fitted on training data so the warm-start dataset
        is normalised consistently.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_warmstart = pretrained_model_path is not None

    # Dataset – use existing scaler if doing warm-start, otherwise fit new one
    dataset    = MeterDataset(df, window_size=window_size,
                              train=(scaler is None), scaler=scaler)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    scaler_out = dataset.scaler

    model = GRUNet(input_size=6, hidden_size=hidden_size, num_layers=1, dropout=0.1)
    model = model.to(device)

    if is_warmstart:
        try:
            ckpt = torch.load(pretrained_model_path, map_location=device,
                              weights_only=False)
        except TypeError:
            ckpt = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    step = max(2, epochs // 2) if is_warmstart else 10
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        scheduler.step()
        if verbose and ((epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0):
            logger.info(f"  Epoch {epoch+1}/{epochs}  loss={avg:.6f}")

    if model_path:
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler":           scaler_out,
            "hidden_size":      hidden_size,
            "window_size":      window_size,
        }, model_path)

    return model, scaler_out, losses


# ============================================================================
# PREDICTION
# ============================================================================

def predict_with_gru(
    model: GRUNet,
    df_predict: pd.DataFrame,
    scaler: StandardScaler,
    window_size: int,
    df_context: pd.DataFrame,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Rolling-window prediction over df_predict.

    To avoid NaN predictions at the start of the prediction window (which
    would occur if window_size > len(df_predict)), we prepend the last
    window_size rows of df_context as history.

    Returns
    -------
    np.ndarray
        Predicted values (unscaled), one per row of df_predict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a combined frame: context tail + predict window
    context_tail = df_context.iloc[-window_size:].copy()
    combined = pd.concat([context_tail, df_predict], ignore_index=True)
    combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)

    # Fill NaNs
    combined["Diff"] = combined["Diff"].fillna(method="ffill").fillna(0.0)

    # Feature engineering
    combined["time_delta"] = combined["timestamp_utc"].diff().dt.total_seconds() / 3600.0
    combined["time_delta"].fillna(0, inplace=True)
    combined["time_delta_norm"] = np.clip(combined["time_delta"], 0, 24) / 24.0
    combined["Diff_norm"] = scaler.transform(combined[["Diff"]])
    hours = combined["timestamp_utc"].dt.hour + combined["timestamp_utc"].dt.minute / 60.0
    combined["sin_tod"] = np.sin(2 * np.pi * hours / 24.0)
    combined["cos_tod"] = np.cos(2 * np.pi * hours / 24.0)
    dow = combined["timestamp_utc"].dt.dayofweek
    combined["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    combined["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    feature_cols = ["Diff_norm", "time_delta_norm", "sin_tod",
                    "cos_tod", "sin_dow", "cos_dow"]

    n_context = len(context_tail)
    n_total   = len(combined)

    predictions_norm = np.full(n_total, np.nan)

    model.eval()
    with torch.no_grad():
        for t in range(window_size, n_total):
            w = combined.iloc[t - window_size : t][feature_cols].values
            X = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)
            predictions_norm[t] = model(X).squeeze().cpu().numpy()

    # Slice predictions to the prediction-window rows only
    pred_norm_slice = predictions_norm[n_context:]

    # Denormalise
    predictions = scaler.inverse_transform(
        pred_norm_slice.reshape(-1, 1)
    ).flatten()

    # Clip negatives (consumption can't be negative)
    predictions = np.clip(predictions, 0.0, None)

    return predictions


# ============================================================================
# PREDICTION — ONLINE (one step at a time with anomaly masking)
# ============================================================================

def predict_with_gru_online(
    model: GRUNet,
    df_predict: pd.DataFrame,
    scaler: StandardScaler,
    window_size: int,
    df_context: pd.DataFrame,
    device: Optional[torch.device] = None,
    ref_residuals: Optional[np.ndarray] = None,
    z_threshold: float = 3,
) -> np.ndarray:
    """
    One-step-ahead GRU prediction with online anomaly masking.

    At each prediction step:
      1. Predict y_hat from the current context window.
      2. Compute residual |y_t - y_hat| and z-score it using statistics
         estimated from ref_residuals (second training segment).
      3. If |z| > z_threshold OR y_t is NaN: write the predicted Diff_norm
         back into the feature matrix at position t so that future windows
         use the model's own output as context instead of the anomalous value.
      4. Otherwise: leave the actual value in the feature matrix as usual.

    This mirrors production behaviour where a detected anomaly is not
    allowed to corrupt the input context for subsequent predictions.

    Parameters:
    -----------
    ref_residuals : np.ndarray or None
        Residuals from the second training segment used to compute reference
        mean and std for online z-score thresholding.
    z_threshold : float
        Z-score threshold above which an observation is masked (default: 3).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-compute reference statistics from second training residuals
    if ref_residuals is not None:
        ref = np.asarray(ref_residuals, dtype=float)
        valid_ref = ref[~np.isnan(ref)]
        ref_mean = float(np.nanmean(valid_ref)) if valid_ref.size > 0 else 0.0
        ref_std  = float(np.nanstd(valid_ref, ddof=1)) if valid_ref.size > 1 else None
    else:
        ref_mean = None
        ref_std  = None

    # Build combined frame: context tail + prediction window
    context_tail = df_context.iloc[-window_size:].copy()
    combined = pd.concat([context_tail, df_predict], ignore_index=True)
    combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)

    # Preserve actual Diff values before NaN filling (used for residual computation)
    actual_diff = combined["Diff"].values.astype(float)

    # Fill NaNs for feature engineering
    combined["Diff"] = combined["Diff"].fillna(method="ffill").fillna(0.0)

    # Feature engineering
    combined["time_delta"] = combined["timestamp_utc"].diff().dt.total_seconds() / 3600.0
    combined["time_delta"].fillna(0, inplace=True)
    combined["time_delta_norm"] = np.clip(combined["time_delta"], 0, 24) / 24.0
    combined["Diff_norm"] = scaler.transform(combined[["Diff"]])
    hours = combined["timestamp_utc"].dt.hour + combined["timestamp_utc"].dt.minute / 60.0
    combined["sin_tod"] = np.sin(2 * np.pi * hours / 24.0)
    combined["cos_tod"] = np.cos(2 * np.pi * hours / 24.0)
    dow = combined["timestamp_utc"].dt.dayofweek
    combined["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    combined["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    feature_cols = ["Diff_norm", "time_delta_norm", "sin_tod",
                    "cos_tod", "sin_dow", "cos_dow"]
    DIFF_NORM_COL = 0  # index of Diff_norm within feature_cols

    # Copy to numpy for efficient in-place updates when masking anomalies
    features = combined[feature_cols].values.copy()   # (n_total, 6)

    n_context = len(context_tail)
    n_total   = len(combined)
    n_predict = n_total - n_context

    predictions = np.full(n_predict, np.nan)

    model.eval()
    with torch.no_grad():
        for t in range(window_size, n_total):
            # ── Forward pass ─────────────────────────────────────────────
            w = features[t - window_size : t]
            X = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)
            pred_norm = float(model(X).squeeze().cpu().numpy())

            # Only score and store predictions for the prediction window
            if t < n_context:
                continue

            pred_idx = t - n_context

            # Denormalise and clip
            y_hat = max(float(scaler.inverse_transform([[pred_norm]])[0][0]), 0.0)
            predictions[pred_idx] = y_hat

            # ── Decide whether to mask ────────────────────────────────────
            y_t = actual_diff[t]
            skip = bool(np.isnan(y_t))

            if not skip and ref_std is not None and ref_std > 0:
                residual = abs(y_t - y_hat)
                z = (residual - ref_mean) / ref_std
                if abs(z) > z_threshold:
                    skip = True

            # ── If anomalous: replace actual with predicted in feature matrix
            # so future windows use the model's output as context, not the spike
            if skip:
                features[t, DIFF_NORM_COL] = pred_norm

    return predictions


# ============================================================================
# SINGLE-METER PROCESSING
# ============================================================================

def process_single_meter(
    csv_filepath: str,
    device: Optional[torch.device] = None,
    epochs_train: int = 10,
    epochs_warmstart: int = 5,
    verbose: bool = False,
    predictions_output_csv: Optional[str] = None,
    threshold_z_score: int = 3,
    threshold_z_score_robust: int = 3,
    temp_dir: str = "./temp_models_gru",
) -> Dict:
    """
    Process one water meter CSV through the sliding-window GRU pipeline.

    The split, seed, resampling, and metrics are identical to the UC model
    so that results are directly comparable.

    Anomaly scoring:
    - Thresholding statistics (mean, std, median, MAD) are estimated from the
      second training segment residuals and then applied to score the prediction
      segment residuals, so that the calibration window is anomaly-free.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {
        "filename": Path(csv_filepath).stem,
        "filepath": csv_filepath,
        "status":   "processing",
        "error":    None,
    }

    try:
        if verbose:
            logger.info(f"Processing {result['filename']}...")

        df_raw = pd.read_csv(csv_filepath)

        if "timestamp_utc" not in df_raw.columns:
            raise ValueError(f"No timestamp_utc column. Available: {df_raw.columns.tolist()}")
        if "hodnota" not in df_raw.columns:
            raise ValueError(f"No 'hodnota' column. Available: {df_raw.columns.tolist()}")
        if len(df_raw) < 2:
            raise ValueError("DataFrame too short (len < 2)")

        df_raw["timestamp_utc"] = pd.to_datetime(df_raw["timestamp_utc"], utc=True)
        df_raw.dropna(subset=["timestamp_utc"], inplace=True)

        # ===================================================================
        # SPLIT  – identical seed / logic to UC model
        # ===================================================================
        seed = int(result["filename"])
        df_train, df_second, df_predict = split_df_sliding_weeks(
            df_raw=df_raw, seed=seed, result=result
        )

        seasonal_period_steps = result["seasonal_period_steps"]
        if seasonal_period_steps < 2:
            raise ValueError("Seasonal period steps < 2.")

        # Use one daily cycle as the GRU sequence window (same as UC's seasonal period)
        window_size = seasonal_period_steps
        result["window_size"] = window_size

        if len(df_train) < window_size or len(df_second) < window_size or len(df_predict) < window_size:
            raise ValueError(
                f"A data split has fewer rows ({len(df_train)}, {len(df_second)}, "
                f"{len(df_predict)}) than window_size ({window_size})."
            )

        if verbose:
            logger.info(
                f"  window_size={window_size}, train={len(df_train)}, "
                f"second={len(df_second)}, predict={len(df_predict)}"
            )

        os.makedirs(temp_dir, exist_ok=True)
        temp_model_path      = os.path.join(temp_dir, f"{result['filename']}_initial.pt")
        warmstart_model_path = os.path.join(temp_dir, f"{result['filename']}_warmstart.pt")

        # ===================================================================
        # INITIAL TRAINING
        # ===================================================================
        t0 = time.time()
        model, scaler, losses_train = train_gru(
            df_train,
            window_size=window_size,
            model_path=temp_model_path,
            epochs=epochs_train,
            batch_size=32,
            hidden_size=32,
            learning_rate=1e-3,
            device=device,
            verbose=verbose,
        )
        result["train_train_time_seconds"] = time.time() - t0
        result["train_final_loss"]         = losses_train[-1] if losses_train else None

        if verbose:
            logger.info(f"  Initial training done.")

        # ===================================================================
        # WARM-START (second training)
        # ===================================================================
        t0 = time.time()
        model, scaler, losses_warmstart = train_gru(
            df_second,
            window_size=window_size,
            model_path=warmstart_model_path,
            epochs=epochs_warmstart,
            batch_size=32,
            hidden_size=32,
            learning_rate=5e-4,
            device=device,
            pretrained_model_path=temp_model_path,
            scaler=scaler,          # keep the same scaler from initial training
            verbose=verbose,
        )
        result["second_train_time_seconds"] = time.time() - t0
        result["warmstart_final_loss"]      = losses_warmstart[-1] if losses_warmstart else None

        if verbose:
            logger.info(f"  Warm-start done.")

        # ===================================================================
        # COMPUTE REFERENCE RESIDUALS FROM SECOND TRAINING SEGMENT
        # These residuals are used to estimate thresholding statistics
        # (mean, std, median, MAD) for anomaly scoring, ensuring the
        # calibration window is anomaly-free.
        # df_train is used as context (mirrors UC's init_second_mean/cov,
        # which carried state from the end of initial training into df_second).
        # ===================================================================
        if verbose:
            logger.info(f"  Computing reference residuals from second training segment...")

        t0 = time.time()
        second_predictions = predict_with_gru(
            model=model,
            df_predict=df_second,
            scaler=scaler,
            window_size=window_size,
            df_context=df_train,   # initial training segment provides history
            device=device,
        )
        result["second_prediction_time_seconds"] = time.time() - t0

        second_actuals  = df_second["Diff"].values.astype(float)
        second_residuals = np.abs(second_actuals - second_predictions)

        if verbose:
            logger.info(f"  Reference residuals computed.")

        # ===================================================================
        # INJECT ANOMALIES INTO PREDICTION SEGMENT
        # Injection happens after all training and after reference residuals
        # are computed, so model parameters and thresholding statistics are
        # not affected by the injected anomalies.
        # ===================================================================
        #df_predict = inject_spike_anomalies_diff(df_predict, random_state=int(result["filename"]))

        if verbose:
            logger.info(f"  Anomalies injected.")

        # ===================================================================
        # PREDICTION
        # df_second is used as context so the first prediction timestep has
        # a full window of history (mirrors UC's state carry-over).
        # Online masking: if a step is flagged as anomalous (|z| > threshold),
        # the predicted value is fed back as context for subsequent steps so
        # the spike does not corrupt future predictions.
        # ===================================================================
        t0 = time.time()
        predictions = predict_with_gru_online(
            model=model,
            df_predict=df_predict,
            scaler=scaler,
            window_size=window_size,
            df_context=df_second,
            device=device,
            ref_residuals=second_residuals,
            z_threshold=threshold_z_score,
        )
        result["prediction_time_seconds"] = time.time() - t0

        if verbose:
            logger.info(f"  Prediction done.")

        # ===================================================================
        # BUILD RESULTS DATAFRAME  (same helper as UC model)
        # Scores applied to prediction residuals, statistics estimated from
        # second training segment residuals.
        # ===================================================================
        predictions_df = build_results_df(
            df_predict, predictions, result,
            ref_residuals=second_residuals,
            threshold_z_score=threshold_z_score,
            threshold_z_score_robust=threshold_z_score_robust,
        )
        
        result["z_scores"] = predictions_df["z_score"].tolist()

        if predictions_output_csv is not None:
            predictions_df.to_csv(predictions_output_csv, index=False)

        # ===================================================================
        # METRICS  (same functions as UC model)
        # ===================================================================
        if len(predictions_df) < 2:
            raise ValueError(
                "Less than 2 data points in one of the prediction dfs "
                "for metric calculation."
            )

        metrics = calculate_metrics(predictions_df)
        for key, value in metrics.items():
            result[f"metric_{key}"] = value

        result["status"] = "success"
        result['converged_train'] = True
        result['converged_second'] = True
        
        if verbose:
            logger.info(
                f"Processed successfully (GRU): {result['filename']} - "
                f"RMSE: {metrics['rmse']:.4f}, "
                f"MAE: {metrics['mae']:.4f}, "
                f"R2: {metrics['r2']:.4f}"
            )

        # Cleanup temp files
        for p in [temp_model_path, warmstart_model_path]:
            if os.path.exists(p):
                os.remove(p)

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"]  = str(e)
        logger.error(f"Failed: {result['filename']} - Error: {e}")
        if verbose:
            logger.debug(traceback.format_exc())
        return result
    
    
from multiprocessing import Process, Queue
from collections import deque
import queue as py_queue
import time
from pathlib import Path
from typing import List
from tqdm import tqdm
import pandas as pd
import torch


def worker(
    csv_path: str,
    result_queue: Queue,
    epochs_train: int,
    epochs_warmstart: int,
    verbose: bool,
):
    filename = Path(csv_path).stem

    try:
        result = process_single_meter(
            csv_path,
            device=None,   # each child creates its own device
            epochs_train=epochs_train,
            epochs_warmstart=epochs_warmstart,
            verbose=verbose,
        )
    except Exception as e:
        result = {
            "filename": filename,
            "filepath": csv_path,
            "status": "failed",
            "error": str(e),
        }

    result_queue.put(result)


def process_batch_manual(
    csv_filepaths: List[str],
    output_csv: str,
    num_workers: int = 7,
    epochs_train: int = 5,
    epochs_warmstart: int = 5,
    verbose: bool = False,
    per_file_timeout: int = 300,
):
    """
    Process multiple meter files in parallel using manual multiprocessing.Process
    workers with hard per-file termination on timeout.

    Windows-friendly approach:
      - at most `num_workers` child processes alive at once
      - each file gets its own child process
      - timed-out child is terminated and replaced by a new one
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
        logger.info(f"Processing {len(csv_filepaths)} files with {num_workers} workers")
        logger.info(f"Per-file timeout: {per_file_timeout} seconds")

    pending = deque(csv_filepaths)
    active = []
    all_results = []
    result_queue = Queue()

    success_count = 0
    failed_count = 0
    timeout_count = 0
    processed_count = 0
    total_files = len(csv_filepaths)

    with tqdm(total=total_files, desc="Processing meters", unit="file") as pbar:
        while pending or active:
            # Fill free worker slots
            while pending and len(active) < num_workers:
                filepath = pending.popleft()
                p = Process(
                    target=worker,
                    args=(filepath, result_queue, epochs_train, epochs_warmstart, verbose),
                )
                p.start()

                active.append({
                    "proc": p,
                    "file": filepath,
                    "filename": Path(filepath).stem,
                    "start_time": time.time(),
                })

                if verbose:
                    logger.info(
                        f"Started {Path(filepath).stem} "
                        f"(pid={p.pid}, active={len(active)}/{num_workers}, queued={len(pending)})"
                    )

            new_active = []

            for entry in active:
                p = entry["proc"]
                filepath = entry["file"]
                filename = entry["filename"]
                start_time = entry["start_time"]

                # Finished normally
                if not p.is_alive():
                    p.join(timeout=0.2)

                    result = None
                    try:
                        while True:
                            candidate = result_queue.get_nowait()
                            if candidate.get("filename") == filename:
                                result = candidate
                                break
                            else:
                                all_results.append(candidate)
                                processed_count += 1
                                if candidate.get("status") == "success":
                                    success_count += 1
                                else:
                                    failed_count += 1
                                pbar.update(1)
                    except py_queue.Empty:
                        pass

                    if result is None:
                        result = {
                            "filename": filename,
                            "filepath": filepath,
                            "status": "failed",
                            "error": "Worker exited without returning a result",
                        }

                    all_results.append(result)
                    processed_count += 1

                    if result.get("status") == "success":
                        success_count += 1
                    else:
                        failed_count += 1

                    pbar.update(1)
                    pbar.set_postfix(
                        success=success_count,
                        failed=failed_count,
                        timeout=timeout_count,
                        running=len(new_active),
                        queued=len(pending),
                    )

                    if verbose:
                        logger.info(f"[{processed_count}/{total_files}] Completed {filename}")

                # Timed out
                elif time.time() - start_time > per_file_timeout:
                    logger.error(
                        f"[{processed_count + 1}/{total_files}] Timeout processing {filename} "
                        f"(>{per_file_timeout}s). Terminating child process pid={p.pid}."
                    )

                    p.terminate()
                    p.join(timeout=1)

                    result = {
                        "filename": filename,
                        "filepath": filepath,
                        "status": "failed",
                        "error": f"Timeout after {per_file_timeout}s",
                    }

                    all_results.append(result)
                    processed_count += 1
                    failed_count += 1
                    timeout_count += 1

                    pbar.update(1)
                    pbar.set_postfix(
                        success=success_count,
                        failed=failed_count,
                        timeout=timeout_count,
                        running=len(new_active),
                        queued=len(pending),
                    )

                # Still running and within timeout
                else:
                    new_active.append(entry)

            active = new_active
            time.sleep(0.2)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    
    zscore_rows = []
    for r in all_results:
        if r.get("status") == "success" and "z_scores" in r:
            for z in r["z_scores"]:
                zscore_rows.append({"filename": r["filename"], "z_score": z})
    
    pd.DataFrame(zscore_rows).to_csv(output_csv.replace(".csv", "_zscores.csv"), index=False)

    logger.info("=" * 70)
    logger.info("BATCH PROCESSING COMPLETE (GRU Sliding Window)")
    logger.info("=" * 70)
    logger.info(f"Total processed : {len(results_df)}")
    logger.info(f"Successful      : {(results_df['status'] == 'success').sum()}")
    logger.info(f"Failed          : {(results_df['status'] == 'failed').sum()}")
    logger.info(f"Metrics saved to: {output_csv}")

    return results_df


# ============================================================================
# BATCH PROCESSING  (ProcessPoolExecutor with timeout – mirrors UC model)
# ============================================================================

def process_batch(
    csv_filepaths: List[str],
    output_csv: str,
    num_workers: int = 7,
    epochs_train: int = 5,
    epochs_warmstart: int = 5,
    verbose: bool = False,
    per_file_timeout: int = 300,
):
    """
    Process multiple meter files in parallel using ProcessPoolExecutor
    (same executor type and timeout mechanism as the UC model).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        logger.info(f"Using device: {device}")
        logger.info(f"Processing {len(csv_filepaths)} files with {num_workers} workers")
        logger.info(f"Per-file timeout: {per_file_timeout} seconds")

    all_results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for filepath in csv_filepaths:
            future = executor.submit(
                process_single_meter,
                filepath,
                device=None,          # each worker creates its own
                epochs_train=epochs_train,
                epochs_warmstart=epochs_warmstart,
                verbose=verbose,
            )
            futures[future] = Path(filepath).stem

        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Processing meters"), 1
        ):
            filename = futures[future]
            try:
                result = future.result(timeout=per_file_timeout)
                all_results.append(result)
                if verbose:
                    logger.info(f"[{i}/{len(futures)}] Completed {filename}")
            except TimeoutError:
                logger.error(
                    f"[{i}/{len(futures)}] Timeout processing {filename} "
                    f"(>{per_file_timeout}s). Marking as failed."
                )
                future.cancel()
                all_results.append({
                    "filename": filename,
                    "status":   "failed",
                    "error":    f"Timeout after {per_file_timeout}s",
                })
            except Exception as e:
                logger.error(f"[{i}/{len(futures)}] Failed {filename}: {e}")
                all_results.append({
                    "filename": filename,
                    "status":   "failed",
                    "error":    str(e),
                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)

    logger.info("=" * 70)
    logger.info("BATCH PROCESSING COMPLETE (GRU Sliding Window)")
    logger.info("=" * 70)
    logger.info(f"Total processed : {len(results_df)}")
    logger.info(f"Successful      : {(results_df['status'] == 'success').sum()}")
    logger.info(f"Failed          : {(results_df['status'] == 'failed').sum()}")
    logger.info(f"Metrics saved to: {output_csv}")

    return results_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GRU Sliding-Window Batch Processor (comparable to UC model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workers",          type=int, default=7)
    parser.add_argument("--epochs-train",     type=int, default=5)
    parser.add_argument("--epochs-warmstart", type=int, default=5)
    parser.add_argument("--samples",          type=int, default=1000)
    parser.add_argument("--verbose",          action="store_true", default=False)

    args = parser.parse_args()

    global logger
    logger = setup_logging(verbose=args.verbose)


    seed_value = 42
    directory = "../../../data/sensor_data"

    with open("../../pickles/test_set.pkl", "rb") as f:
        csv_filenames = pickle.load(f)

    csv_filepaths = [os.path.join(directory, name) for name in csv_filenames]

    #with open("../pickles/common_sensors.pkl", "rb") as f:
    #    csv_filepaths = pickle.load(f)
    
    logger.info(f"Loaded {len(csv_filepaths)} CSV filepaths")

    output_csv = (
        f"./6_weeks_results_gru_{args.samples}_seed_42_"
        f"epochs_{args.epochs_train}_ws_{args.epochs_warmstart}_sliding_window_anomalies_common_z_score.csv"
    )

    _ = process_batch_manual(
        csv_filepaths,
        output_csv,
        num_workers=args.workers,
        epochs_train=args.epochs_train,
        epochs_warmstart=args.epochs_warmstart,
        verbose=args.verbose,
    )

    logger.info(f"Output saved to: {output_csv}")


if __name__ == "__main__":
    main()