import numpy as np
import logging
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import math
import pandas as pd
from io import StringIO

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


def _binary_classification_metrics(y_true, y_pred):
    """
    Compute TP, TN, FP, FN, accuracy, precision, recall, f1
    for binary labels (0/1). Returns a dict.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Drop NaNs if present
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)

    if y_true.size == 0:
        return {
            "tp": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
        }

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

    }


def calculate_metrics(predictions_df):
    """
    Calculate regression metrics and anomaly detection metrics.

    Expected columns:
        "timestamp_utc"
        "actual"
        "predicted"
        "residual"
        "is_anomaly_actual"             # float, 0 = normal, non-zero = anomaly
        "is_anomaly_predicted"          # int, 0/1
        "is_anomaly_robust_predicted"   # int, 0/1
    """
    metrics = {}

    actuals = predictions_df["actual"].values.astype(float)
    timestamps = predictions_df["timestamp_utc"].values
    predictions = predictions_df["predicted"].values
    residuals = predictions_df["residual"].values

    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)
    residuals = np.asarray(residuals)
    timestamps = pd.to_datetime(timestamps, utc=True)

    valid_mask = ~(np.isnan(actuals) | np.isnan(predictions) | np.isnan(residuals))
    valid_actuals = actuals[valid_mask]
    valid_predictions = predictions[valid_mask]
    valid_residuals = residuals[valid_mask]

    metrics["total_count"] = len(actuals)
    metrics["valid_count"] = np.sum(valid_mask)
    metrics["nan_count"] = np.sum(~valid_mask)
    metrics["valid_percentage"] = (
        (np.sum(valid_mask) / len(actuals)) * 100 if len(actuals) > 0 else 0
    )

    if len(valid_actuals) == 0:
        logger.warning("No valid (non-NaN) data points for metrics calculation")
        metrics.update({
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "mape": np.nan,
            "wape": np.nan,
            "mean_residual": np.nan,
            "std_residual": np.nan,
            "max_residual": np.nan,
            "min_residual": np.nan,
            "normalized_rmse": np.nan,
            "median_ape": np.nan,
            "prediction_bias": np.nan,
            "mpe": np.nan,
            "direction_accuracy": np.nan,
            "medae": np.nan,
            "medae_nz": np.nan,
            "p75_ae": np.nan,
            "p90_ae": np.nan,
        })
        # classification metrics placeholders
        for prefix in ["pred", "robust_pred"]:
            metrics.update({
                f"{prefix}_tp": np.nan,
                f"{prefix}_tn": np.nan,
                f"{prefix}_fp": np.nan,
                f"{prefix}_fn": np.nan,
                f"{prefix}_accuracy": np.nan,
                f"{prefix}_precision": np.nan,
                f"{prefix}_recall": np.nan,
                f"{prefix}_f1": np.nan,
            })
        return metrics

    # ===== Regression metrics =====
    metrics["rmse"] = np.sqrt(mean_squared_error(valid_actuals, valid_predictions))
    metrics["mae"] = mean_absolute_error(valid_actuals, valid_predictions)
    metrics["r2"] = r2_score(valid_actuals, valid_predictions)
    metrics["medae"] = np.nanmedian(np.abs(valid_actuals - valid_predictions))

    # ===== Prepare anomaly labels =====
    if "is_anomaly_actual" in predictions_df.columns:
        # ground truth: 1 if non-zero, 0 otherwise
        y_true = (predictions_df["is_anomaly_actual"].fillna(0).values != 0).astype(int)
    else:
        y_true = None

    has_pred = "is_anomaly_predicted" in predictions_df.columns
    has_robust = "is_anomaly_robust_predicted" in predictions_df.columns

    if y_true is not None and has_pred:
        y_pred = predictions_df["is_anomaly_predicted"].values.astype(int)
        m_pred = _binary_classification_metrics(y_true, y_pred)
        metrics.update({
            "pred_tp": m_pred["tp"],
            "pred_tn": m_pred["tn"],
            "pred_fp": m_pred["fp"],
            "pred_fn": m_pred["fn"],
        })
    else:
        metrics.update({
            "pred_tp": np.nan,
            "pred_tn": np.nan,
            "pred_fp": np.nan,
            "pred_fn": np.nan,
        })

    if y_true is not None and has_robust:
        y_pred_rob = predictions_df["is_anomaly_robust_predicted"].values.astype(int)
        m_rob = _binary_classification_metrics(y_true, y_pred_rob)
        metrics.update({
            "robust_pred_tp": m_rob["tp"],
            "robust_pred_tn": m_rob["tn"],
            "robust_pred_fp": m_rob["fp"],
            "robust_pred_fn": m_rob["fn"],
        })
    else:
        metrics.update({
            "robust_pred_tp": np.nan,
            "robust_pred_tn": np.nan,
            "robust_pred_fp": np.nan,
            "robust_pred_fn": np.nan,
        })

    # MAPE
    try:
        metrics["mape"] = mean_absolute_percentage_error(
            valid_actuals, valid_predictions
        )
    except Exception:
        metrics["mape"] = np.nan
        
    # WAPE
    denom = np.sum(np.abs(valid_actuals))
    if denom == 0:
        metrics["wape"] = np.nan
    else:
        metrics["wape"] = np.sum(np.abs(valid_actuals - valid_predictions)) / denom

    # Residual statistics
    metrics["mean_residual"] = np.nanmean(valid_residuals)
    metrics["std_residual"] = np.nanstd(valid_residuals)
    metrics["max_residual"] = np.nanmax(valid_residuals)
    metrics["min_residual"] = np.nanmin(valid_residuals)
    
    # 75th and 90th percentile of absolute error (size of “typical worst” errors)
    abs_errors = np.abs(valid_actuals - valid_predictions)
    metrics["p75_ae"] = np.nanpercentile(abs_errors, 75)
    metrics["p90_ae"] = np.nanpercentile(abs_errors, 90)

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
    
    # Signed percentage bias (MPE)
    try:
        pe_values = (valid_actuals - valid_predictions) / (np.abs(valid_actuals) + 1e-8)
        metrics["mpe"] = np.nanmean(pe_values)
    except Exception:
        metrics["mpe"] = np.nan

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

    if non_zero_count > 2:
        actuals_nz = valid_actuals[non_zero_mask]
        predictions_nz = valid_predictions[non_zero_mask]
        residuals_nz = valid_residuals[non_zero_mask]

        metrics["rmse_nz"] = np.sqrt(mean_squared_error(actuals_nz, predictions_nz))
        metrics["mae_nz"] = mean_absolute_error(actuals_nz, predictions_nz)
        metrics["medae_nz"] = np.nanmedian(np.abs(actuals_nz - predictions_nz))

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
            
        denom_nz = np.sum(np.abs(actuals_nz))
        if denom_nz == 0:
            metrics["wape_nz"] = np.nan
        else:
            metrics["wape_nz"] = np.sum(np.abs(actuals_nz - predictions_nz)) / denom_nz

        metrics["mean_residual_nz"] = np.nanmean(residuals_nz)
        metrics["std_residual_nz"] = np.nanstd(residuals_nz)
        metrics["max_residual_nz"] = np.nanmax(residuals_nz)
        metrics["min_residual_nz"] = np.nanmin(residuals_nz)
        
        # 75th and 90th percentile AE on non-zero subset
        abs_errors_nz = np.abs(actuals_nz - predictions_nz)
        metrics["p75_ae_nz"] = np.nanpercentile(abs_errors_nz, 75)
        metrics["p90_ae_nz"] = np.nanpercentile(abs_errors_nz, 90)

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

        try:
            pe_values_nz = (actuals_nz - predictions_nz) / (np.abs(actuals_nz) + 1e-8)
            metrics["mpe_nz"] = np.nanmean(pe_values_nz)
        except Exception:
            metrics["mpe_nz"] = np.nan

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
        metrics["medae_nz"] = np.nan
        metrics["r2_nz"] = np.nan
        metrics["mape_nz"] = np.nan
        metrics["wape_nz"] = np.nan
        metrics["mean_residual_nz"] = np.nan
        metrics["std_residual_nz"] = np.nan
        metrics["max_residual_nz"] = np.nan
        metrics["min_residual_nz"] = np.nan
        metrics["normalized_rmse_nz"] = np.nan
        metrics["median_ape_nz"] = np.nan
        metrics["prediction_bias_nz"] = np.nan
        metrics["mpe_nz"] = np.nan
        metrics["direction_accuracy_nz"] = np.nan
        metrics["p75_ae_nz"] = np.nan
        metrics["p90_ae_nz"] = np.nan

    return metrics


def calculate_metrics_unresampled(df_predict_unresampled, predictions_df):
    actual = df_predict_unresampled.copy()
    pred = predictions_df.copy()

    # Ensure datetime and sort
    for df in (actual, pred):
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])

    # Build aligned dataframe:
    # for each actual point at time t_i, sum predictions in (t_{i-1}, t_i]
    rows = []
    prev_ts = None
    
    for ts, diff in actual[['timestamp_utc', 'Diff']].itertuples(index=False):
        if prev_ts is None:
            prev_ts = ts
            continue

        mask = (pred['timestamp_utc'] > prev_ts) & (pred['timestamp_utc'] <= ts)
        if not mask.any():
            prev_ts = ts
            continue

        pred_sum = pred.loc[mask, 'predicted'].sum()
        rows.append({
            'timestamp_utc': ts,
            'actual': diff,
            'predicted': pred_sum,
            'residual': 0.0,
            'is_anomaly': 0.0
        })
        prev_ts = ts

    aligned = pd.DataFrame(rows)
    if aligned.empty:
        raise ValueError("No overlapping intervals between actual and predictions.")
    
    return calculate_metrics(aligned)

    # Metrics
    #mae = (y_true.sub(y_pred).abs()).mean()
    #mse = ((y_true - y_pred) ** 2).mean()
    #rmse = math.sqrt(mse)
#
    ## MAPE: ignore actual==0 to avoid division by zero
    #nonzero = y_true != 0
    #if nonzero.any():
    #    mape = (y_true[nonzero].sub(y_pred[nonzero]).abs() /
    #            y_true[nonzero].abs()).mean()
    #else:
    #    mape = float('nan')
#
    #return {
    #    "aligned_df": aligned,  # you can inspect which points were used
    #    "mae": mae,
    #    "mse": mse,
    #    "rmse": rmse,
    #    "mape": mape,
    #}
    #return aligned

def test():
    csv_pred = """timestamp_utc,actual,predicted,residual,z_score,z_score_robust,is_anomaly,is_anomaly_robust
    2023-07-24 08:44:00+00:00,,0.005723836807693697,,,,0,0
    2023-07-24 09:13:52+00:00,,0.005648684790657317,,,,0,0
    2023-07-24 09:43:52+00:00,0.014999999999986358,0.004805537884248945,0.010194462115737412,,,0,0
    2023-07-24 10:13:52+00:00,0.010000000000047748,0.01009693806298883,9.69380629410807e-05,,,0,0
    2023-07-24 10:43:52+00:00,0.0,0.012151712499822356,0.012151712499822356,,,0,0
    2023-07-24 11:13:52+00:00,0.007000000000005002,0.008482838116734403,0.0014828381167294007,,,0,0
    2023-07-24 11:43:52+00:00,0.0,0.0074744249774939715,0.0074744249774939715,,,0,0
    2023-07-24 12:13:52+00:00,,0.0070024023864948114,,,,0,0
    2023-07-24 12:43:52+00:00,,0.005981688276656296,,,,0,0
    2023-07-24 13:13:52+00:00,,0.00781099571627746,,,,0,0
    2023-07-24 13:43:52+00:00,0.0,0.005565898191237439,0.005565898191237439,,,0,0
    2023-07-24 14:13:52+00:00,0.0,0.0035001773971945404,0.0035001773971945404,,,0,0
    2023-07-24 14:43:52+00:00,0.0,0.0022452658768449566,0.0022452658768449566,,,0,0
    2023-07-24 15:13:52+00:00,0.03000000000002956,0.0011721478163443178,0.02882785218368524,,,0,0
    2023-07-24 15:43:52+00:00,0.0,0.007625534007573686,0.007625534007573686,,,0,0
    2023-07-24 16:13:58+00:00,0.007000000000005002,0.005709336979652932,0.00129066302035207,,,0,0
    """
    
    predictions_df = pd.read_csv(StringIO(csv_pred))

    # Irregular actual DF (df_predict_unresampled)
    df_predict_unresampled = pd.DataFrame(
        {
            "timestamp_utc": [
                "2023-07-24 09:43:52+00:00",
                "2023-07-24 11:13:52+00:00",
                "2023-07-24 13:43:52+00:00",
                "2023-07-24 16:13:58+00:00",
            ],
            "Diff": [
                0.015,  # at 11:13:52 we will compare to predictions in (09:43:52, 11:13:52]
                0.007,  # at 13:43:52 compare to (11:13:52, 13:43:52]
                0.0,    # at 16:13:58 compare to (13:43:52, 16:13:58]
                0.007,
            ],
        }
    )

    aligned = calculate_metrics_unresampled(df_predict_unresampled, predictions_df)

    # Expected aligned_df (values rounded for readability)
    expected = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2023-07-24 11:13:52+00:00",
                    "2023-07-24 13:43:52+00:00",
                    "2023-07-24 16:13:58+00:00",
                ]
            ),
            "actual": [0.007, 0.0, 0.007],
            "predicted_agg": [
                # (09:43:52, 11:13:52] → 10:13:52 + 10:43:52 + 11:13:52
                0.01009693806298883
                + 0.012151712499822356
                + 0.008482838116734403,
                # (11:13:52, 13:43:52] → 11:43:52 + 12:13:52 + 12:43:52 + 13:13:52 + 13:43:52
                0.0074744249774939715
                + 0.0070024023864948114
                + 0.005981688276656296
                + 0.00781099571627746
                + 0.005565898191237439,
                # (13:43:52, 16:13:58] → 14:13:52 + 14:43:52 + 15:13:52 + 15:43:52 + 16:13:58
                0.0035001773971945404
                + 0.0022452658768449566
                + 0.0011721478163443178
                + 0.007625534007573686
                + 0.005709336979652932,
            ],
        }
    )

    # Assert almost equal (float round)
    pd.testing.assert_frame_equal(
        aligned.sort_values("timestamp_utc").reset_index(drop=True).round(6),
        expected.sort_values("timestamp_utc").reset_index(drop=True).round(6),
    )
