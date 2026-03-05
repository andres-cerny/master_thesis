import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(actuals, predictions, residuals):
    """
    Calculate comprehensive metrics for predictions, including both all-data
    and non-zero-only metrics.

    Properly handles NaN values in actuals and predictions.
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
        return metrics

    # ====================
    # ALL VALID DATA METRICS
    # ====================

    # Basic metrics
    metrics["rmse"] = np.sqrt(mean_squared_error(valid_actuals, valid_predictions))
    metrics["mae"] = mean_absolute_error(valid_actuals, valid_predictions)
    metrics["r2"] = r2_score(valid_actuals, valid_predictions)
    metrics["medae"] = np.nanmedian(np.abs(valid_actuals - valid_predictions))

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

    if non_zero_count > 0:
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