import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats


class KalmanFilterMetrics:
    """
    Metrics for evaluating Kalman Filter performance on water meter data.
    
    Compares filtered estimates to raw measurements to assess:
    - Prediction accuracy
    - Smoothing effectiveness
    - Anomaly detection quality
    - Filter stability
    """
    
    @staticmethod
    def compute_all_metrics(df_filtered: pd.DataFrame,
                           measurement_col: str = 'Diff',
                           prediction_col: str = 'consumption_pred',
                           filtered_col: str = 'consumption_filtered',
                           anomaly_col: str = 'is_anomaly',
                           nonzero_threshold: float = 0.0) -> Dict:
        """
        Compute comprehensive set of filter performance metrics.
        
        Parameters:
        -----------
        df_filtered : pd.DataFrame
            Filtered dataframe from KalmanFilter.filter_dataframe()
        measurement_col : str
            Column name for raw measurements
        prediction_col : str
            Column name for predictions
        filtered_col : str
            Column name for filtered estimates
        anomaly_col : str
            Column name for anomaly flags
        nonzero_threshold : float
            Minimum absolute measurement value to be considered non-zero
        
        Returns:
        --------
        metrics : dict
            Dictionary containing all computed metrics (both full and non-zero)
        """
        measurements = df_filtered[measurement_col].values
        predictions = df_filtered[prediction_col].values
        filtered = df_filtered[filtered_col].values
        
        metrics = {}
        
        # 1. Prediction Error Metrics (all data)
        metrics.update(KalmanFilterMetrics._prediction_error_metrics(
            measurements, predictions
        ))
        
        # 2. Filtering/Smoothing Metrics (all data)
        metrics.update(KalmanFilterMetrics._smoothing_metrics(
            measurements, filtered
        ))
        
        # 3. NON-ZERO Metrics
        metrics.update(KalmanFilterMetrics._nonzero_metrics(
            measurements, predictions, filtered, nonzero_threshold
        ))
        
        # 4. Innovation Metrics
        if 'innovation' in df_filtered.columns:
            metrics.update(KalmanFilterMetrics._innovation_metrics(
                df_filtered['innovation'].values
            ))
        
        # 5. Anomaly Detection Metrics (if anomalies present)
        if anomaly_col in df_filtered.columns and df_filtered[anomaly_col].sum() > 0:
            metrics.update(KalmanFilterMetrics._anomaly_metrics(
                df_filtered, measurement_col, filtered_col, anomaly_col
            ))
        
        # 6. Filter Stability Metrics
        if 'P_pred_offset_var' in df_filtered.columns:
            metrics.update(KalmanFilterMetrics._stability_metrics(
                df_filtered
            ))
        
        return metrics
    
    @staticmethod
    def _prediction_error_metrics(measurements: np.ndarray,
                                   predictions: np.ndarray) -> Dict:
        """
        Metrics measuring prediction accuracy.
        """
        errors = measurements - predictions
        
        return {
            # Mean Absolute Error
            'MAE_prediction': np.mean(np.abs(errors)),
            
            # Root Mean Squared Error
            'RMSE_prediction': np.sqrt(np.mean(errors**2)),
            
            # Mean Error (bias)
            'ME_prediction': np.mean(errors),
            
            # Median Absolute Error (robust to outliers)
            'MedAE_prediction': np.median(np.abs(errors)),
            
            # Mean Absolute Percentage Error (if no zeros)
            'MAPE_prediction': np.mean(np.abs(errors / (measurements + 1e-10))) * 100,
            
            # Normalized RMSE (by measurement std)
            'NRMSE_prediction': np.sqrt(np.mean(errors**2)) / (np.std(measurements) + 1e-10),
            
            # R-squared (coefficient of determination)
            'R2_prediction': 1 - (np.sum(errors**2) / (np.sum((measurements - np.mean(measurements))**2) + 1e-10))
        }
    
    @staticmethod
    def _smoothing_metrics(measurements: np.ndarray,
                          filtered: np.ndarray) -> Dict:
        """
        Metrics measuring smoothing/filtering effectiveness.
        """
        errors = measurements - filtered
        
        # Smoothness: variance of first differences
        meas_smoothness = np.var(np.diff(measurements))
        filt_smoothness = np.var(np.diff(filtered))
        
        return {
            # Filtering Error Metrics
            'MAE_filtered': np.mean(np.abs(errors)),
            'RMSE_filtered': np.sqrt(np.mean(errors**2)),
            'ME_filtered': np.mean(errors),
            'MedAE_filtered': np.median(np.abs(errors)),
            
            # Smoothness Metrics
            'measurement_variance': np.var(measurements),
            'filtered_variance': np.var(filtered),
            'variance_reduction_ratio': np.var(filtered) / (np.var(measurements) + 1e-10),
            
            # First-difference smoothness (lower = smoother)
            'measurement_smoothness': meas_smoothness,
            'filtered_smoothness': filt_smoothness,
            'smoothness_improvement': (meas_smoothness - filt_smoothness) / (meas_smoothness + 1e-10),
            
            # Correlation with measurements (should be high)
            'correlation_meas_filt': np.corrcoef(measurements, filtered)[0, 1],
            
            # Lag-1 autocorrelation (measure of smoothness)
            'filtered_autocorr_lag1': np.corrcoef(filtered[:-1], filtered[1:])[0, 1]
        }
    
    @staticmethod
    def _nonzero_metrics(measurements: np.ndarray,
                        predictions: np.ndarray,
                        filtered: np.ndarray,
                        threshold: float = 0.0) -> Dict:
        """
        Metrics calculated only on non-zero measurements.
        
        Parameters:
        -----------
        measurements : np.ndarray
            True measurement values
        predictions : np.ndarray
            Predicted values
        filtered : np.ndarray
            Filtered values
        threshold : float
            Minimum absolute value to be considered non-zero
        
        Returns:
        --------
        metrics : dict
            Metrics computed only on non-zero measurements
        """
        # Identify non-zero measurements
        is_nonzero = np.abs(measurements) > threshold
        n_nonzero = np.sum(is_nonzero)
        n_total = len(measurements)
        
        metrics = {
            'n_nonzero_measurements': int(n_nonzero),
            'n_total_measurements': int(n_total),
            'nonzero_percentage': (n_nonzero / n_total) * 100 if n_total > 0 else 0.0
        }
        
        if n_nonzero == 0:
            metrics['nonzero_note'] = "No non-zero measurements found"
            return metrics
        
        # Extract non-zero data
        nz_meas = measurements[is_nonzero]
        nz_pred = predictions[is_nonzero]
        nz_filt = filtered[is_nonzero]
        
        # Prediction errors on non-zero data
        errors_pred = nz_meas - nz_pred
        errors_filt = nz_meas - nz_filt
        
        # Prediction metrics (non-zero)
        metrics.update({
            'MAE_prediction_nonzero': np.mean(np.abs(errors_pred)),
            'RMSE_prediction_nonzero': np.sqrt(np.mean(errors_pred**2)),
            'ME_prediction_nonzero': np.mean(errors_pred),
            'MedAE_prediction_nonzero': np.median(np.abs(errors_pred)),
            'R2_prediction_nonzero': 1 - (np.sum(errors_pred**2) / (np.sum((nz_meas - np.mean(nz_meas))**2) + 1e-10)),
        })
        
        # Filtering metrics (non-zero)
        metrics.update({
            'MAE_filtered_nonzero': np.mean(np.abs(errors_filt)),
            'RMSE_filtered_nonzero': np.sqrt(np.mean(errors_filt**2)),
            'ME_filtered_nonzero': np.mean(errors_filt),
            'MedAE_filtered_nonzero': np.median(np.abs(errors_filt)),
            'R2_filtered_nonzero': 1 - (np.sum(errors_filt**2) / (np.sum((nz_meas - np.mean(nz_meas))**2) + 1e-10)),
        })
        
        # Variance and smoothing (non-zero)
        meas_smoothness_nz = np.var(np.diff(nz_meas)) if len(nz_meas) > 1 else 0
        filt_smoothness_nz = np.var(np.diff(nz_filt)) if len(nz_filt) > 1 else 0
        
        metrics.update({
            'measurement_variance_nonzero': np.var(nz_meas),
            'filtered_variance_nonzero': np.var(nz_filt),
            'variance_reduction_ratio_nonzero': np.var(nz_filt) / (np.var(nz_meas) + 1e-10),
            'measurement_smoothness_nonzero': meas_smoothness_nz,
            'filtered_smoothness_nonzero': filt_smoothness_nz,
            'smoothness_improvement_nonzero': (meas_smoothness_nz - filt_smoothness_nz) / (meas_smoothness_nz + 1e-10),
            'correlation_meas_filt_nonzero': np.corrcoef(nz_meas, nz_filt)[0, 1] if len(nz_meas) > 1 else 0.0
        })
        
        return metrics
    
    @staticmethod
    def _innovation_metrics(innovations: np.ndarray) -> Dict:
        """
        Metrics for innovation sequence (should be white noise if filter is optimal).
        """
        # Normality test
        _, p_value_normality = stats.normaltest(innovations)
        
        # Autocorrelation at lag 1 (should be near 0 for white noise)
        if len(innovations) > 1:
            autocorr_lag1 = np.corrcoef(innovations[:-1], innovations[1:])[0, 1]
        else:
            autocorr_lag1 = 0.0
        
        return {
            'innovation_mean': np.mean(innovations),
            'innovation_std': np.std(innovations),
            'innovation_skewness': stats.skew(innovations),
            'innovation_kurtosis': stats.kurtosis(innovations),
            'innovation_normality_pvalue': p_value_normality,
            'innovation_autocorr_lag1': autocorr_lag1,
            'innovation_is_white_noise': (np.abs(autocorr_lag1) < 0.1 and p_value_normality > 0.05)
        }
    
    @staticmethod
    def _anomaly_metrics(df: pd.DataFrame,
                        measurement_col: str,
                        filtered_col: str,
                        anomaly_col: str) -> Dict:
        """
        Metrics for anomaly detection performance.
        """
        anomalies = df[anomaly_col]
        measurements = df[measurement_col].values
        filtered = df[filtered_col].values
        
        n_anomalies = anomalies.sum()
        n_total = len(df)
        
        # Error magnitude at anomalies vs normal points
        anomaly_errors = np.abs(measurements[anomalies] - filtered[anomalies])
        normal_errors = np.abs(measurements[~anomalies] - filtered[~anomalies])
        
        return {
            'n_anomalies_detected': int(n_anomalies),
            'anomaly_rate_percent': (n_anomalies / n_total) * 100,
            
            # Error at anomalies
            'MAE_at_anomalies': np.mean(anomaly_errors) if len(anomaly_errors) > 0 else 0,
            'MAE_at_normal': np.mean(normal_errors) if len(normal_errors) > 0 else 0,
            
            # Anomaly suppression (how much filter reduces anomaly magnitude)
            'anomaly_suppression_ratio': np.mean(anomaly_errors) / (np.mean(normal_errors) + 1e-10) if len(anomaly_errors) > 0 else 0,
            
            # Standardized innovation at anomalies
            'avg_std_innovation_at_anomalies': df.loc[anomalies, 'standardized_innovation'].abs().mean() if 'standardized_innovation' in df.columns else 0
        }
    
    @staticmethod
    def _stability_metrics(df: pd.DataFrame) -> Dict:
        """
        Metrics for filter stability (covariance and gain evolution).
        """
        metrics = {}
        
        if 'K_offset' in df.columns:
            K = df['K_offset'].values
            metrics.update({
                'K_mean': np.mean(K),
                'K_std': np.std(K),
                'K_min': np.min(K),
                'K_max': np.max(K),
                'K_is_stable': np.std(K) < 0.1  # Stable if gain doesn't vary much
            })
        
        if 'P_pred_offset_var' in df.columns:
            P = df['P_pred_offset_var'].values
            metrics.update({
                'P_pred_mean': np.mean(P),
                'P_pred_std': np.std(P),
                'P_pred_min': np.min(P),
                'P_pred_max': np.max(P),
                'P_is_stable': (np.max(P) / (np.min(P) + 1e-10)) < 10  # Stable if ratio < 10
            })
        
        return metrics
    
    @staticmethod
    def print_metrics_report(metrics: Dict, title: str = "Kalman Filter Performance Metrics"):
        """
        Print a formatted report of all metrics.
        """
        print("\n" + "="*70)
        print(title)
        print("="*70)
        
        # Group metrics by category
        categories = {
            'Data Summary': [k for k in metrics.keys() if k.startswith('n_') or 'percentage' in k.lower()],
            'Prediction Accuracy (All Data)': [k for k in metrics.keys() if 'prediction' in k.lower() and 'nonzero' not in k.lower()],
            'Prediction Accuracy (Non-Zero Only)': [k for k in metrics.keys() if 'prediction' in k.lower() and 'nonzero' in k.lower()],
            'Filtering/Smoothing (All Data)': [k for k in metrics.keys() if ('filtered' in k.lower() or 'variance' in k.lower() or 'smooth' in k.lower() or 'correlation' in k.lower()) and 'nonzero' not in k.lower()],
            'Filtering/Smoothing (Non-Zero Only)': [k for k in metrics.keys() if ('filtered' in k.lower() or 'variance' in k.lower() or 'smooth' in k.lower() or 'correlation' in k.lower()) and 'nonzero' in k.lower()],
            'Innovation (Residuals)': [k for k in metrics.keys() if 'innovation' in k.lower()],
            'Anomaly Detection': [k for k in metrics.keys() if 'anomaly' in k.lower()],
            'Filter Stability': [k for k in metrics.keys() if k.startswith('K_') or k.startswith('P_')]
        }
        
        for category, keys in categories.items():
            if not keys:
                continue
            print(f"\n{category}:")
            print("-" * 70)
            for key in keys:
                value = metrics[key]
                if isinstance(value, bool):
                    print(f"  {key:45s}: {value}")
                elif isinstance(value, int):
                    print(f"  {key:45s}: {value}")
                elif isinstance(value, str):
                    print(f"  {key:45s}: {value}")
                elif isinstance(value, float):
                    if abs(value) < 0.01 or abs(value) > 1000:
                        print(f"  {key:45s}: {value:.6e}")
                    else:
                        print(f"  {key:45s}: {value:.6f}")
        
        print("="*70 + "\n")
    
    @staticmethod
    def create_metrics_dataframe(metrics: Dict) -> pd.DataFrame:
        """
        Convert metrics dictionary to a pandas DataFrame for easy export.
        """
        return pd.DataFrame([metrics]).T.rename(columns={0: 'Value'})



# Usage example function
def evaluate_kalman_filter(kf, df, anomaly_threshold=3.0, nonzero_threshold=0.0, save_to_csv=False):
    """
    Complete evaluation pipeline for Kalman Filter.
    
    Parameters:
    -----------
    kf : WaterMeterKalmanFilter
        Initialized Kalman filter
    df : pd.DataFrame
        Data to filter
    anomaly_threshold : float
        Threshold for anomaly detection
    nonzero_threshold : float
        Minimum absolute measurement value to be considered non-zero
    save_to_csv : bool
        Whether to save metrics to CSV
    
    Returns:
    --------
    df_filtered : pd.DataFrame
        Filtered data with diagnostics
    metrics : dict
        Performance metrics
    """
    # Filter data with diagnostics
    df_filtered = kf.filter_dataframe(
        df,
        anomaly_threshold=anomaly_threshold,
        return_diagnostics=True
    )
    
    # Compute metrics
    metrics = KalmanFilterMetrics.compute_all_metrics(
        df_filtered,
        nonzero_threshold=nonzero_threshold
    )
    
    # Print report
    KalmanFilterMetrics.print_metrics_report(metrics)
    
    # Optionally save metrics
    if save_to_csv:
        metrics_df = KalmanFilterMetrics.create_metrics_dataframe(metrics)
        metrics_df.to_csv('kalman_filter_metrics.csv')
        print("Metrics saved to kalman_filter_metrics.csv")
    
    return df_filtered, metrics



if __name__ == "__main__":
    print("Kalman Filter Metrics Module")
    print("="*70)
    print("\nUsage:")
    print("  from kalman_metrics import evaluate_kalman_filter, KalmanFilterMetrics")
    print()
    print("  # Evaluate filter (with non-zero metrics)")
    print("  df_filtered, metrics = evaluate_kalman_filter(")
    print("      kf, df, ")
    print("      anomaly_threshold=3.0,")
    print("      nonzero_threshold=0.0  # Adjust as needed")
    print("  )")
    print()
    print("  # Or compute metrics manually")
    print("  metrics = KalmanFilterMetrics.compute_all_metrics(")
    print("      df_filtered,")
    print("      nonzero_threshold=0.01")
    print("  )")
    print("  KalmanFilterMetrics.print_metrics_report(metrics)")
