"""
Kalman Filter for Water Meter Anomaly Detection

This module implements a Kalman Filter that uses the estimated A, Q, R matrices
from the KalmanFilterWRLSEstimator to filter water consumption data
and detect anomalies in real-time.

Author: For Diploma Thesis - Water Consumption Anomaly Detection
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import warnings


class WaterMeterKalmanFilter:
    """
    Kalman Filter for water meter data with anomaly detection.
    
    Uses pre-estimated matrices (A, Q, R) from WRLS estimation to filter 
    consumption data and identify anomalous readings based on innovation.
    
    Supports both:
    - Batch filtering of historical data
    - Single-measurement filtering for real-time/streaming data
    
    The state vector includes:
    - Consumption level (offset)
    - Seasonal harmonic components (sin/cos pairs)
    
    Attributes:
    -----------
    A : np.ndarray
        State transition matrix (n × n)
    Q : np.ndarray
        Process noise covariance matrix (n × n)
    R : np.ndarray
        Measurement noise covariance matrix (n × n)
    H : np.ndarray
        Measurement matrix (1 × n), observes consumption only
    x : np.ndarray
        Current state estimate
    P : np.ndarray
        Current state covariance estimate
    """
    
    def __init__(self, 
                 A: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 n_harmonics: int = 3,
                 seasonal_period: float = 24.0,
                 initial_state: Optional[np.ndarray] = None,
                 initial_covariance: Optional[np.ndarray] = None):
        """
        Initialize the Kalman Filter.
        
        Parameters:
        -----------
        A : np.ndarray
            State transition matrix (n × n)
            Obtained from KalmanFilterWRLSEstimator
        Q : np.ndarray
            Process noise covariance (n × n)
            Obtained from KalmanFilterWRLSEstimator
        R : np.ndarray
            Measurement noise covariance (n × n)
            Obtained from KalmanFilterWRLSEstimator
        n_harmonics : int, default=3
            Number of harmonic components (must match A matrix)
        seasonal_period : float, default=24.0
            Seasonal period in hours (must match A matrix estimation)
        initial_state : np.ndarray, optional
            Initial state estimate (n,)
            If None, uses zeros
        initial_covariance : np.ndarray, optional
            Initial state covariance (n × n)
            If None, uses Q (process noise as initial uncertainty)
        """
        self.A = A
        self.Q = Q
        self.R = R
        self.state_dim = A.shape[0]
        self.n_harmonics = n_harmonics
        self.seasonal_period = seasonal_period
        
        # Validate dimensions
        if Q.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Q shape {Q.shape} doesn't match A shape {A.shape}")
        if R.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"R shape {R.shape} doesn't match A shape {A.shape}")
        
        # Measurement matrix (observe consumption only - first component)
        self.H = np.zeros((1, self.state_dim))
        self.H[0, 0] = 1.0
        
        # Initialize state and covariance
        if initial_state is None:
            self.x = np.zeros(self.state_dim)
        else:
            if len(initial_state) != self.state_dim:
                raise ValueError(f"Initial state size {len(initial_state)} doesn't match state_dim {self.state_dim}")
            self.x = initial_state.copy()
        
        if initial_covariance is None:
            # Use Q as initial uncertainty (reasonable for first iteration)
            self.P = self.Q.copy()
        else:
            if initial_covariance.shape != (self.state_dim, self.state_dim):
                raise ValueError(f"Initial covariance shape doesn't match state_dim")
            self.P = initial_covariance.copy()
        
        # Store initial values for reset
        self.x_init = self.x.copy()
        self.P_init = self.P.copy()
        
        # For time tracking (needed for seasonal components)
        self.start_time = None
    
    def _compute_seasonal_components(self, hours_from_start: float) -> np.ndarray:
        """
        Compute seasonal harmonic components for given time.
        
        Parameters:
        -----------
        hours_from_start : float
            Hours elapsed since reference start time
        
        Returns:
        --------
        seasonal : np.ndarray
            Seasonal components [sin_1, cos_1, sin_2, cos_2, ..., sin_n, cos_n]
        """
        seasonal = np.zeros(2 * self.n_harmonics)
        
        for k in range(1, self.n_harmonics + 1):
            omega_k = 2 * np.pi * k / self.seasonal_period
            sin_idx = 2 * (k - 1)
            cos_idx = 2 * (k - 1) + 1
            
            seasonal[sin_idx] = np.sin(omega_k * hours_from_start)
            seasonal[cos_idx] = np.cos(omega_k * hours_from_start)
        
        return seasonal
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of Kalman filter.
        
        Computes:
        - x_pred = A @ x
        - P_pred = A @ P @ A.T + Q
        
        Returns:
        --------
        x_pred : np.ndarray
            Predicted state (n,)
        P_pred : np.ndarray
            Predicted covariance (n × n)
        """
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q
        
        return x_pred, P_pred
    
    def update(self, 
               measurement: float,
               x_pred: np.ndarray,
               P_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Update step of Kalman filter.
        
        Parameters:
        -----------
        measurement : float
            Observed consumption value (Diff)
        x_pred : np.ndarray
            Predicted state from predict()
        P_pred : np.ndarray
            Predicted covariance from predict()
        
        Returns:
        --------
        x_updated : np.ndarray
            Updated state estimate
        P_updated : np.ndarray
            Updated covariance estimate
        innovation : float
            Prediction error (measurement - predicted measurement)
        innovation_std : float
            Standard deviation of innovation
        """
        # Innovation (prediction error)
        z = measurement
        z_pred = (self.H @ x_pred)[0]
        innovation = z - z_pred
        
        # Innovation covariance
        # S = H @ P_pred @ H.T + R
        # Since H observes only first component, this simplifies:
        S = P_pred[0, 0] + self.R[0, 0]
        innovation_std = np.sqrt(S) if S > 0 else 1e-6
        
        # Kalman gain
        # K = P_pred @ H.T @ inv(S)
        K = P_pred[:, 0:1] / S  # (n, 1)
        
        # State update
        x_updated = x_pred + K.flatten() * innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ self.H
        P_updated = I_KH @ P_pred @ I_KH.T + K @ K.T * self.R[0, 0]
        
        # Ensure P remains symmetric and positive definite
        P_updated = (P_updated + P_updated.T) / 2
        
        return x_updated, P_updated, innovation, innovation_std
    
    def filter_single_measurement(self,
                               measurement: float,
                               timestamp: pd.Timestamp,
                               reference_start_time: Optional[pd.Timestamp] = None,
                               anomaly_threshold: float = 3.0,
                               return_diagnostics: bool = False) -> Dict:
        """
        Filter a single measurement in real-time.

        (rest of docstring)

        Parameters:
        -----------
        return_diagnostics : bool, default=False
            If True, includes detailed diagnostic information in the returned dict
        """
        # Set or use reference start time
        if reference_start_time is not None:
            self.start_time = reference_start_time
        elif self.start_time is None:
            self.start_time = timestamp

        # Compute hours from start
        hours_from_start = (timestamp - self.start_time).total_seconds() / 3600

        # Store state before prediction (for diagnostics)
        x_before = self.x.copy()
        P_before = self.P.copy()

        # Prediction step
        x_pred, P_pred = self.predict()

        # Update step
        x_updated, P_updated, innovation, innovation_std = self.update(
            measurement, x_pred, P_pred
        )

        # Compute Kalman gain (for diagnostics)
        S = P_pred[0, 0] + self.R[0, 0]
        K = P_pred[:, 0] / S

        # Update internal state
        self.x = x_updated
        self.P = P_updated

        # Extract consumption components
        consumption_pred = x_pred[0]
        consumption_filtered = x_updated[0]

        # Standardized innovation (anomaly score)
        standardized_innovation = innovation / innovation_std if innovation_std > 1e-10 else 0.0

        # Anomaly detection
        is_anomaly = np.abs(standardized_innovation) > anomaly_threshold

        # Base result
        result = {
            'timestamp': timestamp,
            'measurement': measurement,
            'consumption_pred': consumption_pred,
            'consumption_filtered': consumption_filtered,
            'innovation': innovation,
            'innovation_std': innovation_std,
            'standardized_innovation': standardized_innovation,
            'is_anomaly': is_anomaly,
            'state': self.x.copy(),
            'covariance': self.P.copy()
        }

        # Add diagnostics if requested
        if return_diagnostics:
            diagnostics = {
                # Kalman gain
                'K': K.copy(),
                'K_offset': K[0],  # Gain for consumption offset (most important)

                # Covariance before prediction
                'P_before': P_before.copy(),
                'P_before_trace': np.trace(P_before),
                'P_before_offset_var': P_before[0, 0],
                'P_before_diag': np.diag(P_before).copy(),

                # Covariance after prediction
                'P_pred': P_pred.copy(),
                'P_pred_trace': np.trace(P_pred),
                'P_pred_offset_var': P_pred[0, 0],
                'P_pred_diag': np.diag(P_pred).copy(),

                # Covariance after update
                'P_updated': P_updated.copy(),
                'P_updated_trace': np.trace(P_updated),
                'P_updated_offset_var': P_updated[0, 0],
                'P_updated_diag': np.diag(P_updated).copy(),

                # Innovation covariance
                'S': S,

                # Key ratios
                'P_pred_to_R_ratio': P_pred[0, 0] / self.R[0, 0],
                'Q_to_R_ratio': self.Q[0, 0] / self.R[0, 0],

                # State evolution
                'x_before': x_before.copy(),
                'x_pred': x_pred.copy(),
                'x_updated': x_updated.copy(),

                # Filter parameters
                'Q_offset': self.Q[0, 0],
                'R_offset': self.R[0, 0],

                # Time
                'hours_from_start': hours_from_start
            }
            result['diagnostics'] = diagnostics

        return result


    def filter_dataframe(self,
                            df: pd.DataFrame,
                        anomaly_threshold: float = 3.0,
                        reset_on_start: bool = True,
                        return_diagnostics: bool = False) -> pd.DataFrame:
        """
        Filter entire dataframe (batch processing).

        Parameters:
        -----------
        df : pd.DataFrame
            Water consumption data with columns:
            - 'timestamp': timestamp
            - 'Diff': consumption since last reading
        anomaly_threshold : float, default=3.0
            Threshold for anomaly detection (in standard deviations)
        reset_on_start : bool, default=True
            Whether to reset filter state before processing
        return_diagnostics : bool, default=False
            If True, includes diagnostic columns for K and P evolution

        Returns:
        --------
        df_filtered : pd.DataFrame
            Original dataframe with additional columns including diagnostics
        """
        if reset_on_start:
            self.reset()

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Remove rows with NaN in Diff
        df_valid = df.dropna(subset=['Diff']).reset_index(drop=True)

        if len(df_valid) == 0:
            raise ValueError("No valid data points after removing NaN values")

        # Set reference start time
        self.start_time = df_valid['timestamp'].iloc[0]

        # Initialize result lists
        results = []

        # Process each measurement
        for idx, row in df_valid.iterrows():
            result = self.filter_single_measurement(
                measurement=row['Diff'],
                timestamp=row['timestamp'],
                reference_start_time=self.start_time,
                anomaly_threshold=anomaly_threshold,
                return_diagnostics=return_diagnostics
            )

            result_row = {
                'consumption_pred': result['consumption_pred'],
                'consumption_filtered': result['consumption_filtered'],
                'innovation': result['innovation'],
                'innovation_std': result['innovation_std'],
                'standardized_innovation': result['standardized_innovation'],
                'is_anomaly': result['is_anomaly']
            }

            # Add diagnostics if requested
            if return_diagnostics and 'diagnostics' in result:
                diag = result['diagnostics']
                result_row.update({
                    'K_offset': diag['K_offset'],
                    'P_before_offset_var': diag['P_before_offset_var'],
                    'P_pred_offset_var': diag['P_pred_offset_var'],
                    'P_updated_offset_var': diag['P_updated_offset_var'],
                    'P_before_trace': diag['P_before_trace'],
                    'P_pred_trace': diag['P_pred_trace'],
                    'P_updated_trace': diag['P_updated_trace'],
                    'S': diag['S'],
                    'P_pred_to_R_ratio': diag['P_pred_to_R_ratio'],
                    'Q_to_R_ratio': diag['Q_to_R_ratio'],
                    'hours_from_start': diag['hours_from_start']
                })

            results.append(result_row)

        # Convert results to dataframe
        results_df = pd.DataFrame(results)

        # Add results to original dataframe
        for col in results_df.columns:
            df_valid[col] = results_df[col].values

        return df_valid

    def reset(self):
        """Reset filter to initial state."""
        self.x = self.x_init.copy()
        self.P = self.P_init.copy()
        self.start_time = None
         
    def get_state(self) -> Dict:
        """
        Get current filter state (for saving/loading).
        
        Returns:
        --------
        state : dict
            Current state containing:
            - 'x': State vector
            - 'P': Covariance matrix
            - 'start_time': Reference start time
        """
        return {
            'x': self.x.copy(),
            'P': self.P.copy(),
            'start_time': self.start_time
        }
    
    def set_state(self, state: Dict):
        """
        Set filter state (for loading from saved state).
        
        Parameters:
        -----------
        state : dict
            State dictionary from get_state()
        """
        self.x = state['x'].copy()
        self.P = state['P'].copy()
        self.start_time = state['start_time']


if __name__ == "__main__":
    """
    Example usage demonstration
    """
    print("Water Meter Kalman Filter")
    print("=" * 70)
    print("\nThis module provides Kalman filtering with anomaly detection")
    print("for water meter consumption data.")
    print()
    print("Quick start (batch processing):")
    print("  from kalman_filter import WaterMeterKalmanFilter")
    print("  ")
    print("  # Load estimated matrices from WRLS estimator")
    print("  A = np.load('sensor_001_A.npy')")
    print("  Q = np.load('sensor_001_Q.npy')")
    print("  R = np.load('sensor_001_R.npy')")
    print("  ")
    print("  # Initialize filter")
    print("  kf = WaterMeterKalmanFilter(A, Q, R, n_harmonics=3)")
    print("  ")
    print("  # Filter entire dataframe")
    print("  df_filtered = kf.filter_dataframe(df, anomaly_threshold=3.0)")
    print("  ")
    print("  # Detect anomalies")
    print("  anomalies = df_filtered[df_filtered['is_anomaly']]")
    print()
    print("Real-time usage (streaming):")
    print("  # Initialize filter")
    print("  kf = WaterMeterKalmanFilter(A, Q, R, n_harmonics=3)")
    print("  ")
    print("  # Process measurements as they arrive")
    print("  for timestamp, measurement in incoming_stream:")
    print("      result = kf.filter_single_measurement(")
    print("          measurement, timestamp, anomaly_threshold=3.0")
    print("      )")
    print("      if result['is_anomaly']:")
    print("          alert_system.send_alert(result)")
