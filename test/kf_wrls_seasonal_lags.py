"""
Online A Matrix Estimation for Water Meter Kalman Filters using WRLS
with Seasonal Lag State Vector

This module implements Weighted Recursive Least Squares (WRLS) estimation
of the state transition matrix A for Kalman filtering on water 
consumption time series data with potential gaps and irregular timestamps.

State Vector Design:
- Each component represents diff at a seasonal lag: [diff(t-1), diff(t-2), ..., diff(t-n)]
- n = number of measurements per seasonal period (e.g., 48 for daily season with 30-min intervals)
- Captures seasonal patterns directly without Fourier decomposition

This version:
- Uses WRLS for sequential, online A matrix updates
- Naturally handles gaps by only updating on valid consecutive pairs
- Estimates Q (process noise) and R (measurement noise) empirically
- Provides exponential forgetting for adaptive estimation
- Supports loading previous A and P for incremental updates

Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from datetime import timedelta
import warnings
import os


class KalmanFilterWRLSEstimatorSeasonalLags:
    """
    Online A Matrix Estimation using WRLS for Water Meter Kalman Filters
    with Seasonal Lag State Vector
    
    Estimates the state transition matrix A using Weighted Recursive Least Squares
    on water consumption data with potential gaps and irregular timestamps.
    
    State vector consists of seasonal lags: [diff(t-1), diff(t-2), ..., diff(t-n)]
    where n is the number of measurements per seasonal period.
    
    Key Features:
    - Sequential WRLS updates (online learning)
    - Naturally skips invalid pairs (gaps)
    - Estimates Q and R matrices empirically
    - Exponential weighting favors recent data
    - Adaptive to changing consumption patterns
    - Supports incremental updates from previous estimates
    
    Attributes:
    -----------
    lambda_forget : float
        Forgetting factor for exponential weighting (0.95-0.99)
    A_matrix : np.ndarray
        Current estimate of state transition matrix
    P_matrix : np.ndarray
        Current parameter covariance matrix
    Q_matrix : np.ndarray
        Estimated process noise covariance
    R_matrix : np.ndarray
        Estimated measurement noise covariance
    estimation_metadata : dict
        Information about the estimation process
    """
    
    def __init__(self, lambda_forget: float = 0.9999):
        """
        Initialize the WRLS estimator.
        
        Parameters:
        -----------
        lambda_forget : float, default=0.9999
            Forgetting factor for exponential weighting.
            - Higher values (0.99): More historical data influence
            - Lower values (0.95): Faster adaptation to recent changes
            - Recommended range: 0.95-0.99
        """
        if not 0 < lambda_forget < 1:
            raise ValueError("lambda_forget must be between 0 and 1")
        
        self.lambda_forget = lambda_forget
        self.A_matrix = None
        self.P_matrix = None
        self.Q_matrix = None
        self.R_matrix = None
        self.estimation_metadata = {}
    
    def construct_state_vectors(self, 
                                df: pd.DataFrame,
                                expected_interval_seconds: float,
                                seasonal_period_hours: float = 24.0,
                                tolerance_percent: float = 10.0) -> Tuple[np.ndarray, pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Construct state vectors and valid state pairs from water consumption data.
        
        State vector structure:
        [diff(t-1), diff(t-2), diff(t-3), ..., diff(t-n)]
        
        where n = seasonal_period_hours * 3600 / expected_interval_seconds
        (e.g., 24 hours / 30 minutes = 48 components for daily seasonality)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns ['timestamp', 'Value', 'Diff']
            - 'timestamp': timestamp (may have gaps and irregular spacing)
            - 'Value': cumulative meter reading
            - 'Diff': consumption since last reading
        expected_interval_seconds : float
            Expected sampling interval in SECONDS (e.g., 1800 for 30 minutes)
            Used as reference for filtering consecutive pairs
        seasonal_period_hours : float, default=24.0
            Seasonal period in hours (24.0 for daily seasonality)
            Determines the number of lag components in state vector
        tolerance_percent : float, default=10.0
            Tolerance for time difference as percentage of expected interval
            - Only keeps pairs where: expected_interval * (1 - tol/100) <= actual <= expected_interval * (1 + tol/100)
            - Default 10% means ±10% tolerance
        
        Returns:
        --------
        state_sequence : np.ndarray
            Array of shape (n_valid_points, state_dim) containing state vectors
            for all valid (non-NaN) data points
            state_dim = number of measurements per seasonal period
        df_valid : pd.DataFrame
            Cleaned dataframe with only valid (non-NaN) data points
        valid_state_pairs : List[Tuple[np.ndarray, np.ndarray]]
            List of (x_t, x_next) tuples for valid consecutive pairs
            Each element is shape (state_dim, 1)
        """
        # Ensure datetime format
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove rows with NaN in Diff (missing/invalid consumption data)
        df_valid = df.dropna(subset=['Diff']).reset_index(drop=True)
        
        if len(df_valid) == 0:
            raise ValueError("No valid data points after removing NaN values")
        
        # Calculate state dimension based on seasonal period
        measurements_per_period = int((seasonal_period_hours * 3600) / expected_interval_seconds)
        state_dim = measurements_per_period
        
        if len(df_valid) < state_dim + 10:
            warnings.warn(
                f"Only {len(df_valid)} valid points for state dimension {state_dim}. "
                f"Consider using more data or reducing seasonal_period_hours."
            )
        
        # Calculate total time span and expected number of readings
        time_span_seconds = (df_valid['timestamp'].iloc[-1] - df_valid['timestamp'].iloc[0]).total_seconds()
        expected_num_readings = int(time_span_seconds / expected_interval_seconds) + 1
        
        # Calculate time differences between consecutive points (in seconds)
        time_diffs_seconds = df_valid['timestamp'].diff().dt.total_seconds()
        
        # Calculate tolerance bounds
        expected_interval_min = expected_interval_seconds * (1 - tolerance_percent / 100.0)
        expected_interval_max = expected_interval_seconds * (1 + tolerance_percent / 100.0)
        
        # Extract diff values
        diff_values = df_valid['Diff'].values
        
        # Create state vectors with seasonal lags
        # state[i] = [diff[i], diff[i-1], diff[i-2], ..., diff[i-state_dim+1]]
        n_points = len(diff_values)
        n_valid_states = n_points - state_dim + 1
        
        if n_valid_states < 2:
            raise ValueError(
                f"Insufficient data for state construction. "
                f"Need at least {state_dim + 1} points, have {n_points}. "
                f"Consider reducing seasonal_period_hours or using more data."
            )
        
        state_sequence = np.zeros((n_valid_states, state_dim))
        
        for i in range(n_valid_states):
            # State vector: [diff(t), diff(t-1), ..., diff(t-state_dim+1)]
            state_sequence[i, :] = diff_values[i:i+state_dim]
        
        # Construct valid state pairs
        # For state_sequence[i] and state_sequence[i+1] to form a valid pair,
        # the time difference between df_valid[i+state_dim-1] and df_valid[i+state_dim]
        # must be within tolerance
        valid_state_pairs = []
        
        for i in range(n_valid_states - 1):
            # Check if transition from state i to state i+1 is valid
            # This corresponds to time step from df_valid[i+state_dim-1] to df_valid[i+state_dim]
            time_idx = i + state_dim
            time_diff = time_diffs_seconds.iloc[time_idx]
            
            if expected_interval_min <= time_diff <= expected_interval_max:
                x_t = state_sequence[i, :].reshape(-1, 1)      # (state_dim, 1)
                x_next = state_sequence[i+1, :].reshape(-1, 1)  # (state_dim, 1)
                valid_state_pairs.append((x_t, x_next))
        
        # Calculate validity percentage
        n_valid_pairs = len(valid_state_pairs)
        expected_num_pairs = expected_num_readings - state_dim  # Adjusted for state construction
        
        if expected_num_pairs > 0:
            validity_percentage = (n_valid_pairs / expected_num_pairs) * 100
        else:
            validity_percentage = 0.0
        
        # Warn if validity is too low
        if validity_percentage < 70.0:
            warnings.warn(
                f"Data quality warning: Only {validity_percentage:.1f}% of expected pairs are valid.\n"
                f"  Expected pairs (based on time span): {expected_num_pairs}\n"
                f"  Valid pairs found: {n_valid_pairs}\n"
                f"  Missing/invalid: {expected_num_pairs - n_valid_pairs}\n"
                f"  This may indicate significant NaN values, gaps, or irregular sampling."
            )
        
        if len(valid_state_pairs) < 2:
            raise ValueError(
                f"Only {len(valid_state_pairs)} valid consecutive pairs found "
                f"within tolerance {tolerance_percent}% of {expected_interval_seconds} seconds. "
                f"Consider relaxing tolerance or checking data quality."
            )
        
        return state_sequence, df_valid, valid_state_pairs
    
    
    def construct_state_vectors(self, 
                                df: pd.DataFrame,
                                expected_interval_seconds: float,
                                seasonal_period_hours: float = 24.0,
                                tolerance_percent: float = 10.0) -> Tuple[np.ndarray, pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Construct state vectors and valid state pairs from water consumption data.
        
        State vector structure:
        [diff(t-1), diff(t-2), diff(t-3), ..., diff(t-n)]
        
        where n = seasonal_period_hours * 3600 / expected_interval_seconds
        (e.g., 24 hours / 30 minutes = 48 components for daily seasonality)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns ['timestamp', 'Value', 'Diff']
            - 'timestamp': timestamp (may have gaps and irregular spacing)
            - 'Value': cumulative meter reading
            - 'Diff': consumption since last reading
        expected_interval_seconds : float
            Expected sampling interval in SECONDS (e.g., 1800 for 30 minutes)
            Used as reference for filtering consecutive pairs
        seasonal_period_hours : float, default=24.0
            Seasonal period in hours (24.0 for daily seasonality)
            Determines the number of lag components in state vector
        tolerance_percent : float, default=10.0
            Tolerance for time difference as percentage of expected interval
            - Only keeps pairs where: expected_interval * (1 - tol/100) <= actual <= expected_interval * (1 + tol/100)
            - Default 10% means ±10% tolerance
        
        Returns:
        --------
        state_sequence : np.ndarray
            Array of shape (n_valid_points, state_dim) containing state vectors
            for all valid (non-NaN) data points
            state_dim = number of measurements per seasonal period
        df_valid : pd.DataFrame
            Cleaned dataframe with only valid (non-NaN) data points
        valid_state_pairs : List[Tuple[np.ndarray, np.ndarray]]
            List of (x_t, x_next) tuples for valid consecutive pairs
            Each element is shape (state_dim, 1)
        """
        # Ensure datetime format
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove rows with NaN in Diff (missing/invalid consumption data)
        df_valid = df.dropna(subset=['Diff']).reset_index(drop=True)
        
        if len(df_valid) == 0:
            raise ValueError("No valid data points after removing NaN values")
        
        # Calculate state dimension based on seasonal period
        measurements_per_period = int((seasonal_period_hours * 3600) / expected_interval_seconds)
        state_dim = measurements_per_period
        
        if len(df_valid) < state_dim + 10:
            warnings.warn(
                f"Only {len(df_valid)} valid points for state dimension {state_dim}. "
                f"Consider using more data or reducing seasonal_period_hours."
            )
        
        # Calculate total time span and expected number of readings
        time_span_seconds = (df_valid['timestamp'].iloc[-1] - df_valid['timestamp'].iloc[0]).total_seconds()
        expected_num_readings = int(time_span_seconds / expected_interval_seconds) + 1
        
        # Calculate time differences between consecutive points (in seconds)
        time_diffs_seconds = df_valid['timestamp'].diff().dt.total_seconds()
        
        # Calculate tolerance bounds
        expected_interval_min = expected_interval_seconds * (1 - tolerance_percent / 100.0)
        expected_interval_max = expected_interval_seconds * (1 + tolerance_percent / 100.0)
        
        # Extract diff values
        diff_values = df_valid['Diff'].values
        
        # Create state vectors with seasonal lags
        # state[i] = [diff[i], diff[i-1], diff[i-2], ..., diff[i-state_dim+1]]
        n_points = len(diff_values)
        n_valid_states = n_points - state_dim + 1
        
        if n_valid_states < 2:
            raise ValueError(
                f"Insufficient data for state construction. "
                f"Need at least {state_dim + 1} points, have {n_points}. "
                f"Consider reducing seasonal_period_hours or using more data."
            )
        
        state_sequence = np.zeros((n_valid_states, state_dim))
        
        for i in range(n_valid_states):
            # State vector: [diff(t), diff(t-1), ..., diff(t-state_dim+1)]
            state_sequence[i, :] = diff_values[i:i+state_dim]
        
        # Construct valid state pairs
        # For state_sequence[i] and state_sequence[i+1] to form a valid pair,
        # the time difference between df_valid[i+state_dim-1] and df_valid[i+state_dim]
        # must be within tolerance
        valid_state_pairs = []
        
        for i in range(n_valid_states - 1):
            # Check if diffs in x_t vector are within range +1 diff which is a diff between last two elems in x_next
            time_diffs = time_diffs_seconds.iloc[i:i+state_dim+1]
            if not all((expected_interval_min <= time_diffs) & (time_diffs <= expected_interval_max)):
                continue
            
            x_t = state_sequence[i, :].reshape(-1, 1)      # (state_dim, 1)
            x_next = state_sequence[i+1, :].reshape(-1, 1)  # (state_dim, 1)
            valid_state_pairs.append((x_t, x_next))
        
        # Calculate validity percentage
        n_valid_pairs = len(valid_state_pairs)
        expected_num_pairs = expected_num_readings - state_dim  # Adjusted for state construction
        
        if expected_num_pairs > 0:
            validity_percentage = (n_valid_pairs / expected_num_pairs) * 100
        else:
            validity_percentage = 0.0
        
        # Warn if validity is too low
        if validity_percentage < 70.0:
            warnings.warn(
                f"Data quality warning: Only {validity_percentage:.1f}% of expected pairs are valid.\n"
                f"  Expected pairs (based on time span): {expected_num_pairs}\n"
                f"  Valid pairs found: {n_valid_pairs}\n"
                f"  Missing/invalid: {expected_num_pairs - n_valid_pairs}\n"
                f"  This may indicate significant NaN values, gaps, or irregular sampling."
            )
        
        if len(valid_state_pairs) < 2:
            raise ValueError(
                f"Only {len(valid_state_pairs)} valid consecutive pairs found "
                f"within tolerance {tolerance_percent}% of {expected_interval_seconds} seconds. "
                f"Consider relaxing tolerance or checking data quality."
            )
        
        return state_sequence, df_valid, valid_state_pairs
    
    
    def estimate_A_wrls(self,
                       df: pd.DataFrame,
                       expected_interval_seconds: float,
                       seasonal_period_hours: float = 24.0,
                       tolerance_percent: float = 10.0,
                       initial_P_scale: float = 1000.0,
                       initial_A_filepath: Optional[str] = None,
                       initial_P_filepath: Optional[str] = None,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate A matrix using Weighted Recursive Least Squares (WRLS).
        
        This method processes valid consecutive pairs sequentially, naturally
        skipping gaps. More recent pairs receive higher weight through the
        forgetting factor.
        
        Can load previous A and P matrices to continue estimation from a
        previous run (useful for monthly updates).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Water consumption data with columns ['timestamp', 'Value', 'Diff']
        expected_interval_seconds : float
            Expected sampling interval in SECONDS
        seasonal_period_hours : float, default=24.0
            Seasonal period in hours (determines state dimension)
        tolerance_percent : float, default=10.0
            Tolerance for time difference filtering (±%)
        initial_P_scale : float, default=1000.0
            Initial uncertainty scale for P matrix (larger = less certain)
            Only used if initial_P_filepath is not provided
        initial_A_filepath : str, optional
            Path to load initial A matrix from previous run
            If None, starts with identity matrix
        initial_P_filepath : str, optional
            Path to load initial P matrix from previous run
            If None, uses identity * initial_P_scale
        verbose : bool, default=True
            Whether to print progress information
        
        Returns:
        --------
        A : np.ndarray
            Estimated state transition matrix (state_dim × state_dim)
        Q : np.ndarray
            Estimated process noise covariance (state_dim × state_dim)
        R : np.ndarray
            Estimated measurement noise covariance (state_dim × state_dim)
        """
        # Step 1: Construct state vectors and valid pairs
        state_sequence, df_valid, valid_state_pairs = self.construct_state_vectors(
            df,
            expected_interval_seconds,
            seasonal_period_hours,
            tolerance_percent
        )
        
        state_dim = state_sequence.shape[1]
        N = len(state_sequence)
        n_updates = len(valid_state_pairs)
        n_total_possible_pairs = N - 1
        n_skipped = n_total_possible_pairs - n_updates
        
        # Calculate time span for reporting
        time_span_seconds = (df_valid['timestamp'].iloc[-1] - df_valid['timestamp'].iloc[0]).total_seconds()
        expected_num_readings = int(time_span_seconds / expected_interval_seconds) + 1
        expected_num_pairs = expected_num_readings - state_dim
        validity_percentage = (n_updates / expected_num_pairs) * 100 if expected_num_pairs > 0 else 0.0
        
        if verbose:
            print(f"✓ Data Processing Summary:")
            print(f"  Time span: {time_span_seconds / 3600:.1f} hours")
            print(f"  Expected readings (based on {expected_interval_seconds}s interval): {expected_num_readings}")
            print(f"  Seasonal period: {seasonal_period_hours} hours")
            print(f"  State dimension (measurements per period): {state_dim}")
            print(f"  Expected pairs: {expected_num_pairs}")
            print(f"  Valid pairs found: {n_updates}")
            print(f"  Data validity: {validity_percentage:.1f}%")
            print()
        
        # Step 2: Initialize or load A and P matrices
        loaded_from_file = False
        
        if initial_A_filepath is not None and os.path.exists(initial_A_filepath):
            try:
                A = np.load(initial_A_filepath)
                if A.shape != (state_dim, state_dim):
                    raise ValueError(
                        f"Loaded A matrix shape {A.shape} doesn't match "
                        f"expected state dimension {state_dim}x{state_dim}"
                    )
                loaded_from_file = True
                if verbose:
                    print(f"✓ Loaded initial A matrix from: {initial_A_filepath}")
            except Exception as e:
                warnings.warn(f"Failed to load A matrix from {initial_A_filepath}: {e}. Using identity.")
                A = np.eye(state_dim)
        else:
            A = np.eye(state_dim)  # Start with identity (persistence assumption)
            if verbose and initial_A_filepath is not None:
                print(f"  Initial A file not found: {initial_A_filepath}. Using identity matrix.")
        
        if initial_P_filepath is not None and os.path.exists(initial_P_filepath):
            try:
                P = np.load(initial_P_filepath)
                if P.shape != (state_dim, state_dim):
                    raise ValueError(
                        f"Loaded P matrix shape {P.shape} doesn't match "
                        f"expected state dimension {state_dim}x{state_dim}"
                    )
                loaded_from_file = True
                if verbose:
                    print(f"✓ Loaded initial P matrix from: {initial_P_filepath}")
            except Exception as e:
                warnings.warn(f"Failed to load P matrix from {initial_P_filepath}: {e}. Using default.")
                P = np.eye(state_dim) * initial_P_scale
        else:
            P = np.eye(state_dim) * initial_P_scale  # Large initial uncertainty
            if verbose and initial_P_filepath is not None:
                print(f"  Initial P file not found: {initial_P_filepath}. Using default uncertainty.")
        
        if verbose:
            if loaded_from_file:
                print(f"  Continuing estimation from previous run (incremental update)")
            print()
            print(f"Starting WRLS estimation...")
            print(f"  State dimension: {state_dim}")
            print(f"  Forgetting factor λ = {self.lambda_forget}")
            print()
        
        # Step 3: Per-row WRLS loop over valid pairs
        # Initialize separate P matrices for each output dimension
        P_matrices = [np.eye(state_dim) * initial_P_scale for _ in range(state_dim)]
        
        for idx, (x_t, x_next) in enumerate(valid_state_pairs):
            # Update each row of A separately using scalar WRLS
            for i in range(state_dim):
                # Target: i-th component of next state
                y_i = x_next[i, 0]  # scalar
                
                # Current prediction for i-th component
                prediction_i = A[i, :] @ x_t  # scalar (dot product of i-th row with x_t)
                prediction_i = prediction_i[0, 0] if prediction_i.shape == (1, 1) else prediction_i
                
                # Innovation (prediction error) for this dimension
                innovation_i = y_i - prediction_i  # scalar
                
                # Compute Kalman gain for i-th row
                # K_i = P_i @ x_t / (x_t.T @ P_i @ x_t + 1/λ)
                P_i = P_matrices[i]
                denominator_i = x_t.T @ P_i @ x_t + (1.0 / self.lambda_forget)
                denominator_i = denominator_i[0, 0] if denominator_i.shape == (1, 1) else denominator_i
                
                # Regularization to prevent numerical issues
                if denominator_i < 1e-10:
                    denominator_i = 1e-10
                    if verbose and idx < 10:  # Only warn for first few iterations
                        warnings.warn(f"Small denominator at update {idx}, row {i}. Regularizing.")
                
                K_i = (P_i @ x_t) / denominator_i  # (state_dim, 1)
                
                # Update i-th row of A
                # a_i <- a_i + K_i * innovation_i
                A[i, :] = A[i, :] + (K_i.flatten() * innovation_i)
                
                # Update covariance P_i for this row
                # P_i <- (P_i - K_i @ x_t.T @ P_i) / λ
                P_matrices[i] = (P_i - K_i @ x_t.T @ P_i) / self.lambda_forget
            
            # Optional: Monitor stability periodically
            if verbose and (idx + 1) % 100 == 0:
                eigvals = np.linalg.eigvals(A)
                max_eig = np.max(np.abs(eigvals))
                print(f"  Processed {idx + 1}/{len(valid_state_pairs)} updates... Max |eigval(A)| = {max_eig:.4f}")
                if max_eig > (1.0 + 1e-6):
                    warnings.warn(f"Potentially unstable A detected at update {idx + 1}. Max eigenvalue: {max_eig:.4f}")
        
        # Store the final P matrices (average for overall uncertainty estimate)
        self.P_matrix = np.mean([P_i for P_i in P_matrices], axis=0)
        self.A_matrix = A
        
        # Step 4: Estimate Q (process noise covariance) using valid pairs
        if verbose:
            print()
            print(f"Estimating Q (process noise covariance)...")
        
        Q = self._estimate_Q_from_valid_pairs(valid_state_pairs, A)
        self.Q_matrix = Q
        
        # Step 5: Estimate R (measurement noise covariance)
        if verbose:
            print(f"Estimating R (measurement noise covariance)...")
        
        R = self._estimate_R_from_measurements(state_sequence, df_valid)
        self.R_matrix = R
        
        # Store metadata
        expected_interval_minutes = expected_interval_seconds / 60.0
        self.estimation_metadata = {
            'method': 'WRLS_Seasonal_Lags',
            'n_updates': n_updates,
            'n_skipped': n_skipped,
            'n_total_possible_pairs': n_total_possible_pairs,
            'expected_num_readings': expected_num_readings,
            'expected_num_pairs': expected_num_pairs,
            'validity_percentage': validity_percentage,
            'lambda_forget': self.lambda_forget,
            'condition_number': np.linalg.cond(A),
            'state_dim': state_dim,
            'seasonal_period_hours': seasonal_period_hours,
            'expected_interval_seconds': expected_interval_seconds,
            'expected_interval_minutes': expected_interval_minutes,
            'tolerance_percent': tolerance_percent,
            'data_start': df_valid['timestamp'].iloc[0],
            'data_end': df_valid['timestamp'].iloc[-1],
            'time_span_hours': time_span_seconds / 3600,
            'Q_trace': np.trace(Q),
            'R_trace': np.trace(R),
            'loaded_from_previous': loaded_from_file,
            'initial_A_filepath': initial_A_filepath,
            'initial_P_filepath': initial_P_filepath,
            'max_eigenvalue': np.max(np.abs(np.linalg.eigvals(A)))
        }
        
        if verbose:
            print()
            print(f"✓ WRLS estimation complete")
            print(f"  Updates performed: {n_updates}")
            print(f"  A matrix shape: {A.shape}")
            print(f"  Condition number: {self.estimation_metadata['condition_number']:.2f}")
            print(f"  Max |eigenvalue|: {self.estimation_metadata['max_eigenvalue']:.4f}")
            print()
            print(f"✓ Noise covariance estimation:")
            print(f"  Q (process noise) trace: {np.trace(Q):.6f}")
            print(f"  R (measurement noise) trace: {np.trace(R):.6f}")
            print(f"  Q/R ratio: {np.trace(Q)/np.trace(R):.3f}")
        
        return A, Q, R
    
    def _estimate_Q_from_valid_pairs(self,
                                     valid_state_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                     A: np.ndarray) -> np.ndarray:
        """
        Estimate Q (process noise covariance) from valid state pairs.
        
        Q represents the uncertainty in the state transition model.
        We estimate it as the sample covariance of (x_{t+1} - A x_t)
        for all valid consecutive pairs.
        
        Parameters:
        -----------
        valid_state_pairs : List[Tuple[np.ndarray, np.ndarray]]
            List of (x_t, x_next) tuples for valid transitions
        A : np.ndarray
            Current estimate of state transition matrix
        
        Returns:
        --------
        Q : np.ndarray
            Process noise covariance matrix (state_dim × state_dim)
        """
        if len(valid_state_pairs) < 2:
            warnings.warn("Insufficient valid pairs for Q estimation. Using identity matrix.")
            state_dim = A.shape[0]
            return np.eye(state_dim) * 0.01

        residuals = []
        for x_t, x_next in valid_state_pairs:
            residual = x_next - A @ x_t
            residuals.append(residual.flatten())

        residuals_array = np.array(residuals)
        # Remove rows containing nan/inf before covariance
        mask = np.isfinite(residuals_array).all(axis=1)
        filtered = residuals_array[mask]

        if filtered.shape[0] < 2:
            warnings.warn("Too few finite residuals for Q. Using identity.")
            state_dim = A.shape[0]
            return np.eye(state_dim) * 0.01
            
        Q = np.cov(filtered.T)
        
        # Ensure Q is positive definite
        if not np.isfinite(Q).all():
            warnings.warn("Q contains non-finite values! Using identity*0.01.")
            Q = np.eye(filtered.shape[1]) * 0.01
        else:
            # Add small regularization to ensure positive definiteness
            min_eigenval = np.min(np.real(np.linalg.eigvals(Q)))
            if min_eigenval <= 0 or not np.isfinite(min_eigenval):
                Q = Q + np.eye(Q.shape[0]) * (abs(min_eigenval) + 1e-6)
        
        return Q
    
    def _estimate_R_from_measurements(self,
                                     state_sequence: np.ndarray,
                                     df_valid: pd.DataFrame) -> np.ndarray:
        """
        Estimate R (measurement noise covariance) from measurement characteristics.
        
        For water meters, R represents sensor noise and measurement uncertainty.
        Since our state vector consists of diff values at different lags, we estimate
        R as a diagonal matrix with variance from the diff measurements.
        
        Parameters:
        -----------
        state_sequence : np.ndarray
            State vectors (n_samples, state_dim)
        df_valid : pd.DataFrame
            Valid data points with timestamps and Diff values
        
        Returns:
        --------
        R : np.ndarray
            Measurement noise covariance matrix (state_dim × state_dim)
        """
        state_dim = state_sequence.shape[1]
        
        # Extract all diff values from state sequence
        all_diff_values = state_sequence.flatten()
        
        # Method 1: Overall variance of diff measurements
        overall_var = np.var(all_diff_values)
        
        # Method 2: Look for stable periods (low variance windows)
        # Use the first component of state vectors (most recent diff)
        current_diff = state_sequence[:, 0]
        window_size = min(10, len(current_diff) // 4)
        
        if window_size >= 3:
            rolling_std = pd.Series(current_diff).rolling(window=window_size).std()
            stable_periods = rolling_std < rolling_std.quantile(0.25)  # Bottom 25% variance
            
            if stable_periods.sum() > 5:
                stable_diff = current_diff[stable_periods]
                sensor_noise_var = np.var(stable_diff)
            else:
                sensor_noise_var = overall_var * 0.1  # Use 10% of total variance
        else:
            sensor_noise_var = overall_var * 0.1
        
        # Construct R matrix as diagonal
        # All state components are diff measurements, so use same noise variance
        R = np.eye(state_dim) * sensor_noise_var
        
        # Ensure minimum variance to avoid numerical issues
        min_var = 1e-6
        if sensor_noise_var < min_var:
            R = np.eye(state_dim) * min_var
            warnings.warn(f"Very low measurement noise variance detected. Using minimum {min_var}.")
        
        return R
    
    def estimate_from_dataframe(self,
                                df: pd.DataFrame,
                                expected_interval_seconds: float,
                                seasonal_period_hours: float = 24.0,
                                tolerance_percent: float = 10.0,
                                initial_P_scale: float = 1000.0,
                                initial_A_filepath: Optional[str] = None,
                                initial_P_filepath: Optional[str] = None,
                                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Complete estimation pipeline: A, Q, R from raw dataframe.
        
        This is the main method for WRLS estimation with noise covariance.
        Can load previous A and P matrices for incremental updates.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Water consumption data
        expected_interval_seconds : float
            Expected sampling interval in SECONDS
        seasonal_period_hours : float, default=24.0
            Seasonal period in hours (determines state dimension)
        tolerance_percent : float, default=10.0
            Tolerance for time difference filtering (±%)
        initial_P_scale : float, default=1000.0
            Initial uncertainty scale (if not loading from file)
        initial_A_filepath : str, optional
            Path to load initial A matrix
        initial_P_filepath : str, optional
            Path to load initial P matrix
        verbose : bool, default=True
            Whether to print progress
        
        Returns:
        --------
        A : np.ndarray
            Estimated state transition matrix
        Q : np.ndarray
            Estimated process noise covariance
        R : np.ndarray
            Estimated measurement noise covariance
        metadata : dict
            Comprehensive information about the estimation
            
        Example (monthly incremental update):
        --------
        >>> # First month (no previous data)
        >>> estimator = KalmanFilterWRLSEstimator(lambda_forget=0.98)
        >>> A, Q, R, meta = estimator.estimate_from_dataframe(
        ...     df_month1, 
        ...     expected_interval_seconds=1800,  # 30 minutes
        ...     seasonal_period_hours=24.0       # Daily seasonality
        ... )
        >>> estimator.save_matrices("sensor_001_2025_01")
        >>> 
        >>> # Second month (continue from previous)
        >>> estimator2 = KalmanFilterWRLSEstimator(lambda_forget=0.98)
        >>> A2, Q2, R2, meta2 = estimator2.estimate_from_dataframe(
        ...     df_month2, 
        ...     expected_interval_seconds=1800,
        ...     seasonal_period_hours=24.0,
        ...     initial_A_filepath="sensor_001_2025_01_A.npy",
        ...     initial_P_filepath="sensor_001_2025_01_P.npy"
        ... )
        >>> estimator2.save_matrices("sensor_001_2025_02")
        """
        A, Q, R = self.estimate_A_wrls(
            df,
            expected_interval_seconds,
            seasonal_period_hours,
            tolerance_percent,
            initial_P_scale,
            initial_A_filepath,
            initial_P_filepath,
            verbose
        )
        
        return A, Q, R, self.estimation_metadata
    
    def save_matrices(self, filepath_prefix: str):
        """
        Save all estimated matrices to files.
        
        Parameters:
        -----------
        filepath_prefix : str
            Prefix for output files (e.g., "sensor_001_2025_01")
            Will create:
            - {prefix}_A.npy
            - {prefix}_Q.npy
            - {prefix}_R.npy
            - {prefix}_P.npy
            
        Example:
        --------
        >>> estimator.save_matrices("sensor_001_2025_01")
        # Creates: sensor_001_2025_01_A.npy, sensor_001_2025_01_Q.npy, etc.
        """
        if self.A_matrix is None:
            raise ValueError("No matrices to save. Run estimation first.")
        
        np.save(f"{filepath_prefix}_A.npy", self.A_matrix)
        np.save(f"{filepath_prefix}_Q.npy", self.Q_matrix)
        np.save(f"{filepath_prefix}_R.npy", self.R_matrix)
        np.save(f"{filepath_prefix}_P.npy", self.P_matrix)
        
        print(f"✓ Saved matrices to:")
        print(f"  {filepath_prefix}_A.npy")
        print(f"  {filepath_prefix}_Q.npy")
        print(f"  {filepath_prefix}_R.npy")
        print(f"  {filepath_prefix}_P.npy")
    
    def load_matrices(self, filepath_prefix: str):
        """
        Load previously estimated matrices.
        
        Parameters:
        -----------
        filepath_prefix : str
            Prefix used when saving (e.g., "sensor_001_2025_01")
            
        Returns:
        --------
        A, Q, R : np.ndarray
            Loaded matrices
            
        Example:
        --------
        >>> estimator = KalmanFilterWRLSEstimator()
        >>> A, Q, R = estimator.load_matrices("sensor_001_2025_01")
        """
        self.A_matrix = np.load(f"{filepath_prefix}_A.npy")
        self.Q_matrix = np.load(f"{filepath_prefix}_Q.npy")
        self.R_matrix = np.load(f"{filepath_prefix}_R.npy")
        self.P_matrix = np.load(f"{filepath_prefix}_P.npy")
        
        print(f"✓ Loaded matrices from:")
        print(f"  {filepath_prefix}_A.npy")
        print(f"  {filepath_prefix}_Q.npy")
        print(f"  {filepath_prefix}_R.npy")
        print(f"  {filepath_prefix}_P.npy")
        
        return self.A_matrix, self.Q_matrix, self.R_matrix


if __name__ == "__main__":
    """
    Example usage demonstration
    """
    print("Kalman Filter WRLS Estimator with Seasonal Lag State Vector")
    print("=" * 70)
    print("This version uses seasonal lag components instead of Fourier harmonics.")
    print("State vector: [diff(t-1), diff(t-2), ..., diff(t-n)]")
    print("where n = measurements per seasonal period")
    print()
    print("Quick start:")
    print("  from kalman_wrls_seasonal_lags import KalmanFilterWRLSEstimator")
    print()
    print("  # First run (no previous data)")
    print("  estimator = KalmanFilterWRLSEstimator(lambda_forget=0.98)")
    print("  A, Q, R, metadata = estimator.estimate_from_dataframe(")
    print("      df,")
    print("      expected_interval_seconds=1800,  # 30 minutes")
    print("      seasonal_period_hours=24.0,      # Daily seasonality (48 components)")
    print("      tolerance_percent=10.0")
    print("  )")
    print("  estimator.save_matrices('sensor_001_2025_01')")
    print()
    print("  # For 20-minute intervals with daily seasonality:")
    print("  # State dimension = 24 * 60 / 20 = 72 components")
    print("  A, Q, R, metadata = estimator.estimate_from_dataframe(")
    print("      df,")
    print("      expected_interval_seconds=1200,  # 20 minutes")
    print("      seasonal_period_hours=24.0       # 72 lag components")
    print("  )")
    print()
    print("Key advantages:")
    print("  ✓ Direct seasonal modeling (no Fourier decomposition)")
    print("  ✓ Each component = actual diff at specific lag")
    print("  ✓ Interpretable state transitions")
    print("  ✓ Sequential online updates (adapts to changing patterns)")
    print("  ✓ Naturally handles gaps (skips invalid pairs)")
    print("  ✓ Exponential forgetting (recent data weighted higher)")
    print("  ✓ Incremental updates from previous runs")
    print()
    print("Note: State dimension = seasonal_period_hours * 3600 / expected_interval_seconds")
    print("Example: 24 hours / 30 minutes = 48 components")
