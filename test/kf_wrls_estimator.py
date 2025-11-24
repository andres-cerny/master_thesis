"""
Online A Matrix Estimation for Water Meter Kalman Filters using WRLS

This module implements Weighted Recursive Least Squares (WRLS) estimation
of the state transition matrix A for Kalman filtering on water 
consumption time series data with potential gaps and irregular timestamps.

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


def generate_artificial_state_pairs():
    # Parameters
    n_points = 50
    offset_base = 0.05
    offset_amplitude = 0.04  # simulate some cyclic variation
    period = 24  # e.g., 24 hours for daily pattern
    
    # Generate synthetic timestamps (regular 30 min intervals)
    hours = np.linspace(0, 23.5, n_points)
    offsets = offset_base + offset_amplitude * np.sin(2 * np.pi * hours / period)
    
    # Harmonic 1 (daily: sin, cos)
    sin_terms = np.sin(2 * np.pi * hours / period)
    cos_terms = np.cos(2 * np.pi * hours / period)
    
    # Stack state vectors [offset, sin, cos]
    states = np.vstack((offsets, sin_terms, cos_terms)).T  # shape (50, 3)
    
    # Make consecutive state pairs for WRLS
    state_pairs = [(states[i].reshape(-1, 1), states[i+1].reshape(-1, 1)) for i in range(n_points - 1)]
    
    # Print first 3 state vectors and pairs
    print("Example state vector pairs:")
    for i in range(3):
        print(f"{state_pairs[i][0].flatten()} -> {state_pairs[i][1].flatten()}")
        
    return state_pairs, 3


class KalmanFilterWRLSEstimator:
    """
    Online A Matrix Estimation using WRLS for Water Meter Kalman Filters
    
    Estimates the state transition matrix A using Weighted Recursive Least Squares
    on water consumption data with potential gaps and irregular timestamps.
    
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
        lambda_forget : float, default=0.98
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
                                n_harmonics: int = 3,
                                seasonal_period: float = 24.0,
                                tolerance_percent: float = 10.0,
                                verbose: bool = False) -> Tuple[np.ndarray, pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Construct state vectors and valid state pairs from water consumption data.
        
        State vector structure:
        [offset, sin_1, cos_1, sin_2, cos_2, ..., sin_n, cos_n]
        
        The offset captures the consumption level, while harmonic components
        capture daily and sub-daily seasonal patterns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns ['timestamp', 'Value', 'Diff']
            - 'timestamp': timestamp (may have gaps and irregular spacing)
            - 'Value': cumulative meter reading
            - 'Diff': consumption since last reading
        expected_interval_seconds : float
            Expected sampling interval in SECONDS (e.g., 1500 for 25 minutes)
            Used as reference for filtering consecutive pairs
        n_harmonics : int, default=3
            Number of harmonic components for seasonal modeling
            - 1: Daily cycle only (24h)
            - 2: Daily + 12-hour cycles
            - 3: Daily + 12-hour + 8-hour cycles (recommended)
        seasonal_period : float, default=24.0
            Period in hours for the fundamental frequency (24.0 for daily)
        tolerance_percent : float, default=10.0
            Tolerance for time difference as percentage of expected interval
            - Only keeps pairs where: expected_interval * (1 - tol/100) <= actual <= expected_interval * (1 + tol/100)
            - Default 10% means ±10% tolerance
        
        Returns:
        --------
        state_sequence : np.ndarray
            Array of shape (n_valid_points, state_dim) containing state vectors
            for all valid (non-NaN) data points
            state_dim = 1 + 2*n_harmonics
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
        
        if len(df_valid) < 10:
            warnings.warn(f"Only {len(df_valid)} valid points. Consider using more data.")
        
        # Calculate total time span and expected number of readings
        time_span_seconds = (df_valid['timestamp'].iloc[-1] - df_valid['timestamp'].iloc[0]).total_seconds()
        expected_num_readings = int(time_span_seconds / expected_interval_seconds) + 1
        
        # Calculate time differences between consecutive points (in seconds)
        time_diffs_seconds = df_valid['timestamp'].diff().dt.total_seconds()
        
        # Calculate tolerance bounds
        expected_interval_min = expected_interval_seconds * (1 - tolerance_percent / 100.0)
        expected_interval_max = expected_interval_seconds * (1 + tolerance_percent / 100.0)
        
        # Create full state sequence for all valid (non-NaN) points
        start_time = df_valid['timestamp'].iloc[0]
        df_valid['hours_from_start'] = (
            df_valid['timestamp'] - start_time
        ).dt.total_seconds() / 3600
        
        # State dimension: 1 (offset) + 2*n_harmonics (sin/cos pairs)
        state_dim = 1 + 2 * n_harmonics
        state_sequence = np.zeros((len(df_valid), state_dim))
        
        # First component: offset (consumption value)
        state_sequence[:, 0] = df_valid['Diff'].values
        
        # Seasonal components (Fourier harmonics)
        for k in range(1, n_harmonics + 1):
            omega_k = 2 * np.pi * k / seasonal_period
            sin_idx = 2 * k - 1
            cos_idx = 2 * k
            
            state_sequence[:, sin_idx] = np.sin(omega_k * df_valid['hours_from_start'].values)
            state_sequence[:, cos_idx] = np.cos(omega_k * df_valid['hours_from_start'].values)
        
        # Construct valid state pairs
        valid_state_pairs = []
        N = len(state_sequence)
        
        for i in range(1, N):
            time_diff = time_diffs_seconds.iloc[i]
            
            # Check if this is a valid consecutive pair
            if expected_interval_min <= time_diff <= expected_interval_max:
                x_t = state_sequence[i-1, :].reshape(-1, 1)      # (state_dim, 1)
                x_next = state_sequence[i, :].reshape(-1, 1)     # (state_dim, 1)
                valid_state_pairs.append((x_t, x_next))
        
        # Calculate validity percentage
        # This accounts for both NaN values and gaps/irregular spacing
        n_valid_pairs = len(valid_state_pairs)
        expected_num_pairs = expected_num_readings - 1  # pairs = readings - 1
        
        if expected_num_pairs > 0:
            validity_percentage = (n_valid_pairs / expected_num_pairs) * 100
        else:
            validity_percentage = 0.0
        
        # Warn if validity is too low
        if validity_percentage < 80.0 and verbose:
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
                       n_harmonics: int = 3,
                       seasonal_period: float = 24.0,
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
        n_harmonics : int, default=3
            Number of harmonic components for seasonal modeling
        seasonal_period : float, default=24.0
            Seasonal period in hours (24.0 for daily)
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
            n_harmonics,
            seasonal_period,
            tolerance_percent,
            verbose
        )
        
        state_dim = state_sequence.shape[1]
        N = len(state_sequence)
        n_updates = len(valid_state_pairs)
        n_total_possible_pairs = N - 1
        n_skipped = n_total_possible_pairs - n_updates
        
        #valid_state_pairs, state_dim = generate_artificial_state_pairs()
        
        # Calculate time span for reporting
        time_span_seconds = (df_valid['timestamp'].iloc[-1] - df_valid['timestamp'].iloc[0]).total_seconds()
        expected_num_readings = int(time_span_seconds / expected_interval_seconds) + 1
        expected_num_pairs = expected_num_readings - 1
        validity_percentage = (n_updates / expected_num_pairs) * 100 if expected_num_pairs > 0 else 0.0
        
        if verbose:
            print(f"✓ Data Processing Summary:")
            print(f"  Time span: {time_span_seconds / 3600:.1f} hours")
            print(f"  Expected readings (based on {expected_interval_seconds}s interval): {expected_num_readings}")
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
            A = np.zeros((state_dim, state_dim))  # Start with identity (no change assumption)
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
            #print(f"Vector pairs: {valid_state_pairs}")
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
                
                K_i = (P_i @ x_t) / denominator_i  # (state_dim, 1)
                
                # Update i-th row of A
                # a_i <- a_i + K_i * innovation_i
                A[i, :] = A[i, :] + (K_i.flatten() * innovation_i)
                
                # Update covariance P_i for this row
                # P_i <- (P_i - K_i @ x_t.T @ P_i) / λ
                P_matrices[i] = (P_i - K_i @ x_t.T @ P_i) / self.lambda_forget

            
                if verbose and denominator_i < 1e-6:
                    print(f"WARNING: Denominator nearly zero! x_t: {x_t.flatten()} on index: {idx}")

            # Optional: Monitor stability periodically
            if verbose and (idx + 1) % 100 == 0:
                eigvals = np.linalg.eigvals(A)
                max_eig = np.max(np.abs(eigvals))
                print(f"  Processed {idx + 1}/{len(valid_state_pairs)} updates... Max |eigval(A)| = {max_eig:.4f}")
                #if max_eig >= 1.0:
                #    print(f"  WARNING: Unstable A! Eigenvalues: {eigvals}")
                if max_eig > (1.0 + 1e-6):
                    print(max_eig)
                    print(f"  WARNING: Unstable A! Eigenvalues: {eigvals}")
        
        # After the loop, you can store the final P matrices if needed
        #self.P_matrices = P_matrices  # List of P matrices, one per output dimension
        self.A_matrix = A
        self.P_matrix = P
        
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
            'method': 'WRLS',
            'n_updates': n_updates,
            'n_skipped': n_skipped,
            'n_total_possible_pairs': n_total_possible_pairs,
            'expected_num_readings': expected_num_readings,
            'expected_num_pairs': expected_num_pairs,
            'validity_percentage': validity_percentage,
            'lambda_forget': self.lambda_forget,
            'condition_number': np.linalg.cond(A),
            'state_dim': state_dim,
            'n_harmonics': n_harmonics,
            'seasonal_period': seasonal_period,
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
            'initial_P_filepath': initial_P_filepath
        }
        
        if verbose:
            print()
            print(f"✓ WRLS estimation complete")
            print(f"  Updates performed: {n_updates}")
            print(f"  A matrix shape: {A.shape}")
            print(f"  Condition number: {self.estimation_metadata['condition_number']:.2f}")
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
            print("DEBUG: Q matrix non-finite. Residuals summary:")
            print(np.nan_to_num(filtered)[:10])
            state_dim = A.shape[0]
            return np.eye(state_dim) * 0.01
            
        Q = np.cov(filtered.T)
        # Check again
        if not np.isfinite(Q).all():
            warnings.warn("Q contains non-finite values! Using identity*0.01.")
            Q = np.eye(filtered.shape[1]) * 0.01
        else:
            min_eigenval = np.nanmin(np.linalg.eigvals(Q))
            if min_eigenval <= 0 or not np.isfinite(min_eigenval):
                Q = Q + np.eye(Q.shape[0]) * (abs(min_eigenval) + 1e-6)
        return Q

    
    def _estimate_R_from_measurements(self,
                                     state_sequence: np.ndarray,
                                     df_valid: pd.DataFrame) -> np.ndarray:
        """
        Estimate R (measurement noise covariance) from measurement characteristics.
        
        For water meters, R represents sensor noise and measurement uncertainty.
        We estimate it from:
        1. Variance of consumption differences (Diff column)
        2. Small stable periods where consumption should be near-constant
        
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
        
        # Method 1: Use variance of offset component (consumption)
        # This captures the natural variability in sensor readings
        consumption_values = state_sequence[:, 0]  # Offset component
        consumption_var = np.var(consumption_values)
        
        # Method 2: Look for stable periods (low variance windows)
        # Use rolling window to find periods of stable consumption
        window_size = min(10, len(consumption_values) // 4)
        if window_size >= 3:
            rolling_std = pd.Series(consumption_values).rolling(window=window_size).std()
            stable_periods = rolling_std < rolling_std.quantile(0.25)  # Bottom 25% variance
            
            if stable_periods.sum() > 5:
                stable_consumption = consumption_values[stable_periods]
                sensor_noise_var = np.var(stable_consumption)
            else:
                sensor_noise_var = consumption_var * 0.1  # Use 10% of total variance
        else:
            sensor_noise_var = consumption_var * 0.1
        
        # Construct R matrix
        # Diagonal matrix: measurement noise is independent across state components
        R = np.eye(state_dim) * sensor_noise_var
        
        # For seasonal components (sin/cos), use smaller noise
        # since they're deterministic functions of time
        R[1:, 1:] = R[1:, 1:] * 0.01  # 1% of consumption noise for harmonics
        
        return R
    
    def estimate_from_dataframe(self,
                                df: pd.DataFrame,
                                expected_interval_seconds: float,
                                n_harmonics: int = 3,
                                seasonal_period: float = 24.0,
                                tolerance_percent: float = 10.0,
                                initial_P_scale: float = 1000.0,
                                initial_A_filepath: Optional[str] = None,
                                initial_P_filepath: Optional[str] = None,
                                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
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
        n_harmonics : int, default=3
            Number of harmonic components
        seasonal_period : float, default=24.0
            Seasonal period in hours
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
        >>> A, Q, R, meta = estimator.estimate_from_dataframe(df_month1, 1500)
        >>> estimator.save_matrices("sensor_001_2025_01")
        >>> 
        >>> # Second month (continue from previous)
        >>> estimator2 = KalmanFilterWRLSEstimator(lambda_forget=0.98)
        >>> A2, Q2, R2, meta2 = estimator2.estimate_from_dataframe(
        ...     df_month2, 
        ...     1500,
        ...     initial_A_filepath="sensor_001_2025_01_A.npy",
        ...     initial_P_filepath="sensor_001_2025_01_P.npy"
        ... )
        >>> estimator2.save_matrices("sensor_001_2025_02")
        """
        A, Q, R = self.estimate_A_wrls(
            df,
            expected_interval_seconds,
            n_harmonics,
            seasonal_period,
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
    print("Kalman Filter WRLS Estimator with Q and R Estimation")
    print("=" * 70)
    print("This version uses Weighted Recursive Least Squares (WRLS) for online")
    print("A matrix estimation and provides empirical Q and R estimates.")
    print()
    print("Quick start:")
    print("  from kalman_wrls_estimator import KalmanFilterWRLSEstimator")
    print()
    print("  # First run (no previous data)")
    print("  estimator = KalmanFilterWRLSEstimator(lambda_forget=0.98)")
    print("  A, Q, R, metadata = estimator.estimate_from_dataframe(")
    print("      df,")
    print("      expected_interval_seconds=1500,  # 25 minutes")
    print("      n_harmonics=3,")
    print("      tolerance_percent=10.0")
    print("  )")
    print("  estimator.save_matrices('sensor_001_2025_01')")
    print()
    print("  # Monthly update (continue from previous)")
    print("  estimator2 = KalmanFilterWRLSEstimator(lambda_forget=0.98)")
    print("  A2, Q2, R2, meta2 = estimator2.estimate_from_dataframe(")
    print("      df_new_month,")
    print("      expected_interval_seconds=1500,")
    print("      initial_A_filepath='sensor_001_2025_01_A.npy',")
    print("      initial_P_filepath='sensor_001_2025_01_P.npy'")
    print("  )")
    print("  estimator2.save_matrices('sensor_001_2025_02')")
    print()
    print("Key advantages:")
    print("  ✓ Sequential online updates (adapts to changing patterns)")
    print("  ✓ Naturally handles gaps (skips invalid pairs)")
    print("  ✓ Provides Q and R estimates (needed for Kalman filtering)")
    print("  ✓ Exponential forgetting (recent data weighted higher)")
    print("  ✓ Single validity metric (accounts for NaN + gaps)")
    print("  ✓ Incremental updates from previous runs (monthly updates)")
    print()
    print("Output:")
    print("  - A: State transition matrix")
    print("  - Q: Process noise covariance")
    print("  - R: Measurement noise covariance")
    print("  - metadata: Estimation details and quality metrics")
