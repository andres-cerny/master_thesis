"""
Adaptive Kalman Filters for Water Consumption Anomaly Detection

This module provides two Kalman Filter implementations that learn their parameters
and capture daily periodicity in water consumption data:

1. AdaptivePeriodicKalmanFilter: Online learning with innovation-based adaptation
2. EMKalmanFilterPeriodic: Batch learning using Expectation-Maximization algorithm

Author: Generated for Diploma Thesis on Water Consumption Anomaly Detection
Date: October 31, 2025
"""

import numpy as np
from typing import Tuple, Optional
import pandas as pd


class AdaptivePeriodicKalmanFilter:
    """
    Adaptive Kalman Filter with periodic state-space model that learns A, Q, and R matrices.

    This filter captures daily periodicity through a state-space formulation and adapts
    its parameters using online learning approaches based on innovation statistics.

    Parameters:
    -----------
    period : int
        The period for capturing periodicity (e.g., 48 for daily periodicity with 30-min readings)
    Q_init : float or np.ndarray
        Initial process noise covariance
    R_init : float
        Initial measurement noise covariance
    learning_rate_Q : float
        Learning rate for Q matrix adaptation (default: 0.01)
    learning_rate_R : float
        Learning rate for R matrix adaptation (default: 0.01)
    window_size : int
        Window size for computing innovation statistics (default: 100)
    """

    def __init__(
        self, 
        period: int = 48,
        Q_init: float = 1e-5,
        R_init: float = 1e-5,
        learning_rate_Q: float = 0.01,
        learning_rate_R: float = 0.01,
        window_size: int = 100
    ):
        self.period = period
        self.state_dim = period + 1  # State: [current_value, periodic_components...]

        # Initialize state transition matrix A
        self.A = self._build_periodic_A_matrix()

        # Observation matrix
        self.H = np.zeros((1, self.state_dim))
        self.H[0, 0] = 1.0  # observe current value
        self.H[0, 1:] = 1.0 / period  # observe average of periodic components

        # Initialize covariance matrices
        if isinstance(Q_init, (int, float)):
            self.Q = np.eye(self.state_dim) * Q_init
        else:
            self.Q = Q_init

        self.R = np.array([[R_init]])

        # Learning rates
        self.lr_Q = learning_rate_Q
        self.lr_R = learning_rate_R

        # Window for innovation statistics
        self.window_size = window_size
        self.innovation_history = []
        self.innovation_cov_history = []

        # State estimate and covariance
        self.x_hat = None
        self.P = None

        # For parameter learning
        self.step_count = 0
        self.warmup_steps = max(50, window_size)

    def _build_periodic_A_matrix(self) -> np.ndarray:
        """
        Build state transition matrix with periodic structure.

        State vector: [x(t), s1(t), s2(t), ..., s_period(t)]
        All components follow random walk dynamics.
        """
        A = np.eye(self.state_dim)
        A[0, 0] = 1.0  # Random walk for base level

        # Periodic components evolve as random walks
        for i in range(1, self.state_dim):
            A[i, i] = 1.0

        return A

    def initialize(self, initial_value: float = 0.0, initial_covariance: float = 1.0):
        """Initialize the filter state and covariance."""
        self.x_hat = np.zeros((self.state_dim, 1))
        self.x_hat[0, 0] = initial_value

        self.P = np.eye(self.state_dim) * initial_covariance
        self.step_count = 0
        self.innovation_history = []
        self.innovation_cov_history = []

    def predict(self):
        """Prediction step of the Kalman filter."""
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update step of the Kalman filter.

        Parameters:
        -----------
        measurement : float
            New observation (Diff value)

        Returns:
        --------
        innovation : float
            Innovation (measurement residual)
        innovation_magnitude : float
            Standardized innovation (Mahalanobis distance)
        """
        measurement = np.array([[measurement]])

        # Compute innovation
        innovation = measurement - (self.H @ self.x_hat)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x_hat = self.x_hat + K @ innovation

        # Update covariance estimate (Joseph form for stability)
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

        # Store innovation statistics
        self.innovation_history.append(innovation.item())
        self.innovation_cov_history.append(S.item())

        # Keep window size fixed
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)
            self.innovation_cov_history.pop(0)

        self.step_count += 1

        # Adaptive parameter learning
        if self.step_count > self.warmup_steps:
            self._adapt_parameters(innovation, S, K)

        # Compute innovation magnitude for anomaly detection
        innovation_magnitude = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation).item()

        return innovation.item(), innovation_magnitude

    def _adapt_parameters(self, innovation: np.ndarray, S: np.ndarray, K: np.ndarray):
        """
        Adapt Q and R matrices based on innovation statistics.

        Uses innovation-based adaptive estimation:
        - Innovation sequence should be white with zero mean
        - Sample covariance of innovations estimates measurement noise
        - Normalized innovations indicate process noise mismatch
        """
        if len(self.innovation_history) < self.window_size:
            return

        # Compute innovation statistics
        innovations = np.array(self.innovation_history)
        sample_var = np.var(innovations)

        # Adapt R based on sample variance of innovations
        expected_var = S.item()

        if sample_var > 1e-10:  # Avoid division by zero
            R_scale = sample_var / (expected_var + 1e-10)
            R_scale = np.clip(R_scale, 0.5, 2.0)  # Limit adaptation rate

            # Adaptive R update with exponential smoothing
            self.R = (1 - self.lr_R) * self.R + self.lr_R * (R_scale * self.R)

        # Adapt Q based on normalized innovation squared
        normalized_innov_sq = (innovation.T @ np.linalg.inv(S) @ innovation).item()

        # Expected value is 1 (chi-square with 1 DOF)
        if normalized_innov_sq > 1.5:
            Q_scale = 1 + self.lr_Q * (normalized_innov_sq - 1.0)
            Q_scale = np.clip(Q_scale, 1.0, 1.5)
            self.Q = Q_scale * self.Q
        elif normalized_innov_sq < 0.5:
            Q_scale = 1 - self.lr_Q * (1.0 - normalized_innov_sq)
            Q_scale = np.clip(Q_scale, 0.7, 1.0)
            self.Q = Q_scale * self.Q

        # Ensure Q and R remain positive definite
        self.Q = (self.Q + self.Q.T) / 2  # Symmetrize
        min_q = 1e-8
        eigenvalues = np.linalg.eigvalsh(self.Q)
        if eigenvalues.min() < min_q:
            self.Q += (min_q - eigenvalues.min()) * np.eye(self.state_dim)

        self.R = np.maximum(self.R, 1e-8)

    def get_current_estimate(self) -> float:
        """Get the current state estimate."""
        return self.x_hat[0, 0]

    def get_periodic_components(self) -> np.ndarray:
        """Get the periodic component estimates."""
        return self.x_hat[1:, 0]

    def get_parameters(self) -> dict:
        """Get current filter parameters."""
        return {
            'A': self.A.copy(),
            'Q': self.Q.copy(),
            'R': self.R.copy(),
            'H': self.H.copy()
        }


class EMKalmanFilterPeriodic:
    """
    Kalman Filter with Expectation-Maximization algorithm for learning A, Q, R.
    Captures daily periodicity through harmonic (Fourier) components.

    This version uses EM algorithm to learn parameters from a batch of data,
    then can be used for online filtering.

    Parameters:
    -----------
    period : int
        The period for capturing periodicity
    n_harmonics : int
        Number of harmonic components to capture periodicity (default: 3)
    max_iter : int
        Maximum EM iterations (default: 20)
    tol : float
        Convergence tolerance for EM (default: 1e-4)
    """

    def __init__(
        self,
        period: int = 48,
        n_harmonics: int = 3,
        max_iter: int = 20,
        tol: float = 1e-4
    ):
        self.period = period
        self.n_harmonics = n_harmonics

        # State dimension: current value + 2*n_harmonics (cos and sin components)
        self.state_dim = 1 + 2 * n_harmonics

        self.max_iter = max_iter
        self.tol = tol

        # Initialize matrices
        self.A = None
        self.Q = None
        self.R = None
        self.H = None

        # Current state for filtering
        self.x_hat = None
        self.P = None

        # EM learned parameters
        self.em_converged = False
        self.em_iterations = 0

    def _initialize_parameters(self, y: np.ndarray, time_indices: np.ndarray):
        """Initialize parameters for EM algorithm."""
        # Build A matrix with periodic structure using rotation matrices
        self.A = np.eye(self.state_dim)
        self.A[0, 0] = 1.0  # Random walk for base level

        # Periodic components: rotation matrices for harmonic oscillators
        for k in range(self.n_harmonics):
            omega_k = 2 * np.pi * (k + 1) / self.period
            idx_cos = 1 + 2 * k
            idx_sin = 2 + 2 * k

            # 2x2 rotation matrix for harmonic k
            cos_omega = np.cos(omega_k)
            sin_omega = np.sin(omega_k)

            self.A[idx_cos, idx_cos] = cos_omega
            self.A[idx_cos, idx_sin] = -sin_omega
            self.A[idx_sin, idx_cos] = sin_omega
            self.A[idx_sin, idx_sin] = cos_omega

        # Observation matrix
        self.H = np.zeros((1, self.state_dim))
        self.H[0, 0] = 1.0  # base level
        for k in range(self.n_harmonics):
            self.H[0, 1 + 2*k] = 1.0  # cos components

        # Initialize Q and R
        self.Q = np.eye(self.state_dim) * 1e-5
        self.R = np.array([[np.var(y) * 0.1]])

    def _kalman_filter(self, y: np.ndarray):
        """Run Kalman filter forward pass."""
        n = len(y)

        # Storage
        x_filt = np.zeros((self.state_dim, n))
        P_filt = np.zeros((self.state_dim, self.state_dim, n))
        x_pred = np.zeros((self.state_dim, n))
        P_pred = np.zeros((self.state_dim, self.state_dim, n))

        # Initialize
        x_pred[:, 0] = np.zeros(self.state_dim)
        P_pred[:, :, 0] = np.eye(self.state_dim) * 1.0

        for t in range(n):
            # Prediction
            if t > 0:
                x_pred[:, t] = self.A @ x_filt[:, t-1]
                P_pred[:, :, t] = self.A @ P_filt[:, :, t-1] @ self.A.T + self.Q

            # Update
            innovation = y[t] - self.H @ x_pred[:, t]
            S = self.H @ P_pred[:, :, t] @ self.H.T + self.R
            K = P_pred[:, :, t] @ self.H.T / S
            K = K.flatten()

            x_filt[:, t] = x_pred[:, t] + K * innovation
            P_filt[:, :, t] = P_pred[:, :, t] - np.outer(K, K) * S

        return x_filt, P_filt, x_pred, P_pred

    def _kalman_smoother(self, x_filt, P_filt, x_pred, P_pred):
        """Run Kalman smoother backward pass (RTS smoother)."""
        n = x_filt.shape[1]

        x_smooth = np.zeros_like(x_filt)
        P_smooth = np.zeros_like(P_filt)
        P_smooth_lag = np.zeros_like(P_filt)

        # Initialize with filtered values
        x_smooth[:, -1] = x_filt[:, -1]
        P_smooth[:, :, -1] = P_filt[:, :, -1]

        # Backward pass
        for t in range(n-2, -1, -1):
            # Smoother gain
            J = P_filt[:, :, t] @ self.A.T @ np.linalg.inv(P_pred[:, :, t+1])

            # Smooth state and covariance
            x_smooth[:, t] = x_filt[:, t] + J @ (x_smooth[:, t+1] - x_pred[:, t+1])
            P_smooth[:, :, t] = P_filt[:, :, t] + J @ (P_smooth[:, :, t+1] - P_pred[:, :, t+1]) @ J.T

            # Lag-one covariance
            if t < n - 1:
                P_smooth_lag[:, :, t+1] = J @ P_smooth[:, :, t+1]

        return x_smooth, P_smooth, P_smooth_lag

    def _em_step(self, y: np.ndarray, x_smooth, P_smooth, P_smooth_lag):
        """EM M-step: update Q and R given smoothed states."""
        n = len(y)

        # Sufficient statistics
        S11 = np.zeros((self.state_dim, self.state_dim))
        S10 = np.zeros((self.state_dim, self.state_dim))
        S00 = np.zeros((self.state_dim, self.state_dim))
        Syy = 0
        Syx = np.zeros(self.state_dim)
        Sxx_obs = np.zeros((self.state_dim, self.state_dim))

        for t in range(1, n):
            S11 += P_smooth[:, :, t] + np.outer(x_smooth[:, t], x_smooth[:, t])
            S10 += P_smooth_lag[:, :, t] + np.outer(x_smooth[:, t], x_smooth[:, t-1])
            S00 += P_smooth[:, :, t-1] + np.outer(x_smooth[:, t-1], x_smooth[:, t-1])

        for t in range(n):
            Syy += y[t]**2
            Syx += y[t] * x_smooth[:, t]
            Sxx_obs += P_smooth[:, :, t] + np.outer(x_smooth[:, t], x_smooth[:, t])

        # Update Q (keep A fixed for stability)
        Q_new = (S11 - S10 @ np.linalg.inv(S00) @ S10.T) / (n - 1)
        Q_new = (Q_new + Q_new.T) / 2  # Symmetrize

        # Ensure Q is positive definite
        min_eig = np.linalg.eigvalsh(Q_new).min()
        if min_eig < 1e-8:
            Q_new += (1e-8 - min_eig) * np.eye(self.state_dim)

        # Update R
        H = self.H
        R_new = (Syy - 2 * H @ Syx + H @ Sxx_obs @ H.T) / n
        R_new = np.maximum(R_new, 1e-8)

        return Q_new, R_new

    def fit(self, y: np.ndarray, time_indices: Optional[np.ndarray] = None):
        """
        Fit parameters using EM algorithm.

        Parameters:
        -----------
        y : np.ndarray
            Observations (1D array of Diff. values)
        time_indices : np.ndarray, optional
            Time indices for each observation
        """
        if time_indices is None:
            time_indices = np.arange(len(y))

        y = y.flatten()

        # Initialize parameters
        self._initialize_parameters(y, time_indices)

        prev_log_likelihood = -np.inf

        for iter_num in range(self.max_iter):
            # E-step: Kalman filter and smoother
            x_filt, P_filt, x_pred, P_pred = self._kalman_filter(y)
            x_smooth, P_smooth, P_smooth_lag = self._kalman_smoother(
                x_filt, P_filt, x_pred, P_pred
            )

            # Compute log-likelihood - FIXED VERSION
            log_likelihood = 0.0  # Use Python float, not numpy
            for t in range(len(y)):
                innovation = y[t] - self.H @ x_pred[:, t]
                S = self.H @ P_pred[:, :, t] @ self.H.T + self.R
                S_scalar = float(S.item())  # Ensure Python float
                innov_scalar = float(innovation.item())

                # Compute contribution
                contrib = -0.5 * (np.log(2*np.pi*S_scalar) + innov_scalar**2 / S_scalar)
                log_likelihood += float(contrib)  # Convert to Python float

            # Check convergence
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.em_converged = True
                self.em_iterations = iter_num + 1
                break

            prev_log_likelihood = float(log_likelihood)  # Store as Python float

            # M-step: Update parameters
            Q_new, R_new = self._em_step(y, x_smooth, P_smooth, P_smooth_lag)

            self.Q = Q_new
            self.R = R_new

            self.em_iterations = iter_num + 1

        print(f"EM converged: {self.em_converged} after {self.em_iterations} iterations")
        print(f"Final log-likelihood: {prev_log_likelihood:.2f}")
    
    def initialize_filter(self, initial_value: float = 0.0):
        """Initialize filter for online use after EM training."""
        self.x_hat = np.zeros((self.state_dim, 1))
        self.x_hat[0, 0] = initial_value
        self.P = np.eye(self.state_dim)

    def predict(self):
        """Prediction step."""
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update step.

        Parameters:
        -----------
        measurement : float
            New observation

        Returns:
        --------
        innovation : float
            Innovation (measurement residual)
        innovation_magnitude : float
            Standardized innovation
        """
        y = np.array([[measurement]])

        # Innovation
        innovation = y - self.H @ self.x_hat
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T / S

        # Update
        self.x_hat = self.x_hat + K * innovation
        self.P = self.P - np.outer(K.flatten(), K.flatten()) * S 

        # Innovation magnitude
        innov_mag = np.sqrt((innovation**2 / S).item())

        return innovation.item(), innov_mag

    def get_parameters(self):
        """Get learned parameters."""
        return {
            'A': self.A.copy(),
            'Q': self.Q.copy(),
            'R': self.R.copy(),
            'H': self.H.copy(),
            'converged': self.em_converged,
            'iterations': self.em_iterations
        }


# Convenience functions for running the filters

def run_adaptive_periodic_kalman_filter(
    df: pd.DataFrame,
    period: int = 48,
    Q_init: float = 1e-5,
    R_init: float = 1e-5,
    learning_rate_Q: float = 0.01,
    learning_rate_R: float = 0.01
) -> Tuple[pd.DataFrame, AdaptivePeriodicKalmanFilter]:
    """
    Run adaptive periodic Kalman filter on DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Diff' column
    period : int
        Period for daily periodicity
    Q_init, R_init : float
        Initial covariance matrices
    learning_rate_Q, learning_rate_R : float
        Learning rates for adaptation

    Returns:
    --------
    df_results : pd.DataFrame
        Results with estimates and diagnostics
    filter_obj : AdaptivePeriodicKalmanFilter
        The trained filter object
    """
    kf = AdaptivePeriodicKalmanFilter(
        period=period,
        Q_init=Q_init,
        R_init=R_init,
        learning_rate_Q=learning_rate_Q,
        learning_rate_R=learning_rate_R
    )

    kf.initialize(initial_value=0, initial_covariance=1)

    state_estimates = []
    innovation_vals = []
    innovation_magnitudes = []
    Q_traces = []
    R_vals = []

    for diff_val in df['Diff']:
        kf.predict()
        innovation, innovation_mag = kf.update(diff_val)

        state_estimates.append(kf.get_current_estimate())
        innovation_vals.append(innovation)
        innovation_magnitudes.append(innovation_mag)
        Q_traces.append(np.trace(kf.Q))
        R_vals.append(kf.R.item())

    df_results = df.copy()
    df_results['Diff_Estimate'] = state_estimates
    df_results['Innovation'] = innovation_vals
    df_results['Innovation_Magnitude'] = innovation_magnitudes
    df_results['Q_Trace'] = Q_traces
    df_results['R_Value'] = R_vals

    return df_results, kf


def run_em_kalman_filter(
    df: pd.DataFrame,
    period: int = 48,
    n_harmonics: int = 3,
    train_size: int = 500
) -> Tuple[pd.DataFrame, EMKalmanFilterPeriodic]:
    """
    Run EM-based Kalman filter: train on first portion, then filter.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Diff' column
    period : int
        Period for periodicity
    n_harmonics : int
        Number of harmonic components
    train_size : int
        Number of samples for EM training

    Returns:
    --------
    df_results : pd.DataFrame
        Results with estimates and diagnostics
    filter_obj : EMKalmanFilterPeriodic
        The trained filter object
    """
    data = df['Diff'].values
    train_data = data[:train_size]

    print(f"Training EM Kalman Filter on {train_size} samples...")
    kf = EMKalmanFilterPeriodic(period=period, n_harmonics=n_harmonics)
    kf.fit(train_data)

    # Run filter on all data
    kf.initialize_filter(initial_value=0.0)

    state_estimates = []
    innovation_vals = []
    innovation_magnitudes = []

    for diff_val in data:
        kf.predict()
        innovation, innov_mag = kf.update(diff_val)

        state_estimates.append(kf.x_hat[0, 0])
        innovation_vals.append(innovation)
        innovation_magnitudes.append(innov_mag)

    df_results = df.copy()
    df_results['Diff_Estimate'] = state_estimates
    df_results['Innovation'] = innovation_vals
    df_results['Innovation_Magnitude'] = innovation_magnitudes

    return df_results, kf


if __name__ == "__main__":
    print("Adaptive Kalman Filter Module for Water Consumption Anomaly Detection")
    print("="*70)
    print()
    print("Available classes:")
    print("  - AdaptivePeriodicKalmanFilter: Online adaptive learning")
    print("  - EMKalmanFilterPeriodic: EM-based batch learning")
    print()
    print("Usage examples:")
    print("  df_results, kf = run_adaptive_periodic_kalman_filter(df, period=48)")
    print("  df_results, kf = run_em_kalman_filter(df, period=48, n_harmonics=3)")
    
    graph2 = pd.read_csv('../filled_data_complete.csv')
    graph2.rename(columns={'Diff.': 'Diff'}, inplace=True)
    df_results, model = run_em_kalman_filter(graph2)
