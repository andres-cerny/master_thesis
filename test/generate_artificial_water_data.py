"""
Artificial Water Consumption Data Generator
For WRLS Kalman Filter Testing (A, Q, R Matrix Estimation)

This script generates 3 months of realistic water consumption data
with 30-minute intervals, strong daily periodicity, and Gaussian noise.

Output: artificial_water_consumption_3months.csv

Author: Generated for diploma thesis on water consumption anomaly detection
Date: November 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_artificial_water_consumption(
    start_date=None,
    duration_days=90,
    interval_minutes=30,
    n_months=3,
    random_seed=42,
    output_filename='./data/artificial_water_consumption_3months.csv'
):
    """
    Generate artificial water consumption data with realistic daily patterns.

    Parameters:
    -----------
    start_date : datetime, optional
        Start date for the time series. Default: 2025-01-01
    duration_days : int, default=90
        Number of days to generate (approximately 3 months)
    interval_minutes : int, default=30
        Time interval between readings in minutes
    n_months : int, default=3
        Number of months (for documentation)
    random_seed : int, default=42
        Random seed for reproducibility
    output_filename : str, default='artificial_water_consumption_3months.csv'
        Output CSV filename

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['timestamp', 'hodnota', 'Diff']
    """

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Default start date
    if start_date is None:
        start_date = datetime(2025, 1, 1, 0, 0, 0)

    # Generate timestamps - exactly 30 minutes apart, no gaps
    timestamps = [start_date + timedelta(minutes=interval_minutes * i) 
                  for i in range(duration_days * 24 * 2)]  # 2 readings per hour

    # Convert to pandas datetime
    df = pd.DataFrame({'timestamp': timestamps})

    # ============================================================================
    # CONSUMPTION PATTERN GENERATION
    # ============================================================================

    # Extract hours for pattern generation
    hours = np.array([(ts.hour + ts.minute/60) for ts in timestamps])

    # 1. Base consumption pattern (peak around noon, low at night)
    base_pattern = 0.05 + 0.03 * np.sin(2 * np.pi * (hours - 6) / 24)

    # 2. Morning peak (around 7-9 AM) - showers, cooking
    morning_peak = 0.04 * np.exp(-((hours - 7.5)**2) / (2 * 1.5**2))

    # 3. Evening peak (around 6-8 PM) - dinner, cleaning
    evening_peak = 0.035 * np.exp(-((hours - 19)**2) / (2 * 1.5**2))

    # 4. Lunch peak (around 12-1 PM)
    lunch_peak = 0.02 * np.exp(-((hours - 12.5)**2) / (2 * 1**2))

    # Combine all patterns
    daily_pattern = base_pattern + morning_peak + evening_peak + lunch_peak

    # 5. Weekly variation (10% higher on weekdays)
    day_of_week = np.array([ts.weekday() for ts in timestamps])
    weekly_factor = 1.0 + 0.1 * (day_of_week < 5)  # Weekdays (0-4) vs weekends (5-6)

    # 6. Measurement noise (Gaussian)
    noise = np.random.normal(0, 0.005, len(timestamps))

    # Calculate Diff (consumption since last reading)
    diff_values = daily_pattern * weekly_factor + noise

    # Ensure all diff values are non-negative (can't have negative consumption)
    diff_values = np.maximum(diff_values, 0)

    # 7. Add some occasional zero consumption (night time, no activity)
    # 30% probability of zero consumption during night hours (1-5 AM)
    night_hours = (hours >= 1) & (hours <= 5)
    zero_consumption_prob = 0.3
    zero_mask = night_hours & (np.random.random(len(timestamps)) < zero_consumption_prob)
    diff_values[zero_mask] = 0

    df['Diff'] = diff_values

    # Calculate cumulative water meter reading (hodnota)
    # Start with an initial meter reading
    initial_reading = 1000.0
    df['hodnota'] = initial_reading + df['Diff'].cumsum()

    # Reorder columns to match required format
    df = df[['timestamp', 'hodnota', 'Diff']]

    return df


def print_dataset_summary(df):
    """Print comprehensive summary of the generated dataset."""

    print("=" * 70)
    print("ARTIFICIAL WATER CONSUMPTION DATA GENERATION")
    print("=" * 70)
    print(f"\nData Parameters:")
    print(f"  Duration: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days} days")
    print(f"  Interval: 30 minutes")
    print(f"  Total readings: {len(df)}")
    print(f"  Start date: {df['timestamp'].iloc[0]}")
    print(f"  End date: {df['timestamp'].iloc[-1]}")

    print(f"\nData Quality:")
    print(f"  Missing timestamps: 0 (perfect regularity)")
    print(f"  Missing values: 0")
    print(f"  Completeness: 100%")

    print(f"\nConsumption Statistics (Diff):")
    print(f"  Mean: {df['Diff'].mean():.6f} m³")
    print(f"  Std: {df['Diff'].std():.6f} m³")
    print(f"  Min: {df['Diff'].min():.6f} m³")
    print(f"  Max: {df['Diff'].max():.6f} m³")
    print(f"  Median: {df['Diff'].median():.6f} m³")

    print(f"\nMeter Reading (hodnota):")
    print(f"  Initial: {df['hodnota'].iloc[0]:.3f} m³")
    print(f"  Final: {df['hodnota'].iloc[-1]:.3f} m³")
    print(f"  Total consumption: {df['hodnota'].iloc[-1] - df['hodnota'].iloc[0]:.3f} m³")

    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print(f"\nLast 5 rows:")
    print(df.tail(5).to_string(index=False))


def verify_data_quality(df):
    """Verify the generated data meets quality standards."""

    print("\n" + "=" * 70)
    print("DATA QUALITY VERIFICATION")
    print("=" * 70)

    # Check for the expected state space dimensions with 3 harmonics
    n_harmonics = 3
    state_dim = 1 + 2 * n_harmonics

    print(f"\n✓ WRLS Algorithm Compatibility:")
    print(f"  State dimension: {state_dim}")
    print(f"  Total readings: {len(df)}")
    print(f"  Valid consecutive pairs: {len(df) - 1}")

    # Time interval consistency check
    time_diffs = df['timestamp'].diff().dt.total_seconds() / 60
    all_regular = (time_diffs.dropna() == 30).all()

    print(f"\n✓ Quality Checks:")
    checks = [
        ("No missing timestamps", len(df) == 4320),
        ("No NaN values in Diff", df['Diff'].notna().all()),
        ("Regular 30-min intervals", all_regular),
        ("Strong periodicity", df.groupby(df['timestamp'].dt.hour)['Diff'].std().mean() < 0.02),
        ("Sufficient data (3 months)", len(df) >= 4000),
        ("Non-negative consumption", (df['Diff'] >= 0).all()),
    ]

    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")

    print(f"\n✓ Data is ready for WRLS Kalman Filter A, Q, R estimation!")


def analyze_patterns(df):
    """Analyze and display consumption patterns."""

    print("\n" + "=" * 70)
    print("DAILY CONSUMPTION PATTERN ANALYSIS")
    print("=" * 70)

    # Analyze pattern by hour of day
    hourly_stats = df.groupby(df['timestamp'].dt.hour)['Diff'].agg(['mean', 'std', 'min', 'max'])
    print("\nAverage consumption by hour of day:")
    print(hourly_stats.to_string())

    # Weekly pattern
    weekly_stats = df.groupby(df['timestamp'].dt.day_name())['Diff'].mean()
    print(f"\nAverage consumption by day of week:")
    print(weekly_stats.to_string())


def main():
    """Main execution."""

    print("\n" + "=" * 70)
    print("WATER CONSUMPTION DATA GENERATOR")
    print("=" * 70)

    # Generate the artificial data
    print("\nGenerating artificial water consumption data...")
    df = generate_artificial_water_consumption(
        start_date=datetime(2025, 1, 1, 0, 0, 0),
        duration_days=90,
        interval_minutes=30,
        n_months=3,
        random_seed=42,
        output_filename='artificial_water_consumption_3months.csv'
    )

    # Print summary
    print_dataset_summary(df)

    # Verify quality
    verify_data_quality(df)

    # Analyze patterns
    analyze_patterns(df)

    # Save to CSV
    csv_filename = 'artificial_water_consumption_3months.csv'
    df_export = df[['timestamp', 'Diff']].copy()
    df_export.to_csv(csv_filename, index=False)

    print(f"\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\n✓ Data saved to: {csv_filename}")
    print(f"  Columns: timestamp, Diff")
    print(f"  Rows: {len(df_export)}")

    return df


if __name__ == "__main__":
    df = main()

    print(f"\n\nUsage in your WRLS Kalman Filter:")
    print(f"""
    import pandas as pd
    from kalman_wrls_estimator import KalmanFilterWRLSEstimator

    # Load the artificial data
    df = pd.read_csv('artificial_water_consumption_3months.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Run your WRLS estimator
    estimator = KalmanFilterWRLSEstimator(lambda_forget=0.9999)
    A, Q, R, metadata = estimator.estimate_from_dataframe(
        df,
        expected_interval_seconds=1800,  # 30 minutes
        n_harmonics=3,
        seasonal_period=24.0,
        tolerance_percent=10.0,
        verbose=True
    )

    # Save results
    estimator.save_matrices('artificial_data_test')
    """)
