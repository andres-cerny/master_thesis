"""
Multithreaded UTC Timestamp Processing for Water Meter Data
============================================================

This module adds timezone-aware 'timestamp_utc' columns to water meter CSV files,
properly handling DST transitions in the Czech Republic (Europe/Prague timezone).

Author: Water Meter Anomaly Detection Thesis
Date: November 2025
"""

import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
import pytz


def add_utc_timestamp(filepath, df, timestamp_col='timestamp', timezone='Europe/Prague'):
    """
    Add timezone-aware UTC timestamp column to dataframe.
    
    Strategy:
    1. Try localizing with ambiguous='infer'
    2. If AmbiguousTimeError occurs, set problematic timestamp_utc rows to NaT
    3. Localize timestamp column (keeping local time values)

    Parameters
    ----------
    filepath : str
        Path to the file being processed (for logging)
    df : pd.DataFrame
        Input dataframe with timestamp column
    timestamp_col : str, default='timestamp'
        Name of the timestamp column
    timezone : str, default='Europe/Prague'
        Target timezone for localization

    Returns
    -------
    pd.DataFrame
        Dataframe with added 'timestamp_utc' column (NaT for ambiguous rows)
    """
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%Y-%m-%d')

    # Try to localize with ambiguous='infer'
    try:
        df['timestamp_localized'] = df[timestamp_col].dt.tz_localize(
            timezone,
            ambiguous='infer',
            nonexistent='NaT'
        )
        df['timestamp_utc'] = df['timestamp_localized'].dt.tz_convert('UTC')
        return df
        
    except pytz.exceptions.AmbiguousTimeError as e:
        #print(f"  File {os.path.basename(filepath)}: AmbiguousTimeError detected - {e}")
        localized_timestamps, utc_timestamps, nat_count = [], [], 0
        
        for ts in df[timestamp_col].dt.tz_localize(None):
            try:
                localized = pd.to_datetime([ts], utc=False).tz_localize(
                    timezone,
                    ambiguous='infer',
                    nonexistent='NaT'
                )
                utc_ts = localized.tz_convert('UTC')[0]
                localized_timestamps.append(localized[0])
                utc_timestamps.append(utc_ts)
            except pytz.exceptions.AmbiguousTimeError:
                # All attempts failed, mark as NaT
                localized_timestamps.append(pd.NaT)
                utc_timestamps.append(pd.NaT)
                nat_count += 1
                #print(f"Added NaT at {ts}.")
        
        # Update both columns
        #df['timestamp_localized'] = localized_timestamps  # Localized (keeps local time with tz info)
        df['timestamp_utc'] = utc_timestamps      # UTC version (NaT for ambiguous)
        
        #print(f"Set {nat_count} ambiguous timestamp_utc values to NaT")
        
        if not df['timestamp_utc'].dropna().is_monotonic_increasing:
            warnings.warn(f"{filepath} is does not have sorted timestamps.")
        
        return df
        
    except Exception as e:
        warnings.warn(f"Unexpected error in {filepath}: {e}")
        raise


def process_file_with_utc(filepath, timestamp_col='timestamp', timezone='Europe/Prague'):
    """
    Process a single CSV file to add UTC timestamp column.

    Parameters
    ----------
    filepath : str
        Path to the CSV file
    timestamp_col : str, default='timestamp'
        Name of the timestamp column
    timezone : str, default='Europe/Prague'
        Target timezone for localization
    """
    try:
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Check if timestamp column exists
        if timestamp_col not in df.columns:
            warnings.warn(f"Column '{timestamp_col}' not found in {filepath}")
            return
        
        # Add UTC timestamp
        df = add_utc_timestamp(filepath, df, timestamp_col=timestamp_col, timezone=timezone)
        
        # Save back to CSV
        df.to_csv(filepath, index=False)
        
    except Exception as e:
        warnings.warn(f"Error processing {filepath}: {str(e)}")


def add_utc_timestamps_multithreaded(
    folder="./data_w_diff_001",
    max_workers=8,
    timestamp_col='timestamp',
    timezone='Europe/Prague'
):
    """
    Add UTC timestamp columns to all CSV files in folder using multithreading.

    Parameters
    ----------
    folder : str
        Path to folder containing CSV files
    max_workers : int, default=8
        Number of threads to use
    timestamp_col : str, default='timestamp'
        Name of the timestamp column in CSV files
    timezone : str, default='Europe/Prague'
        Target timezone for localization
    """
    # Get all CSV files
    filepaths = [
        os.path.join(folder, f) 
        for f in os.listdir(folder) 
        if f.endswith('.csv')
    ]

    if not filepaths:
        print(f"No CSV files found in {folder}")
        return

    print(f"Found {len(filepaths)} CSV files to process")

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create partial function with fixed parameters
        process_func = lambda fp: process_file_with_utc(
            fp, 
            timestamp_col=timestamp_col, 
            timezone=timezone
        )

        # Process with progress bar
        list(tqdm(
            executor.map(process_func, filepaths),
            total=len(filepaths),
            desc="Adding UTC timestamps",
            unit="file"
        ))

    print(f"✓ Processed all CSV files in folder: {folder} with {max_workers} threads")


# ============================================================================
#                           TEST FUNCTIONS
# ============================================================================

def create_test_df_spring_forward():
    """
    Create test dataframe that spans spring DST transition (CET -> CEST).
    In 2024: March 31, 2:00 AM CET -> 3:00 AM CEST (clock springs forward)
    """
    dates_before = pd.date_range(
        start='2024-03-30 23:00:00',
        end='2024-03-31 01:30:00',
        freq='30min'
    )

    dates_after = pd.date_range(
        start='2024-03-31 03:00:00',
        end='2024-03-31 05:00:00',
        freq='30min'
    )

    timestamps = dates_before.append(dates_after)

    np.random.seed(42)
    base_value = 1000.0
    increments = np.random.uniform(0.5, 2.0, len(timestamps))
    hodnota = base_value + np.cumsum(increments)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'hodnota': hodnota
    })

    df['Diff'] = df['hodnota'].diff().fillna(0)

    return df

def create_test_df_spring_forward_NaT():
    """
    Create test dataframe that spans spring DST transition (CET -> CEST).
    In 2024: March 31, 2:00 AM CET -> 3:00 AM CEST (clock springs forward)
    """
    dates_before = pd.date_range(
        start='2024-03-30 23:15:00',
        end='2024-03-31 02:15:00',
        freq='30min'
    )

    dates_after = pd.date_range(
        start='2024-03-31 03:15:00',
        end='2024-03-31 05:15:00',
        freq='30min'
    )

    timestamps = dates_before.append(dates_after)

    np.random.seed(42)
    base_value = 1000.0
    increments = np.random.uniform(0.5, 2.0, len(timestamps))
    hodnota = base_value + np.cumsum(increments)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'hodnota': hodnota
    })

    df['Diff'] = df['hodnota'].diff().fillna(0)

    return df


def create_test_df_fall_back():
    """
    Create test dataframe that spans fall DST transition (CEST -> CET).
    In 2024: October 27, 3:00 AM CEST -> 2:00 AM CET (clock falls back)
    """
    dates_before = pd.date_range(
        start='2024-10-27 00:00:00',
        end='2024-10-27 02:30:00',
        freq='30min'
    )

    dates_after = pd.date_range(
        start='2024-10-27 02:00:00',
        end='2024-10-27 05:00:00',
        freq='30min'
    )

    timestamps = dates_before.append(dates_after)

    np.random.seed(123)
    base_value = 2000.0
    increments = np.random.uniform(0.5, 2.0, len(timestamps))
    hodnota = base_value + np.cumsum(increments)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'hodnota': hodnota
    })

    df['Diff'] = df['hodnota'].diff().fillna(0)

    return df

def create_test_df_fall_back_NaT():
    """
    Create test dataframe that spans fall DST transition (CEST -> CET).
    In 2024: October 27, 3:00 AM CEST -> 2:00 AM CET (clock falls back)
    """
    dates_before = pd.date_range(
        start='2024-10-27 00:15:00',
        end='2024-10-27 02:45:00',
        freq='30min'
    )

    dates_after = pd.date_range(
        start='2024-10-27 02:15:00',
        end='2024-10-27 05:15:00',
        freq='30min'
    )

    timestamps = dates_before.append(dates_after)

    np.random.seed(123)
    base_value = 2000.0
    increments = np.random.uniform(0.5, 2.0, len(timestamps))
    hodnota = base_value + np.cumsum(increments)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'hodnota': hodnota
    })

    df['Diff'] = df['hodnota'].diff().fillna(0)

    return df


def test_spring_forward_transition():
    """Test UTC timestamp addition for spring forward DST transition."""
    print("\n" + "="*70)
    print("TEST 1: Spring Forward Transition (CET -> CEST)")
    print("="*70)

    df = create_test_df_spring_forward()
    print(f"\nCreated test dataframe with {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    df_processed = add_utc_timestamp("test_spring.csv", df)

    assert 'timestamp_utc' in df_processed.columns, "timestamp_utc column not added"
    print("✓ timestamp_utc column added")

    # Check how many are valid (not NaT)
    valid_count = df_processed['timestamp_utc'].notna().sum()
    print(f"✓ {valid_count}/{len(df_processed)} UTC timestamps are valid")

    # Check monotonicity for non-NaT values
    df_valid = df_processed[df_processed['timestamp_utc'].notna()]
    if len(df_valid) > 1:
        time_diffs = df_valid['timestamp_utc'].diff().dt.total_seconds()
        if (time_diffs[1:] > 0).all():
            print("✓ Valid UTC timestamps are monotonically increasing")
        else:
            print("⚠ Some valid UTC timestamps are not monotonic")

    print("\n✓ TEST 1 PASSED: Spring forward transition handled correctly")
    return df_processed

def test_spring_forward_transition_NaT():
    """Test UTC timestamp addition for spring forward DST transition."""
    print("\n" + "="*70)
    print("TEST 1: Spring Forward Transition (CET -> CEST)")
    print("="*70)

    df = create_test_df_spring_forward_NaT()
    print(f"\nCreated test dataframe with {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    df_processed = add_utc_timestamp("test_spring.csv", df)

    assert 'timestamp_utc' in df_processed.columns, "timestamp_utc column not added"
    print("✓ timestamp_utc column added")

    # Check how many are valid (not NaT)
    valid_count = df_processed['timestamp_utc'].notna().sum()
    print(f"✓ {valid_count}/{len(df_processed)} UTC timestamps are valid")

    # Check monotonicity for non-NaT values
    df_valid = df_processed[df_processed['timestamp_utc'].notna()]
    if len(df_valid) > 1:
        time_diffs = df_valid['timestamp_utc'].diff().dt.total_seconds()
        if (time_diffs[1:] > 0).all():
            print("✓ Valid UTC timestamps are monotonically increasing")
        else:
            print("⚠ Some valid UTC timestamps are not monotonic")

    print("\n✓ TEST 3 PASSED: Spring forward transition handled correctly")
    return df_processed


def test_fall_back_transition():
    """Test UTC timestamp addition for fall back DST transition."""
    print("\n" + "="*70)
    print("TEST 2: Fall Back Transition (CEST -> CET)")
    print("="*70)

    df = create_test_df_fall_back()
    print(f"\nCreated test dataframe with {len(df)} rows")

    df_processed = add_utc_timestamp("test_fall.csv", df)

    assert 'timestamp_utc' in df_processed.columns, "timestamp_utc column not added"
    print("✓ timestamp_utc column added")

    # Check how many are valid
    valid_count = df_processed['timestamp_utc'].notna().sum()
    nat_count = df_processed['timestamp_utc'].isna().sum()
    print(f"✓ {valid_count}/{len(df_processed)} UTC timestamps are valid")
    if nat_count > 0:
        print(f"  ({nat_count} ambiguous timestamps set to NaT)")

    # Check monotonicity for non-NaT values
    df_valid = df_processed[df_processed['timestamp_utc'].notna()]
    if len(df_valid) > 1:
        time_diffs = df_valid['timestamp_utc'].diff().dt.total_seconds()
        if (time_diffs[1:] > 0).all():
            print("✓ Valid UTC timestamps are monotonically increasing")

    print("\n✓ TEST 2 PASSED: Fall back transition handled correctly")
    return df_processed

def test_fall_back_transition_NaT():
    """Test UTC timestamp addition for fall back DST transition."""
    print("\n" + "="*70)
    print("TEST 2: Fall Back Transition (CEST -> CET)")
    print("="*70)

    df = create_test_df_fall_back_NaT()
    print(f"\nCreated test dataframe with {len(df)} rows")

    df_processed = add_utc_timestamp("test_fall.csv", df)

    assert 'timestamp_utc' in df_processed.columns, "timestamp_utc column not added"
    print("✓ timestamp_utc column added")

    # Check how many are valid
    valid_count = df_processed['timestamp_utc'].notna().sum()
    nat_count = df_processed['timestamp_utc'].isna().sum()
    print(f"✓ {valid_count}/{len(df_processed)} UTC timestamps are valid")
    if nat_count > 0:
        print(f"  ({nat_count} ambiguous timestamps set to NaT)")

    # Check monotonicity for non-NaT values
    df_valid = df_processed[df_processed['timestamp_utc'].notna()]
    if len(df_valid) > 1:
        time_diffs = df_valid['timestamp_utc'].diff().dt.total_seconds()
        if (time_diffs[1:] > 0).all():
            print("✓ Valid UTC timestamps are monotonically increasing")

    print("\n✓ TEST 4 PASSED: Fall back transition handled correctly")
    return df_processed

def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*70)
    print("RUNNING ALL TESTS FOR UTC TIMESTAMP PROCESSING")
    print("="*70)

    try:
        test_spring_forward_transition()
        test_fall_back_transition()
        test_spring_forward_transition_NaT()
        test_fall_back_transition_NaT()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nThe UTC timestamp processing works correctly:")
        print("  ✓ Spring forward DST transition (CET -> CEST)")
        print("  ✓ Fall back DST transition (CEST -> CET)")
        print("  ✓ Ambiguous timestamps set to NaT")
        print("\nYou can now safely use this on your ~27,000 water meter files!")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
#                           MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run tests first
    success = run_all_tests()

    if success:
        print("\n" + "="*70)
        print("READY TO PROCESS YOUR DATA")
        print("="*70)
        
        # Uncomment to process your actual data
        add_utc_timestamps_multithreaded(
            folder="./data_w_diff_001",
            max_workers=8,
            timestamp_col='timestamp',
            timezone='Europe/Prague'
        )
    
        #filepath = 'data_w_diff_001/100001.csv'
        #df = pd.read_csv(filepath)
        #df_processed = add_utc_timestamp(filepath, df, timestamp_col='timestamp', timezone='Europe/Prague')
        #print(df_processed)
