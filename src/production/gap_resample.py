import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
from datetime import datetime

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

    common_periodicity_mode= time_diffs.mode()
    #print(f"Periodicity found {common_periodicity_mode.iloc[0]/60} minutes.")
    return common_periodicity_mode.iloc[0]
    


def fill_gaps_with_periodicity_adaptive(
    df,
    timestamp_col: str = 'timestamp_utc',
    tolerance_percentage=10
):
    """
    Fill gaps in time-series data for a single id sensor using adaptive timestamp generation.
    Instead of creating a full expected range upfront, this function builds the expected timeline
    iteratively based on actual readings, adjusting for timing drift.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns: 'timestamp_utc', 'hodnota', 'Diff'
        Must contain data for a single sensor only.
        Optionally may contain 'is_anomaly' (float or int).
    tolerance_percentage : float
        Tolerance as percentage of periodicity (e.g., 10 means 10% of periodicity_seconds)

    Returns
    -------
    pandas.DataFrame
        DataFrame with filled gaps (NaN for missing hodnota, is_anomaly = 0 for synthetic rows)
    dict
        Diagnostics
    """
    df = df.copy()

    if len(df) < 2:
        raise ValueError("DataFrame passed is too short len < 2 in resample script")

    if timestamp_col not in df.columns:
        raise ValueError(
            f"No {timestamp_col} column found. Available: {df.columns.tolist()}"
        )

    if 'hodnota' not in df.columns:
        raise ValueError(
            f"No 'hodnota' column found. Available: {df.columns.tolist()}"
        )

    # Ensure is_anomaly exists; if not, create it as 0
    if 'is_anomaly' not in df.columns:
        df['is_anomaly'] = 0.0

    try:
        periodicity_seconds = get_periodicity(df, timestamp_col=timestamp_col)
    except ValueError as e:
        print(f"Couldn't find periodicity, skipping this df and getting an error: {e}")
        return df, {}
    
    if periodicity_seconds < 3*60:
        raise ValueError("Periodicity is too small")
    
    if periodicity_seconds < 5*60:
        periodicity_seconds = 5*periodicity_seconds
    
    if periodicity_seconds < 7*60:
        periodicity_seconds = 3*periodicity_seconds
        
    if periodicity_seconds < 10*60:
        periodicity_seconds = 2*periodicity_seconds
    
    tolerance_seconds = (tolerance_percentage / 100.0) * periodicity_seconds

    result_rows = []
    unmatched_detail = []
    matched_indices = set()

    actual_reading_time = df.loc[0, timestamp_col]
    data_end_time = df[timestamp_col].max()

    def get_candidate_indeces(df, within_tolerance, matched_indices):
        if not within_tolerance.any():
            return []
        candidate_indices = df.index[within_tolerance].tolist()
        candidate_indices = [idx for idx in candidate_indices if idx not in matched_indices]
        return candidate_indices

    while (actual_reading_time + pd.Timedelta(seconds=periodicity_seconds)) <= data_end_time:
        expected_next = actual_reading_time + pd.Timedelta(seconds=periodicity_seconds)
        time_diffs = abs((df[timestamp_col] - expected_next).dt.total_seconds())
        within_tolerance = time_diffs <= tolerance_seconds
        candidate_indices = get_candidate_indeces(df, within_tolerance, matched_indices)

        if not candidate_indices:
            time_diffs_real = (df[timestamp_col] - actual_reading_time).dt.total_seconds()
            sorted_diffs = time_diffs_real[time_diffs_real > 0].sort_values()
            min_positive = sorted_diffs.iloc[0]

            min_allowed_diff = periodicity_seconds * 0.5
            idx_sorted_diffs = 0
            not_enough_data = False
            while min_positive < min_allowed_diff:
                idx_sorted_diffs += 1
                if len(sorted_diffs) <= idx_sorted_diffs:
                    not_enough_data = True
                    break
                min_positive = sorted_diffs.iloc[idx_sorted_diffs]

            next_real_index = time_diffs_real[time_diffs_real == min_positive].index[0]
            nans_needed = round(sorted_diffs.iloc[idx_sorted_diffs] / periodicity_seconds)

            if not_enough_data:
                break

            if nans_needed >= 2:
                gap_period = min_positive / nans_needed
                prev_reading_time = actual_reading_time

                # synthetic rows in the gap: hodnota = NaN, is_anomaly = 0
                for _ in range(nans_needed - 1):
                    expected_next = prev_reading_time + pd.Timedelta(seconds=gap_period)
                    expected_next = expected_next.round('s')
                    row_data = {
                        timestamp_col: expected_next,
                        'hodnota': np.nan,
                        'is_anomaly': 0.0,
                    }
                    result_rows.append(row_data)
                    prev_reading_time = expected_next

            actual_reading_time = df.loc[next_real_index, timestamp_col]
            row_data = {
                timestamp_col: actual_reading_time,
                'hodnota': df.loc[next_real_index, 'hodnota'],
                'is_anomaly': df.loc[next_real_index, 'is_anomaly'],
            }
            result_rows.append(row_data)
            matched_indices.add(next_real_index)
            continue

        # There are candidates within tolerance: choose the closest
        closest_idx = min(candidate_indices, key=lambda idx: time_diffs[idx])

        row_data = {
            timestamp_col: df.loc[closest_idx, timestamp_col],
            'hodnota': df.loc[closest_idx, 'hodnota'],
            'is_anomaly': df.loc[closest_idx, 'is_anomaly'],
        }
        result_rows.append(row_data)
        matched_indices.add(closest_idx)

        actual_reading_time = df.loc[closest_idx, timestamp_col]

    # Create the filled DataFrame
    filled_df = pd.DataFrame(result_rows)

    # Ensure column order and presence
    filled_df = filled_df[[timestamp_col, 'hodnota', 'is_anomaly']]

    # Identify unmatched readings and diagnostics (unchanged, uses original df)
    unmatched_indices = [idx for idx in df.index if idx not in matched_indices]
    unmatched_detail = []
    try:
        for idx in unmatched_indices:
            row = df.loc[idx]
            if len(filled_df) > 0:
                time_diffs_to_result = abs(
                    (filled_df[timestamp_col] - row[timestamp_col]).dt.total_seconds()
                )
                nearest_result_idx = time_diffs_to_result.argmin()
                expected_slot = filled_df.loc[nearest_result_idx, timestamp_col]
                time_diff = time_diffs_to_result.iloc[nearest_result_idx]
            else:
                expected_slot = None
                time_diff = None

            prev_idx = idx - 1 if idx > 0 else None
            next_idx = idx + 1 if idx < len(df) - 1 else None

            prev_row = df.loc[prev_idx] if prev_idx is not None else None
            next_row = df.loc[next_idx] if next_idx is not None else None

            unmatched_detail.append({
                'reading_timestamp': str(row[timestamp_col]),
                'hodnota': float(row['hodnota']) if pd.notna(row['hodnota']) else None,
                'expected_slot': str(expected_slot) if expected_slot else None,
                'time_difference_seconds': float(time_diff) if time_diff is not None else None,
                'prev_reading_timestamp': str(prev_row[timestamp_col]) if prev_row is not None else None,
                'prev_reading_value': float(prev_row['hodnota']) if prev_row is not None and pd.notna(prev_row['hodnota']) else None,
                'next_reading_timestamp': str(next_row[timestamp_col]) if next_row is not None else None,
                'next_reading_value': float(next_row['hodnota']) if next_row is not None and pd.notna(next_row['hodnota']) else None
            })
    except Exception as e:
        print("Didnt manage to build unmatched reading list.")

    filled_diffs = filled_df[timestamp_col].diff()
    min_allowed_diff = pd.Timedelta(seconds=periodicity_seconds * 0.5)
    max_allowed_diff = pd.Timedelta(seconds=periodicity_seconds * 1.5)
    if ((min_allowed_diff > filled_diffs) | (filled_diffs > max_allowed_diff)).any():
        for idx, diff in enumerate(filled_diffs):
            if min_allowed_diff > diff:
                print(f"Periodicity used: {periodicity_seconds}")
                print(f"Diff is smaller then possible on idx: {idx} which is datetime {filled_df[timestamp_col].loc[idx]}")
            elif diff > max_allowed_diff:
                print(f"Periodicity used: {periodicity_seconds}")
                print(f"Diff is larger then possible on idx: {idx} which is datetime {filled_df[timestamp_col].loc[idx]}")

    # Recompute Diff
    filled_df['Diff'] = filled_df['hodnota'].diff()
    filled_df.loc[filled_df['Diff'] <= -0.002, 'Diff'] = np.nan
    filled_df.loc[(filled_df['Diff'] > -0.002) & (filled_df['Diff'] < 0), 'Diff'] = 0

    total_filled = len(filled_df)
    missing_count = filled_df['hodnota'].isna().sum()
    diff_nan_count = filled_df['Diff'].isna().sum()

    timespan = (df.iloc[-1][timestamp_col] - df.iloc[0][timestamp_col]).total_seconds()
    expected_timestamps = timespan / periodicity_seconds

    diagnostics = {
        'total_expected_timestamps': expected_timestamps,
        'total_actual_readings': len(df),
        'total_after_filling': total_filled,
        'total_matched_readings': len(matched_indices),
        'missing_values_filled': missing_count,
        'nan_diff_count': diff_nan_count,
        'unmatched_readings_count': len(unmatched_indices),
        'unmatched_readings_detail': unmatched_detail,
        'date_range': f"{df[timestamp_col].min().date()} to {df[timestamp_col].max().date()}",
        'periodicity_used_seconds': periodicity_seconds,
        'tolerance_percentage': tolerance_percentage,
        'tolerance_used_seconds': tolerance_seconds
    }

    return filled_df, diagnostics
