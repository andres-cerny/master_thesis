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
        raise ValueError("No time difference calculated.")

    common_periodicity_mode= time_diffs.mode()
    #print(f"Periodicity found {common_periodicity_mode.iloc[0]/60} minutes.")
    return common_periodicity_mode.iloc[0]
    


def fill_gaps_with_periodicity_adaptive(df, timestamp_col: str = 'timestamp_utc', tolerance_percentage=10):
    """
    Fill gaps in time-series data for a single id sensor using adaptive timestamp generation.
    Instead of creating a full expected range upfront, this function builds the expected timeline
    iteratively based on actual readings, adjusting for timing drift.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: 'timestamp_utc', 'hodnota', 'Diff'
        Must contain data for a single sensor only.
    tolerance_percentage : float
        Tolerance as percentage of periodicity (e.g., 10 means 10% of periodicity_seconds)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with filled gaps (NaN for missing values)
    dict
        Dictionary with diagnostic information including detailed unmatched readings
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    if len(df) < 2:
        raise ValueError("DataFrame passed is too short len < 2")
    
    if timestamp_col not in df.columns:
            raise ValueError(
                f"No {timestamp_col} column found. Available: {df.columns.tolist()}"
            )
            
    if 'hodnota' not in df.columns:
            raise ValueError(
                f"No 'hodnota' column found. Available: {df.columns.tolist()}"
            )
    
    try:
        periodicity_seconds = get_periodicity(df, timestamp_col=timestamp_col)
    except ValueError as e:
        print(f"Couldn't find periodicity, skipping this df and getting an error: {e}")
        return df, {}
    
    # Calculate tolerance in seconds based on percentage
    tolerance_seconds = (tolerance_percentage / 100.0) * periodicity_seconds
    
    # Initialize variables for adaptive timeline building
    result_rows = []
    unmatched_detail = []
    matched_indices = set()
    
    # Start with the first timestamp
    actual_reading_time = df.loc[0, timestamp_col]
    
    # Track the original data end time
    data_end_time = df[timestamp_col].max()
    
    def get_candidate_indeces(df, within_tolerance, matched_indices):
        if not within_tolerance.any():
            return []
        # Get indices of all readings within tolerance
        candidate_indices = df.index[within_tolerance].tolist()
        # Remove already matched indices
        candidate_indices = [idx for idx in candidate_indices if idx not in matched_indices]
        return candidate_indices
    
    while (actual_reading_time + pd.Timedelta(seconds=periodicity_seconds)) <= data_end_time:
        # Calculate next expected timestamp from the ACTUAL reading
        expected_next = actual_reading_time + pd.Timedelta(seconds=periodicity_seconds)
        # Find readings within tolerance of expected_next
        time_diffs = abs((df[timestamp_col] - expected_next).dt.total_seconds())
        within_tolerance = time_diffs <= tolerance_seconds
        candidate_indices = get_candidate_indeces(df, within_tolerance, matched_indices)
        
        if not candidate_indices:
            time_diffs_real = (df[timestamp_col] - actual_reading_time).dt.total_seconds()
            sorted_diffs = time_diffs_real[time_diffs_real > 0].sort_values()
            min_positive = sorted_diffs.iloc[0]
            
            min_allowed_diff = periodicity_seconds*0.5
            idx_sorted_diffs = 0
            not_enough_data = False
            while min_positive < min_allowed_diff:
                idx_sorted_diffs += 1
                if len(sorted_diffs) <= idx_sorted_diffs:
                    not_enough_data = True
                    break
                min_positive = sorted_diffs.iloc[idx_sorted_diffs]
            
            next_real_index = time_diffs_real[time_diffs_real == min_positive].index[0]
            nans_needed = round(sorted_diffs.iloc[0] / periodicity_seconds)
            
            if not_enough_data:
                break
            
            if nans_needed >= 2:
                gap_period = min_positive / nans_needed
                prev_reading_time = actual_reading_time

                for _ in range(nans_needed - 1):
                    expected_next = prev_reading_time + pd.Timedelta(seconds=gap_period)
                    expected_next = expected_next.round('s')
                    row_data = {
                        timestamp_col: expected_next,
                        'hodnota': np.nan
                    }
                    result_rows.append(row_data)
                    prev_reading_time = expected_next
            
            actual_reading_time = df.loc[next_real_index, timestamp_col]
            row_data = {
                    timestamp_col: actual_reading_time,
                    'hodnota': df.loc[next_real_index, 'hodnota']
            }
            result_rows.append(row_data)
            matched_indices.add(next_real_index)
            continue
        
        # Find the closest one among candidates
        closest_idx = min(candidate_indices, key=lambda idx: time_diffs[idx])
        
        # Add this reading to results
        row_data = {
            timestamp_col: df.loc[closest_idx, timestamp_col],
            'hodnota': df.loc[closest_idx, 'hodnota']
        }                
        result_rows.append(row_data)
        matched_indices.add(closest_idx)
        
        # Calculate next expected timestamp from the ACTUAL reading
        actual_reading_time = df.loc[closest_idx, timestamp_col]
            
            
    # Create the filled DataFrame
    filled_df = pd.DataFrame(result_rows)
    

    filled_df = filled_df[[timestamp_col, 'hodnota']]
    
    # Identify unmatched readings
    unmatched_indices = [idx for idx in df.index if idx not in matched_indices]
    
    # Build detailed unmatched readings list
    for idx in unmatched_indices:
        row = df.loc[idx]
        
        # Find what the expected slot would have been (closest in our result)
        if len(filled_df) > 0:
            time_diffs_to_result = abs((filled_df[timestamp_col] - row[timestamp_col]).dt.total_seconds())
            nearest_result_idx = time_diffs_to_result.argmin()
            expected_slot = filled_df.loc[nearest_result_idx, timestamp_col]
            time_diff = time_diffs_to_result.iloc[nearest_result_idx]
        else:
            expected_slot = None
            time_diff = None
        
        # Get previous and next readings in the original data
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
        
    filled_diffs = filled_df[timestamp_col].diff()
    min_allowed_diff = pd.Timedelta(seconds=periodicity_seconds*0.5)
    max_allowed_diff = pd.Timedelta(seconds=periodicity_seconds*1.5)
    if ((min_allowed_diff > filled_diffs) | (filled_diffs > max_allowed_diff)).any():
        raise ValueError("Some timestamp differences in the new resampled df are larger or smaller then allowed.")
        #for idx, diff in enumerate(filled_diffs):
        #    if min_allowed_diff > diff:
        #        print(f"Diff is smaller then possible on idx: {idx} which is datetime {filled_df[timestamp_col].loc[idx]}")
        #    elif diff > max_allowed_diff:
        #        print(f"Diff is larger then possible on idx: {idx} which is datetime {filled_df[timestamp_col].loc[idx]}")
    
     # Add Diff (diff needs to be recalculated so when there is a gap there is no diff after a gap)
    filled_df['Diff'] = filled_df['hodnota'].diff()
    # This is for metadata_001 (we allow 0.001 negative difference and set it to 0)
    # Set Diff to np.nan where it is smaller than -0.001
    filled_df.loc[filled_df['Diff'] <= -0.002, 'Diff'] = np.nan
    filled_df.loc[(filled_df['Diff'] > -0.002) & (filled_df['Diff'] < 0), 'Diff'] = 0
    
    # Calculate diagnostics
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


def process_file_with_gap_filling(input_file: str, output_folder: str, output_folder_metadata: str, timestamp_col: str = 'timestamp_utc', tolerance_percentage=10):
    """
    Process a single CSV file: fill gaps with adaptive periodicity.
    
    Parameters:
    -----------
    input_file : str
        Filepath to file to be resampled
    output_folder : str
        Folder where the processed file will be saved
    output_folder_metadata : str
        Folder where diagnostics of the precessing will be saved
    timestamp_col: str
        Column name with timestamps
    tolerance_percentage : float
        Tolerance as percentage of periodicity
        
    Returns:
    --------
    dict
        Summary statistics for this file
    """
    filepath = input_file
    
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Ensure timestamp column exists
        if timestamp_col not in df.columns:
            raise ValueError(f"File {filepath} missing {timestamp_col} column")
        
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        time_now = pd.Timestamp.now(tz='UTC')
        one_year_ago = time_now - pd.DateOffset(years=1)

        mask = df[timestamp_col] > one_year_ago
        df = df.loc[mask].reset_index(drop=True)
        
        if df.empty:
            raise ValueError(f"File {filepath} doesn't have any data in past year")
        
        # Fill gaps
        #try:
        filled_df, diagnostics = fill_gaps_with_periodicity_adaptive(
            df, 
            timestamp_col=timestamp_col, 
            tolerance_percentage=tolerance_percentage
        )
        #except Exception as e:
        #    print(f"Failed {input_file} with {e}.")
        
        # Create output filepath
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_folder, filename)
        
        # Save the filled dataframe
        filled_df.to_csv(output_filepath, index=False)
        
        # Create output filepath metadata
        filename = os.path.basename(filepath)
        json_path = os.path.join(output_folder_metadata, filename.replace('.csv', '.json'))
        
        def convert_np_types(obj):
            if isinstance(obj, dict):
                return {k: convert_np_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_np_types(elem) for elem in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            else:
                return obj

        metadata_clean = convert_np_types(diagnostics)  # Convert all np types to native Python types
        with open(json_path, 'w') as f:
            json.dump(metadata_clean, f, indent=4)
        
        # Return summary
        return {
            'filename': filename,
            'success': True,
            'periodicity_seconds': diagnostics['periodicity_used_seconds'],
            'original_rows': diagnostics['total_actual_readings'],
            'filled_rows': diagnostics['total_after_filling'],
            'missing_filled': diagnostics['missing_values_filled'],
            'unmatched': diagnostics['unmatched_readings_count'],
            'tolerance_seconds': diagnostics['tolerance_used_seconds']
        }
        
    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'success': False,
            'error': str(e)
        }


def fill_gaps_multithreaded(input_folder="./data_w_diff_001",
                           metadata_folder="./metadata_resample",
                           output_folder="./data_w_diff_001_resampled",
                           tolerance_percentage=10,
                           max_workers=8,
                           timestamp_col: str = 'timestamp_utc'):
    """
    Process all CSV files in input folder using multithreading to fill gaps.
    
    Parameters:
    -----------
    input_folder : str
        Folder containing input CSV files
    metadata_folder : str
        Folder containing metadata for diagnostics data
    output_folder : str
        Folder where processed files will be saved
    tolerance_percentage : float
        Tolerance as percentage of periodicity (default 10%)
    max_workers : int
        Number of threads to use
    timestamp_col : str
        Name of the DataFrame column where timestamps are located
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all CSV files and metadata files
    csv_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')])
    
    if len(csv_files) == 0:
        print(f"No CSV files found in {input_folder}")
        return []
        
    print(f"Processing {len(csv_files)} files with:")
    print(f"  - Tolerance: {tolerance_percentage}%")
    print(f"  - Threads: {max_workers}")
    print(f"  - Input folder: {input_folder}")
    print(f"  - Output folder: {output_folder}")
    print(f"  - Metadata folder: {metadata_folder}")
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create partial function with fixed parameters
        from functools import partial
        process_func = partial(
            process_file_with_gap_filling,
            output_folder=output_folder,
            output_folder_metadata=metadata_folder,
            timestamp_col=timestamp_col,
            tolerance_percentage=tolerance_percentage
        )
        
        # Execute with progress bar
        results = list(tqdm(
            executor.map(process_func, csv_files),
            total=len(csv_files),
            desc="Filling gaps",
            unit="file"
        ))
    
    # Summarize results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  - Successful: {len(successful)}/{len(results)}")
    print(f"  - Failed: {len(failed)}/{len(results)}")
    
    if successful:
        total_original = sum(r['original_rows'] for r in successful)
        total_filled = sum(r['filled_rows'] for r in successful)
        total_missing = sum(r['missing_filled'] for r in successful)
        total_unmatched = sum(r['unmatched'] for r in successful)
        
        print(f"\nAggregate statistics:")
        print(f"  - Total original rows: {total_original:,}")
        print(f"  - Total filled rows: {total_filled:,}")
        print(f"  - Total missing values filled: {total_missing:,}")
        print(f"  - Total unmatched readings: {total_unmatched:,}")
    
    if failed:
        print(f"\nFailed files:")
        for r in failed:
            print(f"  - {r['filename']}: {r['error']}")
    
    return results


def main():
    """
    Example usage
    """
    # Configure your parameters here
    INPUT_FOLDER = "./data_w_diff_001"
    METADATA_FOLDER = "./metadata_resample"
    OUTPUT_FOLDER = "./data_w_diff_001_resampled"
    TOLERANCE_PERCENTAGE = 10  # 10% of periodicity
    MAX_WORKERS = 8
    TIMESTAMP_COL = 'timestamp_utc'
    
    # Run the multithreaded gap filling
    results = fill_gaps_multithreaded(
        input_folder=INPUT_FOLDER,
        metadata_folder=METADATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        tolerance_percentage=TOLERANCE_PERCENTAGE,
        max_workers=MAX_WORKERS,
        timestamp_col=TIMESTAMP_COL
    )
    
    if not results:
        print("No results to save.")
        return
    
    # Create results filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"gap_filling_results_{timestamp}.json"
    results_filepath = os.path.join(OUTPUT_FOLDER, results_filename)
    
    # Get average periodicity from successful results
    successful_results = [r for r in results if r['success']]
    if successful_results:
        avg_periodicity_seconds = sum(r['periodicity_seconds'] for r in successful_results) / len(successful_results)
        avg_periodicity_minutes = sum(r['periodicity_minutes'] for r in successful_results) / len(successful_results)
    else:
        avg_periodicity_seconds = None
        avg_periodicity_minutes = None
    
    # Prepare results for JSON serialization
    results_for_json = {
        'processing_parameters': {
            'input_folder': INPUT_FOLDER,
            'metadata_folder': METADATA_FOLDER,
            'output_folder': OUTPUT_FOLDER,
            'average_periodicity_seconds': avg_periodicity_seconds,
            'average_periodicity_minutes': avg_periodicity_minutes,
            'tolerance_percentage': TOLERANCE_PERCENTAGE,
            'max_workers': MAX_WORKERS,
            'timestamp': timestamp
        },
        'summary': {
            'total_files': len(results),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'total_original_rows': sum(r.get('original_rows', 0) for r in results if r['success']),
            'total_filled_rows': sum(r.get('filled_rows', 0) for r in results if r['success']),
            'total_missing_filled': sum(r.get('missing_filled', 0) for r in results if r['success']),
            'total_unmatched': sum(r.get('unmatched', 0) for r in results if r['success'])
        },
        'file_results': results
    }
    
    # Save to JSON file
    with open(results_filepath, 'w') as json_file:
        json.dump(results_for_json, json_file, indent=4)
    
    print(f"\nResults saved to: {results_filepath}")

def test():
    #df = pd.read_csv()
    try:
        diagnostics = process_file_with_gap_filling('../data_w_diff_001/137402.csv', '.', '../metadata_resample')
    except Exception as e:
        print(f"Fuck {e}")
    print(diagnostics)
    return diagnostics
    

if __name__ == "__main__":
    test()
