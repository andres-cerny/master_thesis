import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
from datetime import datetime


def fill_gaps_with_periodicity_adaptive(df, periodicity_seconds, tolerance_percentage=10):
    """
    Fill gaps in time-series data for a single id sensor using adaptive timestamp generation.
    Instead of creating a full expected range upfront, this function builds the expected timeline
    iteratively based on actual readings, adjusting for timing drift.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: 'timestamp', 'hodnota', 'Diff' (and optionally 'id')
        Must contain data for a single sensor only.
    periodicity_seconds : int
        Expected time interval between consecutive readings in seconds (e.g., 1800 for 30 minutes)
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
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if len(df) == 0:
        return df, {}
    
    # Calculate tolerance in seconds based on percentage
    tolerance_seconds = (tolerance_percentage / 100.0) * periodicity_seconds
    
    # Initialize variables for adaptive timeline building
    result_rows = []
    unmatched_detail = []
    matched_indices = set()
    
    # Start with the first timestamp
    expected_next = df.loc[0, 'timestamp']
    
    # Track the original data end time
    data_end_time = df['timestamp'].max()
    
    def get_candidate_indeces(df, within_tolerance, matched_indices):
        if not within_tolerance.any():
            return []
        # Get indices of all readings within tolerance
        candidate_indices = df.index[within_tolerance].tolist()
        # Remove already matched indices
        candidate_indices = [idx for idx in candidate_indices if idx not in matched_indices]
        return candidate_indices
    
    while expected_next <= data_end_time:
        # Find readings within tolerance of expected_next
        time_diffs = abs((df['timestamp'] - expected_next).dt.total_seconds())
        within_tolerance = time_diffs <= tolerance_seconds
        candidate_indices = get_candidate_indeces(df, within_tolerance, matched_indices)
        
        if not candidate_indices:
            # No reading within tolerance, add NaN row
            row_data = {
                'id': df.loc[0, 'id'],
                'timestamp': expected_next,
                'hodnota': np.nan,
                'Diff': np.nan
            }
                        
            result_rows.append(row_data)
            expected_next = expected_next + pd.Timedelta(seconds=periodicity_seconds)
            continue
        
        # Find the closest one among candidates
        closest_idx = min(candidate_indices, key=lambda idx: time_diffs[idx])
        
        # Add this reading to results
        row_data = {
            'id': df.loc[closest_idx, 'id'],
            'timestamp': expected_next,
            'hodnota': df.loc[closest_idx, 'hodnota'],
            'Diff': df.loc[closest_idx, 'Diff']
        }                
        result_rows.append(row_data)
        matched_indices.add(closest_idx)
        
        # Calculate next expected timestamp from the ACTUAL reading
        actual_reading_time = df.loc[closest_idx, 'timestamp']
        expected_next = actual_reading_time + pd.Timedelta(seconds=periodicity_seconds)
            
            
    # Create the filled DataFrame
    filled_df = pd.DataFrame(result_rows)
    
    # Reorder columns to match original
    if 'id' in df.columns:
        filled_df = filled_df[['id', 'timestamp', 'hodnota', 'Diff']]
    else:
        filled_df = filled_df[['timestamp', 'hodnota', 'Diff']]
    
    # Identify unmatched readings
    unmatched_indices = [idx for idx in df.index if idx not in matched_indices]
    
    # Build detailed unmatched readings list
    for idx in unmatched_indices:
        row = df.loc[idx]
        
        # Find what the expected slot would have been (closest in our result)
        if len(filled_df) > 0:
            time_diffs_to_result = abs((filled_df['timestamp'] - row['timestamp']).dt.total_seconds())
            nearest_result_idx = time_diffs_to_result.argmin()
            expected_slot = filled_df.loc[nearest_result_idx, 'timestamp']
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
            'reading_timestamp': str(row['timestamp']),
            'hodnota': float(row['hodnota']) if pd.notna(row['hodnota']) else None,
            'diff': float(row['Diff']) if pd.notna(row['Diff']) else None,
            'expected_slot': str(expected_slot) if expected_slot else None,
            'time_difference_seconds': float(time_diff) if time_diff is not None else None,
            'prev_reading_timestamp': str(prev_row['timestamp']) if prev_row is not None else None,
            'prev_reading_value': float(prev_row['hodnota']) if prev_row is not None and pd.notna(prev_row['hodnota']) else None,
            'next_reading_timestamp': str(next_row['timestamp']) if next_row is not None else None,
            'next_reading_value': float(next_row['hodnota']) if next_row is not None and pd.notna(next_row['hodnota']) else None
        })
    
    # Calculate diagnostics
    total_filled = len(filled_df)
    missing_count = filled_df['hodnota'].isna().sum()
    expected_per_day = (24 * 60 * 60) / periodicity_seconds
    
    filled_df['date'] = filled_df['timestamp'].dt.date
    samples_per_day = filled_df.groupby('date').size()
    filled_df = filled_df.drop(columns=['date'])
    
    diagnostics = {
        'total_expected_timestamps': total_filled,
        'total_actual_readings': len(df),
        'total_after_filling': total_filled,
        'total_matched_readings': len(matched_indices),
        'missing_values_filled': missing_count,
        'unmatched_readings_count': len(unmatched_indices),
        'unmatched_readings_detail': unmatched_detail,
        'expected_samples_per_day': expected_per_day,
        'actual_samples_per_day': {str(k): int(v) for k, v in samples_per_day.to_dict().items()},
        'date_range': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
        'periodicity_used_seconds': periodicity_seconds,
        'tolerance_percentage': tolerance_percentage,
        'tolerance_used_seconds': tolerance_seconds
    }
    
    return filled_df, diagnostics


def process_file_with_gap_filling(file_metadata_pair, output_folder, tolerance_percentage=10):
    """
    Process a single CSV file: fill gaps with adaptive periodicity.
    
    Parameters:
    -----------
    file_metadata_pair : tuple
        Tuple of (filepath, metadata_filepath)
    output_folder : str
        Folder where the processed file will be saved
    tolerance_percentage : float
        Tolerance as percentage of periodicity
        
    Returns:
    --------
    dict
        Summary statistics for this file
    """
    filepath, metadata_filepath = file_metadata_pair
    
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            raise ValueError(f"File {filepath} missing 'timestamp' column")
        
        # Get periodicity from metadata
        with open(metadata_filepath, 'r') as file:
            metadata = json.load(file)
            
        periodicity_seconds = metadata["common_periodicity_seconds"]
        
        # Fill gaps
        filled_df, diagnostics = fill_gaps_with_periodicity_adaptive(
            df, 
            periodicity_seconds=periodicity_seconds, 
            tolerance_percentage=tolerance_percentage
        )
        
        # Create output filepath
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_folder, filename)
        
        # Save the filled dataframe
        filled_df.to_csv(output_filepath, index=False)
        
        # Return summary
        return {
            'filename': filename,
            'success': True,
            'periodicity_seconds': periodicity_seconds,
            'periodicity_minutes': round(periodicity_seconds / 60, 2),
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
                           metadata_folder="./metadata_001",
                           output_folder="./data_w_diff_001_resampled",
                           tolerance_percentage=10,
                           max_workers=8):
    """
    Process all CSV files in input folder using multithreading to fill gaps.
    
    Parameters:
    -----------
    input_folder : str
        Folder containing input CSV files
    metadata_folder : str
        Folder containing metadata for input data
    output_folder : str
        Folder where processed files will be saved
    tolerance_percentage : float
        Tolerance as percentage of periodicity (default 10%)
    max_workers : int
        Number of threads to use
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all CSV files and metadata files
    csv_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.csv')])
    metadata_files = sorted([f for f in os.listdir(metadata_folder) if f.endswith('.json')])
    
    if len(csv_files) == 0:
        print(f"No CSV files found in {input_folder}")
        return []
    
    if len(csv_files) != len(metadata_files):
        print(f"Warning: Number of metadata files ({len(metadata_files)}) does not equal number of CSV files ({len(csv_files)}).")
        print(f"Will only process files that have matching metadata.")
    
    # Create pairs of (csv_filepath, metadata_filepath) by matching filenames
    file_pairs = []
    for csv_file in csv_files:
        # Assume metadata has same name but .json extension
        base_name = os.path.splitext(csv_file)[0]
        metadata_file = base_name + '.json'
        
        if metadata_file in metadata_files:
            csv_path = os.path.join(input_folder, csv_file)
            metadata_path = os.path.join(metadata_folder, metadata_file)
            file_pairs.append((csv_path, metadata_path))
        else:
            print(f"Warning: No metadata found for {csv_file}, skipping.")
    
    if len(file_pairs) == 0:
        print("No matching file-metadata pairs found!")
        return []
    
    print(f"Processing {len(file_pairs)} files with:")
    print(f"  - Tolerance: {tolerance_percentage}%")
    print(f"  - Threads: {max_workers}")
    print(f"  - Output folder: {output_folder}")
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create partial function with fixed parameters
        from functools import partial
        process_func = partial(
            process_file_with_gap_filling,
            output_folder=output_folder,
            tolerance_percentage=tolerance_percentage
        )
        
        # Execute with progress bar
        results = list(tqdm(
            executor.map(process_func, file_pairs),
            total=len(file_pairs),
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
    METADATA_FOLDER = "./metadata_001"
    OUTPUT_FOLDER = "./data_w_diff_resampled"
    TOLERANCE_PERCENTAGE = 10  # 10% of periodicity
    MAX_WORKERS = 8
    
    # Run the multithreaded gap filling
    results = fill_gaps_multithreaded(
        input_folder=INPUT_FOLDER,
        metadata_folder=METADATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        tolerance_percentage=TOLERANCE_PERCENTAGE,
        max_workers=MAX_WORKERS
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
    df = pd.read_csv('data_w_diff_001/2328.csv')
    filled_df, diagnostics = fill_gaps_with_periodicity_adaptive(df, 30*60)
    print(diagnostics)
    

if __name__ == "__main__":
    test()
