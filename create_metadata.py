import os
import pandas as pd
import numpy as np
import json
import sys
from multiprocessing import Pool, cpu_count
from statsmodels.tsa.stattools import acf

input_folder = './data_sorted_timestamp'
metadata_folder = './metadata'
os.makedirs(metadata_folder, exist_ok=True)

# Base metadata generation (only if you want to regenerate all)
def generate_metadata(df):
    metadata = {
        "num_records": len(df),
        "start_time": df['timestamp'].min(),
        "end_time": df['timestamp'].max(),
        "value_min": df['hodnota'].min(),
        "value_max": df['hodnota'].max(),
        "value_mean": df['hodnota'].mean(),
        "value_std": df['hodnota'].std(),
        "diff_min": df['Diff'].min(),
        "diff_max": df['Diff'].max(),
        "diff_mean": df['Diff'].mean(),
        "diff_std": df['Diff'].std()
    }
    return metadata

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

def save_metadata(metadata, json_filepath):
    metadata_clean = convert_np_types(metadata)  # Convert all np types to native Python types
    with open(json_filepath, 'w') as f:
        json.dump(metadata_clean, f, indent=4)

def load_metadata(json_filepath):
    try:
        with open(json_filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {json_filepath}: {e}", file=sys.stderr)
        raise

def create_all_metadata(filename):
    if filename.endswith('.csv'):
        csv_path = os.path.join(input_folder, filename)
        df = pd.read_csv(csv_path)
        metadata = generate_metadata(df)
        json_path = os.path.join(metadata_folder, filename.replace('.csv', '.json'))
        save_metadata(metadata, json_path)
        return filename

def augment_metadata(filename, augmentor):
    json_path = os.path.join(metadata_folder, filename.replace('.csv', '.json'))
    csv_path = os.path.join(input_folder, filename)
    if not os.path.exists(json_path) or not os.path.exists(csv_path):
        return f"Missing file for {filename}"
    metadata = load_metadata(json_path)
    df = pd.read_csv(csv_path)
    metadata = augmentor(df, metadata)
    save_metadata(metadata, json_path)
    return filename

def add_nan_counts(df, metadata):
    metadata['nan_value_count'] = df['hodnota'].isna().sum()
    metadata['nan_diff_count'] = df['Diff'].isna().sum()
    return metadata

def add_periodicity_info(df, metadata):
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
        
    time_diffs = df['timestamp'].diff().dropna().dt.total_seconds()
    
    if time_diffs.empty:
        # If single reading or no diff, set periodicity as None and gaps as 0
        metadata['common_periodicity_seconds'] = None
        metadata['num_gaps_over_5min'] = 0
        return metadata

    common_periodicity = time_diffs.mode().iloc[0]

    gaps_5min = time_diffs[(time_diffs > (common_periodicity + 300)) | (time_diffs < (common_periodicity - 300))].count()
    gaps_5percent = time_diffs[(time_diffs > common_periodicity * 1.05) | (time_diffs < common_periodicity * 0.95)].count()

    metadata['common_periodicity_seconds'] = common_periodicity
    metadata['num_gaps_5min'] = int(gaps_5min)
    metadata['num_gaps_5percent'] = int(gaps_5percent)

    return metadata

def add_autocorrelation_info(df, metadata):
    """
    Adds autocorrelation related metadata to the metadata dictionary.
    Computes the autocorrelation function of the 'diff' column and finds the lag
    with the highest autocorrelation peak beyond lag 0.
    """
    diffs = df['Diff'].dropna()
    periodicity = metadata['common_periodicity_seconds']
    max_lag = int(60*60*24*7 // periodicity) if periodicity and periodicity > 0 else 0
    
    if len(diffs) < 2*max_lag or max_lag < 1 or diffs.var() == 0:
        metadata['max_autocorr'] = None
        metadata['max_autocorr_lag'] = None
        metadata['autocorr_mean'] = None
        metadata['autocorr_std'] = None
        return metadata
    
    acf_values = acf(diffs, nlags=max_lag, fft=True, missing='drop')
    
    if np.any(np.isnan(acf_values)) or np.any(np.isinf(acf_values)):
        metadata['max_autocorr'] = None
        metadata['max_autocorr_lag'] = None
        metadata['autocorr_mean'] = None
        metadata['autocorr_std'] = None
        return metadata
    
    # Ignore lag 0 (which is always 1)
    acf_lags = np.arange(len(acf_values))
    acf_values_no_lag0 = acf_values[1:]
    acf_lags_no_lag0 = acf_lags[1:]
    
    max_peak_idx = np.argmax(np.abs(acf_values_no_lag0))
    max_peak_value = acf_values_no_lag0[max_peak_idx]
    max_peak_lag = acf_lags_no_lag0[max_peak_idx]

    # Summary statistics on autocorrelation values (excluding lag 0)
    acf_mean = np.mean(acf_values_no_lag0)
    acf_std = np.std(acf_values_no_lag0)

    metadata['max_autocorr'] = float(max_peak_value)
    metadata['max_autocorr_lag'] = int(max_peak_lag)
    metadata['autocorr_mean'] = float(acf_mean)
    metadata['autocorr_std'] = float(acf_std)
    
    return metadata


if __name__ == '__main__':
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    chunksize = len(files) // cpu_count() + 1

    # Create all base metadata (run once)
    # with Pool(processes=cpu_count()) as pool:
    #     for i, fname in enumerate(pool.imap_unordered(create_all_metadata, files, chunksize=chunksize)):
    #         if i % 5000 == 0:
    #             print(f"Created base metadata for {i} files")

    # Add NaN any other new metadata in parallel
    with Pool(processes=cpu_count()) as pool:
        # Use partial to fix the augmentor parameter
        from functools import partial
        augment_with_nans = partial(augment_metadata, augmentor=add_periodicity_info)
        for i, fname in enumerate(pool.imap_unordered(augment_with_nans, files, chunksize=chunksize)):
            if i % 5000 == 0:
                print(f"Augmented metadata for {i} files")

    print("Metadata augmentation completed.")
