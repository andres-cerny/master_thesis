import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_timestamp_file(args):
    """
    Process a single CSV file for timestamp sorting.
    """
    filename, input_dir, output_dir, output_dir_failed, timestamp_col, value_col = args
    
    input_filepath = os.path.join(input_dir, filename)
    output_filepath = os.path.join(output_dir, filename)
    output_filepath_failed = os.path.join(output_dir_failed, filename)

    df = pd.read_csv(input_filepath)
    
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%Y-%m-%d')
    
    if df[timestamp_col].is_monotonic_increasing:
        df.to_csv(output_filepath, index=False)
        return
    
    unsorted_indices = df.index[df[timestamp_col].diff() < pd.Timedelta(0)].tolist()
    for idx in unsorted_indices:
        if df.iloc[idx-1][value_col] - df.iloc[idx][value_col] <= 0:
            # Swap timestamp values
            tmp = df.at[idx, timestamp_col]
            df.at[idx, timestamp_col] = df.at[idx-1, timestamp_col]
            df.at[idx-1, timestamp_col] = tmp
        else:
            # Swap whole rows
            df.iloc[[idx-1, idx]] = df.iloc[[idx, idx-1]].values
            
    df.to_csv(output_filepath, index=False)
        
    values_no_na = df[value_col].dropna()
    if not values_no_na.is_monotonic_increasing:
        df.to_csv(output_filepath_failed, index=False)


def sort_dfs_timestamp(input_dir="./data", output_dir="./data_sorted_timestamp", 
                       timestamp_col='timestamp', value_col="hodnota", 
                       output_dir_failed="./data_sorted_timestamp_decrease_value",
                       max_workers=8):
    '''
    Check and sort CSV files by timestamp column if needed,
    and save sorted DataFrames to a new directory using multithreading.
    '''
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    # Prepare arguments for each file
    args_list = [(f, input_dir, output_dir, output_dir_failed, timestamp_col, value_col) 
                 for f in files]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_timestamp_file, args_list), 
                  total=len(args_list), 
                  desc="Sorting timestamps", 
                  unit="file"))
    
    assert len(os.listdir(input_dir)) == len(os.listdir(output_dir)), \
        f"Number of dfs in input directory ({len(os.listdir(input_dir))}) does not equal number of dfs in the directory ({len(os.listdir(output_dir))})."
    
    print(f"Processed all timestamp sorting with {max_workers} threads")


def process_file(filepath):
    df = pd.read_csv(filepath)

    df['Diff'] = df['hodnota'].diff()
    df['Diff'] = df['Diff'].fillna(0)
    #df.loc[df['Diff'] < -0.002, 'Diff'] = 0 # maybe exclude
    df.loc[df['Diff'] < 0, 'Diff'] = np.nan

    df.to_csv(filepath, index=False)


def calculate_diff_multithreaded(folder="./data_sorted_timestamp", max_workers=8):
    '''
    Calculate Diff values from Values in all CSV files in the specified folder using multithreading.
    Adds 'Diff' column, replacing negative differences with NaN,
    and overwrites the original files.
    '''
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_file, filepaths), 
                  total=len(filepaths), 
                  desc="Calculating diffs", 
                  unit="file"))

    print(f"Processed all CSV files in folder: {folder} with {max_workers} threads")

    
def main():
    #sort_dfs_timestamp()
    calculate_diff_multithreaded()
    
if __name__ == "__main__":
    main()
