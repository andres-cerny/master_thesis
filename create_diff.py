import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def sort_dfs_timestamp(input_dir="./data", output_dir="./data_sorted_timestamp", timestamp_col='timestamp', value_col="hodnota", output_dir_failed="./data_sorted_timestamp_decrease_value"):
    '''
    Check and sort CSV files by timestamp column if needed,
    and save sorted DataFrames to a new directory.
    '''
    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv'):
            continue
        
        input_filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, filename)
        output_filepath_failed = os.path.join(output_dir_failed, filename)
    
        df = pd.read_csv(input_filepath)
        
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="%Y-%m-%d %H:%M:%S")
        if not df[timestamp_col].is_monotonic_increasing:
            df = df.sort_values(by=timestamp_col).reset_index(drop=True)
        
        df.to_csv(output_filepath, index=False)
            
        values_no_na = df[value_col].dropna()
        if not values_no_na.is_monotonic_increasing:
            df.to_csv(output_filepath_failed, index=False)
    
    assert len(os.listdir(input_dir)) == len(os.listdir(output_dir)), \
        f"Number of dfs in input directory ({len(os.listdir(input_dir))}) does not equal number of dfs in the directory ({len(os.listdir(output_dir))})."
    

def process_file(filepath):
    df = pd.read_csv(filepath)

    df['Diff'] = df['Value'].diff()
    df['Diff'] = df['Diff'].fillna(0)
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
        executor.map(process_file, filepaths)

    print(f"Processed all CSV files in folder: {folder} with {max_workers} threads")

    
def main():
    #sort_dfs_timestamp()
    calculate_diff_multithreaded()
    
if __name__ == "__main__":
    main()