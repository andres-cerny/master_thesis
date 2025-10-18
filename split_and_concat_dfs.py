import os
import pandas as pd
from collections import defaultdict

def split_and_save_df(df: pd.DataFrame) -> int:
    new_num_rows = 0
    split_dfs = {group_id: group_df for group_id, group_df in df.groupby('id')}
    
    first_df = list(split_dfs.values())[0]
    year = first_df["timestamp"].iloc[0].split("-")[0]
    print(f"Processing year: {year}")
    
    for df_id, df_values in split_dfs.items():
        new_num_rows += df_values.shape[0]
        df_values.to_csv(f"split_data/{df_id}-{year}.csv", index=False)
    
    return new_num_rows

def concat_and_save_dfs_same_id(save_dir: str, filepaths: list, id):
    dfs = [pd.read_csv(filepath) for filepath in filepaths]
        
    concat_df = pd.concat(dfs, axis=0, ignore_index=True)
    concat_df.to_csv(f"{save_dir}/{id}.csv", index=False)

def concat_split_dfs(split_dfs_dir: str, save_dir: str):
    files_dict = defaultdict(list)
    for filename in os.listdir(split_dfs_dir):
        id = filename.split("-", 1)[0]
        files_dict[id].append(os.path.join(split_dfs_dir, filename))
    
    num_files_dict = sum(len(sublist) for sublist in files_dict.values())
        
    assert num_files_dict == len(os.listdir(split_dfs_dir)), \
        f"Number of dfs in dictionary ({num_files_dict}) does not equal number of dfs in the directory ({len(os.listdir(split_dfs_dir))})."
    
    print(f"Starting to concatenate all dfs and saving them in {save_dir} directory.")
    for id, filepaths in files_dict.items():
        concat_and_save_dfs_same_id(save_dir, filepaths, id)
    print("All dfs concatenated.")
    

def main():
    for root, dirs, filenames in os.walk("./zasilka-TP4VGT9M89DDMFIM"):
        for filename in filenames:
            print(f"\nFile {os.path.join(root, filename)} is being processed.")
            df = pd.read_csv(f"{os.path.join(root, filename)}", sep=";")
            
            original_num_rows = df.shape[0]
            new_num_rows = split_and_save_df(df)
            
            print(f"File {os.path.join(root, filename)} is split and saved into multiple dfs based on their ids.")
            
            if original_num_rows != new_num_rows:
                raise ValueError(f"Mismatch in row counts: original={original_num_rows}, new={new_num_rows} after splitting {os.path.join(root, filename)}")
            
    concat_split_dfs("./split_data", "./data")
    
if __name__ == "__main__":
    main()
            