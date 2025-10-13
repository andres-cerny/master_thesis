import os
import pandas as pd
from collections import defaultdict

def concat_and_save_dfs_same_id(save_dir: str, filepaths: list, id):
    dfs = [pd.read_csv(filepath) for filepath in filepaths]
        
    concat_df = pd.concat(dfs, axis=0, ignore_index=True)
    concat_df.to_csv(f"{save_dir}/{id}.csv", index=False)

def concat_split_dfs(split_dfs_dir: str, save_dir: str):
    files_dict = defaultdict(list)
    for filename in os.listdir(split_dfs_dir):
        id = filename.split("-", 1)[0]
        files_dict[id].append(os.path.join(split_dfs_dir, filename))
        
    assert len(files_dict.values()) != len(os.listdir(split_dfs_dir)), \
        f"Number of dfs in dictionary ({len(files_dict.values())}) does not equal number of dfs in the directory ({len(os.listdir(split_dfs_dir))})."
    
    print(f"Starting to concatenate all dfs and saving them in {save_dir} directory.")
    for id, filepaths in files_dict.items():
        concat_and_save_dfs_same_id(save_dir, filepaths, id)
    print("All dfs concatenated.")
        
def main():
    concat_split_dfs("./split_data", "./data")
        
if __name__ == "__main__":
    main()
            