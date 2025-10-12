import os
import pandas as pd
from collections import defaultdict

def concat_and_save_dfs_same_id(save_dir: str, filepaths: list, id):
    dfs = []
    for filepath in filepaths:
        dfs.append(pd.read_csv(filepath))
    
    concat_df = pd.concat(dfs, axis=0, ignore_index=True)
    concat_df.to_csv(f"{save_dir}/{id}.csv", index=False)

def concat_split_dfs(split_dfs_dir: str, save_dir: str):
    ids = set()
    files_dict = defaultdict(list)
    
    for filename in os.listdir(split_dfs_dir):
        id = filename.split("-", 1)[0]
        ids.add(id)
        
    for filename in os.listdir(split_dfs_dir):
        id = filename.split("-", 1)[0]
        if id in ids:
            files_dict[id].append(os.path.join(split_dfs_dir, filename))
    
    print(f"Starting to concatenate all dfs and saving them in {save_dir} directory.")
    for id, filepaths in files_dict.items():
        concat_and_save_dfs_same_id(save_dir, filepaths, id)
    print("All dfs concatenated.")
        
def main():
    concat_split_dfs("./split_data", "./data")
        
if __name__ == "__main__":
    main()
            