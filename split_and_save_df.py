import pandas as pd
import os

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


if __name__ == "__main__":
    main()