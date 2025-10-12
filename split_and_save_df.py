import pandas as pd
import os

def split_and_save_df(df: pd.DataFrame):
    split_dfs = {group_id: group_df for group_id, group_df in df.groupby('id')}
    
    first_df = list(split_dfs.values())[0]
    year = first_df["timestamp"].iloc[0].split("-")[0]
    print(f"Processing year: {year}")
    
    for df_id, df_values in split_dfs.items():
        df_values.to_csv(f"split_data/{df_id}-{year}.csv", index=False)

def main():
    for root, dirs, filenames in os.walk("./zasilka-TP4VGT9M89DDMFIM"):
        for filename in filenames:
            df = pd.read_csv(f"{os.path.join(root, filename)}", sep=";")
            print(f"File {os.path.join(root, filename)} is being processed.")
            split_and_save_df(df)
            print(f"File {os.path.join(root, filename)} is split and saved into multiple dfs based on their ids.")

if __name__ == "__main__":
    main()