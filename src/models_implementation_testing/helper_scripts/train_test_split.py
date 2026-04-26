import os
import random
import pickle

# 1. Collect all file paths from a root directory
root_dir = "./data_w_diff_001"  # <- change this
all_filepaths = []
random.seed(42)

for dirpath, dirnames, filenames in os.walk(root_dir):
    for fname in filenames:
        full_path = os.path.join("../data_w_diff_001", fname)
        all_filepaths.append(full_path)

# 2. Shuffle the file paths (for random split)
random.shuffle(all_filepaths)

# 3. Split into 80% train and 20% test
n_total = len(all_filepaths)
n_train = int(0.8 * n_total)

train_filepaths = all_filepaths[:n_train]
test_filepaths = all_filepaths[n_train:]

# 4. Save to pickle files
with open("./pickles/train_set.pkl", "wb") as f:
    pickle.dump(train_filepaths, f)

with open("./pickles/test_set.pkl", "wb") as f:
    pickle.dump(test_filepaths, f)

print(f"Saved {len(train_filepaths)} train paths to train.pkl")
print(f"Saved {len(test_filepaths)} test paths to test.pkl")