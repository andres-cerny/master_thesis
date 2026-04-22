import time
from multiprocessing import Process, Queue
from pathlib import Path
import pickle
from tqdm import tqdm
import pandas as pd
import random

def worker(csv_path, result_queue, epochs_train, epochs_warmstart, verbose):
    # Your existing single-meter logic here:
    # result = process_single_meter(csv_path, ...)
    # For demo:
    try:
        if random.random() < 0.1:
            sleep_for = 10  # deliberately longer than per_file_timeout
            print(f"[TEST] Simulating stuck worker, sleeping {sleep_for}s", flush=True)
            time.sleep(sleep_for)
            
        result = {"filename": Path(csv_path).stem,
                  "status": "failed",
                  "error": str(e)}
    except Exception as e:
        result = {"filename": Path(csv_path).stem,
                  "status": "failed",
                  "error": str(e)}
    result_queue.put(result)

def process_batch_manual(
    csv_filepaths,
    output_csv,
    num_workers=7,
    epochs_train=5,
    epochs_warmstart=5,
    verbose=False,
    per_file_timeout=5,
):
    pending = list(csv_filepaths)
    active = []
    result_queue = Queue()
    all_results = []

    success_count = 0
    failed_count = 0
    timeout_count = 0

    with tqdm(total=len(csv_filepaths), desc="Processing meters", unit="file") as pbar:
        while pending or active:
            # Fill free worker slots
            while pending and len(active) < num_workers:
                filepath = pending.pop(0)
                p = Process(
                    target=worker,
                    args=(filepath, result_queue, epochs_train, epochs_warmstart, verbose),
                )
                p.start()
                active.append({
                    "proc": p,
                    "start": time.time(),
                    "file": filepath,
                })

            new_active = []

            for entry in active:
                p = entry["proc"]
                filepath = entry["file"]
                start = entry["start"]

                if not p.is_alive():
                    p.join(timeout=0.2)

                    if not result_queue.empty():
                        result = result_queue.get()
                    else:
                        result = {
                            "filename": Path(filepath).stem,
                            "filepath": filepath,
                            "status": "failed",
                            "error": "Worker exited without returning result",
                        }

                    all_results.append(result)

                    if result["status"] == "success":
                        success_count += 1
                    else:
                        failed_count += 1

                    pbar.update(1)
                    pbar.set_postfix(
                        success=success_count,
                        failed=failed_count,
                        timeout=timeout_count,
                        running=len(new_active),
                        queued=len(pending),
                    )

                elif time.time() - start > per_file_timeout:
                    p.terminate()
                    p.join(timeout=1)

                    result = {
                        "filename": Path(filepath).stem,
                        "filepath": filepath,
                        "status": "failed",
                        "error": f"Timeout after {per_file_timeout}s",
                    }
                    all_results.append(result)

                    timeout_count += 1
                    failed_count += 1

                    pbar.update(1)
                    pbar.set_postfix(
                        success=success_count,
                        failed=failed_count,
                        timeout=timeout_count,
                        running=len(new_active),
                        queued=len(pending),
                    )

                else:
                    new_active.append(entry)

            active = new_active
            time.sleep(0.2)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    return results_df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GRU Sliding-Window Batch Processor (comparable to UC model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workers",          type=int, default=7)
    parser.add_argument("--epochs-train",     type=int, default=5)
    parser.add_argument("--epochs-warmstart", type=int, default=5)
    parser.add_argument("--samples",          type=int, default=1000)
    parser.add_argument("--verbose",          action="store_true", default=False)

    args = parser.parse_args()


    # ── File selection: identical seed + logic to UC model ──────────────────
    directory  = "../data_w_diff_001"
    seed_value = 42

    #all_files = [
    #    os.path.join(directory, f)
    #    for f in os.listdir(directory)
    #    if os.path.isfile(os.path.join(directory, f))
    #]
    #random.seed(seed_value)
    #csv_filepaths = random.sample(all_files, min(args.samples, len(all_files)))

    
    with open("../pickles/train_set.pkl", "rb") as f:
        csv_filepaths = pickle.load(f)
        
    csv_filepaths = csv_filepaths[8980:9000]

    #logger.info(f"Loaded {len(csv_filepaths)} CSV filepaths")

    output_csv = (
        f"test_it.csv"
    )

    _ = process_batch_manual(
        csv_filepaths,
        output_csv
    )



if __name__ == "__main__":
    main()