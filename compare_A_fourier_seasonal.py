
import os
import json
import random
import pickle
from datetime import datetime, timedelta
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import pandas as pd
from tqdm import tqdm

# Your imports (adjust based on your actual module structure)
# from your_module import get_KF_model_fourier, WaterMeterKalmanFilter, KalmanFilterMetrics

from test.kf_wrls_seasonal_lags import KalmanFilterWRLSEstimatorSeasonalLags
from test.kf_seasonal_lags import WaterMeterKalmanFilterSeasonalLags
from test.kf_metrics import evaluate_kalman_filter, KalmanFilterMetrics

from test.kf_wrls_estimator import KalmanFilterWRLSEstimator
from test.kf import WaterMeterKalmanFilter


def get_KF_model_seasonal(df, expected_interval_seconds, tolerance_percent = 10.0, lambda_forget=0.9999):
    estimator = KalmanFilterWRLSEstimatorSeasonalLags(lambda_forget=lambda_forget)

    A, Q, R, metadata = estimator.estimate_from_dataframe(
        df,
        expected_interval_seconds=expected_interval_seconds,
        tolerance_percent=tolerance_percent,
        verbose=False
    )

    return A, Q, R, metadata

def get_KF_model_fourier(df, expected_interval_seconds, tolerance_percent = 10.0, n_harmonics=3, lambda_forget=0.9999):
    estimator = KalmanFilterWRLSEstimator(lambda_forget=lambda_forget)

    A, Q, R, metadata = estimator.estimate_from_dataframe(
        df,
        expected_interval_seconds=expected_interval_seconds,
        n_harmonics=n_harmonics,
        tolerance_percent=tolerance_percent,
        verbose=False
    )

    return A, Q, R, metadata


def process_single_file(file, df_dir, metadata_dir, shared_skip_small, shared_skip_error):
    """
    Process a single file and return results.

    Parameters:
    -----------
    file : str
        Filename to process
    df_dir : str
        Directory containing data files
    metadata_dir : str
        Directory containing metadata files
    shared_skip_small : multiprocessing.Manager.list
        Shared list for files with small periods
    shared_skip_error : multiprocessing.Manager.list
        Shared list for files with errors

    Returns:
    --------
    tuple : (file, estimator_metadata, metrics) or (file, None, None) if skipped/error
    """
    #print(f"Loading file: {file}")
    df_filepath = os.path.join(df_dir, file)
    metadata_filepath = os.path.join(metadata_dir, file.split('.')[0] + '.json')

    try:
        # Load data
        df = pd.read_csv(df_filepath)

        with open(metadata_filepath) as f:
            metadata_file = json.load(f)

        start = datetime.fromisoformat(metadata_file['start_time'])
        end = datetime.fromisoformat(metadata_file['end_time'])

        # Check time range
        if (end - start) < timedelta(days=365):
            print(f"Skipping {file}: Less than 365 days of data")
            return (file, None, None)

        # Check periodicity
        if metadata_file['common_periodicity_seconds'] > 12*60*60:
            shared_skip_small.append(file)
            print(f"Skipping file {file}: Too long periodicity")
            return (file, None, None)

        # Get Kalman Filter model
        try:
            A, Q, R, metadata_estimator = get_KF_model_fourier(
                df, 
                expected_interval_seconds=metadata_file['common_periodicity_seconds']
            )
        except Exception as e:
            shared_skip_error.append(file)
            print(f"Error {e} occurred when estimating file: {file}.")
            return (file, None, None)

        # Create Kalman Filter
        kf = WaterMeterKalmanFilter(A, Q, R, n_harmonics=3, seasonal_period=24.0)

        # Filter dataframe
        try:
            df_filtered = kf.filter_dataframe(
                df,
                anomaly_threshold=3.0,
                return_diagnostics=True
            )
        except Exception as e:
            shared_skip_error.append(file)
            print(f"Error {e} occurred when filtering file: {file}.")
            return (file, None, None)

        # Skip first 480 rows (48*10)
        df_filtered = df_filtered[48*10:]

        # Compute metrics
        metrics = KalmanFilterMetrics.compute_all_metrics(
            df_filtered,
            nonzero_threshold=0.01
        )

        #print(f"Successfully processed {file}")
        return (file, metadata_estimator, metrics)

    except Exception as e:
        shared_skip_error.append(file)
        print(f"Unexpected error processing {file}: {e}")
        return (file, None, None)


def save_results(estimator_metadatas, all_metrics, skipped_small, skipped_error, 
                 output_dir='./results'):
    """
    Save all results to pickle files.

    Parameters:
    -----------
    estimator_metadatas : dict
        Dictionary of estimator metadata
    all_metrics : dict
        Dictionary of all metrics
    skipped_small : list
        List of files skipped due to small period
    skipped_error : list
        List of files with errors
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save estimator metadata
    with open(os.path.join(output_dir, 'estimator_metadatas_fourier.pkl'), 'wb') as f:
        pickle.dump(estimator_metadatas, f)
    print(f"Saved estimator_metadatas_fourier.pkl ({len(estimator_metadatas)} entries)")

    # Save metrics
    with open(os.path.join(output_dir, 'all_metrics_fourier.pkl'), 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f"Saved all_metrics_fourier.pkl ({len(all_metrics)} entries)")

    # Save skipped files
    with open(os.path.join(output_dir, 'skipped_files_small_period.pkl'), 'wb') as f:
        pickle.dump(skipped_small, f)
    print(f"Saved skipped_files_small_period.pkl ({len(skipped_small)} entries)")

    with open(os.path.join(output_dir, 'skipped_files_error.pkl'), 'wb') as f:
        pickle.dump(skipped_error, f)
    print(f"Saved skipped_files_error.pkl ({len(skipped_error)} entries)")


def load_results(output_dir='./results'):
    """
    Load previously saved results from pickle files.

    Parameters:
    -----------
    output_dir : str
        Directory containing saved results

    Returns:
    --------
    tuple : (estimator_metadatas, all_metrics, skipped_small, skipped_error)
    """
    with open(os.path.join(output_dir, 'estimator_metadatas_fourier.pkl'), 'rb') as f:
        estimator_metadatas = pickle.load(f)

    with open(os.path.join(output_dir, 'all_metrics_fourier.pkl'), 'rb') as f:
        all_metrics = pickle.load(f)

    with open(os.path.join(output_dir, 'skipped_files_small_period.pkl'), 'rb') as f:
        skipped_small = pickle.load(f)

    with open(os.path.join(output_dir, 'skipped_files_error.pkl'), 'rb') as f:
        skipped_error = pickle.load(f)

    return estimator_metadatas, all_metrics, skipped_small, skipped_error


def main(random_df_filepaths, df_dir='./data_w_diff_001', metadata_dir='./metadata_001',
         output_dir='./results', n_processes=None, save_interval=50):
    """
    Main function to process files in parallel.

    Parameters:
    -----------
    random_df_filepaths : list
        List of file paths to process
    df_dir : str
        Directory containing data files
    metadata_dir : str
        Directory containing metadata files
    output_dir : str
        Directory to save results
    n_processes : int or None
        Number of processes to use (None = cpu_count - 1)
    save_interval : int
        Save intermediate results every N files
    """
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)

    print(f"Starting multiprocessing with {n_processes} processes")
    print(f"Processing {len(random_df_filepaths)} files")

    # Create manager for shared lists
    with Manager() as manager:
        shared_skip_small = manager.list()
        shared_skip_error = manager.list()

        # Create partial function with fixed arguments
        process_func = partial(
            process_single_file,
            df_dir=df_dir,
            metadata_dir=metadata_dir,
            shared_skip_small=shared_skip_small,
            shared_skip_error=shared_skip_error
        )

        # Initialize result dictionaries
        estimator_metadatas_fourier = {}
        all_metrics_fourier = {}

        # Process files in parallel
        with Pool(processes=n_processes) as pool:
            # Use imap_unordered for better performance and progress tracking
            results = pool.imap_unordered(process_func, random_df_filepaths)

            # Process results as they come in
            for idx, (file, metadata_estimator, metrics) in enumerate(tqdm(results, 
                                                                            total=len(random_df_filepaths),
                                                                            desc="Processing files")):
                if metadata_estimator is not None and metrics is not None:
                    estimator_metadatas_fourier[file] = metadata_estimator
                    all_metrics_fourier[file] = metrics

                # Save intermediate results periodically
                if (idx + 1) % save_interval == 0:
                    print(f"\nSaving intermediate results after {idx + 1} files...")
                    save_results(
                        estimator_metadatas_fourier,
                        all_metrics_fourier,
                        list(shared_skip_small),
                        list(shared_skip_error),
                        output_dir
                    )

        # Convert shared lists to regular lists
        skipped_files_small_period = list(shared_skip_small)
        skipped_files_error = list(shared_skip_error)

    # Final save
    print("\nSaving final results...")
    save_results(
        estimator_metadatas_fourier,
        all_metrics_fourier,
        skipped_files_small_period,
        skipped_files_error,
        output_dir
    )

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(random_df_filepaths)}")
    print(f"Successfully processed: {len(estimator_metadatas_fourier)}")
    print(f"Skipped (small period): {len(skipped_files_small_period)}")
    print(f"Skipped (errors): {len(skipped_files_error)}")
    print("="*60)

    return estimator_metadatas_fourier, all_metrics_fourier, skipped_files_small_period, skipped_files_error


if __name__ == "__main__":
    # Example usage:
    # Assuming you have your random_df_filepaths defined

    # Option 1: Define your file list here
    # random_df_filepaths = [...]

    # Option 2: Load from a saved list
    # with open('random_df_filepaths.pkl', 'rb') as f:
    #     random_df_filepaths = pickle.load(f)

    directory = "data_w_diff_001"
    seed_value = 53

    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    random.seed(seed_value)
    random_df_filepaths = random.sample(all_files, 1000)
    # Run the multiprocessing
    estimator_metadatas_fourier, all_metrics_fourier, \
        skipped_files_small_period, skipped_files_error = main(
            random_df_filepaths,
            df_dir='./data_w_diff_001',
            metadata_dir='./metadata_001',
            output_dir='./results',
            n_processes=None,  # Will use cpu_count - 1
            save_interval=500   # Save every 50 files
        )

    # Results are now saved in ./results/ directory and also returned as variables
    # You can load them later using:
    # estimator_metadatas_fourier, all_metrics_fourier, \
    #     skipped_files_small_period, skipped_files_error = load_results('./results')
