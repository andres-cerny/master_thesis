"""
make_test_segments.py — Extract a random N-week (default 8) segment from each
meter CSV listed in a pickle of *filenames*, for testing the streaming UC
detector (train.py / predict.py).

The pickle holds filenames only, so a --data-dir is prepended to each.

Window selection mirrors the project's split_df_sliding_weeks: a random
N-week window is drawn (seeded per-file for reproducibility) and accepted only
if it has at least one measurement per day, so the resulting segments are
actually usable downstream. All original columns are preserved (timestamp_utc,
hodnota, and Diff/is_anomaly if present).

Optionally (--train-weeks K) it also writes, per meter, a train/stream split of
the same window — the first K weeks to train/ and the remaining N-K weeks to
stream/ — which is exactly the shape the two-script pipeline consumes:
fit on train/<file>, then replay stream/<file> row-by-row through predict.py.

Usage:
    python make_test_segments.py \
        --pickle ./pickles/test_set.pkl \
        --data-dir ./data/sensor_data \
        --output-dir ./test_segments \
        --weeks 8 \
        [--train-weeks 4] \
        [--workers 7] [--seed 42] [--limit 100]
        
        python make_test_segments.py --pickle ../../pickles/test_set.pkl --data-dir ../../../data/sensor_data --output-dir ./test_segments --weeks 8 --timestamp-col timestamp --hodnota-col hodnota --workers 7 --seed 42
"""

import os
import sys
import pickle
import zlib
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numpy.random import default_rng

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    def tqdm(it, **kwargs):
        return it


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================================================
# WINDOW EXTRACTION (adapted from split_df_sliding_weeks)
# ==========================================================================
def extract_random_window(df, seed, total_weeks=8, timestamp_col="timestamp_utc",
                          require_daily_coverage=True,
                          min_days_with_data_per_day=1,
                          max_tries=30):
    """
    Slice a random `total_weeks`-week window out of df. The window is accepted
    only if (when require_daily_coverage) every day in it has at least one
    measurement. The original `timestamp_col` is preserved verbatim in the
    output (a temporary parsed column is used only for the boundary logic, so
    no timezone relabeling leaks into the saved file). Returns
    (segment_df, start_ts, end_ts).
    """
    df = df.copy()
    # Parse for windowing only. utc=True just gives a consistent ordering/day
    # grid for naive local strings; it does NOT alter the preserved column.
    parsed = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    keep = parsed.notna()
    df = df[keep].copy()
    df["_ts"] = parsed[keep].values
    df = df.sort_values("_ts").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Series too short (len < 2).")

    min_ts = df["_ts"].min()
    max_ts = df["_ts"].max()

    total_days_required = 7 * total_weeks
    if max_ts - min_ts < pd.Timedelta(days=total_days_required):
        raise ValueError(
            f"Not enough data: span {(max_ts - min_ts).days}d < required {total_days_required}d."
        )

    df["date"] = df["_ts"].dt.floor("D")
    daily_counts = df.groupby("date").size().rename("count").reset_index()

    def window_has_no_big_gaps(start_ts, end_ts):
        window_days = pd.date_range(
            start=start_ts.floor("D"),
            end=(end_ts - pd.Timedelta(seconds=1)).floor("D"),
            freq="D",
        )
        counts_window = daily_counts[
            (daily_counts["date"] >= window_days.min())
            & (daily_counts["date"] <= window_days.max())
        ]
        counts_full = (
            pd.DataFrame({"date": window_days})
            .merge(counts_window, on="date", how="left")
            .fillna({"count": 0})
        )
        return (counts_full["count"] >= min_days_with_data_per_day).all()

    rng = default_rng(seed)
    max_start = max_ts - pd.Timedelta(days=total_days_required)

    chosen_start = None
    for _ in range(max_tries):
        u = rng.random()
        rand_start = (min_ts + (max_start - min_ts) * u).floor("D")
        rand_end = rand_start + pd.Timedelta(days=total_days_required)
        if rand_end > max_ts:
            continue
        if (not require_daily_coverage) or window_has_no_big_gaps(rand_start, rand_end):
            chosen_start = rand_start
            break

    if chosen_start is None:
        raise ValueError(
            f"No valid {total_weeks}-week window with daily coverage after {max_tries} tries."
        )

    chosen_end = chosen_start + pd.Timedelta(days=total_days_required)
    seg = df[(df["_ts"] >= chosen_start) & (df["_ts"] < chosen_end)].copy()
    seg = seg.drop(columns=["_ts", "date"]).reset_index(drop=True)

    return seg, chosen_start, chosen_end


# ==========================================================================
# PER-FILE WORKER
# ==========================================================================
def _resolve_path(data_dir, name):
    """Join data_dir + name, tolerating filenames with or without .csv."""
    p = os.path.join(data_dir, name)
    if os.path.exists(p):
        return p
    if not name.endswith(".csv"):
        p2 = os.path.join(data_dir, name + ".csv")
        if os.path.exists(p2):
            return p2
    return p  # return the primary candidate (will fail downstream with a clear error)


def _file_seed(global_seed, stem):
    """Stable per-file seed: combines a global seed with a CRC of the filename
    so runs are reproducible (unlike Python's salted hash())."""
    crc = zlib.crc32(str(stem).encode("utf-8")) & 0xFFFFFFFF
    return int((np.uint64(global_seed) * np.uint64(1000003) + np.uint64(crc)) % np.uint64(2**63))


def process_one_file(task):
    (name, data_dir, output_dir, weeks, train_weeks, global_seed,
     require_daily_coverage, timestamp_col, hodnota_col) = task
    stem = Path(name).stem
    src = _resolve_path(data_dir, name)
    out_name = f"{stem}.csv"

    try:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source not found: {src}")

        df = pd.read_csv(src)
        for col in (timestamp_col, hodnota_col):
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}'. Have: {df.columns.tolist()}")

        seed = _file_seed(global_seed, stem)
        seg, start_ts, end_ts = extract_random_window(
            df, seed=seed, total_weeks=weeks, timestamp_col=timestamp_col,
            require_daily_coverage=require_daily_coverage,
        )

        # full N-week segment
        os.makedirs(output_dir, exist_ok=True)
        seg.to_csv(os.path.join(output_dir, out_name), index=False)

        # optional train/stream split of the same window
        if train_weeks and train_weeks > 0:
            split_ts = start_ts + pd.Timedelta(weeks=train_weeks)
            ts = pd.to_datetime(seg[timestamp_col], utc=True, errors="coerce")
            train_df = seg[ts < split_ts]
            stream_df = seg[ts >= split_ts]
            train_dir = os.path.join(output_dir, "train")
            stream_dir = os.path.join(output_dir, "stream")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(stream_dir, exist_ok=True)
            train_df.to_csv(os.path.join(train_dir, out_name), index=False)
            stream_df.to_csv(os.path.join(stream_dir, out_name), index=False)
            split_info = {"n_train": int(len(train_df)), "n_stream": int(len(stream_df))}
        else:
            split_info = {}

        return {
            "filename": stem,
            "status": "success",
            "n_rows": int(len(seg)),
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            **split_info,
        }

    except Exception as e:
        return {"filename": stem, "status": "failed", "error": str(e)}


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Extract random N-week segments from meters listed in a pickle of filenames."
    )
    p.add_argument("--pickle", required=True, help="Pickle holding a list of filenames.")
    p.add_argument("--data-dir", required=True, help="Directory prepended to each filename.")
    p.add_argument("--output-dir", default="./test_segments", help="Where segments are written.")
    p.add_argument("--weeks", type=int, default=8, help="Segment length in weeks (default: 8).")
    p.add_argument("--train-weeks", type=int, default=0,
                   help="If >0, also write a train/ (first K weeks) + stream/ (rest) split.")
    p.add_argument("--workers", type=int, default=7, help="Parallel worker processes.")
    p.add_argument("--seed", type=int, default=42, help="Global seed (combined with each filename).")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N files (debug).")
    p.add_argument("--no-daily-coverage", action="store_true",
                   help="Disable the 'at least one measurement per day' window check.")
    p.add_argument("--timestamp-col", default="timestamp",
                   help="Name of the timestamp column in the source CSVs (default: timestamp).")
    p.add_argument("--hodnota-col", default="hodnota",
                   help="Name of the meter-value column (default: hodnota).")
    args = p.parse_args()

    if args.train_weeks and args.train_weeks >= args.weeks:
        p.error(f"--train-weeks ({args.train_weeks}) must be < --weeks ({args.weeks}).")

    with open(args.pickle, "rb") as f:
        names = pickle.load(f)
    names = [str(n) for n in names]
    if args.limit:
        names = names[: args.limit]
    logger.info(f"Loaded {len(names)} filenames from {args.pickle}")
    logger.info(f"Data dir: {args.data_dir}  ->  output: {args.output_dir}")
    logger.info(f"Window: {args.weeks} weeks"
                + (f", split train/{args.train_weeks}w + stream/{args.weeks - args.train_weeks}w"
                   if args.train_weeks else ""))

    require_daily_coverage = not args.no_daily_coverage
    tasks = [
        (name, args.data_dir, args.output_dir, args.weeks, args.train_weeks,
         args.seed, require_daily_coverage, args.timestamp_col, args.hodnota_col)
        for name in names
    ]

    results = []
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one_file, t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
                results.append(fut.result())
    else:
        for t in tqdm(tasks, desc="Extracting"):
            results.append(process_one_file(t))

    ok = [r for r in results if r["status"] == "success"]
    bad = [r for r in results if r["status"] != "success"]
    logger.info("=" * 60)
    logger.info(f"Done. Success: {len(ok)}  Failed: {len(bad)}")
    if ok:
        rows = np.array([r["n_rows"] for r in ok])
        logger.info(f"Rows per segment — min {rows.min()}, median {int(np.median(rows))}, max {rows.max()}")
    if bad:
        # summarize failure reasons
        from collections import Counter
        reasons = Counter(
            (r.get("error", "").split(":")[0] or "unknown") for r in bad
        )
        logger.info("Failure reasons (top): " + ", ".join(f"{k}×{v}" for k, v in reasons.most_common(5)))

    # write a manifest of what was produced
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "_manifest.csv"), index=False)
    logger.info(f"Manifest: {os.path.join(args.output_dir, '_manifest.csv')}")


if __name__ == "__main__":
    main()
