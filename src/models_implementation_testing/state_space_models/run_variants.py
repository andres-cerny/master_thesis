"""
run_variants.py — Batch runner for UC model variants.
=====================================================

Runs one or more variants from `uc_variants.VARIANTS` over a set of meter CSVs
and writes two artifacts that together support "run once, filter many times":

  summary.csv     one row per (sensor x variant): status, convergence, gate
                  diagnostics, gate booleans, accuracy metrics, and the raw
                  tp/fp/fn/tn counts. Micro-averaged P/R/F1 are NOT stored --
                  they are sums, recomputed at analysis time under whatever
                  cohort filter you choose.

  scores.parquet  long table (filename, variant, idx, z_score, label). This is
                  what makes the DETECTION-threshold sweep free: precision and
                  recall at any z can be recomputed without re-running a thing.

Both outputs are covered by this directory's .gitignore (*.csv / *.parquet).

Pairing note: window selection and anomaly injection are both seeded on
int(filename), so every variant sees identical windows and identical injected
spikes. That makes variant comparison PAIRED per sensor -- do not change the
split or resample logic or this property is lost.

Usage
-----
    # all six variants over the 3 bundled real fixtures
    python run_variants.py --data-dir /path/to/fixtures --out-dir ./results/v1

    # the thesis test set, 4 workers
    python run_variants.py \
        --data-dir ../../../data/sensor_data \
        --pickle ../../pickles/test_set.pkl \
        --limit 500 --workers 4 --out-dir ./results/testset

    # one variant only
    python run_variants.py --data-dir ... --variants baseline log1p
"""

import os
import sys
import time
import pickle
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import uc_variants as uv

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    def tqdm(it, **kwargs):
        return it

logger = logging.getLogger("run_variants")

# Per-reading arrays live in scores.parquet, not in the summary row. The
# per-position reference arrays are large and diagnostic-only, so they are
# dropped entirely rather than bloating a 5.5k-row CSV with list columns.
_DROP_FROM_SUMMARY = (
    "z_scores", "labels",
    "ref_mean_per_position", "ref_std_per_position",
    "ref_median_per_position", "ref_mad_per_position",
    "anomaly_indices", "anomaly_indices_robust",
)


def _job(args):
    """Top-level so ProcessPoolExecutor can pickle it."""
    filepath, variant = args
    return uv.run_variant(filepath, variant)


def collect_files(data_dir, pickle_path=None, limit=None):
    """Resolve the meter CSV list, from a pickle of filenames or a directory."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {data_dir}")

    if pickle_path:
        with open(pickle_path, "rb") as f:
            names = pickle.load(f)
        paths = [data_dir / n for n in names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            logger.warning(
                "%d of %d files from the pickle are missing under %s (skipped). "
                "The full dataset lives on Zenodo, not in this repo.",
                len(missing), len(paths), data_dir,
            )
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(data_dir.glob("*.csv"))

    if not paths:
        raise SystemExit(f"No usable CSV files found under {data_dir}")
    if limit:
        paths = paths[:limit]
    return [str(p) for p in paths]


def run(files, variants, workers, per_file_timeout, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(f, v) for v in variants for f in files]
    logger.info(
        "Running %d jobs (%d files x %d variants) on %d workers",
        len(jobs), len(files), len(variants), workers,
    )

    summary_rows, score_frames = [], []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_job, j): j for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="meters"):
            filepath, variant = futures[fut]
            fname = Path(filepath).stem
            try:
                res = fut.result(timeout=per_file_timeout)
            except TimeoutError:
                # Timeouts are DATA, not noise: a variant that times out more
                # often is worse even if its survivors score better.
                fut.cancel()
                logger.error("timeout %s [%s] > %ss", fname, variant, per_file_timeout)
                summary_rows.append({
                    "filename": fname, "variant": variant,
                    "status": "failed", "error": f"timeout>{per_file_timeout}s",
                })
                continue
            except Exception as e:
                logger.error("crash %s [%s]: %s", fname, variant, e)
                summary_rows.append({
                    "filename": fname, "variant": variant,
                    "status": "failed", "error": str(e),
                })
                continue

            z = res.get("z_scores")
            lab = res.get("labels")
            if res.get("status") == "success" and z is not None and lab is not None:
                score_frames.append(pd.DataFrame({
                    "filename": fname,
                    "variant": variant,
                    "idx": np.arange(len(z), dtype=np.int32),
                    "z_score": np.asarray(z, dtype=np.float32),
                    "label": np.asarray(lab, dtype=np.int8),
                }))

            summary_rows.append({
                k: v for k, v in res.items() if k not in _DROP_FROM_SUMMARY
            })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    scores_path = out_dir / "scores.parquet"
    if score_frames:
        scores = pd.concat(score_frames, ignore_index=True)
        try:
            scores.to_parquet(scores_path, index=False)
        except Exception as e:
            # pyarrow/fastparquet may be absent; CSV keeps the sweep possible.
            scores_path = out_dir / "scores.csv.gz"
            logger.warning("parquet unavailable (%s); writing %s", e, scores_path.name)
            scores.to_csv(scores_path, index=False, compression="gzip")
    else:
        scores = pd.DataFrame()

    _report(summary, files, variants, time.time() - t0, summary_path, scores_path)
    return summary, scores


def _report(summary, files, variants, elapsed, summary_path, scores_path):
    logger.info("=" * 68)
    logger.info("Done in %.1fs -> %s", elapsed, summary_path)
    if len(summary):
        logger.info("%s", scores_path)
        by = summary.groupby("variant")["status"].value_counts().unstack(fill_value=0)
        logger.info("outcomes per variant:\n%s", by.to_string())

        ok = summary[summary["status"] == "success"]
        if len(ok) and "converged_second" in ok.columns:
            conv = ok.groupby("variant")["converged_second"].mean()
            logger.info("second-stage convergence rate:\n%s", conv.to_string())
    logger.info("Next: python analyze_variants.py --results %s", summary_path.parent)
    logger.info("=" * 68)


def main():
    p = argparse.ArgumentParser(
        description="Run UC model variants over a set of meter CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True,
                   help="Directory holding the per-sensor CSVs.")
    p.add_argument("--pickle", default=None,
                   help="Optional pickle of filenames (e.g. test_set.pkl). "
                        "Omit to use every CSV in --data-dir.")
    p.add_argument("--out-dir", default="./results/variants",
                   help="Where summary.csv and scores.parquet are written.")
    p.add_argument("--variants", nargs="+", default=list(uv.VARIANTS),
                   choices=list(uv.VARIANTS),
                   help="Which variants to run.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of sensors (useful for smoke tests).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-(sensor,variant) timeout in seconds.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    files = collect_files(args.data_dir, args.pickle, args.limit)
    run(files, args.variants, args.workers, args.timeout, args.out_dir)


if __name__ == "__main__":
    main()
