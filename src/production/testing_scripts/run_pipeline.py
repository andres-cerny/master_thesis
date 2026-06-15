"""
run_pipeline.py — End-to-end rolling-refit harness for the streaming UC detector.

For each meter's N-week segment (e.g. the 8-week CSVs from make_test_segments.py),
it walks the production refit cadence:

    train weeks [1..4]  -> predict week 5
    train weeks [2..5]  -> predict week 6     (warm-started from the previous fit)
    train weeks [3..6]  -> predict week 7
    train weeks [4..7]  -> predict week 8

Each "predict week" replays the raw readings one at a time through the detector
exactly as a live server would (the detector computes Diff, handles gaps,
anomalies, outages, resets). Every output field is saved — not just the point
prediction, but the next-step band, z-score, status, and which fit window the
row belongs to.

Because train/predict widths and the slide step are configurable, the same
harness covers "train once, predict 4 weeks" (slide = predict-weeks, no overlap)
or a tighter weekly refit (the default).

Window handoff: with the default (train 4, predict 1, slide 1) each fit window
ends exactly where the previous prediction week ended, so the carry-forward
state continues seamlessly and there is no duplicate/skip at the boundary.

The training step is injected (train_callable) so this module can be tested
without statsmodels; main() imports the real train.train lazily.

Usage:
    python run_pipeline.py \
        --segment-dir ./test_segments \
        --state-dir ./pipeline_state \
        --output-dir ./pipeline_results \
        --train-weeks 4 --predict-weeks 1 --slide-weeks 1 \
        --holdout-days 0 --z-threshold 3.5 [--limit 50]
        
    python run_pipeline.py --segment-dir ./test_segments --state-dir ./pipeline_state --output-dir ./pipeline_results --train-weeks 4 --predict-weeks 1 --slide-weeks 1 --holdout-days 0 --z-threshold 3.5 --timestamp-col timestamp --hodnota-col hodnota --timezone Europe/Prague --no-quality-gate --limit 100

Output: ./pipeline_results/<meter>_predictions.csv per meter, plus _summary.csv.
"""

import os
import sys
import shutil
import tempfile
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

parent_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(parent_path)

import shared

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_FIELDS = [
    "status", "flag", "diff", "z_current", "pred_current",
    "pred_next", "lower_band_next", "upper_band_next",
]


def _read_segment(csv_path, timestamp_col="timestamp", hodnota_col="hodnota",
                  timezone="Europe/Prague"):
    """Read a segment, convert local timestamps to UTC (DST-aware) into a
    canonical '_ts_utc' column, and standardize the value column to 'hodnota'.
    Original columns are preserved for the temp training CSVs."""
    df = pd.read_csv(csv_path)
    for col in (timestamp_col, hodnota_col):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}. Have: {df.columns.tolist()}")
    df["_ts_utc"] = shared.localize_to_utc(df[timestamp_col], timezone=timezone)
    df = df.dropna(subset=["_ts_utc"]).sort_values("_ts_utc").reset_index(drop=True)
    return df


def run_meter(meter_id, df, state_dir, train_callable, *,
              train_weeks=4, predict_weeks=1, slide_weeks=1,
              holdout_days=0.0, z_threshold=3.5,
              reload_each_step=False, train_kwargs=None,
              timestamp_col="timestamp", hodnota_col="hodnota",
              timezone="Europe/Prague"):
    """
    Run the rolling train->predict cadence for one meter. Returns
    (records_df, summary_dict). Mutates per-meter state under state_dir/meter_id.

    `df` is expected to come from _read_segment (has the canonical '_ts_utc'
    column). Window boundaries and the prediction replay use '_ts_utc' (UTC);
    the temp training CSVs are written with the original columns and re-converted
    by train_callable, so the whole chain uses the identical conversion.
    """
    train_kwargs = dict(train_kwargs or {})
    meter_dir = os.path.join(state_dir, str(meter_id))
    # fresh state for a clean run
    if os.path.isdir(meter_dir):
        shutil.rmtree(meter_dir)

    df = df.copy()
    ts = df["_ts_utc"]
    t0 = ts.min().floor("D")          # clean week boundaries
    data_end = ts.max()

    train_td = pd.Timedelta(weeks=train_weeks)
    pred_td = pd.Timedelta(weeks=predict_weeks)
    slide_td = pd.Timedelta(weeks=slide_weeks)

    records = []
    n_iters = 0
    it = 0
    rejected_at = None
    while True:
        train_start = t0 + it * slide_td
        train_end = train_start + train_td
        pred_start = train_end
        pred_end = pred_start + pred_td
        # Stop only when the prediction window would start past the data. The
        # final week may be a hair short (segments are half-open [start, start+Nw)),
        # but a week with one fewer reading is still worth predicting.
        if pred_start >= data_end:
            break

        train_slice = df[(df["_ts_utc"] >= train_start) & (df["_ts_utc"] < train_end)]
        pred_slice = df[(df["_ts_utc"] >= pred_start) & (df["_ts_utc"] < pred_end)]
        if len(train_slice) < 2 or len(pred_slice) < 1:
            it += 1
            continue

        # --- TRAIN on the window (writes model + carry-forward state) ------
        # Write the original columns (local timestamps); train re-converts via
        # the same shared.localize_to_utc, so the conversion is identical.
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        tmp.close()
        try:
            train_slice.drop(columns=["_ts_utc"]).to_csv(tmp.name, index=False)
            train_result = train_callable(
                meter_id=str(meter_id),
                csv_path=tmp.name,
                state_dir=state_dir,
                holdout_days=holdout_days,
                z_threshold=z_threshold,
                timestamp_col=timestamp_col,
                hodnota_col=hodnota_col,
                timezone=timezone,
                verbose=False,
                **train_kwargs,
            )
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

        # If the quality gate rejected this sensor, no model was written. Stop
        # this meter's rolling sequence and record why.
        accepted = train_result.get("accepted", True) if isinstance(train_result, dict) else True
        if not accepted or not os.path.exists(os.path.join(meter_dir, shared.MODEL_FILE)):
            rejected_at = {
                "iteration": it,
                "stage": (train_result or {}).get("stage") if isinstance(train_result, dict) else None,
                "reasons": (train_result or {}).get("reasons") if isinstance(train_result, dict) else None,
                "metrics": (train_result or {}).get("metrics") if isinstance(train_result, dict) else None,
            }
            break

        # --- PREDICT the following week, one raw reading at a time --------
        # '_ts_utc' is already UTC, so step() uses it directly (no re-convert).
        if reload_each_step:
            # Faithful version-B flow: load -> step -> save, per reading.
            for _, row in pred_slice.iterrows():
                det = shared.UCStreamingDetector.load(meter_dir, z_threshold_override=z_threshold)
                rec = det.step(row["_ts_utc"], row[hodnota_col])
                if rec["status"] not in ("skipped_early", "skipped_duplicate", "invalid"):
                    det.save_state(meter_dir)
                _annotate(rec, meter_id, it, train_start, train_end, pred_start, pred_end, row)
                records.append(rec)
        else:
            # Load once for the week, step in memory, persist at the end.
            det = shared.UCStreamingDetector.load(meter_dir, z_threshold_override=z_threshold)
            for _, row in pred_slice.iterrows():
                rec = det.step(row["_ts_utc"], row[hodnota_col])
                _annotate(rec, meter_id, it, train_start, train_end, pred_start, pred_end, row)
                records.append(rec)
            det.save_state(meter_dir)  # so the next retrain warm-starts from here

        n_iters += 1
        it += 1

    records_df = pd.DataFrame(records)
    summary = {
        "meter_id": str(meter_id),
        "iterations": n_iters,
        "n_predictions": int(len(records_df)),
        "n_flagged": int(records_df["flag"].sum()) if len(records_df) else 0,
        "rejected": rejected_at is not None,
        "rejected_at_iteration": (rejected_at or {}).get("iteration") if rejected_at else None,
        "rejected_stage": (rejected_at or {}).get("stage") if rejected_at else None,
        "rejected_reasons": "; ".join((rejected_at or {}).get("reasons") or []) if rejected_at else None,
    }
    if len(records_df):
        sc = records_df["status"].value_counts().to_dict()
        for k, v in sc.items():
            summary[f"status_{k}"] = int(v)
    return records_df, summary


def _annotate(rec, meter_id, it, train_start, train_end, pred_start, pred_end, row):
    rec["meter_id"] = str(meter_id)
    rec["iteration"] = it
    rec["train_start"] = train_start.isoformat()
    rec["train_end"] = train_end.isoformat()
    rec["pred_window_start"] = pred_start.isoformat()
    rec["pred_window_end"] = pred_end.isoformat()
    # carry through any reference columns from the raw row if present
    for extra in ("Diff", "is_anomaly"):
        if extra in row.index:
            rec[f"raw_{extra}"] = row[extra]


def main():
    p = argparse.ArgumentParser(description="Rolling train/predict harness for the streaming UC detector.")
    p.add_argument("--segment-dir", required=True, help="Dir of N-week segment CSVs (from make_test_segments.py).")
    p.add_argument("--state-dir", default="./pipeline_state", help="Scratch dir for per-meter model/state.")
    p.add_argument("--output-dir", default="./pipeline_results", help="Where prediction CSVs are written.")
    p.add_argument("--train-weeks", type=int, default=4)
    p.add_argument("--predict-weeks", type=int, default=1)
    p.add_argument("--slide-weeks", type=int, default=1)
    p.add_argument("--holdout-days", type=float, default=0.0)
    p.add_argument("--z-threshold", type=float, default=3.5)
    p.add_argument("--harmonics-daily", type=int, default=2)
    p.add_argument("--harmonics-weekly", type=int, default=2)
    p.add_argument("--k-hours", type=float, default=2.0)
    p.add_argument("--reload-each-step", action="store_true",
                   help="Reload+save state per reading (faithful version-B; slower).")
    p.add_argument("--timestamp-col", default="timestamp", help="Timestamp column in the segment CSVs.")
    p.add_argument("--hodnota-col", default="hodnota", help="Meter-value column in the segment CSVs.")
    p.add_argument("--timezone", default="Europe/Prague", help="Local timezone of the raw timestamps.")
    p.add_argument("--no-quality-gate", action="store_true",
                   help="Compute quality metrics but never reject a sensor.")
    p.add_argument("--mase-metric", default="mase_seasonal", choices=["mase_seasonal", "mase"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    # Real trainer imported lazily (pulls in statsmodels).
    from train import train as train_callable
    train_kwargs = dict(
        harmonics_daily=args.harmonics_daily,
        harmonics_weekly=args.harmonics_weekly,
        k_hours=args.k_hours,
        apply_quality_gate=not args.no_quality_gate,
        mase_metric=args.mase_metric,
    )

    seg_files = sorted(
        f for f in os.listdir(args.segment_dir)
        if f.endswith(".csv") and not f.startswith("_")
    )
    if args.limit:
        seg_files = seg_files[: args.limit]
    logger.info(f"Found {len(seg_files)} segment files in {args.segment_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    summaries = []
    for fname in seg_files:
        meter_id = Path(fname).stem
        try:
            df = _read_segment(os.path.join(args.segment_dir, fname),
                               timestamp_col=args.timestamp_col,
                               hodnota_col=args.hodnota_col,
                               timezone=args.timezone)
            records_df, summary = run_meter(
                meter_id, df, args.state_dir, train_callable,
                train_weeks=args.train_weeks, predict_weeks=args.predict_weeks,
                slide_weeks=args.slide_weeks, holdout_days=args.holdout_days,
                z_threshold=args.z_threshold, reload_each_step=args.reload_each_step,
                train_kwargs=train_kwargs,
                timestamp_col=args.timestamp_col, hodnota_col=args.hodnota_col,
                timezone=args.timezone,
            )
            out_path = os.path.join(args.output_dir, f"{meter_id}_predictions.csv")
            records_df.to_csv(out_path, index=False)
            summary["status"] = "success"
            logger.info(f"{meter_id}: {summary['iterations']} iters, "
                        f"{summary['n_predictions']} preds, {summary['n_flagged']} flagged")
        except Exception as e:
            summary = {"meter_id": meter_id, "status": "failed", "error": str(e)}
            logger.error(f"{meter_id}: FAILED — {e}")
        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(os.path.join(args.output_dir, "_summary.csv"), index=False)
    ok = sum(1 for s in summaries if s.get("status") == "success")
    logger.info("=" * 60)
    logger.info(f"Done. Success: {ok}/{len(summaries)}  -> {args.output_dir}")


if __name__ == "__main__":
    main()
    
    #TODO: python run_pipeline.py --segment-dir ./test_segments --state-dir ./pipeline_state --output-dir ./pipeline_results --train-weeks 4 --predict-weeks 1 --slide-weeks 1 --holdout-days 0 --timestamp-col timestamp --hodnota-col hodnota --timezone Europe/Prague