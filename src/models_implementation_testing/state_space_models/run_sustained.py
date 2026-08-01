"""
run_sustained.py — Batch runner for the sustained-anomaly detection surface.
============================================================================

Sibling of `run_variants.py`. That runner sweeps *model variants* against one
fixed anomaly type; this one sweeps *anomaly types, depths and durations*
against one fixed model, and reports a **detection surface** instead of a single
F1.

The distinction matters. For a sustained fault, "what is the F1?" is not a
well-posed question — detection depends almost entirely on severity, and a
single number averages over exactly the axis worth knowing:

    slow_leak, depth 1.00  -> caught in  7 readings
    slow_leak, depth 0.10  -> caught in 35 readings
    slow_leak, depth 0.02  -> ?

That last row is the one that decides whether this detector is useful in the
field, and no scalar summary can express it.

Outputs (both covered by this directory's .gitignore):

  sustained_matrix.csv    one row per (sensor x archetype x depth x duration):
                          event recall, time-to-detect, FP outside events,
                          recovery, plus the usual gate/accuracy columns.

  sustained_scores.parquet  long table (filename, variant, idx, z_score, label)
                          in the SAME schema `run_variants.py` writes, so
                          `analyze_variants.py --sweep` works on it unchanged.
                          `variant` encodes the injection cell, since that is
                          what varies here.

Pairing note: the window and the injected event are both seeded on
int(filename), so a given sensor gets the same window and the same event
placement in every cell — comparisons across depth are paired per sensor.

Usage
-----
    # smoke run on the three bundled fixtures (about a minute)
    python run_sustained.py --data-dir /home/user/uc-cem/testing_scripts/fixtures/real_segments \\
        --out-dir ./results/sustained_smoke

    # full test set
    python run_sustained.py \\
        --data-dir ../../../data/sensor_data \\
        --pickle ../../pickles/test_set.pkl \\
        --workers 7 --out-dir ./results/sustained_baseline

    # one archetype, custom depth ladder
    python run_sustained.py --data-dir ... --archetypes slow_leak \\
        --depths 0.02 0.05 0.1 0.25 0.5 1.0 --durations 24 48
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import uc_variants as uv
from run_variants import collect_files, _DROP_FROM_SUMMARY, tqdm
from helper_scripts import anomaly_injection as ai
from helper_scripts import event_metrics as em

logger = logging.getLogger("run_sustained")

# Event-level detail is per-event, not per-sensor; it goes nowhere near the
# flat summary row.
_DROP_EXTRA = ("events", "event_rows", "_predictions_df")

# Depth ladder reaching well below the point where detection is expected to
# fail. A ladder that only spans "obviously detectable" magnitudes produces a
# flat surface and tells you nothing.
DEFAULT_DEPTHS = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 3.00)
DEFAULT_DURATIONS_H = (6.0, 24.0, 48.0)

# Archetypes whose severity is set by duration alone — sweeping depth over them
# would multiply runtime by len(depths) while injecting the identical fault.
_SHAPE_ONLY = ("frozen_meter", "zero_consumption", "missing_data",
               "holiday_profile", "null_control")


def build_cells(archetypes, depths, durations):
    """Expand the requested archetypes into (archetype, depth, duration) cells.

    Depth is swept only where it means something (`ai.DEPTH_MEANINGFUL`);
    shape-only archetypes get a single NaN-depth cell per duration.
    `point_spike` and `unit_change` are instantaneous / open-ended, so they take
    one duration each.
    """
    cells = []
    for a in archetypes:
        if a not in ai.ARCHETYPES:
            raise SystemExit(f"unknown archetype {a!r}; known: {sorted(ai.ARCHETYPES)}")
        if a in _SHAPE_ONLY:
            for d in durations:
                cells.append((a, float("nan"), float(d)))
        elif a in ("point_spike", "unit_change"):
            for dep in depths:
                cells.append((a, float(dep), float("nan")))
        else:
            for dep in depths:
                for d in durations:
                    cells.append((a, float(dep), float(d)))
    return cells


def cell_label(archetype, depth, duration_h):
    """Stable identifier used as the `variant` column, so the existing analysis
    layer can group on it without modification."""
    parts = [archetype]
    if np.isfinite(depth):
        parts.append(f"d{depth:g}")
    if np.isfinite(duration_h):
        parts.append(f"h{duration_h:g}")
    return "_".join(parts)


def _job(args):
    """Top-level so ProcessPoolExecutor can pickle it."""
    filepath, archetype, depth, duration_h, variant, threshold_z_score = args
    plan = ai.InjectionPlan(archetype=archetype, depth=depth, duration_h=duration_h)
    return uv.run_variant(filepath, variant, injection_plan=plan,
                          threshold_z_score=threshold_z_score)


def run(files, cells, variant, workers, per_file_timeout, out_dir,
        threshold_z_score=3.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(f, a, dep, dur, variant, threshold_z_score)
            for (a, dep, dur) in cells for f in files]
    logger.info("Running %d jobs (%d files x %d cells) on %d workers",
                len(jobs), len(files), len(cells), workers)

    summary_rows, score_frames, event_rows = [], [], []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_job, j): j for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="runs"):
            filepath, archetype, depth, duration_h, _v, _z = futures[fut]
            fname = Path(filepath).stem
            label = cell_label(archetype, depth, duration_h)
            try:
                res = fut.result(timeout=per_file_timeout)
            except TimeoutError:
                # A cell that times out more often is a worse cell, so this is
                # recorded rather than dropped.
                fut.cancel()
                logger.error("timeout %s [%s] > %ss", fname, label, per_file_timeout)
                summary_rows.append({"filename": fname, "variant": label,
                                     "archetype": archetype, "depth": depth,
                                     "duration_h": duration_h, "status": "failed",
                                     "error": f"timeout>{per_file_timeout}s"})
                continue
            except Exception as e:
                logger.error("crash %s [%s]: %s", fname, label, e)
                summary_rows.append({"filename": fname, "variant": label,
                                     "archetype": archetype, "depth": depth,
                                     "duration_h": duration_h, "status": "failed",
                                     "error": str(e)})
                continue

            z, lab = res.get("z_scores"), res.get("labels")
            if res.get("status") == "success" and z is not None and lab is not None:
                score_frames.append(pd.DataFrame({
                    "filename": fname,
                    "variant": label,
                    "idx": np.arange(len(z), dtype=np.int32),
                    "z_score": np.asarray(z, dtype=np.float32),
                    "label": np.asarray(lab, dtype=np.int8),
                }))

            for er in (res.get("event_rows") or []):
                event_rows.append({"filename": fname, "variant": label, **er})

            row = {k: v for k, v in res.items()
                   if k not in _DROP_FROM_SUMMARY and k not in _DROP_EXTRA}
            row.update({"variant": label, "archetype": archetype,
                        "depth": depth, "duration_h": duration_h})
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "sustained_matrix.csv"
    summary.to_csv(summary_path, index=False)

    events_df = pd.DataFrame(event_rows)
    if len(events_df):
        events_df.to_csv(out_dir / "sustained_events.csv", index=False)

    scores_path = out_dir / "sustained_scores.parquet"
    if score_frames:
        scores = pd.concat(score_frames, ignore_index=True)
        try:
            scores.to_parquet(scores_path, index=False)
        except Exception as e:
            scores_path = out_dir / "sustained_scores.csv.gz"
            logger.warning("parquet unavailable (%s); writing %s", e, scores_path.name)
            scores.to_csv(scores_path, index=False, compression="gzip")

    surface = _surface(summary, events_df)
    if len(surface):
        surface.to_csv(out_dir / "sustained_surface.csv", index=False)

    _report(summary, surface, time.time() - t0, out_dir)
    return summary, surface


def _apply_null_correction(surface):
    """Subtract the coincidental-detection floor measured by `null_control`.

    A detection rate is only evidence of detection to the extent it exceeds what
    the same detector scores on an identically-shaped window containing nothing.
    `excess_detection_rate` is that difference, matched on duration; where no
    control was run for a duration it stays NaN rather than silently equalling
    the raw rate.
    """
    if not len(surface) or "archetype" not in surface.columns:
        return surface
    ctrl = surface[surface["archetype"] == "null_control"]
    if not len(ctrl):
        surface["null_rate"] = np.nan
        surface["excess_detection_rate"] = np.nan
        return surface

    floor = dict(zip(ctrl["duration_h"], ctrl["detection_rate"]))
    surface = surface.copy()
    surface["null_rate"] = surface["duration_h"].map(floor)
    surface["excess_detection_rate"] = (
        surface["detection_rate"] - surface["null_rate"]
    ).clip(lower=0.0)
    surface.loc[surface["archetype"] == "null_control", "excess_detection_rate"] = np.nan
    return surface


def _surface(summary, events_df):
    """Collapse to the archetype x depth x duration surface.

    Built from the per-event frame when available (an event is the unit of
    detection); falls back to per-sensor aggregates otherwise.
    """
    if len(events_df):
        return _apply_null_correction(em.detection_surface(events_df))

    ok = summary[summary.get("status") == "success"] if len(summary) else summary
    if not len(ok) or "event_recall" not in ok.columns:
        return pd.DataFrame()
    g = ok.groupby(["archetype", "depth", "duration_h"], dropna=False)
    out = g.agg(n_sensors=("event_recall", "size"),
                detection_rate=("event_recall", "mean"),
                ttd_readings_median=("ttd_readings_median", "median")).reset_index()
    return _apply_null_correction(out)


def _report(summary, surface, elapsed, out_dir):
    logger.info("=" * 68)
    logger.info("Done in %.1fs -> %s", elapsed, out_dir / "sustained_matrix.csv")
    if len(summary) and "status" in summary.columns:
        by = summary.groupby("variant")["status"].value_counts().unstack(fill_value=0)
        logger.info("outcomes per cell:\n%s", by.to_string())

    if len(surface):
        cols = [c for c in ("archetype", "depth", "duration_h", "n_events",
                            "detection_rate", "null_rate", "excess_detection_rate",
                            "ttd_readings_median")
                if c in surface.columns]
        logger.info("DETECTION SURFACE:\n%s", surface[cols].to_string(index=False))

        # The headline: archetypes nothing ever caught. These are the ones the
        # detector is structurally blind to, and the reason this harness exists.
        # Blindness is judged on EXCESS over the null floor: a cell scoring the
        # same as an empty window has detected nothing, whatever its raw rate.
        rate_col = ("excess_detection_rate" if "excess_detection_rate" in surface.columns
                    else "detection_rate")
        if rate_col in surface.columns:
            blind = surface[surface[rate_col].fillna(1.0) <= 0.0]
            if len(blind):
                logger.info("NEVER DETECTED (%d cells):\n%s", len(blind),
                            blind[cols].to_string(index=False))
    logger.info("=" * 68)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pickle", default=None,
                    help="Pickle of filenames (e.g. ../../pickles/test_set.pkl).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-file-timeout", type=int, default=300)
    ap.add_argument("--out-dir", default="./results/sustained")
    ap.add_argument("--variant", default="baseline",
                    help="Model variant from uc_variants.VARIANTS (default: baseline).")
    ap.add_argument("--threshold-z-score", type=float, default=3.5)
    ap.add_argument("--archetypes", nargs="+", default=sorted(ai.ARCHETYPES))
    ap.add_argument("--depths", nargs="+", type=float, default=list(DEFAULT_DEPTHS))
    ap.add_argument("--durations", nargs="+", type=float,
                    default=list(DEFAULT_DURATIONS_H))
    args = ap.parse_args()

    files = collect_files(args.data_dir, args.pickle, args.limit)
    cells = build_cells(args.archetypes, args.depths, args.durations)
    logger.info("%d files x %d cells", len(files), len(cells))

    run(files, cells, args.variant, args.workers, args.per_file_timeout,
        args.out_dir, threshold_z_score=args.threshold_z_score)
    return 0


if __name__ == "__main__":
    sys.exit(main())
