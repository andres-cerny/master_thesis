"""
event_metrics.py — Event-level scoring for sustained anomalies.

Point-wise P/R/F1 is the wrong instrument for a sustained fault. A 48-hour leak
covering 192 readings that the detector catches on reading 7 is, operationally,
a *success* — the leak was found within two hours. Point-wise scoring records it
as 7 true positives and 185 false negatives, i.e. recall 0.036, and a detector
that fires once and then stays quiet scores worse than one that alarms
continuously for two days. Neither ranking is what an operator wants.

So this module scores three things the existing metrics cannot:

  * **Event recall** — was the event caught at all, once, anywhere inside it.
  * **Time-to-detect** — how long it took, in readings and in hours. This is the
    number that actually distinguishes detectors on sustained faults.
  * **Recovery** — did scoring return to normal after the event ended, or did one
    fault leave the filter permanently disturbed.

Point-wise metrics remain available and unchanged: `label_rows_from_events`
produces the same `is_anomaly_actual` column `calculate_metrics()` already
expects, so both views come from one run.

Reuses `analyze_variants._prf` for precision/recall/F1 rather than restating the
arithmetic.
"""

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SSM = os.path.normpath(os.path.join(_HERE, "..", "state_space_models"))
if _SSM not in sys.path:
    sys.path.insert(0, _SSM)


def _prf(tp, fp, fn):
    """Precision / recall / F1. Imported from the analysis layer when available
    so the two cannot drift; the fallback keeps this module importable on its
    own (the notebook imports it without the runner)."""
    try:
        from analyze_variants import _prf as _impl
        return _impl(tp, fp, fn)
    except Exception:
        p = tp / (tp + fp) if (tp + fp) else np.nan
        r = tp / (tp + fn) if (tp + fn) else np.nan
        f = 2 * p * r / (p + r) if (p and r and np.isfinite(p) and np.isfinite(r)) else np.nan
        return p, r, f


# ============================================================================
# LABELLING
# ============================================================================


def label_rows_from_events(timestamps, events):
    """Per-row 0/1 labels and event ids from a list of `AnomalyEvent` intervals.

    Returns `(labels, event_ids)` where `event_ids[i]` is the index into `events`
    of the event covering row i, or -1. Intervals are inclusive at both ends;
    `point_spike` has `t_start == t_end` and therefore labels exactly one row.

    Overlap is not expected — plans inject one archetype at a time — but if it
    occurs the earlier event in the list wins, deterministically.
    """
    ts = pd.to_datetime(np.asarray(timestamps), utc=True)
    labels = np.zeros(len(ts), dtype=int)
    event_ids = np.full(len(ts), -1, dtype=int)

    for k, ev in enumerate(events):
        t0 = pd.Timestamp(ev.t_start)
        t1 = pd.Timestamp(ev.t_end)
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")
        if t1.tzinfo is None:
            t1 = t1.tz_localize("UTC")
        inside = (ts >= t0) & (ts <= t1)
        fresh = inside & (event_ids < 0)
        labels[fresh] = 1
        event_ids[fresh] = k

    return labels, event_ids


# ============================================================================
# EVENT-LEVEL SCORING
# ============================================================================


def event_level_metrics(timestamps, flags, events, *, periodicity_seconds=None):
    """One row per injected event: caught or missed, and how fast.

    `flags` is the detector's per-row boolean decision — whichever column the
    caller considers an alert (`is_anomaly_predicted`, or a sustained/drift flag).

    Time-to-detect is measured from the first row inside the event, in readings
    and (if `periodicity_seconds` is given) in hours. A missed event reports NaN
    for both, never a sentinel like -1, so aggregations do not silently average
    a magic number into the result.
    """
    ts = pd.to_datetime(np.asarray(timestamps), utc=True)
    flags = np.asarray(flags).astype(bool)
    _, event_ids = label_rows_from_events(ts, events)

    rows = []
    for k, ev in enumerate(events):
        idx = np.flatnonzero(event_ids == k)
        row = {
            "archetype": ev.archetype,
            "depth": ev.depth,
            "duration_h": ev.duration_h,
            "scale": ev.scale,
            "t_start": pd.Timestamp(ev.t_start).isoformat(),
            "t_end": pd.Timestamp(ev.t_end).isoformat(),
            "n_readings": int(idx.size),
            "detected": False,
            "ttd_readings": np.nan,
            "ttd_hours": np.nan,
            "n_flagged_in_event": 0,
        }
        if idx.size:
            hit = np.flatnonzero(flags[idx])
            row["n_flagged_in_event"] = int(hit.size)
            if hit.size:
                row["detected"] = True
                row["ttd_readings"] = int(hit[0])
                if periodicity_seconds:
                    row["ttd_hours"] = float(hit[0]) * float(periodicity_seconds) / 3600.0
        rows.append(row)

    return pd.DataFrame(rows)


def recovery_check(timestamps, flags, events, *, n_after=12):
    """Did the detector settle after each event, or is it still alarming?

    Looks at the `n_after` readings following an event and reports how many are
    still flagged. This is the check that catches a detector which, having been
    disturbed once, never returns to baseline — a real failure mode here, since a
    flagged reading skips the Kalman update, so a stuck alarm also means a stuck
    state. Nothing in the existing suite tests this.

    `clean` is True when no post-event reading is flagged.
    """
    ts = pd.to_datetime(np.asarray(timestamps), utc=True)
    flags = np.asarray(flags).astype(bool)
    _, event_ids = label_rows_from_events(ts, events)

    rows = []
    for k, ev in enumerate(events):
        idx = np.flatnonzero(event_ids == k)
        if not idx.size:
            rows.append({"archetype": ev.archetype, "n_after": 0,
                         "n_flagged_after": 0, "clean": True})
            continue
        lo = idx[-1] + 1
        tail = flags[lo:lo + int(n_after)]
        rows.append({
            "archetype": ev.archetype,
            "n_after": int(tail.size),
            "n_flagged_after": int(tail.sum()),
            "clean": bool(tail.sum() == 0),
        })
    return pd.DataFrame(rows)


def false_alarm_count(timestamps, flags, events):
    """Alerts on rows belonging to no event.

    Denominator for the FP rate. Rows inside the `holiday_profile` probe count
    here by design: the probe is unlabelled precisely so that alarms it provokes
    land in this bucket rather than being scored as detections.
    """
    ts = pd.to_datetime(np.asarray(timestamps), utc=True)
    flags = np.asarray(flags).astype(bool)
    labels, _ = label_rows_from_events(ts, events)
    outside = labels == 0
    return int(np.sum(flags[outside])), int(np.sum(outside))


# ============================================================================
# AGGREGATION -> DETECTION SURFACE
# ============================================================================


def detection_surface(events_df, by=("archetype", "depth", "duration_h")):
    """Collapse per-event rows into the depth x duration surface.

    This is the deliverable shape: for each archetype and each (depth, duration)
    cell, what fraction of injected events were caught and how fast. Reported as
    a surface rather than a single F1 because a sustained-fault detector's
    behaviour is a *function of severity* — one number would hide exactly the
    part that matters, which is where along the depth axis detection begins.
    """
    if events_df.empty:
        return events_df
    by = [c for c in by if c in events_df.columns]
    g = events_df.groupby(by, dropna=False)
    out = g.agg(
        n_events=("detected", "size"),
        n_detected=("detected", "sum"),
        ttd_readings_median=("ttd_readings", "median"),
        ttd_hours_median=("ttd_hours", "median"),
    ).reset_index()
    out["detection_rate"] = out["n_detected"] / out["n_events"]
    return out


def summarize_run(timestamps, flags, events, *, periodicity_seconds=None,
                  n_after=12):
    """Everything for one sensor-run, as flat keys ready to merge into the
    runner's summary row."""
    ev_df = event_level_metrics(timestamps, flags, events,
                                periodicity_seconds=periodicity_seconds)
    rec_df = recovery_check(timestamps, flags, events, n_after=n_after)
    fp, n_clean = false_alarm_count(timestamps, flags, events)

    n_events = len(ev_df)
    n_det = int(ev_df["detected"].sum()) if n_events else 0
    return {
        "n_events": n_events,
        "n_events_detected": n_det,
        "event_recall": (n_det / n_events) if n_events else np.nan,
        "ttd_readings_median": float(ev_df["ttd_readings"].median()) if n_events else np.nan,
        "ttd_hours_median": float(ev_df["ttd_hours"].median()) if n_events else np.nan,
        "fp_outside_events": fp,
        "n_clean_readings": n_clean,
        "recovery_clean": bool(rec_df["clean"].all()) if len(rec_df) else True,
        "n_flagged_after_events": int(rec_df["n_flagged_after"].sum()) if len(rec_df) else 0,
    }, ev_df
