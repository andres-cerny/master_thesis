"""
predict.py — Online prediction for the streaming UC water-meter anomaly detector.

One invocation = one reading. Loads the per-meter model + state, processes a
single (timestamp, cumulative hodnota) reading, and emits a record containing:

  - flag           : whether the reading that just arrived is anomalous
  - z_current      : its z-score (None when not scored, e.g. after a gap)
  - pred_current   : the one-step-ahead prediction made for it before it arrived
  - pred_next      : the one-step-ahead prediction for the next on-cadence slot
  - lower_band_next, upper_band_next : detection envelope for that next slot
  - status         : ok / anomaly / post_gap_skipped / outage / reset /
                     skipped_early / skipped_duplicate / invalid / no_model

The flag is necessarily about the point that just arrived (you can only judge a
reading once you've seen it); the prediction + band are forward-looking, for the
next reading. Gap handling, anomaly masking, outage reset, and meter-reset
detection all live in shared.UCStreamingDetector.step().

State is loaded, advanced by one step, and saved back atomically — except for
the reject statuses (skipped_*/invalid), where nothing changed so nothing is
written. This is the cheap, frequently-run half of the system; it never fits a
model and does not import statsmodels.

Usage:
    python predict.py --meter-id 137402 \
        --timestamp "2024-06-01T08:30:00Z" --hodnota 12345.6 \
        --state-dir ./state

Prints one JSON record to stdout. Exit code 0 on success, 2 if the meter has
not been trained yet.
"""

import os
import sys
import json
import argparse

import pandas as pd

import shared

# Statuses that mean "the reading was rejected and state is unchanged" — no save.
_NO_STATE_CHANGE = {"skipped_early", "skipped_duplicate", "invalid"}


def predict_one(meter_id, timestamp, hodnota, state_dir,
                timezone=None, ambiguous="infer", nonexistent="NaT",
                z_threshold_override=None, persist=True):
    """Process one reading and return a record dict. Persists updated state
    unless the reading was rejected or persist=False.

    `timestamp` may be a naive *local* time (converted using the meter's stored
    timezone, DST-aware) or an already tz-aware / UTC string (used directly).
    """
    meter_dir = os.path.join(state_dir, str(meter_id))

    if not os.path.exists(os.path.join(meter_dir, shared.MODEL_FILE)):
        return {
            "status": "no_model",
            "meter_id": str(meter_id),
            "flag": False,
            "message": "No trained model for this meter. Run train.py first.",
        }

    det = shared.UCStreamingDetector.load(meter_dir, z_threshold_override=z_threshold_override)

    # Local -> UTC using the timezone the model was trained with (overridable).
    tz = timezone or det.timezone
    t_utc = shared.localize_to_utc(timestamp, timezone=tz,
                                   ambiguous=ambiguous, nonexistent=nonexistent)[0]
    if pd.isna(t_utc):
        return {
            "status": "invalid_timestamp",
            "meter_id": str(meter_id),
            "flag": False,
            "message": f"Timestamp '{timestamp}' is ambiguous/nonexistent in {tz}; skipped.",
        }

    rec = det.step(t_utc, hodnota)
    rec["meter_id"] = str(meter_id)

    if persist and rec["status"] not in _NO_STATE_CHANGE:
        det.save_state(meter_dir)

    return rec


def _parse_hodnota(s):
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("", "nan", "none", "null"):
        return float("nan")
    return float(s)


def main():
    p = argparse.ArgumentParser(description="Score one meter reading and forecast the next.")
    p.add_argument("--meter-id", required=True)
    p.add_argument("--timestamp", required=True, help="ISO-8601 timestamp of the reading.")
    p.add_argument("--hodnota", required=True, help="Cumulative meter reading (use 'nan' if missing).")
    p.add_argument("--state-dir", default="./state")
    p.add_argument("--timezone", default=None,
                   help="Local timezone of the timestamp (default: the one the model was trained with). "
                        "Ignored if the timestamp is already tz-aware.")
    p.add_argument("--z-threshold", type=float, default=None,
                   help="Override the stored z-score threshold (optional).")
    args = p.parse_args()

    rec = predict_one(
        meter_id=args.meter_id,
        timestamp=args.timestamp,
        hodnota=_parse_hodnota(args.hodnota),
        state_dir=args.state_dir,
        timezone=args.timezone,
        z_threshold_override=args.z_threshold,
    )
    print(json.dumps(rec))
    sys.exit(0 if rec.get("status") != "no_model" else 2)


if __name__ == "__main__":
    main()
