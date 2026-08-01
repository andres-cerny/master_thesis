"""
anomaly_injection.py — Sustained-anomaly injectors for the UC detection surface.

Companion to `create_anomalies.py`, which is left untouched. That module injects
isolated positive spikes directly onto `Diff`; this one covers the *sustained*
class named as known follow-up in VARIANTS_README.md — leaks, frozen meters,
zero-consumption windows — plus a cumulative-faithful spike and a false-positive
probe that carries no label at all.

Two deliberate differences from `create_anomalies.py`:

1. **Injection is on cumulative `hodnota`, not on `Diff`.** `Diff` is a derived
   quantity; writing to it directly bypasses Diff reconstruction and the
   negative-diff convention, which is exactly the code most likely to be wrong.
   Every archetype here is expressed as a `delta` vector added to the cumulative
   series, and `Diff` is then adjusted exactly (see `apply_delta`).

2. **Ground truth is an interval, not a row index.** A 6-hour burst is one event,
   not 24 independent labels. `AnomalyEvent` carries `[t_start, t_end]`; the
   per-row `is_anomaly_actual` column is derived from the intervals downstream
   (see `event_metrics.label_rows_from_events`), so point-wise metrics remain
   available and `calculate_metrics()` keeps working unchanged.

Depth is expressed in **multiples of the sensor's own scale** (`scale` = median
non-zero `Diff` on the clean calibration segment), never in absolute m³. `Diff`
magnitude reflects meter size rather than statistical structure, so an absolute
depth would mean something different on every sensor. This mirrors how
`uc_variants.Log1pScaled` derives its per-sensor scale, and for the same reason.

The scale MUST come from the calibration segment, never from the prediction
segment — deriving it from data that is about to be injected into would leak the
anomaly into its own magnitude definition.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# Mirrors resample.py's negative-diff convention so an injected series is cleaned
# exactly the way a resampled one is: rounding noise -> 0, real rollback -> NaN.
NEG_DIFF_TOL = 0.002


@dataclass
class AnomalyEvent:
    """One injected fault, labelled by time interval rather than row index.

    `t_start` / `t_end` are inclusive UTC timestamps. For an instantaneous
    archetype (`point_spike`) they are equal.

    `depth` is in multiples of the sensor scale; `duration_h` in hours. Both are
    carried through to the results so the detection surface can be pivoted on
    them without re-deriving anything.
    """
    archetype: str
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    depth: float = np.nan
    duration_h: float = np.nan
    scale: float = np.nan
    params: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["t_start"] = pd.Timestamp(self.t_start).isoformat()
        d["t_end"] = pd.Timestamp(self.t_end).isoformat()
        d["params"] = dict(self.params)
        return d


# ============================================================================
# SCALE
# ============================================================================


def sensor_scale(diff_values) -> float:
    """Median non-zero `Diff` — the sensor's typical consumption per step.

    Falls back to the mean, then to 1.0, so a degenerate calibration segment
    yields a usable (if arbitrary) unit rather than a NaN that would silently
    turn every injected depth into a no-op.
    """
    x = np.asarray(diff_values, dtype=float)
    x = x[np.isfinite(x)]
    nz = x[x > 0]
    if nz.size:
        m = float(np.median(nz))
        if np.isfinite(m) and m > 0:
            return m
    if x.size:
        m = float(np.mean(np.abs(x)))
        if np.isfinite(m) and m > 0:
            return m
    return 1.0


# ============================================================================
# CORE: cumulative delta -> exact Diff adjustment
# ============================================================================


def apply_delta(df, delta, hodnota_col="hodnota", diff_col="Diff"):
    """Add `delta` to the cumulative series and adjust `Diff` exactly.

    `Diff[i] = hodnota[i] - hodnota[i-1]`, so adding `delta` to `hodnota` implies

        Diff_new[i] = Diff_old[i] + (delta[i] - delta[i-1])

    This is computed rather than recomputed from `hodnota.diff()` for one
    specific reason: `df_predict` is a *slice* of the resampled 6-week frame, so
    its first row's `Diff` was derived from the last row of the calibration
    segment, which is not in this frame. Calling `.diff()` here would turn that
    first value into NaN and silently change the series even where nothing was
    injected. Treating the pre-slice row as unmodified (`delta[-1] = 0`) keeps
    row 0 exact.

    NaN-safe by construction: synthetic gap-fill rows carry `hodnota = NaN`, and
    NaN + delta stays NaN, so gaps remain gaps.
    """
    df = df.copy()
    delta = np.asarray(delta, dtype=float)
    if len(delta) != len(df):
        raise ValueError(f"delta length {len(delta)} != frame length {len(df)}")

    prev = np.concatenate(([0.0], delta[:-1]))  # row before the slice is untouched
    step = delta - prev

    df[hodnota_col] = df[hodnota_col].values + delta
    df[diff_col] = df[diff_col].values + step

    # Same cleaning the resampler applies, so an injected frame is
    # indistinguishable in convention from a naturally resampled one.
    d = df[diff_col]
    df[diff_col] = d.mask((d < 0) & (d > -NEG_DIFF_TOL), 0.0)
    d = df[diff_col]
    df[diff_col] = d.mask(d <= -NEG_DIFF_TOL, np.nan)
    return df


def _window(df, rng, duration_steps, timestamp_col="timestamp_utc", margin=2):
    """Pick a random contiguous window of `duration_steps` rows.

    `margin` keeps the window off both edges so an event always has at least one
    clean reading before it (the detector needs a baseline) and after it (the
    recovery check needs somewhere to recover to).
    """
    n = len(df)
    duration_steps = int(max(1, duration_steps))
    lo, hi = margin, n - duration_steps - margin
    if hi <= lo:
        # Frame too short for this duration: centre it and use what there is.
        i0 = max(0, (n - duration_steps) // 2)
        return i0, min(n, i0 + duration_steps)
    i0 = int(rng.integers(lo, hi))
    return i0, i0 + duration_steps


def _steps_for_hours(hours, periodicity_seconds):
    return max(1, int(round(float(hours) * 3600.0 / float(periodicity_seconds))))


def _event(df, i0, i1, archetype, depth, duration_h, scale, timestamp_col, **params):
    ts = pd.to_datetime(df[timestamp_col].values, utc=True)
    return AnomalyEvent(
        archetype=archetype,
        t_start=ts[i0],
        t_end=ts[min(i1, len(df)) - 1],
        depth=float(depth),
        duration_h=float(duration_h),
        scale=float(scale),
        params=params,
    )


# ============================================================================
# ARCHETYPES
# ============================================================================
# Each returns (df_injected, [AnomalyEvent]). All take the same signature so the
# dispatcher can call them uniformly.


def inject_slow_leak(df, rng, *, scale, periodicity_seconds, depth=0.10,
                     duration_h=48.0, timestamp_col="timestamp_utc"):
    """A continuous small extra draw — the headline sustained fault.

    Adds `depth * scale` to every step in the window, i.e. the cumulative series
    gains a constant-slope ramp. This is what a dripping joint looks like, and it
    is precisely what a local-level model absorbs: each individual residual is
    small enough to stay inside the band and to avoid tripping the masking
    threshold, so the level quietly re-baselines around the leak.

    No catch-up on exit — the water really was consumed, so the cumulative offset
    is permanent and `Diff` simply returns to normal afterwards.
    """
    steps = _steps_for_hours(duration_h, periodicity_seconds)
    i0, i1 = _window(df, rng, steps, timestamp_col)
    rate = float(depth) * float(scale)

    delta = np.zeros(len(df))
    delta[i0:i1] = rate * np.arange(1, i1 - i0 + 1)
    delta[i1:] = delta[i1 - 1] if i1 > i0 else 0.0

    ev = _event(df, i0, i1, "slow_leak", depth, duration_h, scale,
                timestamp_col, rate_per_step=rate, steps=i1 - i0)
    return apply_delta(df, delta), [ev]


def inject_burst(df, rng, *, scale, periodicity_seconds, depth=3.0,
                 duration_h=6.0, timestamp_col="timestamp_utc"):
    """A pipe break: same shape as a leak but steep and short.

    Separated from `slow_leak` because the interesting question is where on the
    depth axis detection actually starts, and the two occupy opposite ends of it.
    """
    steps = _steps_for_hours(duration_h, periodicity_seconds)
    i0, i1 = _window(df, rng, steps, timestamp_col)
    rate = float(depth) * float(scale)

    delta = np.zeros(len(df))
    delta[i0:i1] = rate * np.arange(1, i1 - i0 + 1)
    delta[i1:] = delta[i1 - 1] if i1 > i0 else 0.0

    ev = _event(df, i0, i1, "burst", depth, duration_h, scale,
                timestamp_col, rate_per_step=rate, steps=i1 - i0)
    return apply_delta(df, delta), [ev]


def inject_frozen_meter(df, rng, *, scale, periodicity_seconds, depth=np.nan,
                        duration_h=24.0, catch_up=True,
                        timestamp_col="timestamp_utc", diff_col="Diff"):
    """The meter stops advancing: `hodnota` holds, so `Diff` is 0 throughout.

    This is the fault the current detector is structurally blind to. At night the
    model predicts ~0 anyway, so a frozen meter matches the prediction perfectly
    and scores as a *better than average* reading.

    `catch_up=True` (default) releases the accumulated volume as one large step
    when the meter resumes, which is what a stuck mechanical register does. With
    `catch_up=False` the missed volume is lost permanently, which is what a
    telemetry-side freeze looks like. The two differ sharply on exit — the first
    is trivially detectable at the boundary, the second leaves no trace at all —
    so the flag matters, and depth is meaningless here either way.
    """
    steps = _steps_for_hours(duration_h, periodicity_seconds)
    i0, i1 = _window(df, rng, steps, timestamp_col)

    d_orig = np.nan_to_num(np.asarray(df[diff_col].values, dtype=float), nan=0.0)
    # Cancel exactly the consumption that would have accrued in the window.
    delta = np.zeros(len(df))
    delta[i0:i1] = -np.cumsum(d_orig[i0:i1])
    delta[i1:] = 0.0 if catch_up else delta[i1 - 1] if i1 > i0 else 0.0

    ev = _event(df, i0, i1, "frozen_meter", depth, duration_h, scale,
                timestamp_col, catch_up=bool(catch_up), steps=i1 - i0)
    return apply_delta(df, delta), [ev]


def inject_zero_consumption(df, rng, *, scale, periodicity_seconds, depth=np.nan,
                            duration_h=12.0, timestamp_col="timestamp_utc",
                            diff_col="Diff"):
    """Supply genuinely stops — a closed valve or an idle production line.

    Mechanically identical to a non-catching-up frozen meter, but semantically a
    different fault: the meter is healthy and the *consumption* is anomalous.
    Kept as its own archetype so the surface can distinguish "sensor broke" from
    "site stopped", which are different operational responses.
    """
    steps = _steps_for_hours(duration_h, periodicity_seconds)
    i0, i1 = _window(df, rng, steps, timestamp_col)

    d_orig = np.nan_to_num(np.asarray(df[diff_col].values, dtype=float), nan=0.0)
    delta = np.zeros(len(df))
    delta[i0:i1] = -np.cumsum(d_orig[i0:i1])
    delta[i1:] = delta[i1 - 1] if i1 > i0 else 0.0  # no catch-up: never consumed

    ev = _event(df, i0, i1, "zero_consumption", depth, duration_h, scale,
                timestamp_col, steps=i1 - i0)
    return apply_delta(df, delta), [ev]


def inject_point_spike(df, rng, *, scale, periodicity_seconds, depth=5.0,
                       duration_h=0.0, timestamp_col="timestamp_utc"):
    """One oversized reading — the cumulative-faithful version of the existing
    `inject_spike_anomalies_diff`.

    The offset persists in the cumulative series (a real over-draw does not
    un-happen), so exactly one `Diff` is inflated and the rest are untouched.
    Included as the control archetype: it is the one fault the detector is known
    to catch, so it anchors the surface.
    """
    i0, i1 = _window(df, rng, 1, timestamp_col)
    delta = np.zeros(len(df))
    delta[i0:] = float(depth) * float(scale)

    ev = _event(df, i0, i1, "point_spike", depth, 0.0, scale,
                timestamp_col, magnitude=float(depth) * float(scale))
    return apply_delta(df, delta), [ev]


def inject_unit_change(df, rng, *, scale, periodicity_seconds, depth=1000.0,
                       duration_h=np.nan, timestamp_col="timestamp_utc",
                       diff_col="Diff"):
    """The feed switches units mid-stream (m³ -> litres): every subsequent `Diff`
    is multiplied by `depth`.

    A configuration fault rather than a hydraulic one, but it presents as a
    permanent level shift, so it belongs on the same surface. Runs to the end of
    the segment by construction.
    """
    i0, _ = _window(df, rng, 1, timestamp_col)
    d_orig = np.nan_to_num(np.asarray(df[diff_col].values, dtype=float), nan=0.0)

    delta = np.zeros(len(df))
    delta[i0:] = np.cumsum(d_orig[i0:] * (float(depth) - 1.0))

    ev = _event(df, i0, len(df), "unit_change", depth, np.nan, scale,
                timestamp_col, factor=float(depth))
    return apply_delta(df, delta), [ev]


def inject_missing_data(df, rng, *, scale, periodicity_seconds, depth=np.nan,
                        duration_h=12.0, timestamp_col="timestamp_utc",
                        hodnota_col="hodnota", diff_col="Diff"):
    """Telemetry drops out: `hodnota` becomes NaN over the window.

    This is what a real gap looks like *after* resampling — synthetic rows carry
    `hodnota = NaN` and therefore `Diff = NaN` — so it is faithful without
    touching the resampler. Not expressible as a cumulative delta, so it writes
    the columns directly.

    Labelled as an event because a monitoring system should notice a sensor going
    dark, even though the reading itself is absent rather than wrong.
    """
    steps = _steps_for_hours(duration_h, periodicity_seconds)
    i0, i1 = _window(df, rng, steps, timestamp_col)

    out = df.copy()
    out.iloc[i0:i1, out.columns.get_loc(hodnota_col)] = np.nan
    out.iloc[i0:i1, out.columns.get_loc(diff_col)] = np.nan

    ev = _event(df, i0, i1, "missing_data", depth, duration_h, scale,
                timestamp_col, steps=i1 - i0)
    return out, [ev]


def inject_holiday_profile(df, rng, *, scale, periodicity_seconds, depth=np.nan,
                           duration_h=24.0, timestamp_col="timestamp_utc",
                           diff_col="Diff"):
    """FALSE-POSITIVE PROBE — returns NO event, so nothing here is labelled.

    Replaces one weekday's consumption profile with a quieter day's from the same
    segment: a legitimate low-usage day (public holiday, shutdown, weekend
    working pattern), not a fault. Any alert raised inside this window is a false
    positive by construction.

    This exists because a detector can always buy recall on the other archetypes
    by getting twitchier. Without an unlabelled-but-unusual probe in the mix,
    that trade is invisible in the headline numbers.
    """
    steps = _steps_for_hours(duration_h, periodicity_seconds)
    i0, i1 = _window(df, rng, steps, timestamp_col)
    n_win = i1 - i0

    d = np.asarray(df[diff_col].values, dtype=float)
    # Donor: the quietest same-length window elsewhere in the segment.
    best_j, best_sum = None, np.inf
    for j in range(0, len(df) - n_win):
        if abs(j - i0) < n_win:          # must not overlap the target
            continue
        s = np.nansum(d[j:j + n_win])
        if s < best_sum:
            best_sum, best_j = s, j
    if best_j is None:
        return df.copy(), []

    donor = np.nan_to_num(d[best_j:best_j + n_win], nan=0.0)
    target = np.nan_to_num(d[i0:i1], nan=0.0)

    delta = np.zeros(len(df))
    delta[i0:i1] = np.cumsum(donor - target)
    delta[i1:] = delta[i1 - 1] if n_win > 0 else 0.0

    return apply_delta(df, delta), []   # deliberately unlabelled


def inject_null_control(df, rng, *, scale, periodicity_seconds, depth=np.nan,
                        duration_h=24.0, timestamp_col="timestamp_utc"):
    """CONTROL — labels a window but injects nothing at all.

    This is the floor every other archetype must be read against. Event recall
    counts a detection if *any* reading inside the window is flagged, so over a
    24-hour window a detector with a normal false-alarm rate will "detect" a
    fair share of events it was never shown. Without this control, the shallow
    end of the depth ladder reports detection that is purely coincidental, and
    the surface looks like it degrades gracefully toward zero when in fact it
    bottoms out at the false-alarm floor.

    The window is drawn with the same RNG call sequence and the same duration as
    a real injector, so the control interval is directly comparable — same
    length, same seed, same placement distribution.

    Read every detection rate as `rate(archetype) - rate(null_control)` at the
    matching duration.
    """
    steps = _steps_for_hours(duration_h, periodicity_seconds)
    i0, i1 = _window(df, rng, steps, timestamp_col)
    ev = _event(df, i0, i1, "null_control", depth, duration_h, scale,
                timestamp_col, steps=i1 - i0)
    return df.copy(), [ev]


ARCHETYPES = {
    "null_control": inject_null_control,
    "slow_leak": inject_slow_leak,
    "burst": inject_burst,
    "frozen_meter": inject_frozen_meter,
    "zero_consumption": inject_zero_consumption,
    "point_spike": inject_point_spike,
    "unit_change": inject_unit_change,
    "missing_data": inject_missing_data,
    "holiday_profile": inject_holiday_profile,
}

# Archetypes for which `depth` is a meaningful axis of the detection surface.
# The others are shape-defined: their severity is set by duration alone.
DEPTH_MEANINGFUL = {"slow_leak", "burst", "point_spike", "unit_change"}


# ============================================================================
# PLAN DISPATCH
# ============================================================================


@dataclass
class InjectionPlan:
    """What to inject into one sensor's prediction segment.

    A plan is deliberately a *single* archetype at a time. Mixing faults in one
    week would make time-to-detect and the FP rate un-attributable, and the
    surface is built by sweeping plans, not by stacking them.
    """
    archetype: str
    depth: float = np.nan
    duration_h: float = np.nan
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)


def apply_injection_plan(df, plan, *, scale, periodicity_seconds, seed=None,
                         timestamp_col="timestamp_utc"):
    """Apply one `InjectionPlan` and return `(df_injected, [AnomalyEvent])`.

    `seed` is threaded from `int(filename)` exactly as `create_anomalies.py` is
    seeded, so a given sensor gets the same window on every run and across every
    variant. That pairing is what makes the paired bootstrap in
    `analyze_variants.py` valid; breaking it would silently widen every interval.
    """
    if plan is None:
        return df.copy(), []
    if plan.archetype not in ARCHETYPES:
        raise ValueError(
            f"unknown archetype {plan.archetype!r}; known: {sorted(ARCHETYPES)}"
        )

    rng = np.random.default_rng(plan.seed if plan.seed is not None else seed)
    fn = ARCHETYPES[plan.archetype]

    kwargs = dict(scale=scale, periodicity_seconds=periodicity_seconds)
    if np.isfinite(plan.depth):
        kwargs["depth"] = plan.depth
    if np.isfinite(plan.duration_h):
        kwargs["duration_h"] = plan.duration_h
    kwargs.update(plan.params or {})
    kwargs["timestamp_col"] = timestamp_col

    return fn(df, rng, **kwargs)
