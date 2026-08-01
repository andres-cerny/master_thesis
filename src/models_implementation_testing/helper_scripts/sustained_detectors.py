"""
sustained_detectors.py — Read-only detectors for the sustained-anomaly class.

The z-score detector asks one question per reading: *is this value far from its
prediction?* That is the right question for a spike and the wrong one for a leak.
A local-level model absorbs a sustained shift by design — the level component
exists precisely to track slow change — and it absorbs it faster the shallower
the shift is, because small residuals never trip the masking threshold that would
freeze the state. So the fault that most needs catching is the one the model is
built to accommodate.

Each detector here is a **scalar recurrence with explicit state**: one `update()`
per reading, no lookahead, no vectorised pass over the segment. That shape is
deliberate. The production detector
(`uc-cem/model/shared.py::UCStreamingDetector.step()`) processes exactly one
reading per call and persists its state to disk between calls, so a detector
written this way ports across as a field in `state.npz` and a few lines in
`step()`. A detector written as a pandas pass over the week would not port at all.

## Read-only

None of these gates the Kalman measurement update. Masking stays governed by the
existing `mask_z_threshold` alone, for two reasons: changing what freezes the
state changes the state trajectory and therefore changes the *existing* z-score
detector, which would silently invalidate every before/after comparison; and the
masking rule is deliberately two-sided even where detection is not, because an
anomalously low reading corrupts the state as badly as a high one.

So these emit their own columns and touch nothing else.
"""

import numpy as np
import pandas as pd


# ============================================================================
# 1. DIRECTION AND SIGNED z
# ============================================================================


class SignedResidualStats:
    """Keeps the sign the folded z-score throws away.

    The baseline computes `|actual - predicted|` and scores that against
    statistics of past absolute residuals. Two things are lost. First,
    direction: over-consumption and under-consumption are a burst pipe and a
    closed valve, which are different work orders. Second, the folded score is
    strange near zero — a *perfect* prediction gives `z = -mu/sd`, several
    standard deviations from the mean, so an unusually accurate reading looks
    unusual.

    This is pure reporting. The signed innovation is already computed inside the
    Kalman loop and discarded; here it is kept and standardised against the
    calibration residuals.
    """

    def __init__(self, mu=0.0, sd=1.0):
        self.mu = float(mu)
        self.sd = float(sd) if (sd and np.isfinite(sd) and sd > 0) else 1.0

    @classmethod
    def calibrate(cls, signed_residuals):
        r = np.asarray(signed_residuals, dtype=float)
        r = r[np.isfinite(r)]
        if r.size < 2:
            return cls(0.0, 1.0)
        return cls(float(np.mean(r)), float(np.std(r, ddof=1)))

    def update(self, actual, predicted):
        if not (np.isfinite(actual) and np.isfinite(predicted)):
            return {"z_signed": np.nan, "direction": "none"}
        r = float(actual) - float(predicted)
        z = (r - self.mu) / self.sd
        return {
            "z_signed": z,
            "direction": "over" if r > 0 else ("under" if r < 0 else "none"),
        }


# ============================================================================
# 2. SUSTAINED EVENTS
# ============================================================================


class FlagRunCounter:
    """Turns a run of point alarms into one escalating event.

    A 6-hour burst currently emits N independent flags with no grouping. Worse,
    a flagged reading skips the Kalman update, so the state never adapts and the
    run does not self-terminate — the alarm can continue indefinitely.

    Counting consecutive flags costs one integer of state and distinguishes
    "one odd reading" from "something has been wrong for three hours", which is
    the difference between noise and a work order.
    """

    def __init__(self, threshold=3):
        self.threshold = int(threshold)
        self.run = 0

    def update(self, flag):
        self.run = self.run + 1 if bool(flag) else 0
        return {
            "flag_run": self.run,
            "anomaly_sustained": bool(self.run >= self.threshold),
        }


# ============================================================================
# 3. STUCK METER
# ============================================================================


class StuckMeterDetector:
    """Catches a meter that has stopped advancing.

    This is the fault the z-score detector is most completely blind to. A frozen
    register reports `Diff = 0`; at night the model predicts approximately zero
    anyway, so the residual is approximately zero and the reading scores as
    *better than typical*. A dead sensor looks like a well-behaved one, and it
    looks that way indefinitely.

    Residual-based detection cannot see this, so the detector does not use
    residuals: it counts consecutive non-consuming readings and compares against
    how long that sensor genuinely idles. The calibration segment supplies the
    baseline, because a night-shut commercial site and a 24-hour plant have
    completely different normal zero-runs and a fixed threshold would be wrong
    for both.

    `tolerance_factor` gives headroom over the longest idle stretch actually
    observed during calibration, so ordinary quiet periods do not trip it.
    """

    def __init__(self, max_zero_run=0, tolerance_factor=1.5, min_run=3,
                 zero_tol=1e-12):
        self.max_zero_run = int(max_zero_run)
        self.threshold = max(int(min_run),
                             int(np.ceil(max(1, max_zero_run) * float(tolerance_factor))))
        self.zero_tol = float(zero_tol)
        self.run = 0

    @classmethod
    def calibrate(cls, diff_values, **kw):
        """Longest run of zero consumption in the clean calibration segment.

        NaN (a gap) neither extends nor resets the run: a missing reading is not
        evidence either way about whether the meter is advancing.
        """
        d = np.asarray(diff_values, dtype=float)
        longest = run = 0
        for v in d:
            if not np.isfinite(v):
                continue
            if abs(v) <= 1e-12:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        return cls(max_zero_run=longest, **kw)

    def update(self, diff):
        if not np.isfinite(diff):
            return {"zero_run": self.run, "stuck": False}
        if abs(float(diff)) <= self.zero_tol:
            self.run += 1
        else:
            self.run = 0
        return {"zero_run": self.run, "stuck": bool(self.run >= self.threshold)}


# ============================================================================
# 4. SLOW LEAK — EWMA + two-sided CUSUM on SIGNED residuals
# ============================================================================


class LeakDetector:
    """Accumulates a persistent bias that no single residual reveals.

    A leak shifts every residual by a small positive amount. Individually each
    stays inside the band; together they are a mean shift, and a CUSUM is the
    standard instrument for exactly that — it integrates the signed deviation
    and fires when the running sum exceeds a decision interval.

    Signed is essential. Folding to `|residual|` destroys the property being
    measured: a persistent over-draw and a persistent under-draw both increase
    the mean absolute residual, so the folded statistic cannot tell a leak from a
    shortfall, and symmetric noise inflates it as much as either.

    Two statistics, both cheap:

      * **CUSUM** with a slack `k` (in calibration sigmas) that the shift must
        exceed before accumulating, and a decision interval `h`. Standard
        tabular form: `C+ = max(0, C+ + z - k)`, and the mirror for `C-`.
      * **EWMA** of the signed z, reported for interpretability — it says *how
        biased* the recent residuals are, where the CUSUM says *fire now*.

    Both are reset on firing so a detected leak does not latch forever; the
    detector re-arms and can fire again if the condition persists, which is what
    makes a repeat alarm meaningful.

    Defaults `k=0.5`, `h=5.0` are the textbook starting point: a CUSUM tuned to
    detect a 1-sigma shift takes `k` at half the shift, and `h=5` gives a long
    in-control run length. `h` is the knob to move if the false-alarm rate is
    wrong.
    """

    def __init__(self, mu=0.0, sd=1.0, k=0.5, h=5.0, lam=0.05):
        self.mu = float(mu)
        self.sd = float(sd) if (sd and np.isfinite(sd) and sd > 0) else 1.0
        self.k = float(k)
        self.h = float(h)
        self.lam = float(lam)
        self.c_pos = 0.0
        self.c_neg = 0.0
        self.ewma = 0.0

    @classmethod
    def calibrate(cls, signed_residuals, **kw):
        r = np.asarray(signed_residuals, dtype=float)
        r = r[np.isfinite(r)]
        if r.size < 2:
            return cls(0.0, 1.0, **kw)
        return cls(float(np.mean(r)), float(np.std(r, ddof=1)), **kw)

    def update(self, actual, predicted):
        if not (np.isfinite(actual) and np.isfinite(predicted)):
            # A gap is not evidence of drift; hold the accumulators.
            return {"cusum_pos": self.c_pos, "cusum_neg": self.c_neg,
                    "ewma_z": self.ewma, "drift": False, "drift_direction": "none"}

        z = ((float(actual) - float(predicted)) - self.mu) / self.sd
        self.c_pos = max(0.0, self.c_pos + z - self.k)
        self.c_neg = max(0.0, self.c_neg - z - self.k)
        self.ewma = (1 - self.lam) * self.ewma + self.lam * z

        hi, lo = self.c_pos > self.h, self.c_neg > self.h
        drift = bool(hi or lo)
        direction = "over" if hi else ("under" if lo else "none")
        if drift:
            self.c_pos = 0.0
            self.c_neg = 0.0
        return {"cusum_pos": self.c_pos, "cusum_neg": self.c_neg,
                "ewma_z": self.ewma, "drift": drift, "drift_direction": direction}


# ============================================================================
# 4b. SLOW LEAK — CUSUM against a STABLE seasonal reference
# ============================================================================


class SeasonalDriftDetector:
    """CUSUM against the calibration profile instead of the model prediction.

    `LeakDetector` above does not work, and the reason is worth stating plainly
    because it is a property of the model rather than a tuning failure. Measured
    on the fixtures, the number of CUSUM firings under an injected leak is *less
    than or equal to* the number under `null_control` at every (k, h) tried —
    the leak never increases firing. A one-step-ahead residual is the error of an
    adaptive predictor: the local level tracks the leak within a few readings, so
    the residual returns to zero and the CUSUM sees only a brief transient. The
    fault erases its own evidence from the signal being monitored.

    The fix is to monitor something the model has not adapted to. This detector
    compares each reading against a **fixed seasonal profile** — the per-position
    median `Diff` over the clean calibration segment — which is frozen at
    calibration time and cannot drift toward the fault. A leak raises actual
    consumption above that profile for its entire duration, not just at onset.

    Scale is the per-position MAD of the calibration residuals about the same
    profile, so a position with genuinely variable consumption needs a larger
    excursion to fire than a quiet one. MAD rather than standard deviation
    because consumption residuals are right-skewed and a few busy days would
    otherwise inflate the scale everywhere.
    """

    def __init__(self, profile=None, scales=None, daily_steps=1, k=0.5, h=5.0,
                 lam=0.05):
        self.profile = np.asarray(profile, dtype=float) if profile is not None else None
        self.scales = np.asarray(scales, dtype=float) if scales is not None else None
        self.daily_steps = int(max(1, daily_steps))
        self.k = float(k)
        self.h = float(h)
        self.lam = float(lam)
        self.c_pos = 0.0
        self.c_neg = 0.0
        self.ewma = 0.0

    @classmethod
    def calibrate(cls, positions, diff_values, daily_steps, **kw):
        """Per-position median and MAD from the clean calibration segment."""
        pos = np.asarray(positions, dtype=int)
        d = np.asarray(diff_values, dtype=float)
        n = int(daily_steps)
        profile = np.full(n, np.nan)
        scales = np.full(n, np.nan)
        for p in range(n):
            v = d[(pos == p) & np.isfinite(d)]
            if v.size:
                med = float(np.median(v))
                profile[p] = med
                scales[p] = float(np.median(np.abs(v - med))) * 1.4826

        # Positions never observed, or with zero spread, fall back to global
        # values so a sparse daily grid cannot produce an infinitely twitchy slot.
        good = d[np.isfinite(d)]
        g_med = float(np.median(good)) if good.size else 0.0
        g_mad = float(np.median(np.abs(good - g_med))) * 1.4826 if good.size else 1.0
        if not np.isfinite(g_mad) or g_mad <= 0:
            g_mad = float(np.std(good)) if good.size > 1 else 1.0
        if not np.isfinite(g_mad) or g_mad <= 0:
            g_mad = 1.0
        profile = np.where(np.isfinite(profile), profile, g_med)
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, g_mad)
        return cls(profile=profile, scales=scales, daily_steps=n, **kw)

    def update(self, position, actual):
        if self.profile is None or not np.isfinite(actual):
            return {"seasonal_cusum_pos": self.c_pos, "seasonal_cusum_neg": self.c_neg,
                    "seasonal_ewma_z": self.ewma, "seasonal_drift": False,
                    "seasonal_drift_direction": "none"}

        p = int(position) % self.daily_steps
        z = (float(actual) - self.profile[p]) / self.scales[p]
        self.c_pos = max(0.0, self.c_pos + z - self.k)
        self.c_neg = max(0.0, self.c_neg - z - self.k)
        self.ewma = (1 - self.lam) * self.ewma + self.lam * z

        hi, lo = self.c_pos > self.h, self.c_neg > self.h
        drift = bool(hi or lo)
        direction = "over" if hi else ("under" if lo else "none")
        if drift:
            self.c_pos = 0.0
            self.c_neg = 0.0
        return {"seasonal_cusum_pos": self.c_pos, "seasonal_cusum_neg": self.c_neg,
                "seasonal_ewma_z": self.ewma, "seasonal_drift": drift,
                "seasonal_drift_direction": direction}


# ============================================================================
# 5. NIGHT FLOW
# ============================================================================


class NightFlowTracker:
    """Watches the daily minimum — the classic water-industry leak signal.

    A site's *minimum* hourly flow is the part of consumption nobody chose. Daily
    peaks move with production and weather; the overnight trough is close to a
    physical constant, and a leak raises it directly. Utilities have used minimum
    night flow for this since long before anyone applied a Kalman filter to it.

    Complementary to the CUSUM rather than redundant: the CUSUM works on model
    residuals and so inherits whatever the model has already absorbed, while this
    works on raw consumption and is therefore immune to the level component
    quietly re-baselining around the leak. When the model has adapted away a
    leak, this still sees the floor lift.

    Emits at most one alert per day, on the day boundary, since a daily minimum
    is not defined until the day is over.
    """

    def __init__(self, baseline_min=np.nan, factor=3.0, abs_floor=0.0):
        self.baseline = float(baseline_min)
        self.factor = float(factor)
        self.abs_floor = float(abs_floor)
        self._day = None
        self._day_min = np.inf
        self.last_day_min = np.nan

    @classmethod
    def calibrate(cls, timestamps, diff_values, **kw):
        """Median of the per-day minima on the calibration segment.

        Median rather than mean so one already-leaking day in the calibration
        window cannot lift the baseline and mask the very fault being looked for.
        """
        ts = pd.to_datetime(np.asarray(timestamps), utc=True)
        d = np.asarray(diff_values, dtype=float)
        ok = np.isfinite(d)
        if not ok.any():
            return cls(np.nan, **kw)
        s = pd.Series(d[ok], index=ts[ok])
        daily_min = s.groupby(s.index.date).min()
        if not len(daily_min):
            return cls(np.nan, **kw)
        return cls(float(np.median(daily_min.values)), **kw)

    def update(self, timestamp, diff):
        ts = pd.Timestamp(timestamp)
        day = ts.date()
        out = {"night_flow_day_min": np.nan, "night_flow_elevated": False}

        if self._day is None:
            self._day = day
        elif day != self._day:
            # Day rolled over: evaluate the completed day.
            self.last_day_min = self._day_min if np.isfinite(self._day_min) else np.nan
            out["night_flow_day_min"] = self.last_day_min
            if np.isfinite(self.last_day_min) and np.isfinite(self.baseline):
                limit = max(self.baseline * self.factor, self.abs_floor)
                # A zero baseline is common (a site that truly stops overnight);
                # any sustained non-zero floor is then meaningful on its own.
                if self.baseline <= 0:
                    out["night_flow_elevated"] = bool(self.last_day_min > self.abs_floor)
                else:
                    out["night_flow_elevated"] = bool(self.last_day_min > limit)
            self._day = day
            self._day_min = np.inf

        if np.isfinite(diff):
            self._day_min = min(self._day_min, float(diff))
        return out


# ============================================================================
# DRIVER
# ============================================================================


def calibrate_all(cal_timestamps, cal_diff, cal_signed_residuals, *,
                  cal_positions=None, daily_steps=None,
                  flag_run_threshold=3, cusum_k=0.5, cusum_h=5.0,
                  stuck_tolerance=1.5, night_factor=3.0):
    """Build the detector set from the clean calibration segment.

    Calibration MUST come from the calibration segment, never the prediction
    segment: every threshold here is "unusual for this sensor", and deriving that
    from data that may contain the injected fault would let the anomaly define
    its own normal.
    """
    return {
        "signed": SignedResidualStats.calibrate(cal_signed_residuals),
        "run": FlagRunCounter(threshold=flag_run_threshold),
        "stuck": StuckMeterDetector.calibrate(cal_diff, tolerance_factor=stuck_tolerance),
        "leak": LeakDetector.calibrate(cal_signed_residuals, k=cusum_k, h=cusum_h),
        "night": NightFlowTracker.calibrate(cal_timestamps, cal_diff, factor=night_factor),
        "seasonal": (
            SeasonalDriftDetector.calibrate(cal_positions, cal_diff, daily_steps,
                                            k=cusum_k, h=cusum_h)
            if cal_positions is not None and daily_steps
            else SeasonalDriftDetector(k=cusum_k, h=cusum_h)
        ),
    }


def run_detectors(detectors, timestamps, actuals, predictions, flags,
                  positions=None):
    """Step every detector across the prediction segment, in order.

    Returns a DataFrame of the new columns, aligned to the input rows. Strictly
    sequential and causal — reading `i` sees nothing after itself — so the result
    is exactly what an online implementation would have produced.
    """
    ts = pd.to_datetime(np.asarray(timestamps), utc=True)
    actuals = np.asarray(actuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    flags = np.asarray(flags).astype(bool)

    rows = []
    for i in range(len(ts)):
        rec = {}
        rec.update(detectors["signed"].update(actuals[i], predictions[i]))
        rec.update(detectors["run"].update(flags[i]))
        rec.update(detectors["stuck"].update(actuals[i]))
        rec.update(detectors["leak"].update(actuals[i], predictions[i]))
        rec.update(detectors["night"].update(ts[i], actuals[i]))
        if positions is not None:
            rec.update(detectors["seasonal"].update(positions[i], actuals[i]))
        rows.append(rec)

    out = pd.DataFrame(rows, index=pd.RangeIndex(len(ts)))
    # A single "anything fired" column, so event scoring can be run against the
    # union without the caller re-deriving it. The point z-score is deliberately
    # NOT part of this union: these detectors are measured on what they add.
    union = (out["anomaly_sustained"].astype(bool)
             | out["stuck"].astype(bool)
             | out["night_flow_elevated"].astype(bool))
    if "seasonal_drift" in out.columns:
        union = union | out["seasonal_drift"].astype(bool)
    # NOTE: `drift` (the residual CUSUM) is deliberately EXCLUDED from the union.
    # It fires no more often under an injected leak than under null_control, so
    # including it would add false alarms and no detections. It is still emitted
    # as a column so that finding stays visible and auditable.
    out["sustained_alert"] = union
    return out


def crossfit_drift(theta_first, theta_second, mase_first=np.nan,
                   mase_second=np.nan):
    """Compare the two training fits — the batch analogue of a retrain check.

    Production refits weekly. If a sensor's fitted parameters move sharply between
    consecutive fits, something structural changed: new equipment, a schedule
    change, or a fault the model is busy absorbing. That is worth surfacing
    whether or not any individual reading was ever flagged, and it costs nothing
    here because both fits already exist.

    Training-time only; it cannot affect prediction.
    """
    a = np.asarray(theta_first, dtype=float).ravel()
    b = np.asarray(theta_second, dtype=float).ravel()
    out = {"theta_l2_delta": np.nan, "theta_max_rel_delta": np.nan,
           "mase_delta": np.nan}
    if a.size and a.size == b.size:
        out["theta_l2_delta"] = float(np.linalg.norm(b - a))
        denom = np.maximum(np.abs(a), 1e-12)
        out["theta_max_rel_delta"] = float(np.max(np.abs(b - a) / denom))
    if np.isfinite(mase_first) and np.isfinite(mase_second):
        out["mase_delta"] = float(mase_second - mase_first)
    return out
