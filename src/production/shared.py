"""
shared.py — Shared model, Kalman step, and persistence layer for the streaming
Unobserved-Components (local level + daily/weekly Fourier) water-meter anomaly
detector.

Imported by BOTH train.py (offline fit, runs on a schedule) and predict.py
(online, one reading per invocation). The Kalman arithmetic, the position-on-
daily-cycle mapping, the config schema, and the load/save helpers all live here
so that training and prediction can never silently drift apart.

State on disk, per meter, under {state_dir}/{meter_id}/:
  model.npz   — immutable, written once per (re)fit:
                  T, Z, H, Q_full          (state-space system matrices)
                  P0_train                 (end-of-training covariance; outage reset target)
                  means_per_pos, stds_per_pos  (per-position reference-residual stats)
  config.json — immutable, written once per (re)fit:
                  periodicity, tolerance, daily/weekly steps, freq_seasonal,
                  k_positions, z_threshold, global_mean/std, structural signature,
                  theta, schema version, trained_at
  state.npz   — mutable, overwritten atomically every prediction step:
                  x, P                     (Kalman posterior)
                  state_time_epoch         (timestamp the state x,P is aligned to)
                  last_real_hodnota        (last real cumulative meter reading, for Diff)

NOTE: only the classical z-score is used. The robust (median/MAD) variant from
the batch evaluation code has been intentionally dropped.

This module deliberately depends only on numpy + pandas (NOT statsmodels) so the
hot prediction path stays lightweight.
"""

import os
import json
import tempfile

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

MODEL_FILE = "model.npz"
CONFIG_FILE = "config.json"
STATE_FILE = "state.npz"


# ==========================================================================
# POSITION-ON-DAILY-CYCLE MAPPING (carried over from the batch code)
# ==========================================================================
def compute_position_in_period(timestamps, periodicity_seconds, daily_period_steps):
    """
    Map timestamps to their integer position within the daily cycle,
    in [0, daily_period_steps). Position derives from time-of-day (UTC), so two
    timestamps exactly one day apart get the same position regardless of where
    the sampling grid starts. Works for a single timestamp or an array.
    """
    ts = pd.to_datetime(np.atleast_1d(timestamps), utc=True)
    seconds_since_midnight = (
        ts.hour.values.astype(float) * 3600.0
        + ts.minute.values.astype(float) * 60.0
        + ts.second.values.astype(float)
    )
    positions = np.round(seconds_since_midnight / periodicity_seconds).astype(int)
    return positions % daily_period_steps


def position_of(ts, periodicity_seconds, daily_period_steps):
    """Position-in-daily-cycle for a single timestamp, as a Python int."""
    return int(compute_position_in_period([ts], periodicity_seconds, daily_period_steps)[0])


def compute_per_position_stats(ref_residuals, ref_positions, daily_period_steps, k_positions):
    """
    For each position p in [0, daily_period_steps), pool reference residuals from
    positions in [p - k, p + k] (mod period) and return per-position mean and std.

    Returns two arrays of length daily_period_steps; positions whose neighborhood
    has no data hold NaN (callers fall back to the global statistics there).

    The robust (median/MAD) outputs of the batch version are intentionally omitted.
    """
    ref_residuals = np.asarray(ref_residuals, dtype=float)
    ref_positions = np.asarray(ref_positions, dtype=int)

    valid = ~np.isnan(ref_residuals)
    ref_residuals = ref_residuals[valid]
    ref_positions = ref_positions[valid]

    period = int(daily_period_steps)
    by_pos = [[] for _ in range(period)]
    for r, p in zip(ref_residuals, ref_positions):
        by_pos[p].append(r)
    by_pos = [np.asarray(lst, dtype=float) for lst in by_pos]

    means = np.full(period, np.nan)
    stds = np.full(period, np.nan)
    for p in range(period):
        chunks = [
            by_pos[(p + off) % period]
            for off in range(-k_positions, k_positions + 1)
            if by_pos[(p + off) % period].size > 0
        ]
        if not chunks:
            continue
        pooled = np.concatenate(chunks)
        means[p] = pooled.mean()
        stds[p] = pooled.std(ddof=1) if pooled.size > 1 else 0.0
    return means, stds


def compute_k_positions(daily_period_steps, k_hours=2.0, min_k=2):
    """Convert a neighborhood half-width in real hours to a number of positions,
    floored at min_k so coarse-frequency sensors still pool enough samples."""
    return max(min_k, int(round(k_hours * daily_period_steps / 24)))


def structural_signature(periodicity_seconds, daily_period_steps, weekly_period_steps, freq_seasonal):
    """A stable string identifying the model's structural shape. Warm-start from a
    previous fit is only valid when this is unchanged (same state dimension and
    same daily-cycle geometry)."""
    payload = {
        "periodicity_seconds": round(float(periodicity_seconds), 6),
        "daily_period_steps": int(daily_period_steps),
        "weekly_period_steps": int(weekly_period_steps),
        "freq_seasonal": freq_seasonal,
    }
    return json.dumps(payload, sort_keys=True)


# ==========================================================================
# ATOMIC PERSISTENCE HELPERS
# ==========================================================================
def _atomic_savez(path, **arrays):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=d, suffix=".npz", delete=False)
    tmp.close()
    try:
        np.savez(tmp.name, **arrays)  # name ends in .npz -> written verbatim
        os.replace(tmp.name, path)
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def _atomic_write_json(path, obj):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", dir=d, suffix=".tmp", delete=False)
    try:
        json.dump(obj, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def save_model(meter_dir, *, T, Z, H, Q_full, P0_train,
               means_per_pos, stds_per_pos, config,
               x0, P0, state_time, last_real_hodnota):
    """Write the immutable model + config and the initial mutable state.
    Called once per (re)fit by train.py."""
    os.makedirs(meter_dir, exist_ok=True)
    _atomic_savez(
        os.path.join(meter_dir, MODEL_FILE),
        T=np.asarray(T, dtype=float),
        Z=np.asarray(Z, dtype=float),
        H=np.asarray(H, dtype=float),
        Q_full=np.asarray(Q_full, dtype=float),
        P0_train=np.asarray(P0_train, dtype=float),
        means_per_pos=np.asarray(means_per_pos, dtype=float),
        stds_per_pos=np.asarray(stds_per_pos, dtype=float),
    )
    _atomic_write_json(os.path.join(meter_dir, CONFIG_FILE), config)
    _atomic_savez(
        os.path.join(meter_dir, STATE_FILE),
        x=np.asarray(x0, dtype=float).reshape(-1),
        P=np.asarray(P0, dtype=float),
        state_time_epoch=np.asarray(float(pd.Timestamp(state_time).timestamp())),
        last_real_hodnota=np.asarray(float(last_real_hodnota)),
    )


def warmstart_state(meter_dir, expected_signature):
    """Return (x, P) from a prior fit IF it exists and is structurally compatible
    with expected_signature, else (None, None). Used by train.py to warm-start MLE."""
    cfg_path = os.path.join(meter_dir, CONFIG_FILE)
    state_path = os.path.join(meter_dir, STATE_FILE)
    if not (os.path.exists(cfg_path) and os.path.exists(state_path)):
        return None, None
    try:
        cfg = json.load(open(cfg_path))
    except Exception:
        return None, None
    if cfg.get("schema_version") != SCHEMA_VERSION:
        return None, None
    if cfg.get("structural_signature") != expected_signature:
        return None, None
    st = np.load(state_path)
    return st["x"].copy(), st["P"].copy()


# ==========================================================================
# STREAMING DETECTOR
# ==========================================================================
class UCStreamingDetector:
    """One reading in -> one record out. Holds the Kalman posterior (x, P), the
    timestamp the state is aligned to, and the last real cumulative reading.

    Status values returned by step():
      ok               normal reading, scored and state updated
      anomaly          normal-cadence reading flagged (|z| > threshold); update skipped
      post_gap_skipped first reading after a gap; Diff spans the gap so it is
                       neither scored nor used to update the state
      outage           gap longer than outage_cap; state advanced (phase-preserving),
                       covariance reset to the training covariance, not flagged
      reset            negative Diff (meter rollover/replacement); rebaselined, not flagged
      skipped_early    arrived sooner than one period minus tolerance (resample's
                       "unmatched" case); rejected, state untouched
      skipped_duplicate non-increasing timestamp; rejected, state untouched
      invalid          missing/NaN hodnota; rejected, state untouched
    """

    def __init__(self, *, T, Z, H, Q_full, P0_train,
                 means_per_pos, stds_per_pos, global_mean, global_std,
                 periodicity_seconds, tolerance_percentage,
                 daily_period_steps, weekly_period_steps, k_positions, z_threshold,
                 x, P, state_time, last_real_hodnota, outage_cap_steps=None):
        self.T = np.asarray(T, dtype=float)
        self.Z = np.asarray(Z, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q_full = np.asarray(Q_full, dtype=float)
        self.P0_train = np.asarray(P0_train, dtype=float)

        self.means_per_pos = np.asarray(means_per_pos, dtype=float)
        self.stds_per_pos = np.asarray(stds_per_pos, dtype=float)
        self.global_mean = float(global_mean)
        self.global_std = float(global_std) if global_std is not None else 0.0

        self.periodicity = float(periodicity_seconds)
        self.tolerance = (float(tolerance_percentage) / 100.0) * self.periodicity
        self.daily_period_steps = int(daily_period_steps)
        self.weekly_period_steps = int(weekly_period_steps)
        self.k_positions = int(k_positions)
        self.z_threshold = float(z_threshold)
        self.outage_cap_steps = int(outage_cap_steps) if outage_cap_steps else self.weekly_period_steps

        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.P = np.asarray(P, dtype=float)
        self.state_time = self._to_utc(state_time)
        self.last_real_hodnota = float(last_real_hodnota)

        self.k = self.T.shape[0]
        self.I = np.eye(self.k)
        # transient priors set by _kf_predict
        self.x_prior = None
        self.P_prior = None

    # ---- construction from disk ------------------------------------------
    @classmethod
    def load(cls, meter_dir, z_threshold_override=None):
        cfg = json.load(open(os.path.join(meter_dir, CONFIG_FILE)))
        model = np.load(os.path.join(meter_dir, MODEL_FILE))
        state = np.load(os.path.join(meter_dir, STATE_FILE))
        return cls(
            T=model["T"], Z=model["Z"], H=model["H"], Q_full=model["Q_full"],
            P0_train=model["P0_train"],
            means_per_pos=model["means_per_pos"], stds_per_pos=model["stds_per_pos"],
            global_mean=cfg["global_mean"], global_std=cfg["global_std"],
            periodicity_seconds=cfg["periodicity_seconds"],
            tolerance_percentage=cfg["tolerance_percentage"],
            daily_period_steps=cfg["daily_period_steps"],
            weekly_period_steps=cfg["weekly_period_steps"],
            k_positions=cfg["k_positions"],
            z_threshold=(z_threshold_override if z_threshold_override is not None
                         else cfg["z_threshold"]),
            x=state["x"], P=state["P"],
            state_time=pd.Timestamp(float(state["state_time_epoch"]), unit="s", tz="UTC"),
            last_real_hodnota=float(state["last_real_hodnota"]),
            outage_cap_steps=cfg.get("outage_cap_steps"),
        )

    def save_state(self, meter_dir):
        _atomic_savez(
            os.path.join(meter_dir, STATE_FILE),
            x=self.x, P=self.P,
            state_time_epoch=np.asarray(float(self.state_time.timestamp())),
            last_real_hodnota=np.asarray(float(self.last_real_hodnota)),
        )

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _to_utc(ts):
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _stats_at(self, pos):
        mu = self.means_per_pos[pos]
        sd = self.stds_per_pos[pos]
        if np.isnan(mu) or np.isnan(sd) or sd == 0:
            return self.global_mean, self.global_std
        return float(mu), float(sd)

    @staticmethod
    def _scalar(a):
        return float(np.asarray(a).reshape(-1)[0])

    def _kf_predict(self):
        self.x_prior = self.T @ self.x
        self.P_prior = self.T @ self.P @ self.T.T + self.Q_full
        y_raw = self._scalar(self.Z @ self.x_prior)
        return max(y_raw, 0.0), y_raw

    def _commit_no_update(self):
        self.x = self.x_prior
        self.P = self.P_prior

    def _commit_update(self, y_t, y_raw):
        S = self._scalar(self.Z @ self.P_prior @ self.Z.T + self.H)
        K = (self.P_prior @ self.Z.T) / S
        self.x = self.x_prior + K.flatten() * (y_t - y_raw)
        IKZ = self.I - K @ self.Z
        self.P = IKZ @ self.P_prior @ IKZ.T + K * self.H[0, 0] @ K.T

    def _next_step_band(self):
        """One-step-ahead forecast and detection envelope for the slot that the
        next on-cadence reading would occupy (state_time + periodicity)."""
        x_next = self.T @ self.x
        y_next = max(self._scalar(self.Z @ x_next), 0.0)
        next_time = self.state_time + pd.Timedelta(seconds=self.periodicity)
        pos = position_of(next_time, self.periodicity, self.daily_period_steps)
        mu, sd = self._stats_at(pos)
        resid_thresh = mu + self.z_threshold * sd  # |actual - pred| above this is anomalous
        upper = y_next + resid_thresh
        lower = max(y_next - resid_thresh, 0.0)    # Diff is non-negative
        return y_next, lower, upper

    def _blank_record(self, t_utc, hodnota):
        return {
            "timestamp": t_utc.isoformat(),
            "hodnota": (None if hodnota is None else float(hodnota)),
            "status": None,
            "flag": False,
            "diff": None,
            "z_current": None,
            "pred_current": None,
            "pred_next": None,
            "lower_band_next": None,
            "upper_band_next": None,
        }

    # ---- the one-reading entry point -------------------------------------
    def step(self, t_new, hodnota_new):
        """Process a single (timestamp, cumulative hodnota) reading and return a
        record dict. Mutates internal state EXCEPT for the reject statuses
        (skipped_*/invalid), where the caller should not persist."""
        t_new = self._to_utc(t_new)
        rec = self._blank_record(t_new, hodnota_new)

        # --- invalid / missing reading ------------------------------------
        if hodnota_new is None or (isinstance(hodnota_new, float) and np.isnan(hodnota_new)):
            rec["status"] = "invalid"
            return rec
        hodnota_new = float(hodnota_new)

        delta = (t_new - self.state_time).total_seconds()

        # --- out-of-order / duplicate -------------------------------------
        if delta <= 0:
            rec["status"] = "skipped_duplicate"
            return rec

        # --- arrived too early (resample 'unmatched') ---------------------
        if delta < self.periodicity - self.tolerance:
            rec["status"] = "skipped_early"
            return rec

        n_periods = max(1, int(round(delta / self.periodicity)))
        n_missing = n_periods - 1
        diff_new = hodnota_new - self.last_real_hodnota

        # --- meter reset / rollover ---------------------------------------
        if diff_new < 0:
            for _ in range(n_periods):
                self._kf_predict()
                self._commit_no_update()
            self.state_time = t_new
            self.last_real_hodnota = hodnota_new
            y_next, lower, upper = self._next_step_band()
            rec.update(status="reset", diff=diff_new,
                       pred_next=y_next, lower_band_next=lower, upper_band_next=upper)
            return rec

        # --- outage handling ----------------------------------------------
        outage = n_missing > self.outage_cap_steps
        # phase repeats every weekly_period_steps (weekly = 7*daily), so we can
        # advance phase cheaply without applying T thousands of times.
        eff_missing = (n_missing % self.weekly_period_steps) if outage else n_missing

        # --- advance through missing slots (predict-only) ----------------
        for _ in range(eff_missing):
            self._kf_predict()
            self._commit_no_update()

        # --- the arriving reading: always one predict ---------------------
        y_clip, y_raw = self._kf_predict()

        if n_missing == 0 and not outage:
            # clean, on-cadence reading: score it, update unless flagged
            pos = position_of(t_new, self.periodicity, self.daily_period_steps)
            mu, sd = self._stats_at(pos)
            residual = abs(diff_new - y_raw)
            z = (residual - mu) / sd if sd > 0 else 0.0
            flag = abs(z) > self.z_threshold
            if flag:
                self._commit_no_update()      # don't let the spike corrupt the state
            else:
                self._commit_update(diff_new, y_raw)
            status = "anomaly" if flag else "ok"
            rec["pred_current"] = y_clip
            rec["z_current"] = z
            rec["flag"] = bool(flag)
        else:
            # first reading after a gap (or outage): Diff spans the gap -> untrustworthy
            self._commit_no_update()
            if outage:
                self.P = self.P0_train.copy()  # renewed uncertainty, not the inflated one
            status = "outage" if outage else "post_gap_skipped"

        # bookkeeping: state aligned to t_new, baseline advanced for next Diff
        self.state_time = t_new
        self.last_real_hodnota = hodnota_new

        y_next, lower, upper = self._next_step_band()
        rec.update(status=status, diff=diff_new,
                   pred_next=y_next, lower_band_next=lower, upper_band_next=upper)
        return rec
