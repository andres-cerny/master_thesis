"""
uc_variants.py — Parameterized UC model for A/B testing model tweaks.
=====================================================================

Wraps `local_level_unobserved_components_next_one_step_pred.py` (the baseline)
and adds three independently-togglable variants, so improvements can be
measured one at a time rather than bundled:

  1. `transform="log1p_scaled"` — fit in log1p(Diff / s) space, where s is the
     per-sensor median non-zero Diff on the CLEAN calibration segment.
     Plain log1p(Diff) is nearly the identity here (Diff is ~0.005-0.5 m^3, so
     log1p compresses by <5% on some sensors and ~20% on others) — dividing by
     s makes the transform bend over the same *relative* range on every sensor
     regardless of meter size/units. Scoring happens in transformed space (that
     is the point); accuracy metrics are inverted back to RAW units so MAE /
     MASE stay comparable to baseline.

  2. `harmonics_daily_max=H` — adaptive daily Fourier harmonics,
     `min(H, (daily_steps-1)//2)`, which keeps both frequencies strictly below
     Nyquist so no sensor is newly rejected by the periodicity floor.
     Baseline hardcodes 2 harmonics, which captures only ~82-86% of the mean
     daily profile variance on real sensors; H=4 recovers roughly half the rest.

  3. `signed_bands=True` — score the SIGNED residual (actual - predicted) with
     an upper-only detection band, instead of folding to |residual| and doing a
     two-sided test. Injected anomalies are strictly positive, so the negative
     half of a folded band can only ever produce false positives. Folding also
     makes an unusually *accurate* prediction score far from the mean (z =
     -mu/sd at r=0), which is pathological where residuals cluster tightly.

Two more knobs exist to make evaluation honest rather than to change the model:

  * `mask_z_threshold` is decoupled from the detection threshold. In the
    baseline these are the same number, which conflates "what poisons the
    Kalman state" with "what we alert on" — and makes a detection-threshold
    sweep require a full re-run. Masking stays TWO-SIDED even when detection is
    upper-only: a wildly low reading corrupts the state just as badly as a high
    one, and freezing the state is what keeps a sustained deficit visible
    instead of being absorbed by the local level.
  * per-reading z-scores are returned alongside their labels, so precision /
    recall at any *detection* threshold can be recomputed post-hoc with no
    re-run.

Defaults reproduce the baseline exactly — see `baseline_equivalence_test.py`.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Reuse the baseline's unchanged machinery — window splitting, MLE fitting, the
# batch filter pass, and the position-conditional statistics. Only the three
# functions that the variants actually touch are overridden below.
import local_level_unobserved_components_next_one_step_pred as base
from local_level_unobserved_components_next_one_step_pred import (  # noqa: F401
    compute_position_in_period,
    compute_per_position_stats,
    compute_k_positions,
    split_df_sliding_weeks,
    train_model,
    predict_model_batch,
    logger,
)

from helper_scripts.calculate_metrics import calculate_metrics
from helper_scripts.create_anomalies import inject_spike_anomalies_diff
from helper_scripts import anomaly_injection as ai
from helper_scripts import event_metrics as em
from helper_scripts import sustained_detectors as sd
from helper_scripts.series_diagnostics import (
    compute_intermittency,
    compute_seasonal_diagnostics,
    compute_mase,
)

# Production quality gates, mirrored from uc-cem model/shared.py QUALITY_DEFAULTS
# (initial-fit tier). Recorded as booleans per sensor; NEVER used to filter here.
# MASE is model-dependent, so gating on it would let a variant shrink its own
# evaluation cohort and score better for the wrong reason.
# NOTE: uc-cem CLAUDE.md documents 0.50 / 0.90 — the code is authoritative.
PROD_FS_MIN = 0.45
PROD_MASE_MAX = 0.95
PROD_FILL_MAX_PCT = 25.0
PROD_DAILY_STEPS_MIN = 5
PROD_MAX_GAP_DAYS = 2.0


# ============================================================================
# SCALE TRANSFORMS
# ============================================================================


class RawSpace:
    """Identity — the baseline. Model is fit directly on Diff."""

    name = "raw"
    scale = 1.0

    def fwd(self, x):
        return np.asarray(x, dtype=float)

    def inv(self, y):
        return np.asarray(y, dtype=float)


class Log1pScaled:
    """log1p(Diff / s), inverted as s * expm1(y).

    `s` normalizes away the sensor's absolute magnitude so the transform's
    strength does not depend on meter size or units. Diff = s maps to log(2),
    Diff = 10s to log(11), on every sensor. Zeros are preserved exactly
    (log1p(0) = 0) and NaNs propagate untouched, so the Kalman filter's
    missing-observation handling is unaffected.
    """

    name = "log1p_scaled"

    def __init__(self, scale):
        s = float(scale) if scale is not None else np.nan
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        self.scale = s

    def fwd(self, x):
        return np.log1p(np.asarray(x, dtype=float) / self.scale)

    def inv(self, y):
        return self.scale * np.expm1(np.asarray(y, dtype=float))


def build_transform(kind, df_calibration, result):
    """Construct the scale transform from the CLEAN calibration segment.

    `s` must never be derived from the prediction segment — injected anomalies
    would leak into the transform and inflate measured performance.
    """
    if kind in (None, "raw"):
        result["signal_scale"] = np.nan
        result["transform"] = "raw"
        result["transform_scale_used"] = 1.0
        return RawSpace()

    if kind != "log1p_scaled":
        raise ValueError(f"Unknown transform: {kind!r}")

    vals = np.asarray(df_calibration["Diff"].values, dtype=float)
    nz = vals[np.isfinite(vals) & (vals > 0)]
    s = float(np.median(nz)) if nz.size else np.nan
    tf = Log1pScaled(s)
    result["signal_scale"] = s
    result["transform"] = "log1p_scaled"
    result["transform_scale_used"] = tf.scale
    return tf


def _transformed(df, transform):
    """Copy of `df` with Diff mapped into scoring space."""
    out = df.copy()
    out["Diff"] = transform.fwd(out["Diff"].values)
    return out


# ============================================================================
# ADAPTIVE HARMONICS
# ============================================================================


def resolve_daily_harmonics(daily_steps, harmonics_daily_max):
    """Daily Fourier harmonics for this sensor.

    `None` reproduces the baseline exactly (hardcoded 2, no adaptation).
    An integer caps the harmonics at `(daily_steps-1)//2` so the highest
    frequency stays strictly below Nyquist — which is what lets the count rise
    without raising the periodicity floor and rejecting coarse sensors.
    """
    if harmonics_daily_max is None:
        return 2
    return int(max(1, min(int(harmonics_daily_max), (int(daily_steps) - 1) // 2)))


# ============================================================================
# ONLINE PREDICTION (transform-aware, signed-capable, decoupled masking)
# ============================================================================


def predict_model_online(df_predict, result, freq_seasonal, stochastic_freq_seasonal,
                         init_state_mean, init_state_cov, theta,
                         ref_residuals=None, ref_timestamps=None,
                         periodicity_seconds=None, daily_period_steps=None,
                         k_positions=None, mask_z_threshold=3.5,
                         signed_bands=False):
    """One-step-ahead prediction with online anomaly masking, in scoring space.

    Differs from the baseline in three ways:
      * `df_predict["Diff"]` is expected to be ALREADY in scoring space.
      * the residual is signed when `signed_bands=True` (baseline folds to abs).
      * `mask_z_threshold` is the state-protection threshold only; it is NOT the
        detection threshold, and it is applied two-sided in both modes.
    """
    y_pred_segment = np.asarray(df_predict["Diff"].values, dtype=float)
    n = len(y_pred_segment)

    use_conditional = (
        ref_residuals is not None
        and ref_timestamps is not None
        and periodicity_seconds is not None
        and daily_period_steps is not None
        and k_positions is not None
    )

    means_per_pos = stds_per_pos = pred_positions = None
    global_mean = global_std = None

    if ref_residuals is not None:
        ref_arr = np.asarray(ref_residuals, dtype=float)
        valid_ref = ref_arr[~np.isnan(ref_arr)]
        global_mean = float(np.nanmean(valid_ref)) if valid_ref.size > 0 else 0.0
        global_std = float(np.nanstd(valid_ref, ddof=1)) if valid_ref.size > 1 else None

        if use_conditional:
            ref_positions = compute_position_in_period(
                ref_timestamps, periodicity_seconds, daily_period_steps
            )
            pred_positions = compute_position_in_period(
                df_predict["timestamp_utc"].values,
                periodicity_seconds, daily_period_steps,
            )
            means_per_pos, stds_per_pos, _, _ = compute_per_position_stats(
                ref_arr, ref_positions, daily_period_steps, k_positions
            )
            result["ref_mean_per_position"] = means_per_pos.tolist()
            result["ref_std_per_position"] = stds_per_pos.tolist()
            result["k_positions"] = k_positions
        result["ref_mean"] = global_mean
        result["ref_std"] = global_std
    else:
        result["ref_mean"] = None
        result["ref_std"] = None

    t_start_pred = time.time()
    try:
        model_pred = base.UnobservedComponents(
            endog=np.zeros(n),
            level="local level",
            seasonal=None,
            freq_seasonal=freq_seasonal,
            stochastic_level=True,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
        )
        model_pred.update(theta)

        def get_mat(name):
            m = model_pred.ssm[name]
            return m[:, :, 0] if m.ndim == 3 else m

        T = get_mat('transition')
        Z = get_mat('design')
        R = get_mat('selection')
        Q = get_mat('state_cov')
        H = get_mat('obs_cov')

        Q_full = R @ Q @ R.T
        k = T.shape[0]
        I = np.eye(k)

        x = init_state_mean.copy().reshape(-1)
        P = init_state_cov.copy()

        predictions = np.full(n, np.nan)

        for t in range(n):
            x_prior = T @ x
            P_prior = T @ P @ T.T + Q_full

            # .item() not float(): Z @ x_prior has shape (1,), and NumPy >= 2
            # refuses float() on any non-0-d array. Value-identical.
            y_hat = (Z @ x_prior).item()
            # Valid in both spaces: raw Diff >= 0, and log1p(Diff/s) >= 0 too.
            predictions[t] = max(y_hat, 0.0)

            y_t = y_pred_segment[t]
            skip_update = bool(np.isnan(y_t))

            if not skip_update:
                if use_conditional and pred_positions is not None:
                    p = pred_positions[t]
                    mu = means_per_pos[p]
                    sd = stds_per_pos[p]
                    if np.isnan(mu) or np.isnan(sd) or sd == 0:
                        mu, sd = global_mean, global_std
                else:
                    mu, sd = global_mean, global_std

                if sd is not None and sd > 0:
                    residual = (y_t - y_hat) if signed_bands else abs(y_t - y_hat)
                    z = (residual - mu) / sd
                    # Two-sided regardless of detection mode: an anomalously LOW
                    # reading corrupts the state as much as a high one, and
                    # freezing the state here is what keeps a sustained deficit
                    # from being silently absorbed by the local level.
                    if abs(z) > mask_z_threshold:
                        skip_update = True

            if skip_update:
                x = x_prior
                P = P_prior
            else:
                S = (Z @ P_prior @ Z.T + H).item()
                K = (P_prior @ Z.T) / S
                innovation = y_t - y_hat
                x = x_prior + K.flatten() * innovation
                IKZ = I - K @ Z
                P = IKZ @ P_prior @ IKZ.T + K * H[0, 0] @ K.T

    except Exception as e:
        raise ValueError(f"Failed to generate online predictions: {e}")

    result["prediction_time_seconds"] = time.time() - t_start_pred
    return predictions


# ============================================================================
# RESULTS FRAME (scores in scoring space, reports metrics in RAW units)
# ============================================================================


def build_results_df(df_predict_raw, predictions_scoring, transform, result,
                     ref_residuals=None, ref_timestamps=None,
                     periodicity_seconds=None, daily_period_steps=None,
                     k_positions=None, threshold_z_score=3.5,
                     signed_bands=False, save_detection_bands=False):
    """Assemble the per-reading frame.

    `actual` / `predicted` / `residual` are emitted in RAW units so that MAE,
    RMSE and MASE remain comparable across variants; `z_score` is computed in
    scoring space, which is where the transform is supposed to help.

    With `save_detection_bands=True` the frame also carries `upper_band` /
    `lower_band` in RAW units, for plotting. The bands are derived in SCORING
    space and then pushed through `transform.inv`, which is monotone, so the
    drawn envelope is exactly the region the detector treats as normal — under
    a log transform it is visibly asymmetric in raw units, which is the point.
    """
    actual_raw = np.asarray(df_predict_raw["Diff"].values, dtype=float)
    timestamps = pd.to_datetime(df_predict_raw["timestamp_utc"].values, utc=True)

    actual_scoring = transform.fwd(actual_raw)
    predicted_scoring = np.asarray(predictions_scoring, dtype=float)
    predicted_raw = transform.inv(predicted_scoring)

    if signed_bands:
        residual_scoring = actual_scoring - predicted_scoring
    else:
        residual_scoring = np.abs(actual_scoring - predicted_scoring)

    residual_raw = np.abs(actual_raw - predicted_raw)

    if 'is_anomaly' in df_predict_raw.columns:
        anomalies = (np.asarray(df_predict_raw["is_anomaly"].values) != 0.0).astype(int)
    else:
        anomalies = np.zeros(len(df_predict_raw), dtype=int)

    result_df = pd.DataFrame({
        "timestamp_utc": timestamps,
        "actual": actual_raw,
        "predicted": predicted_raw,
        "residual": residual_raw,
    })

    use_conditional = (
        ref_residuals is not None
        and ref_timestamps is not None
        and periodicity_seconds is not None
        and daily_period_steps is not None
        and k_positions is not None
    )

    if use_conditional:
        pred_positions = compute_position_in_period(
            timestamps, periodicity_seconds, daily_period_steps
        )
        ref_positions = compute_position_in_period(
            ref_timestamps, periodicity_seconds, daily_period_steps
        )
        result_df["position_in_period"] = pred_positions
    else:
        pred_positions = ref_positions = None

    z_score, _, mu_used, sd_used, _, _ = base.compute_z_scores(
        residual_scoring,
        result,
        ref_residuals=ref_residuals,
        pred_positions=pred_positions,
        ref_positions=ref_positions,
        daily_period_steps=daily_period_steps,
        k_positions=k_positions,
    )

    result_df["z_score"] = z_score
    result_df["is_anomaly_actual"] = anomalies

    if save_detection_bands:
        # Retained so the bands can be RECOMPUTED at any threshold without
        # re-running the model — this is what makes the notebook's threshold
        # control honest rather than just re-colouring the markers.
        result_df["predicted_scoring"] = predicted_scoring
        result_df["band_mu"] = mu_used
        result_df["band_sd"] = sd_used

        # Half-width of the acceptance region, in scoring space.
        half = mu_used + threshold_z_score * sd_used
        if signed_bands:
            # Upper-only: the acceptance region is everything below the upper
            # bound, so the floor is simply 0 (Diff is non-negative). Drawing a
            # lower boundary here would imply a test that is not being applied.
            upper_scoring = predicted_scoring + half
            lower_scoring = np.zeros_like(upper_scoring)
        else:
            upper_scoring = predicted_scoring + half
            lower_scoring = predicted_scoring - half
        result_df["upper_band"] = transform.inv(upper_scoring)
        result_df["lower_band"] = np.clip(transform.inv(lower_scoring), 0.0, None)

    # Upper-only when signed: every injected anomaly is a positive excursion, so
    # the lower tail can only contribute false positives. Note this changes what
    # `threshold_z_score` MEANS (a signed band at k is ~15x stricter than a
    # folded one at the same k), so variants must be compared on the PR curve /
    # at matched alert volume, never at matched k.
    if signed_bands:
        result_df["is_anomaly_predicted"] = (z_score > threshold_z_score).astype(int)
    else:
        result_df["is_anomaly_predicted"] = (
            np.abs(z_score) > threshold_z_score
        ).astype(int)

    # The robust/MAD path was retired (residuals are right-skewed); the column
    # is kept only so calculate_metrics' expected contract still holds.
    result_df["is_anomaly_robust_predicted"] = 0

    result['number_of_anomalies_actual'] = int(result_df["is_anomaly_actual"].sum())
    result['number_of_anomalies'] = int(result_df["is_anomaly_predicted"].sum())

    return result_df


# ============================================================================
# DIAGNOSTIC HELPERS
# ============================================================================


def _max_real_gap_days(*segments):
    """Largest gap (days) between consecutive REAL readings across segments.

    Synthetic gap-fill rows carry hodnota = NaN, so they are excluded. The
    baseline records nothing comparable, which is why production's
    `max_single_gap_days` gate cannot be reconstructed post-hoc today.
    """
    best = np.nan
    for df in segments:
        if "hodnota" not in df.columns or "timestamp_utc" not in df.columns:
            continue
        real = df.loc[df["hodnota"].notna(), "timestamp_utc"]
        if len(real) < 2:
            continue
        gap = pd.to_datetime(real).sort_values().diff().max()
        if pd.isna(gap):
            continue
        days = gap.total_seconds() / 86400.0
        best = days if (np.isnan(best) or days > best) else best
    return best


def _add_prod_gate_flags(result):
    """Record uc-cem's initial-fit gates as booleans (diagnostic only).

    Structural gates are model-independent and therefore safe to filter on
    per-variant; `would_pass_prod_mase` is model-DEPENDENT and is kept separate
    precisely so it does not get used as a cohort filter.
    """
    def _fin(v):
        return v is not None and np.isfinite(v)

    fs = result.get("f_s_daily_second", np.nan)
    fill = result.get("filled_pct_second", np.nan)
    mase = result.get("mase_seasonal", np.nan)
    gap = result.get("max_single_gap_days", np.nan)
    steps = result.get("daily_period_steps", 0) or 0

    result["would_pass_prod_fs"] = bool(_fin(fs) and fs > PROD_FS_MIN)
    result["would_pass_prod_fill"] = bool(_fin(fill) and fill < PROD_FILL_MAX_PCT)
    result["would_pass_prod_periodicity"] = bool(int(steps) >= PROD_DAILY_STEPS_MIN)
    result["would_pass_prod_gap"] = bool(_fin(gap) and gap <= PROD_MAX_GAP_DAYS)
    result["would_pass_prod_structural"] = bool(
        result["would_pass_prod_fs"]
        and result["would_pass_prod_fill"]
        and result["would_pass_prod_periodicity"]
        and result["would_pass_prod_gap"]
    )
    # Model-dependent — deliberately NOT folded into the structural flag.
    result["would_pass_prod_mase"] = bool(_fin(mase) and mase < PROD_MASE_MAX)


# ============================================================================
# SINGLE METER
# ============================================================================


def process_single_meter(
    csv_filepath,
    transform="raw",
    harmonics_daily_max=None,
    signed_bands=False,
    threshold_z_score=3.5,
    mask_z_threshold=None,
    k_hours=2.0,
    min_k_positions=2,
    variant="baseline",
    verbose=False,
    return_predictions=False,
    injection_plan=None,
    sustained_detectors=False,
    flag_run_threshold=3,
    cusum_k=0.5,
    cusum_h=5.0,
):
    """Run one meter end-to-end under a given variant configuration.

    Defaults (`transform="raw"`, `harmonics_daily_max=None`, `signed_bands=False`,
    `mask_z_threshold=None` -> equals detection threshold, `injection_plan=None`
    -> the original spike injector) reproduce the baseline script exactly.

    `sustained_detectors=True` additionally runs the read-only detectors in
    `helper_scripts/sustained_detectors.py`. They add columns and counts; they
    never alter `is_anomaly_predicted` or the masking rule, so the baseline
    numbers are identical whether or not they are enabled.

    `injection_plan` is an `anomaly_injection.InjectionPlan` selecting one
    sustained archetype at a given depth and duration. It replaces the spike
    injector for that run and adds event-level results (`n_events`,
    `event_recall`, `ttd_*`, `recovery_clean`) to the returned dict. It changes
    only what is fed in, never how scoring works.
    """
    if mask_z_threshold is None:
        mask_z_threshold = threshold_z_score

    result = {
        "filename": Path(csv_filepath).stem,
        "filepath": csv_filepath,
        "variant": variant,
        "status": "processing",
        "error": None,
        "opt_transform": transform or "raw",
        "opt_harmonics_daily_max": harmonics_daily_max,
        "opt_signed_bands": bool(signed_bands),
        "opt_threshold_z_score": threshold_z_score,
        "opt_mask_z_threshold": mask_z_threshold,
        "opt_sustained_detectors": bool(sustained_detectors),
    }

    try:
        df_raw = pd.read_csv(csv_filepath)
        for col in ("timestamp_utc", "hodnota"):
            if col not in df_raw.columns:
                raise ValueError(
                    f"No '{col}' column. Available: {df_raw.columns.tolist()}"
                )
        if len(df_raw) < 2:
            raise ValueError("DataFrame passed is too short len < 2")

        df_raw["timestamp_utc"] = pd.to_datetime(df_raw["timestamp_utc"], utc=True)
        df_raw = df_raw.dropna(subset=["timestamp_utc"]).copy()

        # ---- split (identical windows/seed to baseline, so runs stay paired) --
        seed = int(result['filename'])
        df_train, df_second, df_predict, _ = split_df_sliding_weeks(
            df_raw=df_raw, seed=seed, result=result
        )
        periodicity_seconds = result["periodicity_seconds"]
        daily_period_steps = result["daily_period_steps"]

        if daily_period_steps < 2:
            raise ValueError(
                "Daily period in measurements is < 2 (periodicity too coarse)."
            )

        weekly_period_steps = 7 * daily_period_steps
        h_daily = resolve_daily_harmonics(daily_period_steps, harmonics_daily_max)
        result["harmonics_daily_used"] = h_daily

        freq_seasonal = [
            {"period": weekly_period_steps, "harmonics": 2},
            {"period": daily_period_steps, "harmonics": h_daily},
        ]
        stochastic_freq_seasonal = [True, True]

        k_positions = compute_k_positions(
            daily_period_steps, k_hours=k_hours, min_k=min_k_positions
        )
        result["k_positions"] = k_positions
        result["k_hours"] = k_hours

        # ---- scale transform, from the CLEAN calibration segment only --------
        tf = build_transform(transform, df_second, result)

        df_train_s = _transformed(df_train, tf)
        df_second_s = _transformed(df_second, tf)

        # ---- two-stage warm-start training ----------------------------------
        last_state_mean, last_state_cov, theta_first = train_model(
            df_train=df_train_s, result=result,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            initial_train=True,
        )

        last_state_mean_2, last_state_cov_2, theta_second = train_model(
            df_train=df_second_s, result=result,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            initial_train=False,
            init_state_mean=last_state_mean,
            init_state_cov=last_state_cov,
        )

        # ---- reference residuals from the clean calibration segment ----------
        second_pred_s = predict_model_batch(
            df_predict=df_second_s, result=result,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            init_state_mean=last_state_mean,
            init_state_cov=last_state_cov,
            theta=theta_second,
        )
        second_actual_s = np.asarray(df_second_s["Diff"].values, dtype=float)
        if signed_bands:
            second_residuals = second_actual_s - second_pred_s
        else:
            second_residuals = np.abs(second_actual_s - second_pred_s)

        # ---- inject anomalies (RAW space, before transform) ------------------
        # Same point in the flow as before: after the calibration residuals are
        # fixed, before the transform and the online pass. `injection_plan=None`
        # keeps the original spike injector verbatim, which is what makes the
        # baseline-equivalence check in test_uc_variants.py still meaningful.
        #
        # The sustained injectors need the sensor's own scale. It is taken from
        # the CALIBRATION segment, never from df_predict — deriving it from the
        # segment about to be injected into would let the anomaly set its own
        # magnitude unit. Same reasoning as the log1p transform's scale.
        injected_events = []
        if injection_plan is None:
            df_predict = inject_spike_anomalies_diff(
                df_predict, random_state=int(result['filename'])
            )
        else:
            inj_scale = ai.sensor_scale(df_second["Diff"].values)
            result["injection_scale"] = float(inj_scale)
            df_predict, injected_events = ai.apply_injection_plan(
                df_predict, injection_plan,
                scale=inj_scale,
                periodicity_seconds=periodicity_seconds,
                seed=int(result['filename']),
            )
            # Interval labels -> the same per-row column calculate_metrics()
            # already consumes, so point-wise scoring is unaffected.
            lbl, _ = em.label_rows_from_events(
                df_predict["timestamp_utc"].values, injected_events
            )
            df_predict = df_predict.copy()
            df_predict["is_anomaly"] = lbl.astype(float)

        df_predict_s = _transformed(df_predict, tf)

        # ---- online prediction ----------------------------------------------
        predictions_s = predict_model_online(
            df_predict=df_predict_s, result=result,
            freq_seasonal=freq_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            init_state_mean=last_state_mean_2,
            init_state_cov=last_state_cov_2,
            theta=theta_second,
            ref_residuals=second_residuals,
            ref_timestamps=df_second["timestamp_utc"].values,
            periodicity_seconds=periodicity_seconds,
            daily_period_steps=daily_period_steps,
            k_positions=k_positions,
            mask_z_threshold=mask_z_threshold,
            signed_bands=signed_bands,
        )

        predictions_df = build_results_df(
            df_predict, predictions_s, tf, result,
            ref_residuals=second_residuals,
            ref_timestamps=df_second["timestamp_utc"].values,
            periodicity_seconds=periodicity_seconds,
            daily_period_steps=daily_period_steps,
            k_positions=k_positions,
            threshold_z_score=threshold_z_score,
            signed_bands=signed_bands,
            save_detection_bands=return_predictions,
        )

        # Stored WITH labels so precision/recall at any detection threshold can
        # be recomputed post-hoc — the sweep costs no re-runs.
        result["z_scores"] = predictions_df["z_score"].tolist()
        result["labels"] = predictions_df["is_anomaly_actual"].tolist()

        # ---- sustained detectors (opt-in, read-only) -------------------------
        # Calibrated on the CLEAN calibration segment and stepped across the
        # prediction segment one reading at a time. They emit their own columns
        # and never touch is_anomaly_predicted or the masking rule, so switching
        # them on cannot move the baseline numbers.
        if sustained_detectors:
            cal_signed = second_actual_s - second_pred_s   # signed, pre-fold
            cal_pos = compute_position_in_period(
                df_second["timestamp_utc"].values, periodicity_seconds,
                daily_period_steps)
            pred_pos = compute_position_in_period(
                predictions_df["timestamp_utc"].values, periodicity_seconds,
                daily_period_steps)
            dets = sd.calibrate_all(
                df_second["timestamp_utc"].values,
                np.asarray(df_second["Diff"].values, dtype=float),
                cal_signed,
                cal_positions=cal_pos,
                daily_steps=daily_period_steps,
                flag_run_threshold=flag_run_threshold,
                cusum_k=cusum_k, cusum_h=cusum_h,
            )
            det_df = sd.run_detectors(
                dets,
                predictions_df["timestamp_utc"].values,
                predictions_df["actual"].values,
                predictions_df["predicted"].values,
                predictions_df["is_anomaly_predicted"].values.astype(bool),
                positions=pred_pos,
            )
            for c in det_df.columns:
                predictions_df[c] = det_df[c].values

            result["n_sustained_alerts"] = int(det_df["sustained_alert"].sum())
            result["n_stuck"] = int(det_df["stuck"].sum())
            result["n_drift"] = int(det_df["drift"].sum())
            result["n_seasonal_drift"] = int(det_df.get(
                "seasonal_drift", pd.Series(dtype=bool)).sum())
            result["n_anomaly_sustained"] = int(det_df["anomaly_sustained"].sum())
            result["n_night_flow_elevated"] = int(det_df["night_flow_elevated"].sum())
            result["stuck_threshold"] = int(dets["stuck"].threshold)
            result["night_flow_baseline"] = float(dets["night"].baseline)

        # ---- cross-fit drift (training-time diagnostic) ----------------------
        result.update(sd.crossfit_drift(
            theta_first, theta_second,
            mase_first=result.get("mase_seasonal_first", np.nan),
            mase_second=result.get("mase_seasonal", np.nan),
        ))

        # ---- event-level results (only when a sustained plan was injected) ---
        # Point-wise recall treats a caught 48h leak as ~96% misses; event recall
        # and time-to-detect are what actually separate detectors here. Both
        # views come from the same run — nothing point-wise is displaced.
        if injected_events:
            ev_summary, ev_df = em.summarize_run(
                predictions_df["timestamp_utc"].values,
                predictions_df["is_anomaly_predicted"].values.astype(bool),
                injected_events,
                periodicity_seconds=periodicity_seconds,
            )
            result.update(ev_summary)

            # Same events scored against the sustained detectors' union, so the
            # gain from adding them is a direct delta on one run rather than a
            # comparison across two. Prefixed, never overwriting the z-score
            # numbers above.
            if sustained_detectors and "sustained_alert" in predictions_df.columns:
                comb = (predictions_df["is_anomaly_predicted"].astype(bool)
                        | predictions_df["sustained_alert"].astype(bool))
                for key, flags_col in (("sust", predictions_df["sustained_alert"].astype(bool)),
                                       ("comb", comb)):
                    s_sum, _ = em.summarize_run(
                        predictions_df["timestamp_utc"].values,
                        flags_col.values,
                        injected_events,
                        periodicity_seconds=periodicity_seconds,
                    )
                    for k, v in s_sum.items():
                        result[f"{key}_{k}"] = v

            result["events"] = [e.to_row() for e in injected_events]
            result["event_rows"] = ev_df.to_dict("records")
            result["injection_archetype"] = injection_plan.archetype
            result["injection_depth"] = float(injection_plan.depth)
            result["injection_duration_h"] = float(injection_plan.duration_h)
        elif injection_plan is not None:
            # holiday_profile injects a shape change but labels nothing, so an
            # empty event list is the expected outcome, not a failure. Its FP
            # count is the entire point of the probe, so it is still recorded —
            # with no events, every flagged reading is by definition outside one.
            fp_out, n_clean = em.false_alarm_count(
                predictions_df["timestamp_utc"].values,
                predictions_df["is_anomaly_predicted"].values.astype(bool),
                [],
            )
            result["n_events"] = 0
            result["event_recall"] = np.nan
            result["fp_outside_events"] = fp_out
            result["n_clean_readings"] = n_clean
            result["recovery_clean"] = True   # no event to recover from
            result["injection_archetype"] = injection_plan.archetype
            result["injection_depth"] = float(injection_plan.depth)
            result["injection_duration_h"] = float(injection_plan.duration_h)

        # ---- diagnostics: always RAW, so gate booleans mean the same thing ---
        second_diff = np.asarray(df_second["Diff"].values, dtype=float)

        inter = compute_intermittency(second_diff)
        result["adi_second"] = inter["adi"]
        result["cv2_second"] = inter["cv2"]
        result["nonzero_fraction_second"] = inter["nonzero_fraction"]

        sd_diag = compute_seasonal_diagnostics(
            second_diff, daily_period_steps, weekly_period_steps
        )
        result["f_s_daily_second"] = sd_diag["f_s_daily"]
        result["f_s_weekly_second"] = sd_diag["f_s_weekly"]
        result["resid_cv2_second"] = sd_diag["resid_cv2"]

        def _filled_pct(df_seg):
            if "hodnota" not in df_seg.columns or len(df_seg) == 0:
                return np.nan
            return 100.0 * float(df_seg["hodnota"].isna().mean())

        result["filled_pct_train"] = _filled_pct(df_train)
        result["filled_pct_second"] = _filled_pct(df_second)
        result["filled_pct_predict"] = _filled_pct(df_predict)

        result["max_single_gap_days"] = _max_real_gap_days(df_train, df_second)

        clean = predictions_df["is_anomaly_actual"].values == 0
        mase = compute_mase(
            predictions_df.loc[clean, "actual"].values,
            predictions_df.loc[clean, "predicted"].values,
            second_diff,
            m_seasonal=daily_period_steps,
        )
        result["mae_model_clean"] = mase["mae_model"]
        result["naive_mae_1_second"] = mase["naive_mae_1"]
        result["naive_mae_seasonal_second"] = mase["naive_mae_seasonal"]
        result["mase"] = mase["mase"]
        result["mase_seasonal"] = mase["mase_seasonal"]

        # Production gates as BOOLEANS — recorded, never applied.
        _add_prod_gate_flags(result)

        if len(predictions_df) < 2:
            raise ValueError("Less than 2 data points in prediction df for metrics.")

        for key, value in calculate_metrics(predictions_df).items():
            result[f"metric_{key}"] = value

        result["status"] = "success"
        if return_predictions:
            # Kept out of `result` so the batch runner's summary row stays flat.
            result["_predictions_df"] = predictions_df
        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Failed: {result['filename']} [{variant}] - {e}")
        return result


# ============================================================================
# VARIANT REGISTRY
# ============================================================================

VARIANTS = {
    # Exact reproduction of local_level_unobserved_components_next_one_step_pred.py
    "baseline": dict(),
    # --- single effects -----------------------------------------------------
    "log1p": dict(transform="log1p_scaled"),
    "harmonics4": dict(harmonics_daily_max=4),
    "signed": dict(signed_bands=True),
    # --- interactions -------------------------------------------------------
    # log1p makes residuals roughly symmetric, which is the assumption signed
    # bands need; they may only pay off together.
    "log1p_signed": dict(transform="log1p_scaled", signed_bands=True),
    "all3": dict(
        transform="log1p_scaled", harmonics_daily_max=4, signed_bands=True
    ),
}


def run_variant(csv_filepath, variant, **overrides):
    """Run one meter under a named variant. Picklable for ProcessPoolExecutor."""
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}. Known: {sorted(VARIANTS)}")
    kwargs = dict(VARIANTS[variant])
    kwargs.update(overrides)
    return process_single_meter(csv_filepath, variant=variant, **kwargs)


def run_variant_with_frame(csv_filepath, variant, **overrides):
    """Run one meter and return `(result, predictions_df)`.

    The frame carries per-reading actual/predicted/z_score/labels plus
    `upper_band` / `lower_band` in raw units — everything the inspection
    notebook needs to draw a single sensor's week.
    """
    overrides.setdefault("return_predictions", True)
    res = run_variant(csv_filepath, variant, **overrides)
    return res, res.pop("_predictions_df", None)
