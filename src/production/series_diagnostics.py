"""
Per-sensor series diagnostics for deciding where the UC model is appropriate.

A-priori characteristics on the *clean* second training segment:

    - ADI / CV2       : Syntetos-Boylan intermittency features (RAW, pooled).
                        Kept for reference, but for strongly day/night-seasonal
                        water data they conflate predictable structural zeros /
                        daily value-swings with genuine irregularity. Prefer the
                        residual-based pair below for gating.

    - F_S             : seasonal strength (daily, weekly), Wang-Smith-Hyndman.
    - resid_cv2       : DESEASONALISED CV^2 — dispersion of the decomposition
                        remainder relative to typical consumption. The
                        "is it forecastable" companion to F_S: it ignores the
                        repeating daily/weekly shape and only measures the
                        left-over, unpredictable variation.

Skill score on the prediction segment:

    - MASE            : model MAE / in-sample naive MAE (lag-1 and seasonal).

NO thresholds / class cutoffs are applied here. Raw numbers only.
"""

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL, MSTL


# ---------------------------------------------------------------------------
# Intermittency: RAW ADI and CV^2 (pooled, Syntetos-Boylan) — informational
# ---------------------------------------------------------------------------
def compute_intermittency(diff_values, zero_tol=0.0):
    """
    ADI and pooled CV^2 on the observed (non-NaN) Diff values.

    A NaN is a *missing* reading (not counted as a period); a genuine 0 is a
    no-consumption period and IS counted.

        ADI  = n_periods / n_nonzero
        CV^2 = (std / mean)^2  of the non-zero demand sizes (sample std)

    NOTE: pooled across all times of day, so for a day/night-seasonal sensor
    the predictable nightly zeros inflate ADI and the predictable daily
    value-range inflates CV^2. Use resid_cv2 + F_S for gating such sensors.
    """
    x = np.asarray(diff_values, dtype=float)
    x = x[~np.isnan(x)]
    n_periods = int(x.size)

    if n_periods == 0:
        return {"adi": np.nan, "cv2": np.nan, "n_periods": 0,
                "n_nonzero": 0, "nonzero_fraction": np.nan}

    nz = x[x > zero_tol]
    n_nonzero = int(nz.size)

    if n_nonzero == 0:
        return {"adi": np.inf, "cv2": np.nan, "n_periods": n_periods,
                "n_nonzero": 0, "nonzero_fraction": 0.0}

    adi = n_periods / n_nonzero
    mean_nz = float(nz.mean())
    cv2 = np.nan if mean_nz == 0.0 else (
        (float(nz.std(ddof=1)) if n_nonzero > 1 else 0.0) / mean_nz) ** 2

    return {"adi": float(adi), "cv2": float(cv2), "n_periods": n_periods,
            "n_nonzero": n_nonzero, "nonzero_fraction": n_nonzero / n_periods}


# ---------------------------------------------------------------------------
# Seasonal strength F_S + deseasonalised residual CV^2 (one decomposition)
# ---------------------------------------------------------------------------
def _seasonal_strength(component, remainder):
    """F_S = max(0, 1 - Var(R) / Var(S + R))."""
    component = np.asarray(component, dtype=float)
    remainder = np.asarray(remainder, dtype=float)
    var_r = np.nanvar(remainder)
    var_cr = np.nanvar(component + remainder)
    if not np.isfinite(var_cr) or var_cr <= 0:
        return 0.0
    return float(max(0.0, 1.0 - var_r / var_cr))


def compute_seasonal_diagnostics(diff_values, daily_period_steps,
                                 weekly_period_steps=None,
                                 max_len_for_weekly=20000, robust=False):
    """
    One STL/MSTL decomposition -> daily & weekly seasonal strength AND a
    deseasonalised residual CV^2.

        F_S              : max(0, 1 - Var(R)/Var(S+R)) per seasonal component.
        resid_cv2        : (std(R_obs) / mean_nonzero_obs)^2
        resid_cv2_robust : ((1.4826 * MAD(R_obs)) / median_nonzero_obs)^2

    R is the decomposition remainder (series - trend - all seasonal terms).
    Dispersion is measured over ORIGINALLY-OBSERVED positions only, so
    interpolated gap-fills don't deflate it. The denominator is typical
    non-zero consumption, mirroring the raw CV^2 so the two are comparable -
    but with the predictable daily/weekly shape stripped out of the numerator.

    NaN gaps are linearly interpolated FOR THE DECOMPOSITION ONLY. A period is
    used only if the series is long enough (>= 2*period + 1); the weekly period
    is skipped beyond `max_len_for_weekly` to bound runtime.

    Returns {"f_s_daily","f_s_weekly","resid_cv2","resid_cv2_robust"}.
    """
    out = {"f_s_daily": np.nan, "f_s_weekly": np.nan,
           "resid_cv2": np.nan, "resid_cv2_robust": np.nan}

    raw = np.asarray(diff_values, dtype=float)
    observed = ~np.isnan(raw)

    s = pd.Series(raw).interpolate(limit_direction="both")
    if s.isna().any():
        s = s.fillna(0.0)
    x = s.values
    n = x.size

    daily = int(daily_period_steps) if daily_period_steps else None
    weekly = int(weekly_period_steps) if weekly_period_steps else None

    periods = []
    if daily and daily >= 2 and n >= 2 * daily + 1:
        periods.append(daily)
    if (weekly and weekly >= 2 and weekly != daily
            and n >= 2 * weekly + 1 and n <= max_len_for_weekly):
        periods.append(weekly)
    periods = sorted(set(periods))

    if not periods:
        return out

    try:
        if len(periods) == 1:
            res = STL(x, period=periods[0], robust=robust).fit()
            comp = {periods[0]: np.asarray(res.seasonal)}
            resid = np.asarray(res.resid)
        else:
            res = MSTL(x, periods=periods).fit()
            seas = np.asarray(res.seasonal)          # (n, n_periods), period-sorted
            resid = np.asarray(res.resid)
            comp = {p: seas[:, i] for i, p in enumerate(periods)}

        if daily in comp:
            out["f_s_daily"] = _seasonal_strength(comp[daily], resid)
        if weekly in comp:
            out["f_s_weekly"] = _seasonal_strength(comp[weekly], resid)

        # --- deseasonalised residual CV^2 (observed positions only) ----------
        resid_obs = resid[observed]
        resid_obs = resid_obs[~np.isnan(resid_obs)]
        nz = raw[observed]
        nz = nz[(~np.isnan(nz)) & (nz > 0.0)]

        if resid_obs.size > 1 and nz.size > 0:
            mean_nz = float(nz.mean())
            if mean_nz > 0:
                out["resid_cv2"] = (float(resid_obs.std(ddof=1)) / mean_nz) ** 2
            med_nz = float(np.median(nz))
            if med_nz > 0:
                mad_r = float(np.median(np.abs(resid_obs - np.median(resid_obs))))
                out["resid_cv2_robust"] = ((1.4826 * mad_r) / med_nz) ** 2
    except Exception:
        pass  # leave NaNs

    return out


def compute_seasonal_strength(diff_values, daily_period_steps,
                              weekly_period_steps=None,
                              max_len_for_weekly=20000, robust=False):
    """Backward-compatible wrapper: returns only the two F_S values."""
    d = compute_seasonal_diagnostics(
        diff_values, daily_period_steps, weekly_period_steps,
        max_len_for_weekly=max_len_for_weekly, robust=robust,
    )
    return {"f_s_daily": d["f_s_daily"], "f_s_weekly": d["f_s_weekly"]}


# ---------------------------------------------------------------------------
# MASE: skill relative to an in-sample naive forecast
# ---------------------------------------------------------------------------
def _naive_mae(y, lag):
    """In-sample MAE of a lag-`lag` naive forecast, NaN-safe."""
    y = np.asarray(y, dtype=float)
    if lag is None or lag < 1 or y.size <= lag:
        return np.nan
    d = np.abs(y[lag:] - y[:-lag])
    d = d[~np.isnan(d)]
    return float(d.mean()) if d.size > 0 else np.nan


def compute_mase(actual_pred, predicted, insample_actual, m_seasonal=None):
    """
    MASE = MAE(model on prediction segment) / MAE(naive on in-sample segment).

    `insample_actual` is the clean in-sample series (second training Diff) used
    for the naive scaling denominator. Both lag-1 and seasonal-naive
    (lag = `m_seasonal`) scalings are returned.

    Returns: mae_model, naive_mae_1, naive_mae_seasonal, mase, mase_seasonal.
    """
    a = np.asarray(actual_pred, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ae = np.abs(a - p)
    ae = ae[~np.isnan(ae)]
    mae_model = float(ae.mean()) if ae.size > 0 else np.nan

    naive_mae_1 = _naive_mae(insample_actual, 1)
    naive_mae_seasonal = _naive_mae(insample_actual, m_seasonal) if m_seasonal else np.nan

    def _ratio(num, den):
        if not np.isfinite(num) or not np.isfinite(den) or den == 0:
            return np.nan
        return num / den

    return {
        "mae_model": mae_model,
        "naive_mae_1": naive_mae_1,
        "naive_mae_seasonal": naive_mae_seasonal,
        "mase": _ratio(mae_model, naive_mae_1),
        "mase_seasonal": _ratio(mae_model, naive_mae_seasonal),
    }
