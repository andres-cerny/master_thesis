"""
test_uc_variants.py — Verification suite for the variant harness.
=================================================================

Run this before trusting any variant comparison:

    python test_uc_variants.py --fixtures /path/to/real_segments

The load-bearing check is #1. If `uc_variants` with default arguments does not
reproduce the baseline script bit-for-bit, then every measured "improvement" is
confounded with an accidental refactor difference and the whole comparison is
worthless.

Exit code 0 = all passed, 1 = something failed.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import local_level_unobserved_components_next_one_step_pred as base
import uc_variants as uv
import analyze_variants as av

_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from helper_scripts import anomaly_injection as ai
from helper_scripts import event_metrics as em
from helper_scripts import sustained_detectors as sdet

DEFAULT_FIXTURES = "/home/user/uc-cem/testing_scripts/fixtures/real_segments"

_PASS, _FAIL = [], []


def check(name, ok, detail=""):
    (_PASS if ok else _FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def require_success(result, label):
    """Gate every per-sensor assertion on the run having actually succeeded.

    Without this, a run that fails for an environmental reason (a NumPy 2
    incompatibility did exactly this) is silently `continue`d, and comparisons
    between two failed results report PASS. The suite then claims to be green
    while measuring nothing. A failed run is a test failure, not a skip.
    """
    if result.get("status") == "success":
        return True
    check(f"{label}: run succeeded", False, str(result.get("error"))[:120])
    return False


# ============================================================================


def test_baseline_equivalence(files):
    """Defaults must reproduce the baseline script exactly.

    Known intentional difference: the retired robust/MAD path is dropped, so
    only the classical columns are compared.
    """
    print("\n[1] baseline equivalence (THE linchpin)")
    all_ok = True
    for f in files:
        name = Path(f).stem
        a = base.process_single_meter(f)
        b = uv.process_single_meter(f)

        if a["status"] != b["status"]:
            all_ok &= check(f"{name}: status", False, f'{a["status"]} vs {b["status"]}')
            continue
        if not require_success(a, f"{name}/base"):
            all_ok = False
            continue

        za = np.asarray(a.get("z_scores", []), dtype=float)
        zb = np.asarray(b.get("z_scores", []), dtype=float)
        ok = za.shape == zb.shape and np.allclose(za, zb, equal_nan=True, rtol=0, atol=0)
        all_ok &= check(f"{name}: z-scores bit-identical", ok,
                        f"n={za.size}" if ok else f"{za.shape} vs {zb.shape}")

        for k in ("metric_pred_tp", "metric_pred_fp", "metric_pred_fn",
                  "metric_pred_tn", "metric_mae", "mase_seasonal",
                  "f_s_daily_second", "daily_period_steps", "periodicity_seconds"):
            va, vb = a.get(k), b.get(k)
            ok = (va == vb) or (
                isinstance(va, float) and isinstance(vb, float)
                and np.isclose(va, vb, equal_nan=True)
            )
            if not ok:
                all_ok &= check(f"{name}: {k}", False, f"{va} vs {vb}")
    if all_ok:
        check("all fixtures reproduce baseline", True)
    return all_ok


def test_transforms():
    """Round-trip, zero preservation, NaN propagation, scale invariance."""
    print("\n[2] transform sanity")
    x = np.array([0.0, 0.001, 0.05, 0.5, 12.0, np.nan])

    raw = uv.RawSpace()
    check("raw round-trip", np.allclose(raw.inv(raw.fwd(x)), x, equal_nan=True))

    tf = uv.Log1pScaled(0.014)
    y = tf.fwd(x)
    check("log1p round-trip", np.allclose(tf.inv(y), x, equal_nan=True, rtol=1e-9))
    check("log1p maps 0 -> 0", y[0] == 0.0)
    check("log1p propagates NaN", np.isnan(y[-1]))
    check("log1p is monotone", np.all(np.diff(y[:-1]) > 0))

    # Scale normalisation is the whole point: a meter logging litres instead of
    # cubic metres must get the SAME transformed values.
    a = uv.Log1pScaled(0.014).fwd(np.array([0.014, 0.14, 1.4]))
    b = uv.Log1pScaled(14.0).fwd(np.array([14.0, 140.0, 1400.0]))
    check("log1p is scale-invariant across units", np.allclose(a, b))

    # Degenerate scale must not explode.
    check("log1p guards non-positive scale", uv.Log1pScaled(0.0).scale == 1.0)
    check("log1p guards NaN scale", uv.Log1pScaled(np.nan).scale == 1.0)
    return not _FAIL


def test_harmonics():
    """Adaptive harmonics must never breach Nyquist, and must match baseline
    behaviour when disabled."""
    print("\n[3] adaptive harmonics")
    check("None reproduces baseline literal 2",
          all(uv.resolve_daily_harmonics(d, None) == 2 for d in (5, 24, 96, 144)))

    ok = True
    for daily_steps in range(3, 300):
        for hmax in (2, 3, 4, 6):
            h = uv.resolve_daily_harmonics(daily_steps, hmax)
            # Highest Fourier frequency h/daily_steps must stay < 0.5 (Nyquist),
            # i.e. 2h + 1 <= daily_steps.
            if not (h >= 1 and 2 * h + 1 <= daily_steps + 1 and h <= hmax):
                ok = False
                print(f"      breach: daily_steps={daily_steps} hmax={hmax} -> h={h}")
    check("never breaches Nyquist, never exceeds cap", ok)
    check("caps at requested max on fine grids",
          uv.resolve_daily_harmonics(96, 4) == 4)
    check("degrades gracefully on coarse grids",
          uv.resolve_daily_harmonics(5, 4) == 2)
    return ok


def test_sweep_consistency(files):
    """Recomputing counts from the stored z-scores at a run's own threshold must
    reproduce that run's stored counts — otherwise the free sweep is a lie."""
    print("\n[4] threshold-sweep consistency")
    ok_all = True
    for variant in ("baseline", "signed", "log1p"):
        for f in files:
            r = uv.run_variant(f, variant)
            if not require_success(r, f"{Path(f).stem}/{variant}"):
                ok_all = False
                continue
            z = np.asarray(r["z_scores"], dtype=float)
            y = np.asarray(r["labels"], dtype=int)
            k = r["opt_threshold_z_score"]
            stat = z if r["opt_signed_bands"] else np.abs(z)
            valid = np.isfinite(stat)
            pred = stat[valid] > k
            tp = int(np.sum(pred & (y[valid] == 1)))
            fp = int(np.sum(pred & (y[valid] == 0)))
            ok = (tp == r["metric_pred_tp"]) and (fp == r["metric_pred_fp"])
            ok_all &= ok
            if not ok:
                check(f"{Path(f).stem}/{variant}", False,
                      f"recomputed tp={tp} fp={fp} vs stored "
                      f"{r['metric_pred_tp']}/{r['metric_pred_fp']}")
    if ok_all:
        check("stored counts reproducible from z-scores + labels", True)
    return ok_all


def test_signed_semantics(files):
    """Signed bands must be upper-only and strictly no looser than folded bands
    at the same k (a signed band at k is ~15x stricter for Gaussian residuals)."""
    print("\n[5] signed-band semantics")
    ok_all = True
    for f in files:
        r = uv.run_variant(f, "signed")
        if not require_success(r, f"{Path(f).stem}/signed"):
            ok_all = False
            continue
        z = np.asarray(r["z_scores"], dtype=float)
        k = r["opt_threshold_z_score"]
        flagged_neg = np.sum(z[np.isfinite(z)] < -k)
        # Upper-only: a large NEGATIVE z must never be flagged.
        ok = True  # detection rule is applied inside build_results_df
        n_pred = r["metric_pred_tp"] + r["metric_pred_fp"]
        n_upper = int(np.sum(z[np.isfinite(z)] > k))
        ok = (n_pred == n_upper)
        ok_all &= check(
            f"{Path(f).stem}: detections == upper-tail only", ok,
            f"{n_pred} flagged, {n_upper} above +k, {flagged_neg} below -k (ignored)"
        )
    return ok_all


def test_gate_flags_model_independent(files):
    """Structural gate flags must agree across variants — that is what makes the
    cohort slider safe. MASE flags may differ; that is why MASE is excluded."""
    print("\n[6] gate flags: structural invariance")
    structural = ["would_pass_prod_fs", "would_pass_prod_fill",
                  "would_pass_prod_periodicity", "would_pass_prod_gap"]
    ok_all = True
    for f in files:
        vals = {}
        for variant in ("baseline", "log1p", "harmonics4", "all3"):
            r = uv.run_variant(f, variant)
            if not require_success(r, f"{Path(f).stem}/{variant}"):
                continue
            vals[variant] = tuple(r[c] for c in structural)
        ok = len(set(vals.values())) <= 1
        ok_all &= check(f"{Path(f).stem}: structural flags identical across variants",
                        ok, "" if ok else str(vals))
    return ok_all


def test_analysis_layer(files):
    """The analyzer must never filter on MASE and must fix the cohort."""
    print("\n[7] analysis layer guarantees")
    import ast
    import inspect
    check("select_cohort accepts no mase argument",
          "mase" not in inspect.signature(av.select_cohort).parameters)

    # Assert the real invariant: no MASE column is ever passed to the filter
    # helper. Grepping the source would trip over the docstring that explains
    # exactly why MASE is excluded.
    tree = ast.parse(inspect.getsource(av.select_cohort).lstrip())
    filtered_cols = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "_keep"
        and node.args and isinstance(node.args[0], ast.Constant)
    }
    check("select_cohort filters only model-independent columns",
          not any("mase" in c for c in filtered_cols),
          f"filters on {sorted(filtered_cols)}")

    p, r, f1 = av._prf(10, 0, 0)
    check("_prf perfect case", np.isclose(p, 1.0) and np.isclose(r, 1.0) and np.isclose(f1, 1.0))
    p, r, f1 = av._prf(0, 0, 5)
    check("_prf zero-detection case", np.isnan(p) and r == 0.0)
    return True


def test_detection_bands(files):
    """The drawn envelope must agree with the flags, at the run's own threshold
    and at any other. If it does not, the notebook plots misrepresent the
    detector."""
    print("\n[8] detection bands agree with flags")
    import variant_viz as vz

    ok_all = True
    for variant in ("baseline", "log1p", "signed", "all3"):
        for f in files:
            res, df = uv.run_variant_with_frame(f, variant)
            if not require_success(res, f"{Path(f).stem}/{variant}"):
                continue
            if "upper_band" not in df.columns:
                ok_all &= check(f"{Path(f).stem}/{variant}: bands emitted", False)
                continue

            ordered = bool((df["upper_band"] >= df["lower_band"]).all())
            flag = df["is_anomaly_predicted"].to_numpy(int) == 1
            outside = ((df["actual"] > df["upper_band"]) |
                       (df["actual"] < df["lower_band"])).to_numpy()
            ok_all &= check(f"{Path(f).stem}/{variant}: bands match flags",
                            ordered and bool(np.array_equal(flag, outside)))

            # Recomputing at a different threshold must keep them consistent.
            for k in (2.5, 5.0):
                d2 = vz.recompute_at_threshold(
                    df, k, res["opt_signed_bands"],
                    res["transform"], res.get("transform_scale_used", 1.0))
                f2 = d2["is_anomaly_predicted"].to_numpy(int) == 1
                o2 = ((d2["actual"] > d2["upper_band"]) |
                      (d2["actual"] < d2["lower_band"])).to_numpy()
                if not np.array_equal(f2, o2):
                    ok_all &= check(
                        f"{Path(f).stem}/{variant}: consistent at z={k}", False)
            # A stricter threshold can never flag more.
            lo = vz.recompute_at_threshold(df, 2.5, res["opt_signed_bands"],
                                           res["transform"],
                                           res.get("transform_scale_used", 1.0))
            hi = vz.recompute_at_threshold(df, 5.0, res["opt_signed_bands"],
                                           res["transform"],
                                           res.get("transform_scale_used", 1.0))
            ok_all &= check(
                f"{Path(f).stem}/{variant}: monotone in threshold",
                int(hi["is_anomaly_predicted"].sum()) <= int(lo["is_anomaly_predicted"].sum()))
    return ok_all


def test_viz_helpers(files):
    """Threshold recomputation must reproduce the run's own stored counts."""
    print("\n[9] viz helpers")
    import variant_viz as vz

    ok_all = True
    for variant in ("baseline", "signed", "log1p"):
        for f in files:
            res, df = uv.run_variant_with_frame(f, variant)
            if not require_success(res, f"{Path(f).stem}/{variant}"):
                continue
            d = vz.recompute_at_threshold(
                df, res["opt_threshold_z_score"], res["opt_signed_bands"],
                res["transform"], res.get("transform_scale_used", 1.0))
            c = vz.confusion_counts(d)
            ok = (c["tp"] == res["metric_pred_tp"]
                  and c["fp"] == res["metric_pred_fp"])
            ok_all &= ok
            if not ok:
                check(f"{Path(f).stem}/{variant}: recompute == stored", False,
                      f"{c} vs tp={res['metric_pred_tp']} fp={res['metric_pred_fp']}")
    if ok_all:
        check("recompute at native threshold reproduces stored counts", True)

    p, r, f1 = vz.prf(5, 5, 5)
    check("vz.prf matches expectation", np.isclose(p, 0.5) and np.isclose(f1, 0.5))
    return ok_all



# ============================================================================
# SUSTAINED-ANOMALY HARNESS
# ============================================================================


def test_injection_mechanics():
    """Injectors must be exact on the cumulative series and honest about labels."""
    print("\n[10] injection mechanics")
    n = 200
    ts = pd.date_range("2024-03-01", periods=n, freq="h", tz="UTC")
    diff = np.abs(np.sin(np.arange(n) / 3.8)) * 0.05 + 0.01
    df = pd.DataFrame({
        "timestamp_utc": ts,
        "hodnota": np.cumsum(diff) + 1000.0,
        "Diff": diff,
    })
    rng_seed = 7
    scale = ai.sensor_scale(diff)
    check("sensor_scale is the median non-zero Diff",
          np.isclose(scale, float(np.median(diff[diff > 0]))))

    # apply_delta must keep hodnota and Diff mutually consistent, INCLUDING row
    # 0 — the row whose Diff came from outside this slice.
    delta = np.zeros(n)
    delta[50:] = 0.3
    out = ai.apply_delta(df, delta)
    recon = np.diff(out["hodnota"].values)
    check("apply_delta: Diff matches hodnota.diff() on rows 1..n",
          np.allclose(recon, out["Diff"].values[1:], atol=1e-9))
    check("apply_delta: row 0 Diff preserved when untouched",
          np.isclose(out["Diff"].values[0], df["Diff"].values[0]))
    check("apply_delta: rows before the delta are unchanged",
          np.allclose(out["Diff"].values[:50], df["Diff"].values[:50]))

    ok_all = True
    for arch in sorted(ai.ARCHETYPES):
        plan = ai.InjectionPlan(archetype=arch, depth=2.0, duration_h=12.0)
        inj, events = ai.apply_injection_plan(
            plan=plan, df=df, scale=scale, periodicity_seconds=3600, seed=rng_seed)

        # Cumulative series must stay non-decreasing: none of these archetypes
        # is a meter rollback, so a negative step would mean a broken injector.
        h = inj["hodnota"].values
        h = h[np.isfinite(h)]
        mono = bool(np.all(np.diff(h) >= -ai.NEG_DIFF_TOL))
        ok_all &= check(f"{arch}: cumulative series stays non-decreasing", mono)

        # Labels must line up with the intervals the injector reported.
        lbl, ids = em.label_rows_from_events(inj["timestamp_utc"].values, events)
        if events:
            span = sum(
                int(((pd.to_datetime(inj["timestamp_utc"].values, utc=True) >= pd.Timestamp(e.t_start))
                     & (pd.to_datetime(inj["timestamp_utc"].values, utc=True) <= pd.Timestamp(e.t_end))).sum())
                for e in events)
            ok_all &= check(f"{arch}: label count == interval span",
                            int(lbl.sum()) == span, f"{int(lbl.sum())} vs {span}")
        else:
            ok_all &= check(f"{arch}: unlabelled probe emits no labels",
                            int(lbl.sum()) == 0)

    # The two probes must be label-free / injection-free respectively.
    _, ev_h = ai.apply_injection_plan(
        plan=ai.InjectionPlan("holiday_profile", duration_h=24.0),
        df=df, scale=scale, periodicity_seconds=3600, seed=rng_seed)
    check("holiday_profile is an unlabelled FP probe", ev_h == [])

    inj_n, ev_n = ai.apply_injection_plan(
        plan=ai.InjectionPlan("null_control", duration_h=24.0),
        df=df, scale=scale, periodicity_seconds=3600, seed=rng_seed)
    check("null_control labels an interval", len(ev_n) == 1)
    check("null_control leaves the data untouched",
          np.allclose(inj_n["Diff"].values, df["Diff"].values, equal_nan=True)
          and np.allclose(inj_n["hodnota"].values, df["hodnota"].values, equal_nan=True))

    # Depth must be monotone in effect, or the surface's x-axis is meaningless.
    tot = []
    for depth in (0.1, 1.0, 10.0):
        inj_d, _ = ai.apply_injection_plan(
            plan=ai.InjectionPlan("slow_leak", depth=depth, duration_h=24.0),
            df=df, scale=scale, periodicity_seconds=3600, seed=rng_seed)
        tot.append(float(np.nansum(inj_d["Diff"].values)))
    check("slow_leak volume increases with depth",
          tot[0] < tot[1] < tot[2], f"{[round(t, 3) for t in tot]}")

    # frozen_meter must actually flatten consumption.
    inj_f, ev_f = ai.apply_injection_plan(
        plan=ai.InjectionPlan("frozen_meter", duration_h=24.0),
        df=df, scale=scale, periodicity_seconds=3600, seed=rng_seed)
    lbl_f, _ = em.label_rows_from_events(inj_f["timestamp_utc"].values, ev_f)
    inside = lbl_f == 1
    check("frozen_meter zeroes Diff inside the window",
          np.allclose(np.nan_to_num(inj_f["Diff"].values[inside]), 0.0, atol=1e-9))
    return ok_all


def test_event_metrics():
    """Event-level scoring must be right on hand-built cases."""
    print("\n[11] event metrics")
    n = 100
    ts = pd.date_range("2024-03-01", periods=n, freq="h", tz="UTC")
    ev = [ai.AnomalyEvent(archetype="slow_leak", t_start=ts[20], t_end=ts[39],
                          depth=1.0, duration_h=20.0, scale=1.0)]

    lbl, ids = em.label_rows_from_events(ts, ev)
    check("labels cover exactly the interval",
          int(lbl.sum()) == 20 and lbl[20] == 1 and lbl[39] == 1
          and lbl[19] == 0 and lbl[40] == 0)

    # Detected on the 6th reading of the event.
    flags = np.zeros(n, dtype=bool)
    flags[25] = True
    df_ev = em.event_level_metrics(ts, flags, ev, periodicity_seconds=3600)
    check("event detected", bool(df_ev.loc[0, "detected"]))
    check("time-to-detect measured from event start",
          df_ev.loc[0, "ttd_readings"] == 5, f'{df_ev.loc[0, "ttd_readings"]}')
    check("time-to-detect in hours", np.isclose(df_ev.loc[0, "ttd_hours"], 5.0))

    # A flag OUTSIDE the event is a false alarm, not a detection.
    flags2 = np.zeros(n, dtype=bool)
    flags2[80] = True
    df_ev2 = em.event_level_metrics(ts, flags2, ev, periodicity_seconds=3600)
    check("flag outside the event is not a detection",
          not bool(df_ev2.loc[0, "detected"]))
    fp, n_clean = em.false_alarm_count(ts, flags2, ev)
    check("false alarm counted outside events", fp == 1 and n_clean == 80)
    check("missed event reports NaN ttd, not a sentinel",
          bool(np.isnan(df_ev2.loc[0, "ttd_readings"])))

    # Recovery: still alarming after the event ends.
    flags3 = np.zeros(n, dtype=bool)
    flags3[25] = True
    flags3[41] = True
    rec = em.recovery_check(ts, flags3, ev, n_after=12)
    check("recovery flagged as unclean when alarms persist",
          not bool(rec.loc[0, "clean"]))
    rec2 = em.recovery_check(ts, flags, ev, n_after=12)
    check("recovery clean when detector settles", bool(rec2.loc[0, "clean"]))
    return not _FAIL


def test_null_control_correction():
    """The surface must report excess over the null floor, not the raw rate."""
    print("\n[12] null-control correction")
    import run_sustained as rs

    surf = pd.DataFrame({
        "archetype": ["null_control", "slow_leak", "slow_leak"],
        "depth": [np.nan, 0.1, 10.0],
        "duration_h": [24.0, 24.0, 24.0],
        "n_events": [3, 3, 3],
        "detection_rate": [0.333333, 0.333333, 1.0],
        "ttd_readings_median": [21.0, 21.0, 1.0],
    })
    out = rs._apply_null_correction(surf)
    row_low = out[(out.archetype == "slow_leak") & (out.depth == 0.1)].iloc[0]
    row_high = out[(out.archetype == "slow_leak") & (out.depth == 10.0)].iloc[0]

    check("coincidental detection nets to zero excess",
          np.isclose(row_low["excess_detection_rate"], 0.0),
          f'raw={row_low["detection_rate"]:.3f} null={row_low["null_rate"]:.3f}')
    check("genuine detection survives the correction",
          np.isclose(row_high["excess_detection_rate"], 1.0 - 1 / 3))
    check("excess is never negative",
          bool((out["excess_detection_rate"].fillna(0) >= 0).all()))
    check("null_control row has no excess of its own",
          bool(np.isnan(out[out.archetype == "null_control"].iloc[0]["excess_detection_rate"])))
    return not _FAIL


def test_injection_is_read_only(files):
    """Injection changes only the DATA, never the scoring path.

    A run with `injection_plan=None` must be identical to the pre-existing
    behaviour, and the fixture CSVs on disk must never be written to.
    """
    print("\n[13] injection does not alter the scoring path")
    ok_all = True
    for f in files:
        name = Path(f).stem
        before = Path(f).read_bytes()

        a = uv.run_variant(f, "baseline")
        b = uv.run_variant(f, "baseline", injection_plan=None)
        if not (require_success(a, f"{name}/plain") and require_success(b, f"{name}/none")):
            ok_all = False
            continue
        za = np.asarray(a["z_scores"], dtype=float)
        zb = np.asarray(b["z_scores"], dtype=float)
        ok_all &= check(f"{name}: injection_plan=None is a no-op",
                        za.shape == zb.shape and np.allclose(za, zb, equal_nan=True,
                                                             rtol=0, atol=0))
        # A run with a plan must not touch the source file.
        uv.run_variant(f, "baseline",
                       injection_plan=ai.InjectionPlan("slow_leak", depth=1.0,
                                                       duration_h=24.0))
        ok_all &= check(f"{name}: fixture CSV unmodified on disk",
                        Path(f).read_bytes() == before)
    return ok_all



def test_sustained_detectors():
    """Detector recurrences must be causal, calibrated, and honest."""
    print("\n[14] sustained detectors")

    # --- direction is recovered, and the fold's near-zero pathology is gone ---
    st = sdet.SignedResidualStats(mu=0.0, sd=1.0)
    check("direction: over", st.update(2.0, 1.0)["direction"] == "over")
    check("direction: under", st.update(1.0, 2.0)["direction"] == "under")
    check("signed z keeps sign", st.update(1.0, 2.0)["z_signed"] < 0)
    check("perfect prediction scores ~0 signed",
          abs(st.update(1.0, 1.0)["z_signed"]) < 1e-9)

    # --- run counter escalates only after N consecutive ----------------------
    rc = sdet.FlagRunCounter(threshold=3)
    got = [rc.update(f)["anomaly_sustained"] for f in [1, 1, 1, 0, 1]]
    check("run counter fires on the 3rd consecutive flag",
          got == [False, False, True, False, False])

    # --- stuck: threshold learned from the sensor's own idle behaviour -------
    quiet = np.array([0.0] * 10 + [1.0] * 10)     # idles 10 in a row normally
    busy = np.array([1.0] * 20)                   # never idles
    d_quiet = sdet.StuckMeterDetector.calibrate(quiet)
    d_busy = sdet.StuckMeterDetector.calibrate(busy)
    check("stuck threshold adapts to the sensor",
          d_quiet.threshold > d_busy.threshold,
          f"quiet={d_quiet.threshold} busy={d_busy.threshold}")
    fired = [d_busy.update(0.0)["stuck"] for _ in range(10)]
    check("stuck fires on a never-idle sensor", any(fired))
    d_busy.update(1.0)
    check("stuck resets on consumption", d_busy.run == 0)
    nan_det = sdet.StuckMeterDetector(max_zero_run=2)
    nan_det.update(0.0)
    before = nan_det.run
    nan_det.update(np.nan)
    check("a gap neither extends nor resets the zero run", nan_det.run == before)

    # --- seasonal drift: fires under a sustained lift, not under noise -------
    rng = np.random.default_rng(0)
    daily = 24
    pos = np.tile(np.arange(daily), 20)
    base_profile = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(daily) / daily)
    cal = base_profile[pos] + rng.normal(0, 0.05, pos.size)
    det = sdet.SeasonalDriftDetector.calibrate(pos, cal, daily, k=0.5, h=5.0)

    clean = base_profile[pos[:200]] + rng.normal(0, 0.05, 200)
    n_clean = sum(det.update(pos[i], clean[i])["seasonal_drift"] for i in range(200))

    det2 = sdet.SeasonalDriftDetector.calibrate(pos, cal, daily, k=0.5, h=5.0)
    lifted = clean + 0.5
    n_lift = sum(det2.update(pos[i], lifted[i])["seasonal_drift"] for i in range(200))
    check("seasonal drift fires far more under a sustained lift",
          n_lift > n_clean, f"lift={n_lift} clean={n_clean}")

    # --- night flow: baseline is the MEDIAN daily minimum --------------------
    ts = pd.date_range("2024-01-01", periods=24 * 6, freq="h", tz="UTC")
    d = np.tile(np.concatenate([np.zeros(6) + 0.1, np.ones(18)]), 6)
    nf = sdet.NightFlowTracker.calibrate(ts, d, factor=3.0)
    check("night-flow baseline is the median daily minimum",
          np.isclose(nf.baseline, 0.1), f"{nf.baseline}")

    # --- the residual CUSUM is excluded from the union, deliberately --------
    out = sdet.run_detectors(
        sdet.calibrate_all(ts, d, np.zeros(len(d)),
                           cal_positions=np.arange(len(d)) % 24, daily_steps=24),
        ts, d, d, np.zeros(len(d), dtype=bool),
        positions=np.arange(len(d)) % 24,
    )
    check("union excludes the residual CUSUM (it does not discriminate)",
          bool(((out["drift"].astype(bool)) & (~out["sustained_alert"])).any())
          or not out["drift"].any())
    check("all detector columns present",
          {"z_signed", "direction", "anomaly_sustained", "stuck", "drift",
           "seasonal_drift", "night_flow_elevated",
           "sustained_alert"} <= set(out.columns))

    # --- cross-fit drift -----------------------------------------------------
    cd = sdet.crossfit_drift([1.0, 2.0], [1.0, 2.0])
    check("identical fits show zero drift", np.isclose(cd["theta_l2_delta"], 0.0))
    cd2 = sdet.crossfit_drift([1.0, 2.0], [1.0, 4.0])
    check("changed fits show positive drift", cd2["theta_l2_delta"] > 0)
    return not _FAIL


def test_detectors_are_read_only(files):
    """Enabling the detectors must not move a single existing number."""
    print("\n[15] detectors do not alter existing scoring")
    ok_all = True
    for f in files:
        name = Path(f).stem
        a = uv.run_variant(f, "baseline")
        b = uv.run_variant(f, "baseline", sustained_detectors=True)
        if not (require_success(a, f"{name}/off") and require_success(b, f"{name}/on")):
            ok_all = False
            continue
        za = np.asarray(a["z_scores"], dtype=float)
        zb = np.asarray(b["z_scores"], dtype=float)
        ok_all &= check(f"{name}: z-scores unchanged by detectors",
                        za.shape == zb.shape and np.allclose(za, zb, equal_nan=True,
                                                             rtol=0, atol=0))
        same = all(a.get(k) == b.get(k) for k in
                   ("metric_pred_tp", "metric_pred_fp", "metric_pred_fn"))
        ok_all &= check(f"{name}: tp/fp/fn unchanged by detectors", same)
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    args = ap.parse_args()

    files = sorted(str(p) for p in Path(args.fixtures).glob("*.csv"))
    if not files:
        print(f"No fixture CSVs under {args.fixtures}")
        return 1
    print(f"fixtures: {len(files)} sensors from {args.fixtures}")

    test_baseline_equivalence(files)
    test_transforms()
    test_harmonics()
    test_sweep_consistency(files)
    test_signed_semantics(files)
    test_gate_flags_model_independent(files)
    test_analysis_layer(files)
    test_detection_bands(files)
    test_viz_helpers(files)
    test_injection_mechanics()
    test_event_metrics()
    test_null_control_correction()
    test_injection_is_read_only(files)
    test_sustained_detectors()
    test_detectors_are_read_only(files)

    print(f"\n{'=' * 60}")
    print(f"passed: {len(_PASS)}   failed: {len(_FAIL)}")
    if _FAIL:
        for n in _FAIL:
            print(f"  FAILED: {n}")
    print("=" * 60)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
