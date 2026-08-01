"""
analyze_variants.py — Post-hoc analysis layer for UC variant runs.
==================================================================

Everything here is a re-aggregation of what `run_variants.py` already wrote.
No model is re-run, so gate thresholds and detection thresholds are sliders you
can move freely and instantly.

Two rules keep the comparison honest, and both are enforced here rather than
left to the caller:

1. COHORT IS FIXED ACROSS VARIANTS. The sensor set is resolved once from a
   reference variant (default `baseline`) and then applied to every variant.
   Structural diagnostics (fill_pct, f_s_daily, periodicity, gap) are
   model-independent so they agree across variants anyway -- pinning them makes
   that a guarantee instead of an assumption.

2. NEVER FILTER ON MASE. It is model-dependent: a variant that fits marginal
   sensors slightly worse would drop them from its own cohort and score better
   for the wrong reason. MASE is reported as an OUTCOME, and production's MASE
   gate is surfaced only as a separate `would_pass_prod_mase` coverage count.

Micro-averaging is used throughout (sum tp/fp/fn, then compute once). Per-sensor
averaging hides false-positive floods. `calculate_metrics()` never populates
precision/recall/f1 on its normal path anyway -- only the four counts -- so
summing is both the correct and the only option.

Usage
-----
    python analyze_variants.py --results ./results/variants
    python analyze_variants.py --results ./results/variants --prod-gates
    python analyze_variants.py --results ./results/variants --sweep --baseline baseline
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Production initial-fit gates (uc-cem model/shared.py QUALITY_DEFAULTS).
PROD_GATE = dict(fs_min=0.45, fill_max_pct=25.0, daily_steps_min=5, max_gap_days=2.0)


# ============================================================================
# LOADING
# ============================================================================


def load_results(results_dir):
    results_dir = Path(results_dir)
    summary = pd.read_csv(results_dir / "summary.csv")

    scores = None
    for candidate in ("scores.parquet", "scores.csv.gz"):
        p = results_dir / candidate
        if p.exists():
            scores = (pd.read_parquet(p) if p.suffix == ".parquet"
                      else pd.read_csv(p))
            break
    return summary, scores


# ============================================================================
# COHORT SELECTION (the gate slider)
# ============================================================================


def select_cohort(summary, fs_min=None, fill_max_pct=None, daily_steps_min=None,
                  max_gap_days=None, reference_variant="baseline",
                  require_success_all=True):
    """Resolve the sensor set once, from `reference_variant`, and return the ids.

    All filters here are on MODEL-INDEPENDENT diagnostics. `mase_seasonal` is
    deliberately not accepted as a filter -- see module docstring.
    """
    if reference_variant not in set(summary["variant"]):
        reference_variant = summary["variant"].iloc[0]

    ref = summary[summary["variant"] == reference_variant].copy()
    ref = ref[ref["status"] == "success"]

    def _keep(col, op, val):
        nonlocal ref
        if val is None or col not in ref.columns:
            return
        v = pd.to_numeric(ref[col], errors="coerce")
        ref = ref[op(v, val) & v.notna()]

    _keep("f_s_daily_second", lambda a, b: a > b, fs_min)
    _keep("filled_pct_second", lambda a, b: a < b, fill_max_pct)
    _keep("daily_period_steps", lambda a, b: a >= b, daily_steps_min)
    _keep("max_single_gap_days", lambda a, b: a <= b, max_gap_days)

    ids = set(ref["filename"].astype(str))

    if require_success_all:
        # A sensor only counts if EVERY variant produced a result for it --
        # otherwise variants would be compared on different sensor sets.
        ok = summary[summary["status"] == "success"]
        counts = ok.groupby(ok["filename"].astype(str))["variant"].nunique()
        n_variants = summary["variant"].nunique()
        ids &= set(counts[counts == n_variants].index)

    return ids


# ============================================================================
# MICRO-AVERAGED METRICS
# ============================================================================


def _prf(tp, fp, fn):
    tp, fp, fn = float(tp), float(fp), float(fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if prec is np.nan or rec is np.nan or not np.isfinite(prec) or not np.isfinite(rec) or (prec + rec) == 0:
        f1 = np.nan
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def micro_metrics(summary, cohort=None):
    """Micro-averaged precision/recall/F1 per variant over the fixed cohort."""
    df = summary[summary["status"] == "success"].copy()
    df["filename"] = df["filename"].astype(str)
    if cohort is not None:
        df = df[df["filename"].isin(cohort)]

    rows = []
    for variant, g in df.groupby("variant"):
        tp = pd.to_numeric(g["metric_pred_tp"], errors="coerce").fillna(0).sum()
        fp = pd.to_numeric(g["metric_pred_fp"], errors="coerce").fillna(0).sum()
        fn = pd.to_numeric(g["metric_pred_fn"], errors="coerce").fillna(0).sum()
        prec, rec, f1 = _prf(tp, fp, fn)
        n = len(g)
        rows.append({
            "variant": variant, "n_sensors": n,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": prec, "recall": rec, "f1": f1,
            # The prediction segment is exactly one week, so this is literally
            # alerts per sensor per week -- what Softlink actually feels.
            "fp_per_sensor_week": fp / n if n else np.nan,
            "mae": pd.to_numeric(g.get("metric_mae"), errors="coerce").mean(),
            "median_mase_seasonal": pd.to_numeric(
                g.get("mase_seasonal"), errors="coerce").median(),
            "converged_rate": pd.to_numeric(
                g.get("converged_second"), errors="coerce").mean(),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def fp_flood_distribution(summary, cohort=None, thresholds=(1, 5, 10, 25)):
    """Share of sensors emitting more than N false positives.

    Micro-F1 can look healthy while a handful of near-flat sensors generate most
    of the alert volume. This is a DISTRIBUTIONAL view, not macro-averaging.
    """
    df = summary[summary["status"] == "success"].copy()
    df["filename"] = df["filename"].astype(str)
    if cohort is not None:
        df = df[df["filename"].isin(cohort)]

    rows = []
    for variant, g in df.groupby("variant"):
        fp = pd.to_numeric(g["metric_pred_fp"], errors="coerce").fillna(0)
        row = {"variant": variant, "n_sensors": len(g)}
        for t in thresholds:
            row[f"pct_sensors_fp_gt_{t}"] = 100.0 * float((fp > t).mean())
        total = fp.sum()
        top = fp.sort_values(ascending=False)
        k = max(1, int(round(0.05 * len(fp))))
        row["pct_fp_from_worst_5pct_sensors"] = (
            100.0 * float(top.head(k).sum() / total) if total > 0 else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def coverage_report(summary, reference_variant="baseline"):
    """How many sensors production would actually serve.

    Reported separately from detection quality so a coverage gain can never hide
    inside an F1 number (and vice versa).
    """
    df = summary[summary["status"] == "success"].copy()
    flags = [c for c in df.columns if c.startswith("would_pass_prod_")]
    if not flags:
        return pd.DataFrame()
    rows = []
    for variant, g in df.groupby("variant"):
        row = {"variant": variant, "n_success": len(g)}
        for f in flags:
            row[f] = int(g[f].astype(str).isin(["True", "true", "1"]).sum())
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================================
# DETECTION-THRESHOLD SWEEP (free: no re-runs)
# ============================================================================


def sweep_thresholds(scores, summary, cohort=None, zs=None):
    """Recompute precision/recall/F1 at any detection threshold.

    Honours each variant's own decision rule: signed variants are upper-only
    (`z > k`), folded variants are two-sided (`|z| > k`). This is what lets
    variants be compared at MATCHED ALERT VOLUME rather than at matched k -- a
    signed band at k=3.5 is roughly 15x stricter than a folded one, so a
    fixed-k comparison would misread the difference as lost recall.
    """
    if scores is None or not len(scores):
        return pd.DataFrame()

    if zs is None:
        zs = np.round(np.arange(2.0, 6.01, 0.25), 2)

    signed_map = (
        summary.drop_duplicates("variant")
        .set_index("variant")
        .get("opt_signed_bands", pd.Series(dtype=object))
        .astype(str).str.lower().eq("true").to_dict()
    )

    sc = scores.copy()
    sc["filename"] = sc["filename"].astype(str)
    if cohort is not None:
        sc = sc[sc["filename"].isin(cohort)]

    rows = []
    for variant, g in sc.groupby("variant"):
        z = g["z_score"].to_numpy(dtype=float)
        y = g["label"].to_numpy(dtype=int)
        stat = z if signed_map.get(variant, False) else np.abs(z)
        valid = np.isfinite(stat)
        stat, y = stat[valid], y[valid]
        n_sensors = g["filename"].nunique()

        for k in zs:
            pred = stat > k
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            fn = int(np.sum(~pred & (y == 1)))
            prec, rec, f1 = _prf(tp, fp, fn)
            rows.append({
                "variant": variant, "z": k, "tp": tp, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec, "f1": f1,
                "fp_per_sensor_week": fp / n_sensors if n_sensors else np.nan,
            })
    return pd.DataFrame(rows)


def best_by_f1(sweep_df):
    """Peak F1 per variant across the sweep, with its operating point."""
    if not len(sweep_df):
        return pd.DataFrame()
    idx = sweep_df.groupby("variant")["f1"].idxmax().dropna()
    return (sweep_df.loc[idx]
            .sort_values("f1", ascending=False)
            .reset_index(drop=True))


def match_alert_volume(sweep_df, reference_variant="baseline"):
    """For each variant, the operating point whose alert volume matches the
    reference variant's peak-F1 volume. This is the apples-to-apples row."""
    if not len(sweep_df):
        return pd.DataFrame()
    best = best_by_f1(sweep_df)
    ref = best[best["variant"] == reference_variant]
    if not len(ref):
        return pd.DataFrame()
    target = float(ref.iloc[0]["fp_per_sensor_week"])

    rows = []
    for variant, g in sweep_df.groupby("variant"):
        g = g.dropna(subset=["fp_per_sensor_week"])
        if not len(g):
            continue
        pick = g.iloc[(g["fp_per_sensor_week"] - target).abs().argmin()]
        rows.append(pick)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out.attrs["target_fp_per_sensor_week"] = target
    return out


# ============================================================================
# PAIRED BOOTSTRAP
# ============================================================================


def paired_bootstrap_delta_f1(summary, variant_a, variant_b, cohort=None,
                              n_boot=1000, seed=42):
    """95% CI on micro-F1(variant_b) - micro-F1(variant_a).

    Resamples SENSORS (not readings) with replacement and uses both variants'
    counts for each resampled sensor. The comparison is naturally paired --
    identical windows and identical injected anomalies per sensor -- which makes
    the interval far tighter than an unpaired test would give.
    """
    df = summary[summary["status"] == "success"].copy()
    df["filename"] = df["filename"].astype(str)
    if cohort is not None:
        df = df[df["filename"].isin(cohort)]

    cols = ["metric_pred_tp", "metric_pred_fp", "metric_pred_fn"]
    piv = {}
    for v in (variant_a, variant_b):
        g = df[df["variant"] == v].set_index("filename")
        if not len(g):
            return None
        piv[v] = g[cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    ids = sorted(set(piv[variant_a].index) & set(piv[variant_b].index))
    if len(ids) < 2:
        return None
    A = piv[variant_a].loc[ids].to_numpy(float)
    B = piv[variant_b].loc[ids].to_numpy(float)

    def _f1(mat):
        tp, fp, fn = mat[:, 0].sum(), mat[:, 1].sum(), mat[:, 2].sum()
        return _prf(tp, fp, fn)[2]

    observed = _f1(B) - _f1(A)

    rng = np.random.default_rng(seed)
    n = len(ids)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n, n)
        deltas[i] = _f1(B[pick]) - _f1(A[pick])

    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return {
        "variant_a": variant_a, "variant_b": variant_b, "n_sensors": n,
        "delta_f1": observed, "ci_low": float(lo), "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }


# ============================================================================
# CLI
# ============================================================================


def _show(title, df):
    print(f"\n=== {title} ===")
    if df is None or not len(df):
        print("(no data)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print(df.to_string(index=False))


def main():
    p = argparse.ArgumentParser(
        description="Post-hoc analysis of UC variant runs (no re-runs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", required=True, help="Directory from run_variants.py")
    p.add_argument("--baseline", default="baseline",
                   help="Reference variant for cohort, pairing and volume matching.")
    p.add_argument("--prod-gates", action="store_true",
                   help="Restrict the cohort to production's STRUCTURAL gates "
                        "(fs/fill/periodicity/gap). Never includes MASE.")
    p.add_argument("--fs-min", type=float, default=None)
    p.add_argument("--fill-max-pct", type=float, default=None)
    p.add_argument("--daily-steps-min", type=int, default=None)
    p.add_argument("--max-gap-days", type=float, default=None)
    p.add_argument("--sweep", action="store_true",
                   help="Run the detection-threshold sweep and volume matching.")
    p.add_argument("--bootstrap", action="store_true",
                   help="Paired bootstrap CI on delta-F1 vs the reference.")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    summary, scores = load_results(args.results)

    gate = dict(fs_min=args.fs_min, fill_max_pct=args.fill_max_pct,
                daily_steps_min=args.daily_steps_min, max_gap_days=args.max_gap_days)
    if args.prod_gates:
        for k, v in PROD_GATE.items():
            if gate.get(k) is None:
                gate[k] = v

    cohort = select_cohort(summary, reference_variant=args.baseline, **gate)

    n_all = summary[summary["status"] == "success"]["filename"].astype(str).nunique()
    print(f"\ncohort: {len(cohort)} sensors "
          f"(of {n_all} with any success), filters={ {k: v for k, v in gate.items() if v is not None} }")
    print("note: cohort is fixed across variants; MASE is never used as a filter.")

    _show("outcomes", summary.groupby(["variant", "status"]).size()
          .unstack(fill_value=0).reset_index())
    _show("micro-averaged metrics (at each run's own threshold)",
          micro_metrics(summary, cohort))
    _show("false-positive concentration", fp_flood_distribution(summary, cohort))
    _show("production coverage (reported separately from quality)",
          coverage_report(summary))

    if args.sweep:
        sw = sweep_thresholds(scores, summary, cohort)
        _show("peak F1 per variant (threshold swept)", best_by_f1(sw))
        mv = match_alert_volume(sw, args.baseline)
        if len(mv):
            print(f"\n(matched to {args.baseline} peak-F1 alert volume = "
                  f"{mv.attrs.get('target_fp_per_sensor_week'):.3f} FP/sensor/week)")
        _show("matched alert volume — the apples-to-apples comparison", mv)

    if args.bootstrap:
        rows = []
        for v in sorted(set(summary["variant"])):
            if v == args.baseline:
                continue
            r = paired_bootstrap_delta_f1(summary, args.baseline, v, cohort,
                                          n_boot=args.n_boot)
            if r:
                rows.append(r)
        _show(f"paired bootstrap vs {args.baseline}", pd.DataFrame(rows))

    print()


if __name__ == "__main__":
    main()
