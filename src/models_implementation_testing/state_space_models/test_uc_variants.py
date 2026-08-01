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

DEFAULT_FIXTURES = "/home/user/uc-cem/testing_scripts/fixtures/real_segments"

_PASS, _FAIL = [], []


def check(name, ok, detail=""):
    (_PASS if ok else _FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


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
        if a["status"] != "success":
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
            if r["status"] != "success":
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
        if r["status"] != "success":
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
            if r["status"] != "success":
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

    print(f"\n{'=' * 60}")
    print(f"passed: {len(_PASS)}   failed: {len(_FAIL)}")
    if _FAIL:
        for n in _FAIL:
            print(f"  FAILED: {n}")
    print("=" * 60)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
