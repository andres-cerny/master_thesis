# UC model variant testing harness

A/B testing for UC model tweaks, aimed at the Softlink production deployment.
Measures whether a change improves **prediction accuracy** and/or **detection
quality** — separately, and honestly enough to act on.

## Files

| file | role |
|---|---|
| `uc_variants.py` | Parameterized model. Imports the unchanged machinery from `local_level_unobserved_components_next_one_step_pred.py` and overrides only `predict_model_online`, `build_results_df`, `process_single_meter`. |
| `run_variants.py` | Batch runner → `summary.csv` + `scores.parquet`. |
| `analyze_variants.py` | Post-hoc slider/aggregation. No re-runs. |
| `variant_viz.py` | Plotting + threshold-recompute helpers used by the notebook. |
| `compare_variants.ipynb` | Visual comparison: per-sensor time series, confusion matrices, histograms, adjustable threshold. |
| `test_uc_variants.py` | Verification suite. **Run this first.** |

The two thesis scripts (`..._next_one_step_pred.py`, `..._z_floor.py`) are
untouched and remain reproducible.

## Quick start

```bash
python test_uc_variants.py                         # must be green

python run_variants.py \
    --data-dir ../../../data/sensor_data \
    --pickle ../../pickles/test_set.pkl \
    --workers 4 --out-dir ./results/testset

python analyze_variants.py --results ./results/testset --prod-gates --sweep --bootstrap
```

For visual inspection open `compare_variants.ipynb` and edit the config cell
(`FIXTURES_DIR`, `VARIANTS`, `Z_THRESHOLD`, `SENSOR`), then Run All. It defaults
to the three bundled fixtures and takes about a minute on them.

Optional deps: `pyarrow` (else the runner falls back to `scores.csv.gz`) and
`ipywidgets` (else the notebook's threshold control renders statically instead
of as a slider). Neither is required.

## Variants

| name | transform | daily harmonics | signed bands |
|---|---|---|---|
| `baseline` | raw | 2 (literal) | no |
| `log1p` | `log1p(Diff/s)` | 2 | no |
| `harmonics4` | raw | adaptive ≤4 | no |
| `signed` | raw | 2 | yes |
| `log1p_signed` | `log1p(Diff/s)` | 2 | yes |
| `all3` | `log1p(Diff/s)` | adaptive ≤4 | yes |

**Why `log1p(Diff / s)` and not plain `log1p(Diff)`** — `Diff` is ~0.005–0.5 m³,
so plain `log1p` is nearly the identity: measured compression at p99 was 20.5%
on sensor 100005 but only **4.0%** on 100020, and <2% at the median on all
three. The magnitude reflects **meter size, not statistical structure**, so the
transform's strength would depend on an arbitrary unit choice. Dividing by the
per-sensor median non-zero `Diff` makes it bend over the same *relative* range
everywhere. `s` is computed on the **clean calibration segment only** — deriving
it from the prediction segment would leak injected anomalies into the transform.

**Why adaptive harmonics** — a fixed harmonic count raises the periodicity floor
(`daily_steps ≥ 2H+1`) and would reject coarse sensors, fighting the goal of
serving *more* meters. `min(H, (daily_steps-1)//2)` stays below Nyquist on every
grid, so **no sensor is newly rejected**.

**Why signed bands** — the baseline folds to `|residual|` then does a two-sided
test. Injected anomalies are strictly positive, so the negative half of the band
can only ever produce false positives. Folding also makes an unusually
*accurate* prediction score far from the mean (`z = -mu/sd` at `r = 0`).

## Two rules that keep the comparison honest

**1. The cohort is fixed across variants.** Resolved once from a reference
variant, then applied to all. Enforced in `select_cohort`.

**2. MASE is never a filter.** It is model-dependent — a variant that fits
marginal sensors slightly worse would drop them from its own cohort and score
better for the wrong reason. MASE is reported as an *outcome*; production's MASE
gate appears only as the separate `would_pass_prod_mase` coverage count.
`select_cohort` structurally cannot accept a MASE argument, and
`test_uc_variants.py` asserts this via AST inspection.

Structural gates (`fill_pct`, `f_s_daily`, periodicity, `max_single_gap_days`)
*are* model-independent, so they are safe to slide freely. Test #6 verifies the
structural flags are identical across all variants.

## Compare at matched alert volume, not matched `k`

Switching to signed bands changes what the threshold *means*. For unbiased
Gaussian residuals the folded band at `k=3.5` sits at ≈2.91σ while the signed
band sits at 3.5σ — roughly **15× stricter**. Comparing both at `k=3.5` would
show recall collapsing when nothing has actually got worse.

`analyze_variants.py --sweep` therefore reports peak F1 per variant *and* a
`matched alert volume` table, which is the apples-to-apples row. The sweep is
free because `scores.parquet` stores per-reading z-scores **with labels**.

Note the detection threshold is decoupled from `mask_z_threshold`. Masking
protects the Kalman state and stays **two-sided** even when detection is
upper-only — a wildly low reading corrupts the state as badly as a high one, and
freezing state is what keeps a sustained deficit from being quietly absorbed by
the local level.

## Metrics reported

- **Micro-averaged P/R/F1** — sum `tp`/`fp`/`fn`, then compute once. Per-sensor
  averaging hides FP floods. (`calculate_metrics()` never populates
  precision/recall/f1 on its normal path anyway — only the four counts.)
- **FP per sensor-week** — the prediction segment is exactly one week, so this
  is literally alerts/sensor/week: what Softlink feels operationally.
- **FP concentration** — share of sensors with >N false positives, and the share
  of all FPs coming from the worst 5% of sensors. Micro-F1 can look healthy
  while a few near-flat meters generate most of the alert volume.
- **Paired bootstrap CI on ΔF1** — windows and injected anomalies are seeded on
  `int(filename)`, so variants are naturally paired per sensor; resampling
  sensors gives a much tighter interval than an unpaired test.
- **Coverage** — `would_pass_prod_*` counts, reported separately so a coverage
  gain never hides inside an F1 number.

## Constraints

- **Do not change the split or resample logic.** Pairing depends on
  `seed = int(filename)` producing identical windows and identical injected
  spikes across variants. Only model/scoring may be tweaked.
- **Holdout discipline.** `train_set.pkl` (22,082) and `test_set.pkl` (5,521)
  have zero overlap. Tune on a sample of `train_set`; confirm the winner on
  `test_set` **once**, or the thesis's F1 ≈ 0.684 stops being an honest
  reference.
- **Synthetic anomalies only.** `inject_spike_anomalies_diff` injects isolated
  positive spikes, so every F1 here measures *spike* detection. Sustained
  deficits (a stuck meter, or a production line running dry) are **not measured
  at all** — there are no labels for them.

## Known follow-up: the sustained-deficit class

> **Now implemented** — see [`SUSTAINED_README.md`](SUSTAINED_README.md) for the
> injector, the depth x duration detection surface, and the null control that
> separates real detection from coincidental alarms.

A local-level model absorbs sustained shifts *by design* — that is what the
level component is for — and it gets **worse as the deficit gets shallower**,
since small residuals never trip the masking threshold that would freeze the
state. Point anomalies want a z-score; sustained anomalies want a **CUSUM**, or
comparison against a stable seasonal reference rather than the already-adapted
one-step prediction.

This needs its own injector (depth × duration, reported as a detection surface
rather than one F1) and is deliberately out of scope here.

## Status

Verified on the three real sensors bundled in
`uc-cem/testing_scripts/fixtures/real_segments/` (100005, 100015, 100020 — all
members of `test_set.pkl`). The full dataset is on
[Zenodo](https://zenodo.org/records/19735021) (>10 GB) and is not in this repo.
Everything here is data-volume agnostic.

**Baseline equivalence is verified**: `uc_variants` with default arguments
reproduces the baseline script's z-scores bit-for-bit on all three fixtures. If
that ever fails, every measured "improvement" is confounded with a refactor
difference — treat it as a hard stop.
