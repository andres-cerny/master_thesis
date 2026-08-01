# Sustained-anomaly detection surface

Measures what the UC detector catches **beyond isolated spikes** — leaks, frozen
meters, zero-consumption windows — and how that depends on how bad the fault is.

Companion to [`VARIANTS_README.md`](VARIANTS_README.md), which sweeps *model
variants* against one fixed anomaly type. This sweeps *anomaly types, depths and
durations* against one fixed model, and reports a **detection surface** rather
than a single F1.

## Why a surface and not an F1

Every F1 in this repo — including 0.684 — measures spike detection, because
`create_anomalies.py` injects isolated positive spikes and nothing else. For a
sustained fault a scalar is not just incomplete, it is the wrong shape of answer:
detection depends almost entirely on severity.

```
slow_leak, depth 10.0 -> caught in  1 reading
slow_leak, depth  1.0 -> not caught
slow_leak, depth  0.1 -> not caught
```

The useful output is where along that axis detection begins. One number averages
over exactly the thing worth knowing.

Point-wise scoring is also actively misleading here. A 48-hour leak spanning 192
readings, caught on reading 7, is an operational success — the leak was found in
two hours. Point-wise that is 7 TP and 185 FN, recall 0.036, and a detector that
alarms continuously for two days outranks one that fires once and stops. So this
harness scores **event recall** (caught at all?) and **time-to-detect**, and keeps
point-wise metrics alongside rather than instead.

## Read every rate against the null control

`null_control` labels a window and injects **nothing**. Event recall counts a
detection if *any* reading in the window is flagged, so over 24 hours a detector
with a normal false-alarm rate "detects" a share of events it was never shown.

This is not a rounding error. On the three fixtures:

| archetype | depth | raw rate | null rate | **excess** |
|---|---|---|---|---|
| `slow_leak` | 0.1 | 0.33 | 0.33 | **0.00** |
| `slow_leak` | 1.0 | 0.33 | 0.33 | **0.00** |
| `slow_leak` | 10.0 | 1.00 | 0.33 | **0.67** |
| `frozen_meter` | — | 0.00 | 0.33 | **0.00** |

Read raw, the shallow leaks look one-third detected. They are not detected at
all. **`excess_detection_rate` is the column to read**; the raw rate is kept
only so the correction is auditable.

(`frozen_meter` scoring *below* the floor is real: flattening consumption removes
the variance that would otherwise have produced the coincidental alarm.)

## Setup

```bash
pip install numpy pandas scipy statsmodels scikit-learn tqdm
pip install pyarrow      # optional: else scores fall back to .csv.gz
pip install torch        # imported by the baseline script (unused at runtime)
```

Sensor CSVs are **not in this repo** — the dataset is on
[Zenodo](https://zenodo.org/records/19735021) (>10 GB) and belongs in
`data/sensor_data/`. The three bundled fixtures in
`uc-cem/testing_scripts/fixtures/real_segments/` (100005, 100015, 100020, all
members of `test_set.pkl`) are enough to develop against and run in about a
minute. They are only ever read; nothing here writes to them.

## Run it

```bash
# 0. verify first — nothing below means anything if this is not green
python test_uc_variants.py

# 1. smoke run on the three fixtures (~1 min)
python run_sustained.py \
    --data-dir /home/user/uc-cem/testing_scripts/fixtures/real_segments \
    --out-dir ./results/sustained_smoke

# 1b. same, with the sustained detectors scored alongside
python run_sustained.py \
    --data-dir /home/user/uc-cem/testing_scripts/fixtures/real_segments \
    --out-dir ./results/sustained_smoke_det --sustained-detectors

# 2. full test set
python run_sustained.py \
    --data-dir ../../../data/sensor_data \
    --pickle ../../pickles/test_set.pkl \
    --workers 7 --out-dir ./results/sustained_baseline

# 3. one archetype, custom ladder
python run_sustained.py --data-dir ... \
    --archetypes null_control slow_leak \
    --depths 0.02 0.05 0.1 0.25 0.5 1.0 3.0 10.0 \
    --durations 6 24 48
```

**Always include `null_control` in `--archetypes`.** Without it there is no floor
to subtract and `excess_detection_rate` comes back NaN.

## Archetypes

All are injected on the **cumulative `hodnota`** series, with `Diff` adjusted
exactly — never written to `Diff` directly, which would bypass Diff
reconstruction and the negative-diff convention.

| archetype | what it injects | depth axis |
|---|---|---|
| `null_control` | nothing; labels a window | — (the floor) |
| `slow_leak` | constant extra draw per step | yes |
| `burst` | steep short ramp | yes |
| `frozen_meter` | `hodnota` holds; `Diff` = 0 | — (duration only) |
| `zero_consumption` | supply stops, no catch-up | — |
| `point_spike` | one oversized reading | yes |
| `unit_change` | m³ -> litres, permanent | yes (the factor) |
| `missing_data` | `hodnota` -> NaN | — |
| `holiday_profile` | quiet day swapped in, **unlabelled** | — (FP probe) |

**Depth is in multiples of the sensor's own scale** (median non-zero `Diff` on
the calibration segment), never absolute m³ — `Diff` magnitude reflects meter
size rather than statistical structure, so an absolute depth would mean something
different on every sensor. Same reasoning as the `log1p` transform's per-sensor
scale, and the scale comes from the **calibration** segment so an injected
anomaly cannot define its own units.

`holiday_profile` is a legitimate low-usage day, not a fault, and carries no
label. Any alert inside it is a false positive by construction. It exists because
a detector can always buy recall by getting twitchier, and without an
unlabelled-but-unusual probe that trade stays invisible.

## Outputs

| file | contents |
|---|---|
| `sustained_matrix.csv` | one row per sensor x archetype x depth x duration |
| `sustained_surface.csv` | the surface: detection rate, null rate, **excess**, median TTD |
| `sustained_events.csv` | one row per injected event: detected, TTD |
| `sustained_scores.parquet` | per-reading z + label, same schema as `scores.parquet` |

Because the scores file matches `run_variants.py`'s schema, the existing analysis
layer works on it unchanged:

```bash
python analyze_variants.py --results ./results/sustained_baseline --sweep
```

The runner also prints a **NEVER DETECTED** block listing every cell whose excess
rate is zero. That block is the point of the exercise.

## The detectors

`--sustained-detectors` additionally runs the read-only detectors in
`helper_scripts/sustained_detectors.py` and scores the same events against their
union (`sust_*` columns) and against the union with the z-score (`comb_*`). Off
by default, so the surface measures the existing detector alone.

Each is a **scalar recurrence with explicit state** — one `update()` per reading,
no lookahead. That is not stylistic: the production detector
(`uc-cem/model/shared.py::UCStreamingDetector.step()`) handles one reading per
call and persists state between calls, so a detector in this shape ports across
as a field in `state.npz` and a few lines in `step()`. A pandas pass over the
week would not port at all.

| detector | signal | targets |
|---|---|---|
| `SignedResidualStats` | signed innovation | direction (over/under) |
| `FlagRunCounter` | consecutive z-flags | sustained events |
| `StuckMeterDetector` | run of zero `Diff` | frozen meter |
| `SeasonalDriftDetector` | CUSUM vs **calibration profile** | slow leak |
| `NightFlowTracker` | daily minimum `Diff` | slow leak |
| `LeakDetector` | CUSUM vs **model residual** | *does not work — see below* |

### The residual CUSUM does not work, and that is a finding

The obvious design — CUSUM on the one-step-ahead residual — was implemented,
measured, and **excluded from the alert union**. On the fixtures its firing count
under an injected leak is *less than or equal to* its count under
`null_control`, at every `(k, h)` tried:

| k | h | null | leak 0.1 | leak 1.0 |
|---|---|---|---|---|
| 0.5 | 5 | 12 | 11 | 8 |
| 0.5 | 10 | 4 | 4 | 2 |
| 1.0 | 5 | 3 | 3 | 3 |
| 1.0 | 20 | 0 | 0 | 0 |

No tuning fixes this, because it is not a tuning problem. A one-step-ahead
residual is the error of an *adaptive* predictor: the local level tracks the leak
within a few readings, the residual returns to zero, and the CUSUM sees only a
brief transient. **The fault erases its own evidence from the signal being
monitored.**

The two detectors that do work both avoid model residuals entirely.
`SeasonalDriftDetector` compares against a per-position profile frozen at
calibration time; `NightFlowTracker` watches raw daily minimum consumption. Both
are immune to the level component re-baselining around the fault.

The column is still emitted, so the result stays auditable rather than being
quietly deleted.

### Measured on the three fixtures

| archetype | depth | z-recall | **sust** | resid CUSUM | seasonal | stuck | night |
|---|---|---|---|---|---|---|---|
| `null_control` | — | 0.33 | 0.67 | 12 | 18 | 10 | **0** |
| `slow_leak` | 0.1 | 0.33 | **1.00** | 11 | 19 | 10 | **3** |
| `slow_leak` | 0.5 | 0.33 | **1.00** | 9 | 23 | 10 | **3** |
| `slow_leak` | 1.0 | 0.33 | **1.00** | 8 | 29 | 10 | **3** |
| `frozen_meter` | — | 0.00 | **0.67** | 11 | 19 | **32** | 0 |
| `zero_consumption` | — | 0.00 | **0.67** | 11 | 18 | **34** | 0 |

Night flow separates cleanly (0 on the control, 3 on every leak depth) and
seasonal drift rises monotonically with depth (18 → 19 → 23 → 29). The stuck
detector triples on a frozen meter but carries a non-zero false-alarm count on
the control, so its `tolerance_factor` is the knob to raise if alert volume
matters more than sensitivity.

Three fixtures is not a fleet. These numbers show the detectors respond to the
right things; the fleet run is what sizes the false-alarm rate.

## Invariants

Four properties this harness must not break. All are asserted in
`test_uc_variants.py` (120 checks).

1. **Baseline equivalence.** `uc_variants` with default arguments reproduces the
   baseline script's z-scores bit-for-bit. `injection_plan=None` is a no-op.
   Verify: check [1] and [13].
2. **Injection changes data, never scoring.** No detector, threshold or masking
   rule is altered; `is_anomaly_predicted` comes from the same code as before.
   Verify: check [13].
3. **Fixture CSVs are read-only.** Injection is in-memory on the DataFrame.
   Verify: check [13] compares file bytes before and after a run.
4. **Detectors are read-only.** Enabling them changes no z-score and no
   tp/fp/fn count; masking stays governed by `mask_z_threshold` alone. Changing
   what freezes the Kalman state would change the state trajectory and therefore
   the existing detector, invalidating every before/after comparison.
   Verify: check [15].
5. **Split and resample logic untouched.** Injection happens post-resample, at
   the same point in `process_single_meter` the spike injector always occupied,
   so window selection stays seeded on `int(filename)` and cross-cell comparison
   stays paired per sensor.

## Known limits

- `point_spike` and `unit_change` have no duration, so no null control is matched
  to them and `excess_detection_rate` is NaN. Their windows are one reading wide,
  so the coincidental rate is negligible — but the correction is absent, not zero.
- Archetypes needing injection *before* resampling — real row dropout, clock
  skew, meter replacement, periodicity change — are out of scope, since they
  would require changing the split/resample path and would break cross-cell
  pairing. `missing_data` covers the post-resample shape of a dropout.
- One archetype per run by design. Stacking faults in one week would make
  time-to-detect and the FP rate un-attributable.
