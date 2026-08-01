"""
variant_viz.py — Plotting and threshold-inspection helpers for UC variants.
===========================================================================

The heavy lifting behind `compare_variants.ipynb`. Kept in a module rather than
in notebook cells so it can be imported, tested and reused headlessly.

The central idea is that the DETECTION threshold is a display-time parameter:
`recompute_at_threshold` re-derives both the flags and the drawn bands from
values already stored per reading, so moving the slider never re-runs a model.
That only works because `uc_variants.build_results_df(save_detection_bands=True)`
retains `predicted_scoring`, `band_mu` and `band_sd`.

Colour palette matches `show_softlink.ipynb`.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Palette carried over from show_softlink.ipynb for visual continuity.
COLOR_ACTUAL = "#2E86AB"
COLOR_MODEL = "#A23B72"
COLOR_MODEL_2 = "#F18F01"
COLOR_BAND = "#A23B72"
COLOR_TP = "#2BA84A"
COLOR_FP = "#F18F01"
COLOR_FN = "#FF03CD"


# ============================================================================
# THRESHOLD RECOMPUTATION (no model re-run)
# ============================================================================


def invert(values, transform_name, scale):
    """Map scoring-space values back to raw units."""
    v = np.asarray(values, dtype=float)
    if transform_name == "log1p_scaled":
        return float(scale) * np.expm1(v)
    return v


def recompute_at_threshold(df, z_threshold, signed_bands,
                           transform_name="raw", scale=1.0):
    """Re-flag every reading at `z_threshold` and redraw the bands to match.

    Returns a copy with `is_anomaly_predicted`, `upper_band` and `lower_band`
    updated. Both are recomputed from stored per-reading quantities, so the
    envelope always agrees with the flags -- if only the flags moved, the plot
    would quietly lie about what the detector is doing.
    """
    out = df.copy()
    z = out["z_score"].to_numpy(dtype=float)

    stat = z if signed_bands else np.abs(z)
    out["is_anomaly_predicted"] = (stat > z_threshold).astype(int)

    if {"predicted_scoring", "band_mu", "band_sd"}.issubset(out.columns):
        half = out["band_mu"].to_numpy(float) + z_threshold * out["band_sd"].to_numpy(float)
        pred_s = out["predicted_scoring"].to_numpy(float)
        upper_s = pred_s + half
        lower_s = np.zeros_like(upper_s) if signed_bands else pred_s - half
        out["upper_band"] = invert(upper_s, transform_name, scale)
        out["lower_band"] = np.clip(invert(lower_s, transform_name, scale), 0.0, None)
    return out


def confusion_counts(df):
    """tp / fp / fn / tn for one sensor-frame."""
    y = df["is_anomaly_actual"].to_numpy(int)
    p = df["is_anomaly_predicted"].to_numpy(int)
    return dict(
        tp=int(np.sum((p == 1) & (y == 1))),
        fp=int(np.sum((p == 1) & (y == 0))),
        fn=int(np.sum((p == 0) & (y == 1))),
        tn=int(np.sum((p == 0) & (y == 0))),
    )


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    if not (np.isfinite(prec) and np.isfinite(rec)) or (prec + rec) == 0:
        return prec, rec, np.nan
    return prec, rec, 2 * prec * rec / (prec + rec)


# ============================================================================
# SINGLE-SENSOR TIME SERIES
# ============================================================================


def plot_sensor_timeseries(df, title="", ax=None, figsize=(15, 5),
                           show_bands=True, start=None, end=None):
    """One sensor's prediction week: actual, prediction, acceptance band, and
    every reading classified as TP / FP / FN."""
    d = df.copy()
    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"])
    if start is not None:
        d = d[d["timestamp_utc"] >= pd.to_datetime(start)]
    if end is not None:
        d = d[d["timestamp_utc"] <= pd.to_datetime(end)]

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=figsize, dpi=100)

    t = d["timestamp_utc"]

    if show_bands and "upper_band" in d.columns:
        ax.fill_between(t, d["lower_band"], d["upper_band"],
                        color=COLOR_BAND, alpha=0.13, zorder=1,
                        label="acceptance band")
        ax.plot(t, d["upper_band"], color=COLOR_BAND, lw=0.8, ls="--", alpha=0.55, zorder=2)

    ax.plot(t, d["actual"], color=COLOR_ACTUAL, lw=1.3, label="actual", zorder=3)
    ax.plot(t, d["predicted"], color=COLOR_MODEL, lw=1.1, alpha=0.85,
            label="predicted", zorder=4)

    y = d["is_anomaly_actual"].to_numpy(int)
    p = d["is_anomaly_predicted"].to_numpy(int)
    for mask, colour, marker, lbl, size in (
        ((p == 1) & (y == 1), COLOR_TP, "o", "true positive", 90),
        ((p == 1) & (y == 0), COLOR_FP, "X", "false positive", 80),
        ((p == 0) & (y == 1), COLOR_FN, "v", "missed (FN)", 90),
    ):
        if mask.any():
            ax.scatter(t[mask], d["actual"].to_numpy()[mask], s=size, c=colour,
                       marker=marker, edgecolors="black", linewidths=0.6,
                       zorder=6, label=lbl)

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Diff (m³)")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.9)
    if created:
        plt.tight_layout()
    return ax


def compare_sensor_variants(frames, sensor, z_threshold=None, figsize=(15, 3.6)):
    """Stack the same sensor's week under several variants for visual diffing.

    `frames` maps variant name -> dict(df=..., signed=..., transform=..., scale=...).
    """
    n = len(frames)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n),
                             dpi=100, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (name, meta) in zip(axes, frames.items()):
        d = meta["df"]
        if z_threshold is not None:
            d = recompute_at_threshold(d, z_threshold, meta["signed"],
                                       meta.get("transform", "raw"),
                                       meta.get("scale", 1.0))
        c = confusion_counts(d)
        _, _, f1 = prf(c["tp"], c["fp"], c["fn"])
        ax.set_prop_cycle(None)
        plot_sensor_timeseries(
            d, ax=ax,
            title=f"{sensor} — {name}   "
                  f"(TP {c['tp']}  FP {c['fp']}  FN {c['fn']}  F1 {f1:.2f})",
        )
    plt.tight_layout()
    return fig


# ============================================================================
# CONFUSION MATRICES
# ============================================================================


def plot_confusion_matrices(counts_by_variant, figsize=(3.1, 3.0), ncols=3):
    """Grid of 2x2 confusion matrices, one per variant."""
    names = list(counts_by_variant)
    ncols = min(ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize[0] * ncols, figsize[1] * nrows),
                             dpi=100)
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        c = counts_by_variant[name]
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
        # Row-normalise: class imbalance is extreme (a handful of anomalies
        # against a full week of normal readings), so raw counts would render
        # the anomaly row invisible.
        with np.errstate(invalid="ignore"):
            norm = m / m.sum(axis=1, keepdims=True)
        ax.imshow(np.nan_to_num(norm), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(m[i, j])}\n{norm[i, j]*100:.1f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.5 else "black")
        _, _, f1 = prf(c["tp"], c["fp"], c["fn"])
        ax.set_title(f"{name}\nF1 = {f1:.3f}", fontsize=10)
        ax.set_xticks([0, 1], ["pred normal", "pred anom"], fontsize=8)
        ax.set_yticks([0, 1], ["true normal", "true anom"], fontsize=8)
    for ax in axes[len(names):]:
        ax.axis("off")
    plt.tight_layout()
    return fig


# ============================================================================
# HISTOGRAMS
# ============================================================================


def plot_zscore_histogram(frames, z_threshold=3.5, bins=60, figsize=(13, 4)):
    """z-score distribution split by true label, with the threshold marked.

    Separation between the two distributions is what a better detector actually
    buys; F1 at one threshold is just one vertical slice of this picture.
    """
    fig, axes = plt.subplots(1, len(frames), figsize=figsize, dpi=100, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, meta) in zip(axes, frames.items()):
        d = meta["df"]
        z = d["z_score"].to_numpy(float)
        stat = z if meta["signed"] else np.abs(z)
        y = d["is_anomaly_actual"].to_numpy(int)
        ok = np.isfinite(stat)
        lo, hi = np.nanpercentile(stat[ok], [0.5, 99.5]) if ok.any() else (0, 1)
        rng = (min(lo, -1) if meta["signed"] else 0, max(hi, z_threshold * 1.2))
        ax.hist(stat[ok & (y == 0)], bins=bins, range=rng, color=COLOR_ACTUAL,
                alpha=0.75, label="normal", log=True)
        ax.hist(stat[ok & (y == 1)], bins=bins, range=rng, color=COLOR_FN,
                alpha=0.8, label="anomaly", log=True)
        ax.axvline(z_threshold, color="black", ls="--", lw=1.3,
                   label=f"z = {z_threshold}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("signed z" if meta["signed"] else "|z|")
        ax.grid(alpha=0.2, ls=":")
    axes[0].set_ylabel("count (log)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_metric_histograms(summary, metrics=("mase_seasonal", "metric_mae"),
                           bins=25, figsize=(13, 4)):
    """Per-sensor distribution of accuracy metrics, one panel per metric,
    variants overlaid."""
    ok = summary[summary["status"] == "success"]
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, dpi=100)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        if metric not in ok.columns:
            ax.axis("off")
            continue
        vals = pd.to_numeric(ok[metric], errors="coerce")
        lo, hi = np.nanpercentile(vals.dropna(), [0, 98]) if vals.notna().any() else (0, 1)
        for name, g in ok.groupby("variant"):
            v = pd.to_numeric(g[metric], errors="coerce").dropna()
            ax.hist(v, bins=bins, range=(lo, hi), histtype="step", lw=1.8, label=name)
        ax.set_title(metric, fontsize=10)
        ax.grid(alpha=0.2, ls=":")
    axes[0].set_ylabel("sensors")
    axes[-1].legend(fontsize=7)
    plt.tight_layout()
    return fig


def plot_fp_distribution(summary, figsize=(7, 4)):
    """How concentrated the false positives are across sensors.

    Micro-F1 can look healthy while a few near-flat meters produce most of the
    alert volume -- this is the view that exposes it.
    """
    ok = summary[summary["status"] == "success"]
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    for name, g in ok.groupby("variant"):
        fp = pd.to_numeric(g["metric_pred_fp"], errors="coerce").fillna(0)
        fp = np.sort(fp.to_numpy())[::-1]
        share = np.cumsum(fp) / fp.sum() if fp.sum() > 0 else np.zeros_like(fp)
        ax.plot(np.arange(1, len(fp) + 1) / len(fp) * 100, share * 100,
                lw=1.8, label=name)
    ax.set_xlabel("% of sensors (worst first)")
    ax.set_ylabel("% of all false positives")
    ax.set_title("False-positive concentration", fontsize=11)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ============================================================================
# THRESHOLD SWEEP
# ============================================================================


def sweep_frames(frames, zs=None):
    """Micro P/R/F1 across thresholds, computed from the stored z-scores."""
    if zs is None:
        zs = np.round(np.arange(1.5, 6.01, 0.1), 2)
    rows = []
    for name, meta in frames.items():
        d = meta["df"]
        z = d["z_score"].to_numpy(float)
        stat = z if meta["signed"] else np.abs(z)
        y = d["is_anomaly_actual"].to_numpy(int)
        ok = np.isfinite(stat)
        stat, y = stat[ok], y[ok]
        for k in zs:
            p = stat > k
            tp = int(np.sum(p & (y == 1)))
            fp = int(np.sum(p & (y == 0)))
            fn = int(np.sum(~p & (y == 1)))
            prec, rec, f1 = prf(tp, fp, fn)
            rows.append(dict(variant=name, z=k, tp=tp, fp=fp, fn=fn,
                             precision=prec, recall=rec, f1=f1))
    return pd.DataFrame(rows)


def plot_sweep(sweep_df, z_threshold=None, figsize=(13, 4.4)):
    """F1-vs-threshold and the precision-recall curve.

    The PR curve is the comparison that matters: switching to signed bands
    changes what a given k MEANS (a signed band at k is far stricter than a
    folded one), so a fixed-k comparison confuses a moved operating point with
    a genuinely better detector.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=100)
    for name, g in sweep_df.groupby("variant"):
        g = g.sort_values("z")
        ax1.plot(g["z"], g["f1"], lw=1.8, label=name)
        ax2.plot(g["recall"], g["precision"], lw=1.8, marker="", label=name)
    if z_threshold is not None:
        ax1.axvline(z_threshold, color="black", ls="--", lw=1.2, alpha=0.7)
    ax1.set_xlabel("detection threshold z")
    ax1.set_ylabel("micro F1")
    ax1.set_title("F1 vs threshold", fontsize=11)
    ax1.grid(alpha=0.25, ls=":")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title("Precision-recall (threshold swept)", fontsize=11)
    ax2.grid(alpha=0.25, ls=":")
    plt.tight_layout()
    return fig


# ============================================================================
# TABLES
# ============================================================================


def metrics_table(frames, z_threshold=None):
    """Micro-averaged metrics per variant at one threshold.

    Micro means: sum tp/fp/fn across sensors, then compute once. Averaging
    per-sensor F1 would hide false-positive floods.
    """
    rows = []
    for name, meta in frames.items():
        d = meta["df"]
        if z_threshold is not None:
            d = recompute_at_threshold(d, z_threshold, meta["signed"],
                                       meta.get("transform", "raw"),
                                       meta.get("scale", 1.0))
        c = confusion_counts(d)
        prec, rec, f1 = prf(c["tp"], c["fp"], c["fn"])
        rows.append(dict(variant=name, **c, precision=prec, recall=rec, f1=f1))
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def micro_table_multi(all_frames, z_threshold=None):
    """Micro metrics per variant pooled across every sensor in `all_frames`.

    `all_frames` maps sensor -> {variant -> meta}.
    """
    agg = {}
    for sensor, frames in all_frames.items():
        for name, meta in frames.items():
            d = meta["df"]
            if z_threshold is not None:
                d = recompute_at_threshold(d, z_threshold, meta["signed"],
                                           meta.get("transform", "raw"),
                                           meta.get("scale", 1.0))
            c = confusion_counts(d)
            a = agg.setdefault(name, dict(tp=0, fp=0, fn=0, tn=0, n_sensors=0))
            for k in ("tp", "fp", "fn", "tn"):
                a[k] += c[k]
            a["n_sensors"] += 1

    rows = []
    for name, a in agg.items():
        prec, rec, f1 = prf(a["tp"], a["fp"], a["fn"])
        rows.append(dict(variant=name, **a, precision=prec, recall=rec, f1=f1,
                         fp_per_sensor_week=a["fp"] / a["n_sensors"]))
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def pooled_counts(all_frames, z_threshold=None):
    """Confusion counts per variant pooled over sensors (for the matrix grid)."""
    out = {}
    for sensor, frames in all_frames.items():
        for name, meta in frames.items():
            d = meta["df"]
            if z_threshold is not None:
                d = recompute_at_threshold(d, z_threshold, meta["signed"],
                                           meta.get("transform", "raw"),
                                           meta.get("scale", 1.0))
            c = confusion_counts(d)
            acc = out.setdefault(name, dict(tp=0, fp=0, fn=0, tn=0))
            for k in acc:
                acc[k] += c[k]
    return out


def pool_frames(all_frames):
    """Flatten sensor -> variant -> meta into variant -> meta over all sensors."""
    pooled = {}
    for sensor, frames in all_frames.items():
        for name, meta in frames.items():
            if name not in pooled:
                pooled[name] = dict(df=[], signed=meta["signed"],
                                    transform=meta.get("transform", "raw"),
                                    scale=meta.get("scale", 1.0))
            pooled[name]["df"].append(meta["df"])
    for name, meta in pooled.items():
        meta["df"] = pd.concat(meta["df"], ignore_index=True)
    return pooled
