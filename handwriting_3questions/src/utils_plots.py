"""
utils_plots.py
───────────────
Shared plotting helpers used across report and analysis scripts.

All functions save their output to disk and close the figure to avoid
memory leaks in long-running scripts.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# Global readability defaults for all exported figures.
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
})


# ─── Model order and colours ─────────────────────────────────────────────────

# Canonical display order: linear models first, then non-linear models.
MODEL_ORDER = [
    "Logistic",
    "LinearSVM",
    "Ridge",
    "SVM",
    "MLP",
]


def _sort_models(models: list[str]) -> list[str]:
    """Return models in canonical order (linear first, non-linear after)."""
    known   = [m for m in MODEL_ORDER if m in models]
    unknown = [m for m in models if m not in MODEL_ORDER]
    return known + sorted(unknown)


def _model_colors() -> dict[str, str]:
    return {
        "Logistic": "#4C72B0",
        "LinearSVM": "#DD8452",
        "Ridge": "#55A868",
        "SVM": "#C44E52",
        "MLP": "#8172B3",
    }


def _save_figure_both_formats(save_path: Path, dpi: int = 120) -> None:
    """
    Save current matplotlib figure to both PNG and SVG.
    `save_path` controls basename; suffix is ignored/replaced.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    base = save_path.with_suffix("")
    plt.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(base.with_suffix(".svg"), bbox_inches="tight")


# ─── Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_arr: np.ndarray,
                          classes: Sequence[str],
                          title: str,
                          save_path: Path,
                          cmap: str = "Blues",
                          annotate: bool = True,
                          dpi: int = 120) -> None:
    """
    Plot and save a confusion matrix.

    Parameters
    ----------
    cm_arr    : (n_classes, n_classes) int array
    classes   : ordered list of class name strings
    title     : figure title
    save_path : output file path (.png)
    """
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(7, n * 0.3)))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_title(title, fontsize=11, pad=10)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_ylabel("True label", fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=9)

    if annotate and n <= 40:
        thresh = cm_arr.max() / 2.0
        for i, j in product(range(n), range(n)):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center", fontsize=4,
                    color="white" if cm_arr[i, j] > thresh else "black")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


# ─── Bar chart: flat vs temporal ─────────────────────────────────────────────

def plot_flat_vs_temporal(df: pd.DataFrame,
                          metric: str = "test_accuracy",
                          save_path: Path | None = None,
                          dpi: int = 120) -> None:
    """
    Grouped bar chart comparing flat and temporal features across models and
    split strategies (Q1).

    Expected df columns: model, features, split, <metric>
    """
    splits  = sorted(df["split"].unique())
    models  = _sort_models(list(df["model"].unique()))
    colors  = _model_colors()
    feat_orders = ["flat", "temporal"]

    fig, axes = plt.subplots(1, len(splits),
                             figsize=(7 * len(splits), 5), sharey=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, split_name in zip(axes, splits):
        sub = df[df["split"] == split_name]
        x = np.arange(len(models))
        width = 0.35
        panel_hatch = "" if split_name == "block_aware" else "//"

        for k, feat in enumerate(feat_orders):
            vals = []
            for m in models:
                row = sub[(sub["model"] == m) & (sub["features"] == feat)][metric].values
                vals.append(float(np.mean(row)) if len(row) else 0.0)
            offset = (k - 0.5) * width
            ax.bar(
                x + offset, vals, width,
                label=feat,
                color=[colors.get(m, "#888888") for m in models],
                alpha=(1.0 if feat == "flat" else 0.55),
                hatch=panel_hatch,
                edgecolor="white", linewidth=0.5
            )

        ax.set_title(f"Split: {split_name}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.legend(title="Features", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Q1 – Flat vs Temporal Features", fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


# ─── Bar chart: block-aware vs random-trial ───────────────────────────────────

def plot_block_vs_random(df: pd.DataFrame,
                         metric: str = "test_accuracy",
                         save_path: Path | None = None,
                         dpi: int = 120) -> None:
    """
    Grouped bar chart comparing block_aware and random_trial splits for all
    models (Q2).

    Expected df columns: model, split, features, <metric>
    """
    sub = df[df["features"] == "flat"].copy()
    models = _sort_models(list(sub["model"].unique()))
    split_order = ["block_aware", "random_trial"]
    splits = [s for s in split_order if (sub["split"] == s).any()]
    if len(splits) == 0:
        return
    colors = _model_colors()

    x = np.arange(len(models))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(9, 5))

    for k, split_name in enumerate(splits):
        vals = []
        for m in models:
            row = sub[(sub["model"] == m) & (sub["split"] == split_name)][metric].values
            vals.append(float(np.mean(row)) if len(row) else 0.0)
        offset = (k - (len(splits) - 1) / 2.0) * width
        hatch = "" if split_name == "block_aware" else "//"
        ax.bar(
            x + offset, vals, width,
            label=split_name.replace("_", " "),
            color=[colors.get(m, "#888888") for m in models],
            alpha=1.0,
            hatch=hatch,
            edgecolor="white", linewidth=0.5
        )

    ax.set_title("Q2 – Block-aware vs Random-trial (flat features)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.legend(title="Split", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


# ─── PCA decoder curve ───────────────────────────────────────────────────────

def plot_pca_decoder_curves(df: pd.DataFrame,
                            metric: str = "test_accuracy",
                            split_filter: str = "block_aware",
                            save_path: Path | None = None,
                            dpi: int = 120) -> None:
    """
    Line plot of decoding performance vs number of PCA components (Q3).

    Expected df columns: model, n_components, split, features, <metric>
    Includes a horizontal dashed line for each model's raw-flat baseline.
    """
    sub = df[df["split"] == split_filter].copy()
    models  = _sort_models(list(sub["model"].unique()))
    colors  = _model_colors()

    fig, ax = plt.subplots(figsize=(9, 5))

    for m in models:
        color = colors.get(m, "#888888")
        # PCA curve
        pca_rows = sub[(sub["model"] == m) & (sub["features"] == "flat_pca")]
        if len(pca_rows):
            pca_rows = pca_rows.sort_values("n_components")
            ax.plot(pca_rows["n_components"], pca_rows[metric],
                    marker="o", label=f"{m} (PCA)", color=color)
        # Baseline (raw flat)
        flat_row = sub[(sub["model"] == m) & (sub["features"] == "flat")]
        if len(flat_row):
            ax.axhline(y=float(flat_row[metric].values[0]),
                       color=color, linestyle="--", alpha=0.6,
                       label=f"{m} (flat baseline)")

    ax.set_xscale("log")
    ax.set_xlabel("Number of PCA components (log scale)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Q3 – PCA Decoder Curve ({split_filter})", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


# ─── Explained variance plot ─────────────────────────────────────────────────

def plot_explained_variance(cumvar: np.ndarray,
                            k_list: list[int],
                            save_path: Path | None = None,
                            dpi: int = 120) -> None:
    """Plot cumulative PCA explained variance curve."""
    fig, ax = plt.subplots(figsize=(7, 4))
    total_k = len(cumvar)
    ax.plot(np.arange(1, total_k + 1), cumvar * 100, color="#4C72B0")
    for k in k_list:
        k_eff = min(k, total_k)
        ax.axvline(x=k_eff, color="gray", linestyle="--", alpha=0.5)
        ax.text(k_eff, cumvar[k_eff - 1] * 100 + 1,
                f"k={k_eff}\n{cumvar[k_eff-1]*100:.0f}%",
                fontsize=7, ha="center")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Explained Variance vs PCA Components")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


# ─── Class distribution ──────────────────────────────────────────────────────

def plot_class_distribution(labels: np.ndarray,
                            save_path: Path | None = None,
                            dpi: int = 120) -> None:
    """Bar chart of class frequencies."""
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(len(unique)), counts[order], color="#4C72B0", edgecolor="white")
    ax.set_xticks(np.arange(len(unique)))
    ax.set_xticklabels(unique[order], rotation=90, fontsize=8)
    ax.set_ylabel("Trial count")
    ax.set_title("Class Distribution")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


# ─── PCA trajectory: 2D ──────────────────────────────────────────────────────

def plot_trajectories_2d(mean_trajectories: dict[str, np.ndarray],
                         highlight_chars: list[str] | None = None,
                         title: str = "Mean Neural Trajectories (PC1–PC2)",
                         save_path: Path | None = None,
                         dpi: int = 120) -> None:
    """
    Plot mean PC1-PC2 trajectories for each character.

    Each curve shows how the population state moves over time during the
    movement window.  A dot marks the end of the trajectory.

    Parameters
    ----------
    mean_trajectories : {char: (T, n_pcs)} dict
    highlight_chars   : characters to label prominently (others dimmed)
    """
    chars = list(mean_trajectories.keys())
    cmap_obj = cm.get_cmap("tab20", len(chars))
    if highlight_chars is None:
        highlight_chars = chars

    fig, ax = plt.subplots(figsize=(9, 7))
    for k, char in enumerate(chars):
        traj = mean_trajectories[char]   # (T, n_pcs)
        pc1, pc2 = traj[:, 0], traj[:, 1]
        alpha = 1.0 if char in highlight_chars else 0.2
        lw    = 1.8 if char in highlight_chars else 0.7
        color = cmap_obj(k)
        ax.plot(pc1, pc2, color=color, alpha=alpha, linewidth=lw)
        ax.scatter(pc1[-1], pc2[-1], color=color, alpha=alpha,
                   s=20, zorder=3)
        if char in highlight_chars:
            ax.text(pc1[0], pc2[0], char, fontsize=8,
                    color=color, fontweight="bold")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


# ─── Centroid distance heatmap ────────────────────────────────────────────────

def plot_centroid_distance_heatmap(dist_mat: np.ndarray,
                                   classes: list[str],
                                   save_path: Path | None = None,
                                   dpi: int = 120) -> None:
    """Heatmap of pairwise centroid distances in PC space."""
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.35), max(6, n * 0.3)))
    im = ax.imshow(dist_mat, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Euclidean distance")
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_title("Pairwise Centroid Distance (PC space)", fontsize=11)
    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


# ─── Q1/Q2 enhanced summary figures ───────────────────────────────────────────

def _metric_lookup(df: pd.DataFrame,
                   model: str,
                   split: str,
                   features: str,
                   metric: str) -> float | None:
    row = df[(df["model"] == model) &
             (df["split"] == split) &
             (df["features"] == features)]
    if len(row) == 0:
        return None
    return float(row[metric].values[0])


def plot_q1_slope(df: pd.DataFrame,
                  metric: str = "test_accuracy",
                  save_path: Path | None = None,
                  dpi: int = 120) -> None:
    """
    Slope chart for Q1:
      per model, line from temporal -> flat for each split.
    """
    models = _sort_models(list(df["model"].unique()))
    colors = _model_colors()
    split_styles = {
        "block_aware": ("o", "-"),
        "random_trial": ("s", "--"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x_temporal, x_flat = 0, 1

    for m in models:
        for split, (marker, linestyle) in split_styles.items():
            y_t = _metric_lookup(df, m, split, "temporal", metric)
            y_f = _metric_lookup(df, m, split, "flat", metric)
            if y_t is None or y_f is None:
                continue
            ax.plot([x_temporal, x_flat], [y_t, y_f],
                    color=colors.get(m, "#888888"),
                    linestyle=linestyle, marker=marker,
                    linewidth=1.8, markersize=5, alpha=0.9)

    ax.set_xticks([x_temporal, x_flat])
    ax.set_xticklabels(["temporal", "flat"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Q1 Slope Chart – Temporal → Flat")
    ax.grid(axis="y", alpha=0.3)

    # Compact legend explaining split marker/line style
    dummy1, = ax.plot([], [], color="#444444", marker="o", linestyle="-",
                      label="block_aware")
    dummy2, = ax.plot([], [], color="#444444", marker="s", linestyle="--",
                      label="random_trial")
    ax.legend(handles=[dummy1, dummy2], title="Split", fontsize=8, loc="best")

    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


def plot_q2_slope(df: pd.DataFrame,
                  metric: str = "test_accuracy",
                  save_path: Path | None = None,
                  dpi: int = 120) -> None:
    """
    Slope chart for Q2:
      per model, line from block_aware -> random_trial (flat features).
    """
    sub = df[df["features"] == "flat"].copy()
    models = _sort_models(list(sub["model"].unique()))
    colors = _model_colors()

    fig, ax = plt.subplots(figsize=(8, 5))
    x_ba, x_rt = 0, 1

    for m in models:
        y_ba = _metric_lookup(sub, m, "block_aware", "flat", metric)
        y_rt = _metric_lookup(sub, m, "random_trial", "flat", metric)
        if y_ba is None or y_rt is None:
            continue
        ax.plot([x_ba, x_rt], [y_ba, y_rt],
                color=colors.get(m, "#888888"),
                marker="o", linewidth=2.0, markersize=5, alpha=0.9)

    ax.set_xticks([x_ba, x_rt])
    ax.set_xticklabels(["block_aware", "random_trial"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Q2 Slope Chart – Block-aware → Random-trial (flat)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


def _plot_delta_heatmap(models: list[str],
                        cols: list[str],
                        mat: np.ndarray,
                        title: str,
                        save_path: Path | None = None,
                        dpi: int = 120) -> None:
    """Internal helper for rendering delta heatmaps."""
    vmax = np.nanmax(np.abs(mat)) if np.any(~np.isnan(mat)) else 0.05
    vmax = max(vmax, 0.02)

    fig, ax = plt.subplots(figsize=(max(5.8, 1.9 + 1.1 * len(cols)),
                                    max(3.5, 0.7 * len(models))))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="Delta accuracy")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title(title)

    for i in range(len(models)):
        for j in range(len(cols)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.3f}",
                        ha="center", va="center", fontsize=8, color="#111111")

    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()


def plot_q1_delta_heatmap_random_flat_baseline(df: pd.DataFrame,
                                               metric: str = "test_accuracy",
                                               save_path: Path | None = None,
                                               dpi: int = 120) -> None:
    """
    Q1-only delta heatmap with per-model baseline = random_trial + flat.

    Columns are:
      rt_flat_baseline : always 0 (reference)
      rt_temporal      : random_trial temporal - random_trial flat
      ba_flat          : block_aware flat - random_trial flat
      ba_temporal      : block_aware temporal - random_trial flat
    """
    models = _sort_models(list(df["model"].unique()))
    cols = ["rt_flat_baseline", "rt_temporal", "ba_flat", "ba_temporal"]
    mat = np.full((len(models), len(cols)), np.nan, dtype=float)

    for i, m in enumerate(models):
        rt_flat = _metric_lookup(df, m, "random_trial", "flat", metric)
        rt_temp = _metric_lookup(df, m, "random_trial", "temporal", metric)
        ba_flat = _metric_lookup(df, m, "block_aware", "flat", metric)
        ba_temp = _metric_lookup(df, m, "block_aware", "temporal", metric)
        if rt_flat is None:
            continue
        mat[i, 0] = 0.0
        if rt_temp is not None:
            mat[i, 1] = rt_temp - rt_flat
        if ba_flat is not None:
            mat[i, 2] = ba_flat - rt_flat
        if ba_temp is not None:
            mat[i, 3] = ba_temp - rt_flat

    _plot_delta_heatmap(
        models=models,
        cols=cols,
        mat=mat,
        title="Q1 Delta Heatmap (baseline: random_trial flat)",
        save_path=save_path,
        dpi=dpi,
    )


def plot_q2_delta_heatmap_random_flat_baseline(df: pd.DataFrame,
                                               metric: str = "test_accuracy",
                                               save_path: Path | None = None,
                                               dpi: int = 120) -> None:
    """
    Q2-only delta heatmap with per-model baseline = random_trial + flat.

    Columns are:
      rt_flat_baseline : always 0 (reference)
      ba_flat          : block_aware flat - random_trial flat
    """
    models = _sort_models(list(df["model"].unique()))
    cols = ["rt_flat_baseline", "ba_flat"]
    mat = np.full((len(models), len(cols)), np.nan, dtype=float)

    for i, m in enumerate(models):
        rt_flat = _metric_lookup(df, m, "random_trial", "flat", metric)
        ba_flat = _metric_lookup(df, m, "block_aware", "flat", metric)
        if rt_flat is None:
            continue
        mat[i, 0] = 0.0
        if ba_flat is not None:
            mat[i, 1] = ba_flat - rt_flat

    _plot_delta_heatmap(
        models=models,
        cols=cols,
        mat=mat,
        title="Q2 Delta Heatmap (baseline: random_trial flat)",
        save_path=save_path,
        dpi=dpi,
    )


def plot_rank_shift(df: pd.DataFrame,
                    metric: str = "test_accuracy",
                    save_path: Path | None = None,
                    dpi: int = 120) -> None:
    """
    Rank-shift plot across three conditions:
      Q1_block(flat), Q1_random(flat), Q2_temporal(block_aware temporal)
    showing model stability by rank movement.
    """
    models = _sort_models(list(df["model"].unique()))
    colors = _model_colors()

    conditions = [
        ("Q1_block", ("block_aware", "flat")),
        ("Q1_random", ("random_trial", "flat")),
        ("Q2_temporal", ("block_aware", "temporal")),
    ]

    # Build score table
    score = pd.DataFrame(index=models, columns=[c[0] for c in conditions], dtype=float)
    for cname, (split, feat) in conditions:
        for m in models:
            v = _metric_lookup(df, m, split, feat, metric)
            score.loc[m, cname] = np.nan if v is None else v

    # Rank within each condition: 1 = best
    ranks = score.rank(axis=0, ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.6 * len(models))))
    x = np.arange(len(conditions))

    for m in models:
        y = ranks.loc[m].values.astype(float)
        if np.any(np.isnan(y)):
            continue
        ax.plot(x, y, marker="o", linewidth=2,
                color=colors.get(m, "#888888"), label=m, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions])
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Model Rank Shift Across Conditions")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")

    plt.tight_layout()
    if save_path:
        _save_figure_both_formats(save_path, dpi=dpi)
    plt.close()
