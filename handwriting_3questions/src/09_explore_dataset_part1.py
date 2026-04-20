"""
09_explore_dataset_part1.py
────────────────────────────
Dataset explorer — mirrors the explore_dataset.ipynb notebook.

Generates figures and summary tables and writes everything to
  results/part1_results/

Sections
────────
  1.  Config summary
  2.  Load preprocessed trials
  3.  Class distribution bar chart
  4.  Session & block structure table
  5.  Train / val / test split summary
  6.  Mean neural activity heatmaps (8 highlight characters)
  7.  Single-trial vs mean channel traces
  8.  Feature representations (flat + temporal shapes)
  9.  PCA of flat features — 2-D scatter (PC1v2, PC3v4)
  10. Pre-computed feature NPZ inspection
  11. Saved model-results tables
  12. Preprocessing summary

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/09_explore_dataset_part1.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# ── project root & src/ on path ────────────────────────────────────────────
proj_root = Path(__file__).resolve().parent.parent
os.chdir(proj_root)
sys.path.insert(0, str(proj_root / "src"))

from utils_data     import load_config, load_processed, load_splits
from utils_features import build_flat, build_temporal, temporal_window_boundaries

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


# ── output directory ────────────────────────────────────────────────────────
OUT_DIR = proj_root / "results" / "part1_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "summary.txt"
_log_lines: list[str] = []


def log(msg: str = "") -> None:
    """Print and buffer for the text summary file."""
    print(msg)
    _log_lines.append(msg)


def save_log() -> None:
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_log_lines) + "\n")
    print(f"\nSummary written → {LOG_PATH}")


def savefig(name: str) -> None:
    path = OUT_DIR / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"  [figure] {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Config
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 1 — Config")
log("=" * 70)

cfg = load_config()
log(f"Dataset root   : {cfg['dataset_root']}")
log(f"Sessions       : {len(cfg['sessions'])}")
log(f"Characters     : {len(cfg['characters'])}")
log(f"T_fixed        : {cfg['T_fixed']} bins  ({cfg['T_fixed'] * cfg['dt_ms']:.0f} ms)")
log(f"n_channels     : {cfg['n_channels']}")
log(f"Smoothing σ    : {cfg['sigma_ms']} ms  → {cfg['sigma_ms']/cfg['dt_ms']:.1f} bins")
log(f"Movement window: [{cfg['move_win_start']}, {cfg['move_win_end']})  "
    f"(go-cue at bin {cfg['go_bin']})")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Load preprocessed trials
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 2 — Load preprocessed trials")
log("=" * 70)

trials, labels, session_ids, block_ids, trial_ids = load_processed(cfg)
N, T, C = trials.shape

log(f"trials shape   : {trials.shape}  (N × T × C)")
log(f"dtype          : {trials.dtype}")
log(f"value range    : [{trials.min():.3f}, {trials.max():.3f}]")
log(f"mean / std     : {trials.mean():.4f} / {trials.std():.4f}")
log(f"unique labels  : {np.unique(labels).tolist()}")
log(f"unique sessions: {np.unique(session_ids).tolist()}")

t_axis = np.arange(T) * cfg["dt_ms"]   # ms from movement onset

# ══════════════════════════════════════════════════════════════════════════════
# 3. Class distribution
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 3 — Class distribution")
log("=" * 70)

counts = pd.Series(labels, name="label").value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white", linewidth=0.4)
ax.set_xlabel("Character class")
ax.set_ylabel("Trial count")
ax.set_title(f"Class distribution  (N={N}, {len(counts)} classes)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
savefig("fig01_class_distribution.png")

counts_df = counts.reset_index()
counts_df.columns = ["label", "n_trials"]
counts_df.to_csv(OUT_DIR / "class_counts.csv", index=False)
log(counts.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 4. Session & block structure
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 4 — Session & block structure")
log("=" * 70)

meta_df = pd.DataFrame({
    "trial_id": trial_ids,
    "session":  session_ids,
    "block":    block_ids,
    "label":    labels,
})
sess_summary = meta_df.groupby("session").agg(
    n_trials  = ("trial_id", "count"),
    n_blocks  = ("block",    "nunique"),
    n_classes = ("label",    "nunique"),
).reset_index()
log(sess_summary.to_string(index=False))
sess_summary.to_csv(OUT_DIR / "session_block_summary.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Train / val / test splits
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 5 — Train / val / test splits")
log("=" * 70)

splits = load_splits(cfg)
split_rows = []
for split_name, sp in splits.items():
    if not isinstance(sp, dict):
        continue
    tr = len(sp["train_idx"])
    va = len(sp["val_idx"])
    te = len(sp["test_idx"])
    total = tr + va + te
    log(f"[{split_name:20s}]  train={tr:5d} ({100*tr/total:.0f}%)  "
        f"val={va:4d} ({100*va/total:.0f}%)  "
        f"test={te:4d} ({100*te/total:.0f}%)  total={total}")
    split_rows.append({"split": split_name, "train": tr, "val": va,
                       "test": te, "total": total})

pd.DataFrame(split_rows).to_csv(OUT_DIR / "split_summary.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Mean neural activity heatmaps
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 6 — Mean neural activity heatmaps")
log("=" * 70)

highlight = ["a", "b", "e", "t", "m", "o", "s", "r"]
vmax = 2.5

fig, axes = plt.subplots(2, 4, figsize=(14, 5), sharey=True)
axes = axes.ravel()

for ax, char in zip(axes, highlight):
    mask = labels == char
    mean_tc = trials[mask].mean(axis=0)      # (T, C)
    im = ax.imshow(
        mean_tc.T,
        aspect="auto",
        origin="lower",
        extent=[t_axis[0], t_axis[-1], 0, C],
        vmin=-vmax, vmax=vmax,
        cmap="RdBu_r",
    )
    ax.set_title(f"'{char}'  (n={mask.sum()})")
    ax.set_xlabel("Time from go-cue (ms)")
    if ax is axes[0] or ax is axes[4]:
        ax.set_ylabel("Channel")

fig.colorbar(im, ax=axes, shrink=0.6, label="z-score")
fig.suptitle("Mean neural activity per character  (channels × time)", y=1.01)
plt.tight_layout()
savefig("fig02_mean_neural_activity_heatmaps.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Single-trial vs mean channel traces
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 7 — Single-trial vs mean channel traces")
log("=" * 70)

CHARS_TO_SHOW  = ["a", "t", "o"]
TOP_K_CHANNELS = 5

class_means = np.stack(
    [trials[labels == c].mean(axis=0) for c in np.unique(labels)], axis=0
)  # (n_classes, T, C)
channel_var = class_means.var(axis=0).mean(axis=0)   # (C,)
top_ch = np.argsort(channel_var)[::-1][:TOP_K_CHANNELS]
log(f"Top-{TOP_K_CHANNELS} most discriminative channels: {top_ch.tolist()}")

fig, axes = plt.subplots(
    len(CHARS_TO_SHOW), TOP_K_CHANNELS,
    figsize=(14, 3 * len(CHARS_TO_SHOW)),
    sharey="row",
)
for row, char in enumerate(CHARS_TO_SHOW):
    mask = labels == char
    char_trials = trials[mask]           # (n_trials, T, C)
    for col, ch in enumerate(top_ch):
        ax = axes[row, col]
        for single_tr in char_trials:
            ax.plot(t_axis, single_tr[:, ch], color="steelblue", alpha=0.15, lw=0.7)
        ax.plot(t_axis, char_trials[:, :, ch].mean(axis=0),
                color="darkred", lw=1.8)
        ax.set_title(f"ch {ch}", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"'{char}'  z-score")
        if row == len(CHARS_TO_SHOW) - 1:
            ax.set_xlabel("ms")

fig.suptitle(
    f"Top-{TOP_K_CHANNELS} discriminative channels — trials (blue) + mean (red)",
    y=1.01,
)
plt.tight_layout()
savefig("fig03_single_trial_traces.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Feature representations
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 8 — Feature representations")
log("=" * 70)

all_idx    = np.arange(N)
X_flat     = build_flat(trials, all_idx)       # (N, T*C)
X_temporal = build_temporal(trials, all_idx)   # (N, 3*C)

log(f"Flat feature shape    : {X_flat.shape}   ({T} bins × {C} channels)")
log(f"Temporal feature shape: {X_temporal.shape}  (3 windows × {C} channels)")

windows = temporal_window_boundaries(cfg)
for name, bounds in windows.items():
    s, e = bounds[0], bounds[1]
    log(f"  window '{name}': bins [{s}, {e})  → {(e-s)*cfg['dt_ms']:.0f} ms")

# ══════════════════════════════════════════════════════════════════════════════
# 9. PCA of flat features
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 9 — PCA of flat features")
log("=" * 70)

sp_ba   = splits["block_aware"]
tr_idx  = np.array(sp_ba["train_idx"])

pca = PCA(n_components=10, random_state=42)
pca.fit(X_flat[tr_idx])
Z = pca.transform(X_flat)   # (N, 10)

cum_var = 100 * pca.explained_variance_ratio_.cumsum()
log(f"Explained variance (top 10 PCs): {cum_var[-1]:.1f}%")
log("  " + "  ".join(
    f"PC{i+1}:{100*v:.1f}%"
    for i, v in enumerate(pca.explained_variance_ratio_)
))

le = LabelEncoder()
le.fit(labels)
classes = list(le.classes_)
n_classes = len(classes)
cmap = plt.cm.get_cmap("tab20", n_classes)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (pc_x, pc_y, title) in zip(
    axes,
    [(0, 1, "PC1 vs PC2"), (2, 3, "PC3 vs PC4")],
):
    for ci, char in enumerate(classes):
        mask = labels == char
        ax.scatter(Z[mask, pc_x], Z[mask, pc_y],
                   color=cmap(ci), s=8, alpha=0.5)
    ax.set_xlabel(f"PC{pc_x+1} ({100*pca.explained_variance_ratio_[pc_x]:.1f}%)")
    ax.set_ylabel(f"PC{pc_y+1} ({100*pca.explained_variance_ratio_[pc_y]:.1f}%)")
    ax.set_title(title)

handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap(i), markersize=6)
    for i in range(n_classes)
]
axes[1].legend(handles, classes, fontsize=6, ncol=4,
               bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
fig.suptitle("PCA of flat features (all trials, colour = character class)")
plt.tight_layout()
savefig("fig04_pca_scatter.png")

# PCA scree plot
fig, ax = plt.subplots(figsize=(6, 3.5))
pc_nums = np.arange(1, 11)
ax.bar(pc_nums, 100 * pca.explained_variance_ratio_,
       color="steelblue", edgecolor="white")
ax.plot(pc_nums, cum_var, "o-", color="darkred", lw=1.5, label="cumulative")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance (%)")
ax.set_title("PCA scree plot — flat features (block-aware train split)")
ax.legend()
plt.tight_layout()
savefig("fig05_pca_scree.png")

pd.DataFrame({
    "pc": pc_nums,
    "explained_var_pct": 100 * pca.explained_variance_ratio_,
    "cumulative_var_pct": cum_var,
}).to_csv(OUT_DIR / "pca_scree.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 10. Pre-computed feature NPZ inspection
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 10 — Pre-computed feature NPZ inspection")
log("=" * 70)

processed_dir = Path(cfg["processed_dir"])
npz_rows = []
for feat_type in ["flat", "temporal"]:
    for split_name in ["block_aware", "random_trial"]:
        npz_path = processed_dir / f"features_{feat_type}_{split_name}.npz"
        if not npz_path.exists():
            log(f"  [missing]  {npz_path.name}")
            continue
        d = np.load(npz_path, allow_pickle=True)
        row = {
            "file":       npz_path.name,
            "feat_type":  feat_type,
            "split":      split_name,
            "train_shape": str(d["X_train"].shape),
            "val_shape":   str(d["X_val"].shape),
            "test_shape":  str(d["X_test"].shape),
        }
        npz_rows.append(row)
        log(f"  {npz_path.name:45s}  "
            f"train={d['X_train'].shape}  "
            f"val={d['X_val'].shape}  "
            f"test={d['X_test'].shape}")

if npz_rows:
    pd.DataFrame(npz_rows).to_csv(OUT_DIR / "feature_npz_shapes.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 11. Saved model-results tables
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 11 — Saved model-results tables")
log("=" * 70)

tables_dir = Path(cfg["tables_dir"])
for tbl in ["all_models_summary.csv", "linear_metrics.csv",
            "nonlinear_metrics.csv", "pca_metrics.csv"]:
    p = tables_dir / tbl
    if not p.exists():
        log(f"  [not found] {tbl}")
        continue
    df = pd.read_csv(p)
    log(f"\n{tbl}  ({len(df)} rows, cols: {list(df.columns)})")
    sort_col = next((c for c in ["test_acc", "test_accuracy"] if c in df.columns), None)
    df_show = df.sort_values(sort_col, ascending=False).head(10) if sort_col else df.head(10)
    log(df_show.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 12. Preprocessing summary
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("=" * 70)
log("SECTION 12 — Preprocessing summary")
log("=" * 70)

preproc_path = Path(cfg["metadata_dir"]) / "preprocessing_summary.json"
with open(preproc_path) as f:
    preproc = json.load(f)

log(f"Sessions processed : {preproc['n_sessions']}")
log(f"Total trials       : {preproc['n_trials']}")
log(f"Excluded doNothing : {preproc['excluded_doNothing_trials']}")
log(f"T_fixed / channels : {preproc['T_fixed']} bins / {preproc['n_channels']} ch")
log(f"N classes          : {preproc['n_classes']}")
log("Preprocessing steps:")
for step in preproc["preprocessing"]["steps"]:
    log(f"  {step}")

# ── finalise ────────────────────────────────────────────────────────────────
log("")
log("=" * 70)
log(f"All outputs written to: {OUT_DIR}")
log("=" * 70)
save_log()
