"""
07_make_trajectory_pca_figures.py
───────────────────────────────────
PCA-based neural trajectory analysis and visualization.

This is NOT a classification benchmark.  The goal is to visualize whether
different characters follow separable trajectories in low-dimensional neural
activity space, providing geometric intuition for why decoding works.

Method
──────
  1. Stack all training-set timepoints: (N_train * T_fixed, C)
  2. Fit PCA across channels (scaler + PCA on this matrix)
  3. Project every trial into the top n_pcs PCs → (T_fixed, n_pcs)
  4. Compute per-character mean trajectories (training trials only)
  5. Compute per-character centroids (mean over time of mean trajectory)
  6. Compute pairwise Euclidean centroid distances
  7. Plot 2D and optionally 3D trajectories
  8. Plot centroid heatmap

Outputs
───────
  results/figures/trajectory_pca_2d.png
  results/figures/trajectory_pca_3d.png          (if n_pcs >= 3)
  results/figures/centroid_distance_heatmap.png
  results/pca/trajectory_embeddings.npz
  results/pca/character_centroids.csv
  results/pca/centroid_distance_matrix.csv

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/07_make_trajectory_pca_figures.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_data  import load_config, load_processed, load_splits
from utils_models import make_label_encoder
from utils_plots  import plot_trajectories_2d, plot_centroid_distance_heatmap


def compute_trajectory_pca(
    trials: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    n_pcs: int,
) -> dict:
    """
    Fit PCA on stacked training timepoints and project all trials.

    Parameters
    ----------
    trials    : (N, T_fixed, C)
    labels    : (N,) str
    train_idx : indices of training-set trials
    n_pcs     : number of PCA components

    Returns
    -------
    dict with embeddings, mean trajectories, centroids, distances, etc.
    """
    T = trials.shape[1]

    # Stack all training timepoints: (N_train * T, C)
    print(f"  Stacking {len(train_idx)} training trials × {T} bins …",
          end=" ", flush=True)
    stack = np.vstack([trials[i] for i in train_idx])   # (N*T, C)
    print(f"shape {stack.shape}")

    scaler = StandardScaler()
    stack_sc = scaler.fit_transform(stack)

    k = min(n_pcs, stack_sc.shape[0], stack_sc.shape[1])
    print(f"  Fitting PCA (n_components={k}) …", end=" ", flush=True)
    pca = PCA(n_components=k, random_state=42)
    pca.fit(stack_sc)
    print("done")
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for pc_i, cv in enumerate(cumvar):
        print(f"    PC{pc_i+1}: {pca.explained_variance_ratio_[pc_i]*100:.2f}%  "
              f"(cumulative {cv*100:.1f}%)")

    # Project all trials (train + val + test)
    N_all       = len(trials)
    embeddings  = np.zeros((N_all, T, k), dtype=np.float32)
    for i in range(N_all):
        t_sc        = scaler.transform(trials[i])   # (T, C)
        embeddings[i] = pca.transform(t_sc)          # (T, k)

    # Per-character mean trajectories (training trials only)
    train_labels = labels[train_idx]
    classes      = sorted(set(labels.tolist()))
    mean_trajectories: dict[str, np.ndarray] = {}
    for char in classes:
        char_mask = train_labels == char
        char_idx  = train_idx[char_mask]
        if len(char_idx) == 0:
            mean_trajectories[char] = np.zeros((T, k), dtype=np.float32)
        else:
            mean_trajectories[char] = embeddings[char_idx].mean(axis=0)

    # Per-character centroids (mean over time of mean trajectory)
    centroids = {
        char: mean_trajectories[char].mean(axis=0)
        for char in classes
    }
    centroid_mat = np.vstack([centroids[c] for c in classes])  # (n_classes, k)

    # Pairwise centroid distances
    n_c      = len(classes)
    dist_mat = np.zeros((n_c, n_c), dtype=np.float32)
    for i in range(n_c):
        for j in range(n_c):
            dist_mat[i, j] = float(
                np.linalg.norm(centroid_mat[i] - centroid_mat[j])
            )

    return {
        "embeddings":        embeddings,
        "mean_trajectories": mean_trajectories,
        "centroid_mat":      centroid_mat,
        "dist_mat":          dist_mat,
        "classes":           classes,
        "pca":               pca,
        "scaler":            scaler,
        "explained_var":     pca.explained_variance_ratio_,
        "cumvar":            cumvar,
        "n_pcs":             k,
    }


def plot_trajectories_3d(mean_trajectories: dict[str, np.ndarray],
                         highlight_chars: list[str] | None,
                         save_path: Path,
                         dpi: int = 120) -> None:
    """3D PC1-PC2-PC3 trajectory plot."""
    chars    = list(mean_trajectories.keys())
    cmap_obj = mpl_cm.get_cmap("tab20", len(chars))
    if highlight_chars is None:
        highlight_chars = chars

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for k, char in enumerate(chars):
        traj = mean_trajectories[char]
        if traj.shape[1] < 3:
            break
        pc1, pc2, pc3 = traj[:, 0], traj[:, 1], traj[:, 2]
        alpha = 1.0 if char in highlight_chars else 0.15
        lw    = 1.8 if char in highlight_chars else 0.6
        color = cmap_obj(k)
        ax.plot(pc1, pc2, pc3, color=color, alpha=alpha, linewidth=lw)
        ax.scatter(pc1[-1], pc2[-1], pc3[-1], color=color, alpha=alpha, s=20)
        if char in highlight_chars:
            ax.text(pc1[0], pc2[0], pc3[0], char, fontsize=8,
                    color=color, fontweight="bold")

    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("PC 3")
    ax.set_title("Mean Neural Trajectories (PC1–PC2–PC3)", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main() -> None:
    cfg         = load_config()
    figures_dir = Path(cfg["figures_dir"])
    pca_dir     = Path(cfg["results_dir"]) / "pca"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed trials …")
    trials, labels, _, _, _ = load_processed(cfg)
    N, T, C = trials.shape
    print(f"  trials shape: {trials.shape}")

    splits          = load_splits(cfg)
    split_ba        = splits["block_aware"]
    train_idx       = np.array(split_ba["train_idx"])
    n_pcs           = int(cfg["pca_trajectory_components"])
    highlight_chars = cfg.get("highlight_chars", [])

    print(f"\n{'='*70}")
    print(f"PCA Trajectory Analysis (block_aware split, n_pcs={n_pcs})")

    res = compute_trajectory_pca(trials, labels, train_idx, n_pcs)

    classes      = res["classes"]
    n_c          = len(classes)
    mean_traj    = res["mean_trajectories"]

    # ── Save embeddings NPZ ───────────────────────────────────────────────
    mean_traj_stack = np.stack([mean_traj[c] for c in classes])  # (n_cls, T, k)
    emb_path = pca_dir / "trajectory_embeddings.npz"
    np.savez(
        str(emb_path),
        embeddings=res["embeddings"],
        mean_trajectories=mean_traj_stack,
        centroid_mat=res["centroid_mat"],
        dist_mat=res["dist_mat"],
        classes=np.array(classes),
        labels=labels,
        explained_variance_ratio=res["explained_var"],
        cumulative_variance=res["cumvar"],
    )
    print(f"\nWrote: {emb_path}")

    # ── Save centroid CSV ─────────────────────────────────────────────────
    centroid_df = pd.DataFrame(
        res["centroid_mat"],
        index=classes,
        columns=[f"PC{i+1}" for i in range(res["n_pcs"])],
    )
    centroid_df.index.name = "character"
    centroid_path = pca_dir / "character_centroids.csv"
    centroid_df.to_csv(centroid_path)
    print(f"Wrote: {centroid_path}")

    # ── Save distance matrix CSV ──────────────────────────────────────────
    dist_df = pd.DataFrame(res["dist_mat"], index=classes, columns=classes)
    dist_df.index.name = "character"
    dist_path = pca_dir / "centroid_distance_matrix.csv"
    dist_df.to_csv(dist_path)
    print(f"Wrote: {dist_path}")

    # ── 2D trajectory plot ────────────────────────────────────────────────
    plot_trajectories_2d(
        mean_traj,
        highlight_chars=highlight_chars,
        title=f"Mean Neural Trajectories (PC1–PC2, block_aware training set)",
        save_path=figures_dir / "trajectory_pca_2d.png",
    )
    print(f"Wrote: {figures_dir / 'trajectory_pca_2d.png'}")

    # ── 3D trajectory plot (if enough PCs) ────────────────────────────────
    if n_pcs >= 3:
        plot_trajectories_3d(
            mean_traj,
            highlight_chars=highlight_chars,
            save_path=figures_dir / "trajectory_pca_3d.png",
        )
        print(f"Wrote: {figures_dir / 'trajectory_pca_3d.png'}")

    # ── Centroid distance heatmap ─────────────────────────────────────────
    plot_centroid_distance_heatmap(
        res["dist_mat"], classes,
        save_path=figures_dir / "centroid_distance_heatmap.png",
    )
    print(f"Wrote: {figures_dir / 'centroid_distance_heatmap.png'}")

    # ── Top pairs by distance ─────────────────────────────────────────────
    pairs = [
        (res["dist_mat"][i, j], classes[i], classes[j])
        for i in range(n_c) for j in range(i + 1, n_c)
    ]
    pairs.sort()
    print("\nTop-5 most similar character pairs (smallest centroid distance):")
    for d, a, b in pairs[:5]:
        print(f"  {a:15s} – {b:15s}  d={d:.4f}")
    print("Top-5 most different character pairs (largest centroid distance):")
    for d, a, b in pairs[-5:][::-1]:
        print(f"  {a:15s} – {b:15s}  d={d:.4f}")

    print("\nDone.\n")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    os.chdir(proj_root)
    main()
