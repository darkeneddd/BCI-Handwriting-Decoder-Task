"""
08_make_figures_tables_report.py
──────────────────────────────────
Final report generation script.

Automatically produces all summary figures, tables, and a Markdown report
from the CSV results saved by scripts 04–07.

Figures generated
─────────────────
  1. class_distribution.png
  2. q1_flat_vs_temporal.png
  3. q2_block_vs_random.png
  4. q3_pca_decoder_curves.png
  5. pca_explained_variance.png       (reused from 06)
  6. best_confusion_matrix.png
  (7. trajectory figures reused from 07)

Report
──────
  results/reports/analysis_report.md

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/08_make_figures_tables_report.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_data  import load_config, load_processed, load_splits
from utils_plots import (
    plot_class_distribution,
    plot_flat_vs_temporal,
    plot_block_vs_random,
    plot_pca_decoder_curves,
    plot_explained_variance,
    plot_confusion_matrix,
    plot_q1_slope,
    plot_q2_slope,
    plot_q1_delta_heatmap_random_flat_baseline,
    plot_q2_delta_heatmap_random_flat_baseline,
    plot_rank_shift,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def safe_read_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  ⚠ {label} not found at {path} – skipping")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {label}: {len(df)} rows")
    return df


def merge_all_metrics(tables_dir: Path) -> pd.DataFrame:
    """Combine linear + nonlinear metrics into one DataFrame."""
    frames = []
    for fname in ["linear_metrics.csv", "nonlinear_metrics.csv"]:
        df = safe_read_csv(tables_dir / fname, fname)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─── Markdown report writer ───────────────────────────────────────────────────

def write_markdown_report(
    all_df: pd.DataFrame,
    pca_df: pd.DataFrame | None,
    var_df: pd.DataFrame | None,
    cfg: dict,
    out_path: Path,
) -> None:
    """Generate a structured Markdown analysis report."""

    def fmt_table(df: pd.DataFrame) -> str:
        if df is None or len(df) == 0:
            return "_No data available._\n"
        return df.to_markdown(index=False) + "\n"

    lines = []
    A = lines.append

    A("# Handwriting BCI – 3-Question Analysis Report\n")
    A(f"_Generated automatically by `08_make_figures_tables_report.py`_\n")
    A("")

    # ── Overview ──────────────────────────────────────────────────────────
    A("## 1  Analysis Overview\n")
    A("This report investigates three methodological questions in single-character "
      "handwriting decoding from neural population activity (BrainGate T5 dataset).\n")
    A("**Q1 – Flat vs temporal features:**  "
      "Does fine-grained spatiotemporal structure matter, or do coarse "
      "temporal envelopes suffice?\n")
    A("**Q2 – Block-aware vs random-trial split:**  "
      "How robust is the decoder to neural drift and genuine temporal generalization?\n")
    A("**Q3 – No PCA vs PCA before classification:**  "
      "Does decoding require the full high-dimensional space, or can a "
      "reduced-dimensional latent representation preserve sufficient information?\n")

    # ── Data & Preprocessing ──────────────────────────────────────────────
    A("## 2  Data and Preprocessing\n")
    A("### Dataset\n")
    A(f"- **Sessions:** {', '.join(cfg.get('sessions', []))}\n")
    A(f"- **Characters:** 31 classes (26 letters + 5 punctuation tokens)\n")

    meta_path = Path(cfg["metadata_dir"]) / "preprocessing_summary.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        A(f"- **Total trials:** {meta.get('n_trials', '?')}\n")
        A(f"- **Trial shape after preprocessing:** "
          f"({meta.get('T_fixed','?')}, {meta.get('n_channels','?')})\n")

    A("\n### Preprocessing Steps\n")
    A("The preprocessing replicates the notebook's signal-conditioning philosophy:\n")
    A("1. **A2 – Block-mean subtraction:**  For each trial, the block-level mean "
      "activity (`meansPerBlock` from the MATLAB dataset) is subtracted channel-wise "
      "to remove slow baseline drifts within a recording block.\n")
    A("2. **A3 – Global channel normalization:**  Each channel is divided by the "
      "dataset-wide standard deviation (`stdAcrossAllData` from the MATLAB dataset), "
      f"with ε = {cfg.get('norm_epsilon', 1e-8)} to prevent division by zero.  "
      "This equalises channel scales without fitting any statistic to the test set.\n")
    A(f"3. **A4 – Gaussian temporal smoothing:**  Gaussian smoothing is applied "
      f"along the time axis only (σ = {cfg.get('sigma_ms','?')} ms → "
      f"σ_bins = {float(cfg.get('sigma_ms',20))/float(cfg.get('dt_ms',10)):.1f} bins "
      f"at {cfg.get('dt_ms','?')} ms/bin).  Channels are never mixed.\n")
    A(f"4. **A6 – Movement-window extraction:**  Only bins "
      f"[{cfg.get('move_win_start','?')}, {cfg.get('move_win_end','?')}) are kept, "
      f"yielding T_fixed = {cfg.get('T_fixed','?')} bins (~1000 ms of movement).\n")

    # ── Split definitions ─────────────────────────────────────────────────
    A("\n## 3  Split Definitions\n")
    A("| Split | Method | Description |\n"
      "|-------|--------|-------------|\n"
      "| `block_aware` | Chronological block hold-out | Last 25% of blocks per session → test; preceding 25% → val; remaining → train. Blocks never shared across sets. |\n"
      "| `random_trial` | Stratified random | Trials randomly assigned to train/val/test with class stratification. Easier benchmark. |\n")

    # ── Question 1 ────────────────────────────────────────────────────────
    A("\n## 4  Q1 – Flat vs Temporal Features\n")
    A("_Feature dimensions:_ flat = T×C = 19 200; temporal = 3×C = 576\n\n")
    A("_Temporal windows:_ early [0, 33), middle [33, 66), late [66, 100)\n")
    if all_df is not None and len(all_df):
        q1_df = (all_df[all_df["features"].isin(["flat", "temporal"])]
                 [["model", "features", "split", "test_accuracy", "test_macro_f1"]]
                 .sort_values(["split", "model", "features"]))
        A(fmt_table(q1_df))
    A("![Q1 figure](../figures/q1_flat_vs_temporal.png)\n")

    # ── Question 2 ────────────────────────────────────────────────────────
    A("\n## 5  Q2 – Block-aware vs Random-trial\n")
    A("_Feature:_ flat only (same feature set; split difficulty varies)\n\n")
    if all_df is not None and len(all_df):
        q2_df = (all_df[all_df["features"] == "flat"]
                 [["model", "split", "test_accuracy", "test_macro_f1"]]
                 .sort_values(["model", "split"]))
        A(fmt_table(q2_df))
    A("![Q2 figure](../figures/q2_block_vs_random.png)\n")

    # ── Question 3 ────────────────────────────────────────────────────────
    A("\n## 6  Q3 – No PCA vs PCA before Classification\n")
    A("_Split:_ block_aware (primary).  k ∈ {20, 50, 100, 200} components.\n\n")
    if pca_df is not None and len(pca_df):
        q3_df = (pca_df[pca_df["split"] == "block_aware"]
                 [["model", "features", "n_components", "test_accuracy", "test_macro_f1"]]
                 .sort_values(["model", "n_components"]))
        A(fmt_table(q3_df))
    A("![Q3 figure](../figures/q3_pca_decoder_curves.png)\n")
    if var_df is not None and len(var_df):
        A("\n### Explained Variance\n")
        ba_var = var_df[var_df["split"] == "block_aware"]
        A(fmt_table(ba_var[["n_components", "cumulative_variance"]].drop_duplicates()))
    A("![Explained variance](../figures/pca_explained_variance.png)\n")

    # ── PCA Trajectory Analysis ────────────────────────────────────────────
    A("\n## 7  PCA Trajectory Analysis\n")
    A("Neural trajectories in PC1–PC2 space during the movement window.\n")
    A("Each curve represents the mean population trajectory for one character.\n\n")
    A("![Trajectories 2D](../figures/trajectory_pca_2d.png)\n\n")
    A("![Centroid distances](../figures/centroid_distance_heatmap.png)\n")

    # ── Conclusions ───────────────────────────────────────────────────────
    A("\n## 8  Main Conclusions\n")

    if all_df is not None and len(all_df):
        # Q1 conclusion
        flat_ba = all_df[(all_df["features"] == "flat") &
                         (all_df["split"] == "block_aware")]["test_accuracy"]
        temp_ba = all_df[(all_df["features"] == "temporal") &
                         (all_df["split"] == "block_aware")]["test_accuracy"]
        if len(flat_ba) and len(temp_ba):
            diff = float(flat_ba.mean() - temp_ba.mean())
            if diff > 0.01:
                A(f"- **Q1:** Flat features outperform temporal features by "
                  f"{diff:.3f} accuracy on average (block_aware), indicating that "
                  f"fine-grained spatiotemporal structure carries important decoding "
                  f"information beyond coarse temporal envelopes.\n")
            else:
                A(f"- **Q1:** Flat and temporal features achieve similar performance "
                  f"({diff:+.3f} accuracy difference), suggesting coarse temporal "
                  f"windows may capture most of the discriminative information.\n")

        # Q2 conclusion
        ba_f1 = all_df[(all_df["features"] == "flat") &
                       (all_df["split"] == "block_aware")]["test_accuracy"]
        rt_f1 = all_df[(all_df["features"] == "flat") &
                       (all_df["split"] == "random_trial")]["test_accuracy"]
        if len(ba_f1) and len(rt_f1):
            gap = float(rt_f1.mean() - ba_f1.mean())
            A(f"- **Q2:** Random-trial split exceeds block-aware by {gap:.3f} "
              f"accuracy on average, confirming that neural drift makes temporal "
              f"generalization substantially harder than within-distribution decoding.\n")

    if pca_df is not None and len(pca_df):
        # Q3 conclusion
        k200_f1 = pca_df[(pca_df["features"] == "flat_pca") &
                         (pca_df["n_components"] == 200) &
                         (pca_df["split"] == "block_aware")]["test_accuracy"]
        flat_f1 = pca_df[(pca_df["features"] == "flat") &
                         (pca_df["split"] == "block_aware")]["test_accuracy"]
        if len(k200_f1) and len(flat_f1):
            diff = float(k200_f1.mean() - flat_f1.mean())
            A(f"- **Q3:** PCA(200) vs raw flat: {diff:+.3f} accuracy average.  "
              f"{'PCA provides a regularization benefit.' if diff > 0 else 'Raw flat features retain an advantage, suggesting the full-dimensional structure contains information lost in PCA compression.'}\n")

    A("\n## 9  Limitations\n")
    A("- Single participant (T5); generalisability to other participants is unknown.\n")
    A("- Block-aware test fraction is fixed (25%); other fractions may give "
      "different generalisation gaps.\n")
    A("- MLP and SVM grids are coarse; more exhaustive tuning might improve performance.\n")
    A("- PCA for classification is applied to flattened features; "
      "channel-space PCA (used for trajectories) is a conceptually different compression.\n")
    A("- Temporal window boundaries (early/mid/late) are equal-width; "
      "task-adapted windows might better capture the writing dynamics.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote: {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg         = load_config()
    tables_dir  = Path(cfg["tables_dir"])
    figures_dir = Path(cfg["figures_dir"])
    reports_dir = Path(cfg["reports_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results …")
    all_df = merge_all_metrics(tables_dir)
    pca_df = safe_read_csv(tables_dir / "pca_metrics.csv",     "pca_metrics")
    var_df = safe_read_csv(tables_dir / "pca_explained_variance.csv",
                           "pca_explained_variance")

    # ── Load raw trial labels for class distribution ───────────────────────
    print("\nLoading preprocessed trials for class distribution …")
    trials, labels, _, _, _ = load_processed(cfg)

    # ── Figure 1: Class distribution ──────────────────────────────────────
    print("\nGenerating figures …")
    plot_class_distribution(
        labels,
        save_path=figures_dir / "class_distribution.png",
    )
    print("  class_distribution.png")

    # ── Figure 2: Q1 flat vs temporal ─────────────────────────────────────
    if all_df is not None and len(all_df):
        q1_df = all_df[all_df["features"].isin(["flat", "temporal"])].copy()
        if len(q1_df):
            plot_flat_vs_temporal(
                q1_df,
                metric="test_accuracy",
                save_path=figures_dir / "q1_flat_vs_temporal.png",
            )
            print("  q1_flat_vs_temporal.png")

    # ── Figure 3: Q2 block vs random ──────────────────────────────────────
    if all_df is not None and len(all_df):
        q2_df = all_df[all_df["features"] == "flat"].copy()
        if len(q2_df):
            plot_block_vs_random(
                q2_df,
                metric="test_accuracy",
                save_path=figures_dir / "q2_block_vs_random.png",
            )
            print("  q2_block_vs_random.png")

    # ── Extra Figure: Q1 slope chart ──────────────────────────────────────
    if all_df is not None and len(all_df):
        q1_df = all_df[all_df["features"].isin(["flat", "temporal"])].copy()
        if len(q1_df):
            plot_q1_slope(
                q1_df,
                metric="test_accuracy",
                save_path=figures_dir / "q1_slope_chart.png",
            )
            print("  q1_slope_chart.png")

    # ── Extra Figure: Q2 slope chart ──────────────────────────────────────
    if all_df is not None and len(all_df):
        q2_df = all_df[all_df["features"] == "flat"].copy()
        if len(q2_df):
            plot_q2_slope(
                q2_df,
                metric="test_accuracy",
                save_path=figures_dir / "q2_slope_chart.png",
            )
            print("  q2_slope_chart.png")

    # ── Extra Figure: Q1 delta heatmap (baseline: random_trial flat) ─────
    if all_df is not None and len(all_df):
        plot_q1_delta_heatmap_random_flat_baseline(
            all_df,
            metric="test_accuracy",
            save_path=figures_dir / "q1_delta_heatmap_rtflat_baseline.png",
        )
        print("  q1_delta_heatmap_rtflat_baseline.png")

    # ── Extra Figure: Q2 delta heatmap (baseline: random_trial flat) ─────
    if all_df is not None and len(all_df):
        plot_q2_delta_heatmap_random_flat_baseline(
            all_df,
            metric="test_accuracy",
            save_path=figures_dir / "q2_delta_heatmap_rtflat_baseline.png",
        )
        print("  q2_delta_heatmap_rtflat_baseline.png")

    # ── Extra Figure: model rank shift ────────────────────────────────────
    if all_df is not None and len(all_df):
        plot_rank_shift(
            all_df,
            metric="test_accuracy",
            save_path=figures_dir / "model_rank_shift.png",
        )
        print("  model_rank_shift.png")

    # ── Figure 4: Q3 PCA decoder curves ──────────────────────────────────
    if pca_df is not None and len(pca_df):
        plot_pca_decoder_curves(
            pca_df,
            metric="test_accuracy",
            split_filter="block_aware",
            save_path=figures_dir / "q3_pca_decoder_curves.png",
        )
        print("  q3_pca_decoder_curves.png")

        # Also for random_trial
        plot_pca_decoder_curves(
            pca_df,
            metric="test_accuracy",
            split_filter="random_trial",
            save_path=figures_dir / "q3_pca_decoder_curves_random.png",
        )
        print("  q3_pca_decoder_curves_random.png")

    # ── Figure 5: Explained variance ─────────────────────────────────────
    if var_df is not None and len(var_df):
        ba_var = var_df[var_df["split"] == "block_aware"].drop_duplicates("n_components")
        if len(ba_var):
            k_list = sorted(ba_var["n_components"].tolist())
            # Build a cumvar array from the transformer pkl if available
            pkl = Path(cfg["models_dir"]) / f"pca_transformer_block_aware_k{max(k_list)}.pkl"
            if pkl.exists():
                import pickle
                with open(pkl, "rb") as f:
                    bundle = pickle.load(f)
                transformer = bundle["transformer"]
                plot_explained_variance(
                    transformer.cumulative_variance_,
                    k_list,
                    save_path=figures_dir / "pca_explained_variance_full.png",
                )
                print("  pca_explained_variance_full.png")

    # ── Figure 6: Best block-aware confusion matrix ───────────────────────
    if all_df is not None and len(all_df):
        ba_flat = all_df[(all_df["split"] == "block_aware") &
                         (all_df["features"] == "flat")].copy()
        if len(ba_flat):
            best_row = ba_flat.sort_values("test_accuracy", ascending=False).iloc[0]
            model_n  = best_row["model"]
            feat_n   = best_row["features"]
            npy_candidates = [
                Path(cfg["models_dir"]) / f"linear_cm_{model_n}_{feat_n}_block_aware.npy",
                Path(cfg["models_dir"]) / f"svc_rbf_cm_{feat_n}_block_aware.npy",
                Path(cfg["models_dir"]) / f"mlp_cm_{feat_n}_block_aware.npy",
            ]
            classes_path = Path(cfg["metadata_dir"]) / "preprocessing_summary.json"
            classes = None
            if classes_path.exists():
                with open(classes_path) as f:
                    meta = json.load(f)
                classes = sorted(meta.get("class_counts", {}).keys())

            for npy_path in npy_candidates:
                if npy_path.exists() and classes:
                    cm_arr = np.load(str(npy_path))
                    plot_confusion_matrix(
                        cm_arr, classes,
                        f"Best model: {model_n} | {feat_n} | block_aware\n"
                        f"test acc={best_row['test_accuracy']:.3f}",
                        figures_dir / "best_confusion_matrix.png",
                    )
                    print("  best_confusion_matrix.png")
                    break

    # ── Summary tables ────────────────────────────────────────────────────
    if all_df is not None and len(all_df):
        summary_path = tables_dir / "all_models_summary.csv"
        all_df.to_csv(summary_path, index=False)
        print(f"\nWrote summary table: {summary_path}")

    # ── Markdown report ───────────────────────────────────────────────────
    write_markdown_report(
        all_df, pca_df, var_df, cfg,
        out_path=reports_dir / "analysis_report.md",
    )

    print("\nAll figures and report generated.")
    print(f"  Figures: {figures_dir}")
    print(f"  Reports: {reports_dir}")
    print("Done.\n")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    os.chdir(proj_root)
    main()
