"""
06_run_pca_experiments.py
──────────────────────────
PCA classification experiments to answer Question 3.

  Question 3 (No PCA vs PCA before classification)
  ──────────────────────────────────────────────────
  For every model and split, compare:
    - raw flat features  (T*C = 19 200 dims)
    - flat_pca features  with k ∈ {20, 50, 100, 200} components

  PCA is ALWAYS fitted on the training set only.
  Validation and test data are only transformed.

  This tests whether decoding requires the full high-dimensional feature space
  or whether a lower-dimensional latent representation is sufficient.

Models
──────
  Logistic, LinearSVM, Ridge, SVM, MLP

Splits
──────
  Primary:   block_aware
  Secondary: random_trial

Outputs
───────
  results/tables/pca_metrics.csv
  results/tables/pca_explained_variance.csv
  results/figures/pca_explained_variance.png
  results/models/pca_transformer_<split>_k<k>.pkl

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/06_run_pca_experiments.py
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_data     import load_config, load_processed, load_splits
from utils_features import build_flat, FlatPCATransformer
from utils_models   import (grid_search_linear, grid_search_svc_rbf, grid_search_mlp,
                             evaluate_model, make_label_encoder,
                             hparam_grid_from_config)
from utils_plots    import plot_explained_variance


def run_model_on_features(model_name: str, model_family: str,
                           model_cfg: dict,
                           X_train, y_train,
                           X_val, y_val,
                           X_test, y_test,
                           le) -> tuple[dict, dict]:
    """
    Grid-search + evaluate a model.  Returns (val_metrics, test_metrics).
    """
    if model_family == "linear":
        hgrid    = hparam_grid_from_config(model_name, model_cfg)
        extra_hp = {k: v for k, v in model_cfg.items()
                    if not isinstance(v, list)}
        best_pipe, best_hp, val_acc = grid_search_linear(
            model_name, hgrid, X_train, y_train, X_val, y_val,
            extra_hparams=extra_hp,
        )
    elif model_family == "svc_rbf":
        best_pipe, best_hp, val_acc = grid_search_svc_rbf(
            model_cfg, X_train, y_train, X_val, y_val,
        )
    elif model_family == "mlp":
        best_pipe, best_hp, val_acc = grid_search_mlp(
            model_cfg, X_train, y_train, X_val, y_val,
        )
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    val_eval  = evaluate_model(best_pipe, X_val,  y_val,  le)
    test_eval = evaluate_model(best_pipe, X_test, y_test, le)
    return val_eval, test_eval, best_hp


def main() -> None:
    cfg         = load_config()
    models_dir  = Path(cfg["models_dir"])
    tables_dir  = Path(cfg["tables_dir"])
    figures_dir = Path(cfg["figures_dir"])
    for d in [models_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed trials …")
    trials, labels, _, _, _ = load_processed(cfg)
    N, T, C = trials.shape
    print(f"  trials shape: {trials.shape}")

    splits  = load_splits(cfg)
    le      = make_label_encoder(labels)
    y_all   = le.transform(labels)

    k_list      = [int(k) for k in cfg["pca_components"]]
    linear_cfg  = cfg["linear_models"]
    svc_cfg     = cfg["svc_rbf_models"]
    mlp_cfg     = cfg["mlp_models"]

    all_models = [
        ("LogisticRegression", "Logistic", "linear",  linear_cfg["LogisticRegression"]),
        ("LinearSVC",          "LinearSVM", "linear",  linear_cfg["LinearSVC"]),
        ("RidgeClassifier",    "Ridge", "linear",  linear_cfg["RidgeClassifier"]),
        ("SVC_RBF",            "SVM", "svc_rbf", svc_cfg),
        ("MLP",                "MLP", "mlp",     mlp_cfg),
    ]

    all_rows    = []
    var_rows    = []
    # Store cumvar for the last fitted PCA (for plotting)
    last_cumvar = {}

    for split_name in ["block_aware", "random_trial"]:
        sp      = splits[split_name]
        tr_idx  = np.array(sp["train_idx"])
        val_idx = np.array(sp["val_idx"])
        te_idx  = np.array(sp["test_idx"])
        y_train = y_all[tr_idx]
        y_val   = y_all[val_idx]
        y_test  = y_all[te_idx]

        # Pre-compute raw flat features (used for baseline and PCA input)
        X_train_flat = build_flat(trials, tr_idx)
        X_val_flat   = build_flat(trials, val_idx)
        X_test_flat  = build_flat(trials, te_idx)

        print(f"\n{'='*70}")
        print(f"Split: {split_name}")
        print(f"  X_train_flat={X_train_flat.shape}  "
              f"X_val_flat={X_val_flat.shape}  X_test_flat={X_test_flat.shape}")

        # ── Fit PCA transformers for each k ───────────────────────────────
        # Fit once per split to avoid redundant computation
        pca_transformers: dict[int, FlatPCATransformer] = {}
        for k in k_list:
            print(f"\n  Fitting PCA k={k} on training flat features …",
                  end=" ", flush=True)
            transformer = FlatPCATransformer(k, random_state=42)
            transformer.fit(X_train_flat)
            pca_transformers[k] = transformer
            print(f"done  cumvar={transformer.cumulative_variance_[-1]*100:.1f}%")

            # Save transformer
            pkl_path = models_dir / f"pca_transformer_{split_name}_k{k}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump({
                    "transformer":              transformer,
                    "split":                    split_name,
                    "n_components":             k,
                    "explained_variance_ratio": transformer.explained_variance_ratio_,
                    "cumulative_variance":      transformer.cumulative_variance_,
                }, f)

            # Record explained variance
            var_rows.append({
                "split":              split_name,
                "n_components":       k,
                "cumulative_variance": float(transformer.cumulative_variance_[-1]),
                "explained_variance_ratio": transformer.explained_variance_ratio_.tolist(),
            })

        last_cumvar[split_name] = transformer.cumulative_variance_

        # ── Run each model on each feature set ───────────────────────────
        for model_name, model_display, model_family, model_cfg_item in all_models:
            print(f"\n  --- {model_display} ---")

            # Baseline: raw flat
            print(f"    [flat]", end=" … ", flush=True)
            val_e, test_e, best_hp = run_model_on_features(
                model_name, model_family, model_cfg_item,
                X_train_flat, y_train, X_val_flat, y_val,
                X_test_flat, y_test, le,
            )
            print(f"test acc={test_e['accuracy']:.4f}  f1={test_e['macro_f1']:.4f}")
            all_rows.append({
                "model":          model_display,
                "features":       "flat",
                "n_components":   None,
                "split":          split_name,
                "best_hparams":   str(best_hp),
                "val_accuracy":   val_e["accuracy"],
                "val_macro_f1":   val_e["macro_f1"],
                "test_accuracy":  test_e["accuracy"],
                "test_macro_f1":  test_e["macro_f1"],
            })

            # PCA variants
            for k in k_list:
                transformer = pca_transformers[k]
                X_tr_pca = transformer.transform(X_train_flat)
                X_va_pca = transformer.transform(X_val_flat)
                X_te_pca = transformer.transform(X_test_flat)

                print(f"    [flat_pca k={k}]", end=" … ", flush=True)
                val_e, test_e, best_hp = run_model_on_features(
                    model_name, model_family, model_cfg_item,
                    X_tr_pca, y_train, X_va_pca, y_val,
                    X_te_pca, y_test, le,
                )
                print(f"test acc={test_e['accuracy']:.4f}  "
                      f"f1={test_e['macro_f1']:.4f}")
                all_rows.append({
                    "model":          model_display,
                    "features":       "flat_pca",
                    "n_components":   k,
                    "split":          split_name,
                    "best_hparams":   str(best_hp),
                    "val_accuracy":   val_e["accuracy"],
                    "val_macro_f1":   val_e["macro_f1"],
                    "test_accuracy":  test_e["accuracy"],
                    "test_macro_f1":  test_e["macro_f1"],
                })

    # ── Save metrics CSV ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = tables_dir / "pca_metrics.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\nWrote: {csv_path}")

    # ── Save explained variance CSV ───────────────────────────────────────
    var_df = pd.DataFrame([
        {"split": r["split"],
         "n_components": r["n_components"],
         "cumulative_variance": r["cumulative_variance"]}
        for r in var_rows
    ])
    var_csv = tables_dir / "pca_explained_variance.csv"
    var_df.to_csv(str(var_csv), index=False)
    print(f"Wrote: {var_csv}")

    # ── Plot explained variance ───────────────────────────────────────────
    if "block_aware" in last_cumvar:
        plot_explained_variance(
            last_cumvar["block_aware"],
            k_list,
            save_path=figures_dir / "pca_explained_variance.png",
        )
        print(f"Wrote: {figures_dir / 'pca_explained_variance.png'}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY – PCA Experiments (block_aware | test accuracy)")
    ba_df = df[df["split"] == "block_aware"].copy()
    pivot = ba_df.pivot_table(
        index=["model", "features", "n_components"],
        values="test_accuracy",
        aggfunc="first",
    ).reset_index()
    print(pivot.to_string(index=False))
    print("\nDone.\n")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    os.chdir(proj_root)
    main()
