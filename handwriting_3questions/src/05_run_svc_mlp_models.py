"""
05_run_svc_mlp_models.py
─────────────────────────
Train non-linear classifiers (SVM and MLP) to answer Questions 1 and 2.

  Question 1 (Flat vs temporal features)
  ────────────────────────────────────────
  Run SVM and MLP on flat and temporal features for both splits.

  Question 2 (Block-aware vs random-trial)
  ─────────────────────────────────────────
  Run SVM and MLP on flat features; compare splits.

Rules
─────
  - StandardScaler fitted inside Pipeline on training data only
  - Hyperparameter selection by validation accuracy
  - Test set evaluated once after selection

Outputs
───────
  results/tables/nonlinear_metrics.csv
  results/models/svc_rbf_cm_<features>_<split>.npy
  results/models/mlp_cm_<features>_<split>.npy
  results/figures/svc_rbf_cm_<features>_<split>.png
  results/figures/mlp_cm_<features>_<split>.png
  results/models/svc_rbf_<features>_<split>_best.pkl
  results/models/mlp_<features>_<split>_best.pkl

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/05_run_svc_mlp_models.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_data     import load_config, load_processed, load_splits
from utils_features import build_flat, build_temporal, temporal_window_boundaries
from utils_models   import (grid_search_svc_rbf, grid_search_mlp,
                             evaluate_model, make_label_encoder, save_model)
from utils_plots    import plot_confusion_matrix


def main() -> None:
    cfg         = load_config()
    models_dir  = Path(cfg["models_dir"])
    tables_dir  = Path(cfg["tables_dir"])
    figures_dir = Path(cfg["figures_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed trials …")
    trials, labels, _, _, _ = load_processed(cfg)
    N, T, C = trials.shape
    print(f"  trials shape: {trials.shape}")

    splits  = load_splits(cfg)
    le      = make_label_encoder(labels)
    y_all   = le.transform(labels)
    classes = list(le.classes_)
    windows = temporal_window_boundaries(cfg)

    svc_cfg = cfg["svc_rbf_models"]
    mlp_cfg = cfg["mlp_models"]
    all_rows = []

    for split_name in ["block_aware", "random_trial"]:
        sp      = splits[split_name]
        tr_idx  = np.array(sp["train_idx"])
        val_idx = np.array(sp["val_idx"])
        te_idx  = np.array(sp["test_idx"])
        y_train = y_all[tr_idx]
        y_val   = y_all[val_idx]
        y_test  = y_all[te_idx]

        for feat_name in ["flat", "temporal"]:
            if feat_name == "flat":
                X_train = build_flat(trials, tr_idx)
                X_val   = build_flat(trials, val_idx)
                X_test  = build_flat(trials, te_idx)
            else:
                X_train = build_temporal(trials, tr_idx,  windows)
                X_val   = build_temporal(trials, val_idx, windows)
                X_test  = build_temporal(trials, te_idx,  windows)

            print(f"\n{'='*70}")
            print(f"Split: {split_name}  |  Features: {feat_name}")
            print(f"  X_train={X_train.shape}  X_val={X_val.shape}  "
                  f"X_test={X_test.shape}")

            # ── SVC (RBF) ─────────────────────────────────────────────────────
            print(f"\n  [SVM] Grid search …")
            svc_pipe, svc_hp, svc_val_acc = grid_search_svc_rbf(
                svc_cfg, X_train, y_train, X_val, y_val,
            )
            print(f"  [SVM] Best hparams: {svc_hp}")
            print(f"  [SVM] Val accuracy: {svc_val_acc:.4f}")

            svc_val  = evaluate_model(svc_pipe, X_val,  y_val,  le)
            svc_test = evaluate_model(svc_pipe, X_test, y_test, le)
            print(f"  [SVM] Test acc={svc_test['accuracy']:.4f}  "
                  f"f1={svc_test['macro_f1']:.4f}")

            tag = f"{feat_name}_{split_name}"
            np.save(str(models_dir / f"svc_rbf_cm_{tag}.npy"),
                    svc_test["confusion_matrix"])
            plot_confusion_matrix(
                svc_test["confusion_matrix"], classes,
                f"SVM | {feat_name} | {split_name}",
                figures_dir / f"svc_rbf_cm_{tag}.png",
                cmap="Purples",
            )
            save_model(svc_pipe, svc_hp, le, models_dir / f"svc_rbf_{tag}_best.pkl")

            all_rows.append({
                "model":         "SVM",
                "features":      feat_name,
                "split":         split_name,
                "best_hparams":  str(svc_hp),
                "val_accuracy":  svc_val["accuracy"],
                "val_macro_f1":  svc_val["macro_f1"],
                "test_accuracy": svc_test["accuracy"],
                "test_macro_f1": svc_test["macro_f1"],
                "n_train":       int(len(tr_idx)),
                "n_val":         int(len(val_idx)),
                "n_test":        int(len(te_idx)),
                "feature_dim":   int(X_train.shape[1]),
            })

            # ── MLP ───────────────────────────────────────────────────────────
            print(f"\n  [MLP] Grid search …")
            mlp_pipe, mlp_hp, mlp_val_acc = grid_search_mlp(
                mlp_cfg, X_train, y_train, X_val, y_val,
            )
            print(f"  [MLP] Best hparams: {mlp_hp}")
            print(f"  [MLP] Val accuracy: {mlp_val_acc:.4f}")

            mlp_val  = evaluate_model(mlp_pipe, X_val,  y_val,  le)
            mlp_test = evaluate_model(mlp_pipe, X_test, y_test, le)
            print(f"  [MLP] Test acc={mlp_test['accuracy']:.4f}  "
                  f"f1={mlp_test['macro_f1']:.4f}")

            np.save(str(models_dir / f"mlp_cm_{tag}.npy"),
                    mlp_test["confusion_matrix"])
            plot_confusion_matrix(
                mlp_test["confusion_matrix"], classes,
                f"MLP | {feat_name} | {split_name}",
                figures_dir / f"mlp_cm_{tag}.png",
                cmap="Greens",
            )
            save_model(mlp_pipe, mlp_hp, le, models_dir / f"mlp_{tag}_best.pkl")

            all_rows.append({
                "model":         "MLP",
                "features":      feat_name,
                "split":         split_name,
                "best_hparams":  str(mlp_hp),
                "val_accuracy":  mlp_val["accuracy"],
                "val_macro_f1":  mlp_val["macro_f1"],
                "test_accuracy": mlp_test["accuracy"],
                "test_macro_f1": mlp_test["macro_f1"],
                "n_train":       int(len(tr_idx)),
                "n_val":         int(len(val_idx)),
                "n_test":        int(len(te_idx)),
                "feature_dim":   int(X_train.shape[1]),
            })

    # ── Save metrics CSV ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = tables_dir / "nonlinear_metrics.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\nWrote: {csv_path}")

    print(f"\n{'='*70}")
    print("SUMMARY – Non-linear Models (selected by val accuracy)")
    print(df[["model", "features", "split", "test_accuracy", "test_macro_f1"]]
          .to_string(index=False))
    print("\nDone.\n")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    os.chdir(proj_root)
    main()
