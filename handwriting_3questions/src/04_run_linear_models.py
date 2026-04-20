"""
04_run_linear_models.py
────────────────────────
Train linear classifiers to answer Questions 1 and 2.

  Question 1 (Flat vs temporal features)
  ────────────────────────────────────────
  Run Logistic, LinearSVM, Ridge on:
    - flat     features  (T*C = 100*192 = 19 200 dimensions)
    - temporal features  (3*C = 3*192  =    576 dimensions)
  Evaluated on both block_aware and random_trial splits.

  Question 2 (Block-aware vs random-trial)
  ─────────────────────────────────────────
  Run all three linear models on flat features.
  Compare block_aware and random_trial performance.

Rules
─────
  - StandardScaler fitted on training data only (inside sklearn Pipeline)
  - Hyperparameter selection by validation accuracy
  - Test set evaluated only once per model/feature/split combination

Outputs
───────
  results/tables/linear_metrics.csv
  results/models/linear_cm_<model>_<features>_<split>.npy
  results/figures/linear_cm_<model>_<features>_<split>.png

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/04_run_linear_models.py
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_data     import load_config, load_processed, load_splits
from utils_features import build_flat, build_temporal, temporal_window_boundaries
from utils_models   import (grid_search_linear, evaluate_model,
                             make_label_encoder, hparam_grid_from_config,
                             save_model)
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

    linear_cfg = cfg["linear_models"]
    all_rows   = []
    display_name_map = {
        "LogisticRegression": "Logistic",
        "LinearSVC": "LinearSVM",
        "RidgeClassifier": "Ridge",
    }

    model_names = list(linear_cfg.keys())
    feature_names = ["flat", "temporal"]
    split_names   = ["block_aware", "random_trial"]

    total = len(model_names) * len(feature_names) * len(split_names)
    run   = 0

    for split_name in split_names:
        sp      = splits[split_name]
        tr_idx  = np.array(sp["train_idx"])
        val_idx = np.array(sp["val_idx"])
        te_idx  = np.array(sp["test_idx"])
        y_train = y_all[tr_idx]
        y_val   = y_all[val_idx]
        y_test  = y_all[te_idx]

        for feat_name in feature_names:
            if feat_name == "flat":
                X_train = build_flat(trials, tr_idx)
                X_val   = build_flat(trials, val_idx)
                X_test  = build_flat(trials, te_idx)
            else:
                X_train = build_temporal(trials, tr_idx,   windows)
                X_val   = build_temporal(trials, val_idx,  windows)
                X_test  = build_temporal(trials, te_idx,   windows)

            print(f"\n{'='*70}")
            print(f"Split: {split_name}  |  Features: {feat_name}")
            print(f"  X_train={X_train.shape}  X_val={X_val.shape}  "
                  f"X_test={X_test.shape}")

            for model_name, model_cfg in linear_cfg.items():
                model_display = display_name_map.get(model_name, model_name)
                run += 1
                print(f"\n  [{run}/{total}]  Model: {model_display}")

                # Build hyperparameter grid
                hgrid      = hparam_grid_from_config(model_name, model_cfg)
                extra_hp   = {k: v for k, v in model_cfg.items()
                              if not isinstance(v, list)}
                print(f"    Grid: {hgrid}")

                best_pipe, best_hp, best_val_acc = grid_search_linear(
                    model_name, hgrid,
                    X_train, y_train,
                    X_val, y_val,
                    extra_hparams=extra_hp,
                )
                print(f"    Best hparams: {best_hp}")
                print(f"    Val  accuracy={best_val_acc:.4f}")

                # Evaluate on val and test
                val_eval  = evaluate_model(best_pipe, X_val,  y_val,  le)
                test_eval = evaluate_model(best_pipe, X_test, y_test, le)
                print(f"    Val  acc={val_eval['accuracy']:.4f}  "
                      f"f1={val_eval['macro_f1']:.4f}")
                print(f"    Test acc={test_eval['accuracy']:.4f}  "
                      f"f1={test_eval['macro_f1']:.4f}")

                # Save confusion matrix
                tag     = f"{model_name}_{feat_name}_{split_name}"
                npy_path = models_dir / f"linear_cm_{tag}.npy"
                png_path = figures_dir / f"linear_cm_{tag}.png"
                np.save(str(npy_path), test_eval["confusion_matrix"])
                plot_confusion_matrix(
                    test_eval["confusion_matrix"], classes,
                    f"{model_display} | {feat_name} | {split_name}",
                    png_path,
                )

                # Save model
                model_path = models_dir / f"linear_{tag}.pkl"
                save_model(best_pipe, best_hp, le, model_path)

                all_rows.append({
                    "model":           model_display,
                    "features":        feat_name,
                    "split":           split_name,
                    "best_hparams":    str(best_hp),
                    "val_accuracy":    val_eval["accuracy"],
                    "val_macro_f1":    val_eval["macro_f1"],
                    "test_accuracy":   test_eval["accuracy"],
                    "test_macro_f1":   test_eval["macro_f1"],
                    "n_train":         int(len(tr_idx)),
                    "n_val":           int(len(val_idx)),
                    "n_test":          int(len(te_idx)),
                    "feature_dim":     int(X_train.shape[1]),
                })

    # ── Save metrics CSV ──────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = tables_dir / "linear_metrics.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\nWrote: {csv_path}")

    print(f"\n{'='*70}")
    print("SUMMARY – Linear Models (selected by val accuracy)")
    print(df[["model", "features", "split", "test_accuracy", "test_macro_f1"]]
          .to_string(index=False))
    print("\nDone.\n")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    os.chdir(proj_root)
    main()
