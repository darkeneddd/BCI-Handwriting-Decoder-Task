"""
03_build_features.py
─────────────────────
Pre-compute and cache all feature representations used in downstream
experiments.

Feature types
─────────────
  flat     – flatten (T_fixed, C) → (T_fixed * C,)
  temporal – coarse temporal: mean over 3 windows → (3 * C,)

The PCA-based features (flat_pca) are NOT pre-computed here because PCA must
be fitted separately for each train/val/test split combination (to prevent
leakage).  Instead, FlatPCATransformer in utils_features.py is used inside
the experiment scripts.

Outputs
───────
  processed/features_flat_<split>.npz
      X_train, X_val, X_test, y_train, y_val, y_test, classes

  processed/features_temporal_<split>.npz
      same structure

  metadata/feature_summary.json

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/03_build_features.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_data     import load_config, load_processed, load_splits
from utils_features import build_flat, build_temporal, temporal_window_boundaries


def main() -> None:
    cfg          = load_config()
    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed trials …")
    trials, labels, session_ids, block_ids, trial_ids = load_processed(cfg)
    N, T, C = trials.shape
    print(f"  trials shape: {trials.shape}")

    splits = load_splits(cfg)

    le = LabelEncoder()
    le.fit(labels)
    y_all   = le.transform(labels)
    classes = list(le.classes_)

    windows = temporal_window_boundaries(cfg)
    print(f"  Temporal windows: "
          + "  ".join(f"{k}=[{v[0]},{v[1]})" for k, v in windows.items()))

    summary_rows = []

    for split_name in ["block_aware", "random_trial"]:
        sp        = splits[split_name]
        tr_idx    = np.array(sp["train_idx"])
        val_idx   = np.array(sp["val_idx"])
        te_idx    = np.array(sp["test_idx"])

        y_train = y_all[tr_idx]
        y_val   = y_all[val_idx]
        y_test  = y_all[te_idx]

        print(f"\n[{split_name}]  "
              f"train={len(tr_idx)}  val={len(val_idx)}  test={len(te_idx)}")

        for feat_name, feat_fn in [
            ("flat",     lambda idx: build_flat(trials, idx)),
            ("temporal", lambda idx: build_temporal(trials, idx, windows)),
        ]:
            X_train = feat_fn(tr_idx)
            X_val   = feat_fn(val_idx)
            X_test  = feat_fn(te_idx)
            print(f"  {feat_name}: X_train={X_train.shape}  "
                  f"X_val={X_val.shape}  X_test={X_test.shape}")

            out_path = processed_dir / f"features_{feat_name}_{split_name}.npz"
            np.savez(
                str(out_path),
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                classes=np.array(classes),
                train_idx=tr_idx,
                val_idx=val_idx,
                test_idx=te_idx,
            )
            print(f"  Wrote: {out_path}")

            summary_rows.append({
                "split":         split_name,
                "feature_type":  feat_name,
                "n_train":       int(len(tr_idx)),
                "n_val":         int(len(val_idx)),
                "n_test":        int(len(te_idx)),
                "feature_dim":   int(X_train.shape[1]),
            })

    summary = {
        "T_fixed":           int(T),
        "n_channels":        int(C),
        "flat_dim":          int(T * C),
        "temporal_dim":      int(3 * C),
        "temporal_windows":  {k: list(v) for k, v in windows.items()},
        "classes":           classes,
        "n_classes":         len(classes),
        "splits":            summary_rows,
    }
    summary_path = Path(cfg["metadata_dir"]) / "feature_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {summary_path}")
    print("Done.\n")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    os.chdir(proj_root)
    main()
