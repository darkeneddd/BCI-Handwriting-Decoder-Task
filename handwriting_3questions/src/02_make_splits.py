"""
02_make_splits.py
──────────────────
Create reproducible train / val / test splits.

Two split modes are generated:

  block_aware
    Separate by blocks within each session.  Later blocks are held out for
    test (last ~25% of blocks per session), the preceding blocks for val
    (~25%), and the remaining early blocks for training.

    This is the stricter and more realistic evaluation: neural activity drifts
    over time, so later blocks resemble a genuine held-out recording day.

  random_trial
    Stratified random split at the individual trial level.
    Easier than block_aware (serves as a performance upper bound).

Both splits are stored in a single JSON so downstream scripts can load either
by name.

Output
──────
  metadata/splits_same_as_notebook_preproc.json

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/02_make_splits.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_data import load_config


# ─── helpers ──────────────────────────────────────────────────────────────────

def class_counts(labels: np.ndarray, idx: np.ndarray) -> dict:
    vals, cnts = np.unique(labels[idx], return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, cnts)}


# ─── block-aware split ────────────────────────────────────────────────────────

def block_aware_split(df: pd.DataFrame,
                      test_frac: float,
                      val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-session chronological block split.

    For each session:
      1. Sort blocks by block_id (chronological order)
      2. Hold out the last  test_frac  fraction for test
      3. Hold out the next  val_frac   fraction  for val
      4. All earlier blocks → train

    This prevents any block from appearing in two splits, which would allow
    the model to memorise block-level drift and inflate accuracy.
    """
    train_idx, val_idx, test_idx = [], [], []

    for sess, grp in df.groupby("session_id"):
        blocks   = sorted(grp["block_id"].unique())
        n_blocks = len(blocks)

        n_test = max(1, round(n_blocks * test_frac))
        n_val  = max(1, round(n_blocks * val_frac))
        n_val  = min(n_val, n_blocks - n_test - 1)
        n_val  = max(n_val, 0)

        test_blocks  = set(blocks[-n_test:])
        val_blocks   = set(blocks[-(n_test + n_val): -n_test]) if n_val > 0 else set()
        train_blocks = set(blocks) - test_blocks - val_blocks

        test_idx.extend( grp[grp["block_id"].isin(test_blocks)].index.tolist())
        val_idx.extend(  grp[grp["block_id"].isin(val_blocks) ].index.tolist())
        train_idx.extend(grp[grp["block_id"].isin(train_blocks)].index.tolist())

    return (np.array(sorted(train_idx)),
            np.array(sorted(val_idx)),
            np.array(sorted(test_idx)))


# ─── random-trial split ───────────────────────────────────────────────────────

def random_trial_split(df: pd.DataFrame,
                       test_frac: float,
                       val_frac: float,
                       seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified random split ignoring block / session structure.

    Each class is split proportionally so that class imbalance does not bias
    any individual split.
    """
    all_idx = np.arange(len(df))
    labels  = df["label"].values

    trainval, test = train_test_split(
        all_idx, test_size=test_frac,
        random_state=seed, stratify=labels,
    )
    val_frac_adj = val_frac / (1.0 - test_frac)
    train, val   = train_test_split(
        trainval, test_size=val_frac_adj,
        random_state=seed, stratify=labels[trainval],
    )
    return np.sort(train), np.sort(val), np.sort(test)


# ─── validation ───────────────────────────────────────────────────────────────

def validate_split(N: int, train_idx, val_idx, test_idx,
                   labels: np.ndarray, name: str) -> None:
    """Assert no overlaps and all trials are assigned."""
    all_set  = set(train_idx) | set(val_idx) | set(test_idx)
    overlap  = ((set(train_idx) & set(test_idx)) |
                (set(train_idx) & set(val_idx))  |
                (set(val_idx)   & set(test_idx)))
    assert len(all_set) == N,      f"[{name}] {N - len(all_set)} unassigned trials!"
    assert len(overlap) == 0,      f"[{name}] {len(overlap)} overlapping indices!"
    n_all = len(set(labels))
    for sname, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        n_cls = len(set(labels[idx]))
        flag  = "(all)" if n_cls == n_all else "(MISSING CLASSES ⚠)"
        print(f"    {sname}: {len(idx)} trials, {n_cls} classes {flag}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg          = load_config()
    metadata_dir = Path(cfg["metadata_dir"])
    seed         = int(cfg["random_seed"])
    test_frac    = float(cfg["block_test_frac"])
    val_frac     = float(cfg["block_val_frac"])

    csv_path = metadata_dir / "single_char_trials_metadata.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.  Run script 01 first.")
        sys.exit(1)

    print(f"Loading {csv_path} …")
    df = pd.read_csv(str(csv_path))
    N  = len(df)
    labels = df["label"].values
    print(f"  {N} trials, {len(df['label'].unique())} classes")

    splits: dict = {}

    # ── block-aware ───────────────────────────────────────────────────────
    print(f"\n[block_aware]  test_frac={test_frac}  val_frac={val_frac}")
    train_ba, val_ba, test_ba = block_aware_split(df, test_frac, val_frac)
    validate_split(N, train_ba, val_ba, test_ba, labels, "block_aware")

    # Report block distribution
    for sname, idx in [("train", train_ba), ("val", val_ba), ("test", test_ba)]:
        bc = df.iloc[idx].groupby("session_id")["block_id"].nunique()
        print(f"    {sname} blocks per session: {bc.to_dict()}")

    splits["block_aware"] = {
        "strategy":            "block_aware",
        "seed":                seed,
        "test_frac":           test_frac,
        "val_frac":            val_frac,
        "n_train":             int(len(train_ba)),
        "n_val":               int(len(val_ba)),
        "n_test":              int(len(test_ba)),
        "train_class_counts":  class_counts(labels, train_ba),
        "val_class_counts":    class_counts(labels, val_ba),
        "test_class_counts":   class_counts(labels, test_ba),
        "train_idx":           train_ba.tolist(),
        "val_idx":             val_ba.tolist(),
        "test_idx":            test_ba.tolist(),
    }

    # ── random-trial ──────────────────────────────────────────────────────
    print(f"\n[random_trial]  test_frac={test_frac}  val_frac={val_frac}")
    train_rt, val_rt, test_rt = random_trial_split(df, test_frac, val_frac, seed)
    validate_split(N, train_rt, val_rt, test_rt, labels, "random_trial")

    splits["random_trial"] = {
        "strategy":            "random_trial",
        "seed":                seed,
        "test_frac":           test_frac,
        "val_frac":            val_frac,
        "n_train":             int(len(train_rt)),
        "n_val":               int(len(val_rt)),
        "n_test":              int(len(test_rt)),
        "train_class_counts":  class_counts(labels, train_rt),
        "val_class_counts":    class_counts(labels, val_rt),
        "test_class_counts":   class_counts(labels, test_rt),
        "train_idx":           train_rt.tolist(),
        "val_idx":             val_rt.tolist(),
        "test_idx":            test_rt.tolist(),
    }

    splits["default"] = cfg["split_strategy"]

    out_path = metadata_dir / "splits_same_as_notebook_preproc.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nWrote: {out_path}")
    print(f"Default strategy: {splits['default']}")

    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    for strategy in ["block_aware", "random_trial"]:
        s = splits[strategy]
        print(f"  {strategy}:  "
              f"train={s['n_train']}  val={s['n_val']}  test={s['n_test']}")
    print("Done.\n")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    os.chdir(proj_root)
    main()
