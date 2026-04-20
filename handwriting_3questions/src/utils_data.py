"""
utils_data.py
─────────────
Shared data-loading helpers used across all pipeline scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.io
import yaml


# ─── config ───────────────────────────────────────────────────────────────────

def load_config(cfg_path: str | Path = "configs/config.yaml") -> dict:
    """Load YAML config and return as dict."""
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ─── .mat file loading ────────────────────────────────────────────────────────

def mat_load(path: Path) -> dict[str, Any]:
    """
    Load a MATLAB .mat file.

    Tries scipy.io first (handles v5/v6).  Falls back to h5py for HDF5-based
    MATLAB v7.3 files.  Returns a plain dict of numpy arrays.
    """
    path = Path(path)
    try:
        data = scipy.io.loadmat(str(path), squeeze_me=False)
        # scipy adds meta-keys starting with '__'; keep only real variables
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception:
        pass
    # HDF5 fallback
    out: dict[str, Any] = {}
    with h5py.File(str(path), "r") as hf:
        for key in hf.keys():
            out[key] = hf[key][()]
    return out


# ─── string extraction from MATLAB object arrays ──────────────────────────────

def extract_char_label(cue_val) -> str:
    """
    Pull a plain string out of nested numpy object arrays produced by MATLAB.

    MATLAB stores character cues as cell arrays that scipy loads as nested
    object arrays; we unwrap until we reach a scalar string.
    """
    val = cue_val
    while isinstance(val, np.ndarray):
        val = val.flat[0]
    return str(val).strip()


# ─── block look-up ────────────────────────────────────────────────────────────

def block_id_at_bin(go_bin: int, block_nums_ts: np.ndarray) -> int:
    """
    Return the block number recorded at a given time-series bin.

    go_bin         : integer bin index of the go-cue onset
    block_nums_ts  : 1-D array of block labels over the whole session time axis
    """
    idx = max(0, min(int(go_bin), len(block_nums_ts) - 1))
    return int(block_nums_ts.flat[idx])


# ─── processed dataset loading ────────────────────────────────────────────────

def load_processed(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    """
    Load the preprocessed NPZ produced by script 01.

    Returns
    -------
    trials      : (N, T_fixed, C) float32 array
    labels      : (N,) str array
    session_ids : (N,) str array
    block_ids   : (N,) int32 array
    trial_ids   : (N,) int32 array
    """
    npz_path = Path(cfg["processed_dir"]) / "single_char_trials_preprocessed_same_as_notebook.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {npz_path}\n"
            "Run 01_extract_and_preprocess_trials_same_as_notebook.py first."
        )
    data = np.load(str(npz_path), allow_pickle=True)
    trials      = data["trials"].astype(np.float32)   # (N, T, C)
    labels      = data["labels"].astype(str)
    session_ids = data["session_ids"].astype(str)
    block_ids   = data["block_ids"].astype(np.int32)
    trial_ids   = data["trial_ids"].astype(np.int32)
    return trials, labels, session_ids, block_ids, trial_ids


def load_splits(cfg: dict) -> dict:
    """Load the splits JSON produced by script 02."""
    splits_path = Path(cfg["metadata_dir"]) / "splits_same_as_notebook_preproc.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found at {splits_path}\n"
            "Run 02_make_splits.py first."
        )
    with open(splits_path) as f:
        return json.load(f)


# ─── experiment results helpers ───────────────────────────────────────────────

def append_results_csv(rows: list[dict], csv_path: Path) -> None:
    """Append a list of result dicts to a CSV (create if missing)."""
    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
