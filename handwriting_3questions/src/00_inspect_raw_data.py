"""
00_inspect_raw_data.py
───────────────────────
Inspect the raw singleLetters.mat files across all sessions.

Prints the keys, shapes, dtypes, and sample values for each field that is
relevant to the preprocessing pipeline (neural cubes, block metadata,
normalization arrays, timing).

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/00_inspect_raw_data.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running from project root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_data import load_config, mat_load


# ─── fields to inspect ───────────────────────────────────────────────────────

INSPECT_FIELDS = [
    "characterCues",
    "goPeriodOnsetTimeBin",
    "delayPeriodOnsetTimeBin",
    "blockNumsTimeSeries",
    "blockList",
    "meansPerBlock",
    "stdAcrossAllData",
]

# Also inspect one neural cube per session
CUBE_PREFIX = "neuralActivityCube_"


def inspect_field(data: dict, key: str) -> None:
    """Print shape, dtype, min, max, and a small sample for one field."""
    if key not in data:
        print(f"    [{key}]  NOT FOUND")
        return
    val = data[key]
    if isinstance(val, np.ndarray):
        sample = val.flat[0] if val.size > 0 else "(empty)"
        try:
            print(f"    [{key}]  shape={val.shape}  dtype={val.dtype}  "
                  f"min={float(np.nanmin(val)):.4f}  max={float(np.nanmax(val)):.4f}  "
                  f"sample={sample}")
        except (TypeError, ValueError):
            print(f"    [{key}]  shape={val.shape}  dtype={val.dtype}  "
                  f"sample={repr(sample)[:60]}")
    else:
        print(f"    [{key}]  type={type(val).__name__}  value={repr(val)[:80]}")


def inspect_session(sess_dir: Path, session_id: str) -> None:
    mat_path = sess_dir / "singleLetters.mat"
    if not mat_path.exists():
        print(f"  [skip] no singleLetters.mat in {sess_dir.name}")
        return

    print(f"\n{'='*70}")
    print(f"Session: {session_id}")
    print(f"  File: {mat_path}")

    data = mat_load(mat_path)
    all_keys = sorted(data.keys())
    cube_keys = [k for k in all_keys if k.startswith(CUBE_PREFIX)]
    non_cube_keys = [k for k in all_keys if not k.startswith(CUBE_PREFIX)]

    print(f"\n  All keys ({len(all_keys)} total):")
    print(f"    Non-cube: {non_cube_keys}")
    print(f"    Neural cubes: {cube_keys[:5]}{'...' if len(cube_keys) > 5 else ''}")
    print(f"    Total cube keys: {len(cube_keys)}")

    print("\n  Standard metadata fields:")
    for field in INSPECT_FIELDS:
        inspect_field(data, field)

    if cube_keys:
        first_cube = cube_keys[0]
        print(f"\n  Example cube ({first_cube}):")
        inspect_field(data, first_cube)
        cube = data[first_cube]
        if isinstance(cube, np.ndarray) and cube.ndim == 3:
            n_reps, T, C = cube.shape
            print(f"    → n_reps={n_reps}, T={T}, C={C}")
            print(f"    → mean(abs)={float(np.abs(cube).mean()):.4f}")

    # Check for block index alignment
    if "characterCues" in data and "goPeriodOnsetTimeBin" in data:
        n_trials = data["characterCues"].flatten().shape[0]
        print(f"\n  n_trials (characterCues): {n_trials}")

    # Summarize block structure
    if "blockList" in data and "meansPerBlock" in data:
        blocks = data["blockList"].flatten()
        mpb    = data["meansPerBlock"]
        print(f"\n  Block structure:")
        print(f"    blockList: {blocks[:10]}{'...' if len(blocks) > 10 else ''}")
        print(f"    n_blocks in blockList: {len(blocks)}")
        print(f"    meansPerBlock shape: {mpb.shape}")
        if mpb.shape[0] == len(blocks):
            print(f"    ✓ meansPerBlock rows match blockList length")
        else:
            print(f"    ⚠ shape mismatch: meansPerBlock has {mpb.shape[0]} rows, "
                  f"blockList has {len(blocks)} entries")

    if "stdAcrossAllData" in data:
        std = data["stdAcrossAllData"].flatten()
        print(f"\n  stdAcrossAllData: shape={std.shape}  "
              f"min={std.min():.4f}  max={std.max():.4f}  mean={std.mean():.4f}")
        n_zero = int((std == 0).sum())
        if n_zero > 0:
            print(f"    ⚠ {n_zero} channels have std=0 (will need epsilon in normalization)")


def main() -> None:
    cfg = load_config()
    dataset_root = Path(cfg["dataset_root"])
    sessions     = cfg["sessions"]

    print(f"Dataset root: {dataset_root}")
    print(f"Sessions to inspect: {sessions}")

    summary_rows = []
    for sess in sessions:
        sess_dir = dataset_root / sess
        inspect_session(sess_dir, sess)

    print(f"\n{'='*70}")
    print("Inspection complete.")
    print("\nPreprocessing pipeline uses:")
    print(f"  GO_BIN        = {cfg['go_bin']}")
    print(f"  MOVE_WIN      = [{cfg['move_win_start']}, {cfg['move_win_end']})")
    print(f"  T_fixed       = {cfg['T_fixed']} bins")
    print(f"  sigma_ms      = {cfg['sigma_ms']} ms")
    print(f"  dt_ms         = {cfg['dt_ms']} ms")
    print(f"  sigma_bins    = {cfg['sigma_ms']/cfg['dt_ms']:.1f}")
    print(f"  norm_epsilon  = {cfg['norm_epsilon']}")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    os.chdir(proj_root)
    main()
