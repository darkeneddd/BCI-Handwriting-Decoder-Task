"""
01_extract_and_preprocess_trials_same_as_notebook.py
──────────────────────────────────────────────────────
Load every session's singleLetters.mat, apply the notebook-matching
preprocessing pipeline, and save a clean NPZ for downstream use.

Preprocessing steps (matching the notebook philosophy):
  A1. Load raw session data  – neural cubes, labels, block metadata
  A2. Block-mean subtraction – subtract meansPerBlock from each trial
  A3. Global channel normalization – divide by stdAcrossAllData (+ epsilon)
  A4. Gaussian temporal smoothing – sigma_ms=20 → sigma_bins=2.0 at dt=10ms
  A5. Movement-window extraction  – slice bins [51, 151) → 100 time bins

The block means and global std come directly from the MATLAB dataset objects
(meansPerBlock, stdAcrossAllData), avoiding any recomputation bias.

Outputs
───────
  processed/single_char_trials_preprocessed_same_as_notebook.npz
      trials      – (N, 100, 192) float32
      labels      – (N,) str
      session_ids – (N,) str
      block_ids   – (N,) int32
      trial_ids   – (N,) int32

  metadata/single_char_trials_metadata.csv
  metadata/preprocessing_summary.json

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python src/01_extract_and_preprocess_trials_same_as_notebook.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_data       import load_config, mat_load, extract_char_label, block_id_at_bin
from utils_preprocess import preprocess_trial, compute_global_std


# ─── per-session extraction ───────────────────────────────────────────────────

def extract_and_preprocess_session(
    session_dir: Path,
    session_id: str,
    global_trial_offset: int,
    cfg: dict,
) -> tuple[list[np.ndarray], list[dict], np.ndarray | None, np.ndarray | None, int]:
    """
    Load one session, apply preprocessing, return trial arrays + metadata.

    Returns
    -------
    trials      : list of (T_fixed, C) float32 arrays
    metas       : list of per-trial metadata dicts
    block_means : (n_blocks, C) array of meansPerBlock (for fallback std calc)
    global_std  : (C,) stdAcrossAllData if available, else None
    n_skipped_do_nothing : count of excluded doNothing trials in this session
    """
    mat_path = session_dir / "singleLetters.mat"
    if not mat_path.exists():
        print(f"  [skip] no singleLetters.mat in {session_dir.name}")
        return [], [], None, None, 0

    print(f"\n  Loading {mat_path} …", end=" ", flush=True)
    data = mat_load(mat_path)
    print("ok")

    # ── unpack metadata vectors ────────────────────────────────────────────
    char_cues   = data["characterCues"].flatten()              # (n_trials,)
    go_bins     = data["goPeriodOnsetTimeBin"].flatten()       # (n_trials,)
    delay_bins  = (data["delayPeriodOnsetTimeBin"].flatten()
                   if "delayPeriodOnsetTimeBin" in data
                   else np.full_like(go_bins, -1))
    block_ts    = data["blockNumsTimeSeries"].flatten()        # (n_time,)
    block_list  = data["blockList"].flatten()                  # (n_blocks,)

    # ── block means (meansPerBlock) ────────────────────────────────────────
    # Shape: (n_blocks, C) – one row per block, in the same order as blockList
    if "meansPerBlock" in data:
        means_per_block = data["meansPerBlock"]                # (n_blocks, C)
        if means_per_block.ndim == 1:
            means_per_block = means_per_block[np.newaxis, :]
        print(f"    meansPerBlock shape: {means_per_block.shape}  "
              f"→ {means_per_block.shape[0]} blocks, {means_per_block.shape[1]} channels")
    else:
        print("    ⚠ meansPerBlock not found – will use zero mean (no baseline subtraction)")
        # Determine C from first available cube
        first_cube_key = next(
            (k for k in data if k.startswith("neuralActivityCube_")), None
        )
        C = data[first_cube_key].shape[2] if first_cube_key else cfg["n_channels"]
        means_per_block = np.zeros((len(block_list), C), dtype=np.float64)

    # Build a mapping: block_id_value → row index in meansPerBlock
    block_id_to_idx: dict[int, int] = {
        int(bid): i for i, bid in enumerate(block_list)
    }

    # ── global std (stdAcrossAllData) ─────────────────────────────────────
    global_std_session: np.ndarray | None = None
    if "stdAcrossAllData" in data:
        global_std_session = data["stdAcrossAllData"].flatten().astype(np.float64)
        print(f"    stdAcrossAllData: shape={global_std_session.shape}  "
              f"mean={global_std_session.mean():.4f}")
    else:
        print("    ⚠ stdAcrossAllData not found – will compute globally after loading")

    # ── preprocessing parameters ──────────────────────────────────────────
    sigma_ms   = float(cfg["sigma_ms"])
    dt_ms      = float(cfg["dt_ms"])
    win_start  = int(cfg["move_win_start"])
    win_end    = int(cfg["move_win_end"])
    epsilon    = float(cfg["norm_epsilon"])

    # ── iterate over trials ───────────────────────────────────────────────
    char_rep_counter: dict[str, int] = {}
    trials, metas = [], []
    skipped_do_nothing = 0

    n_trials_session = len(char_cues)
    for t_idx in range(n_trials_session):
        char    = extract_char_label(char_cues[t_idx])
        # Exclude explicit "doNothing" class from this single-character decoder.
        # Handle capitalization/format variants robustly.
        char_norm = char.lower().replace("_", "").replace(" ", "")
        if char_norm == "donothing":
            skipped_do_nothing += 1
            continue
        go_bin  = int(go_bins[t_idx])
        delay_b = int(delay_bins[t_idx])
        block_id = block_id_at_bin(go_bin, block_ts)

        # Locate cube row for this trial
        rep = char_rep_counter.get(char, 0)
        char_rep_counter[char] = rep + 1

        cube_key = f"neuralActivityCube_{char}"
        if cube_key not in data:
            continue
        cube = data[cube_key]  # (n_reps, T_raw, C)
        if rep >= cube.shape[0]:
            continue

        raw_trial = cube[rep].astype(np.float32)  # (T_raw, C)

        # Validate window fits within the raw trial
        if raw_trial.shape[0] < win_end:
            print(f"    ⚠ trial {t_idx} (char={char}, rep={rep}) too short: "
                  f"T={raw_trial.shape[0]} < win_end={win_end}  → skip")
            continue

        # Get block mean for this trial's block
        block_idx = block_id_to_idx.get(block_id)
        if block_idx is None:
            # Fallback: use zeros if block_id not in blockList
            block_mean = np.zeros(raw_trial.shape[1], dtype=np.float64)
        else:
            block_mean = means_per_block[block_idx]  # (C,)

        # Preprocessing: A2–A4 applied now; A3 uses session std if available,
        # otherwise we store the block-subtracted trial and normalize later.
        if global_std_session is not None:
            preprocessed = preprocess_trial(
                raw_trial, block_mean, global_std_session,
                sigma_ms=sigma_ms, dt_ms=dt_ms,
                win_start=win_start, win_end=win_end, epsilon=epsilon,
            )
        else:
            # Defer normalization: apply only A2, A4, A6 for now
            from utils_preprocess import subtract_block_mean, gaussian_smooth, extract_movement_window
            t = subtract_block_mean(raw_trial, block_mean)
            t = gaussian_smooth(t, sigma_ms, dt_ms)
            t = extract_movement_window(t, win_start, win_end)
            preprocessed = t  # normalization applied below

        trials.append(preprocessed)
        metas.append({
            "trial_id":          global_trial_offset + len(metas),
            "session_id":        session_id,
            "block_id":          block_id,
            "session_trial_idx": t_idx,
            "char_rep":          rep,
            "label":             char,
            "go_onset_bin":      go_bin,
            "delay_onset_bin":   delay_b,
            "T":                 preprocessed.shape[0],
            "n_channels":        preprocessed.shape[1],
            "had_global_std":    global_std_session is not None,
        })

    print(f"    → {len(trials)} trials extracted from {session_id}"
          f"  (excluded doNothing: {skipped_do_nothing})")
    return trials, metas, means_per_block, global_std_session, skipped_do_nothing


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg          = load_config()
    dataset_root = Path(cfg["dataset_root"])
    processed_dir = Path(cfg["processed_dir"])
    metadata_dir  = Path(cfg["metadata_dir"])
    sessions      = cfg["sessions"]
    epsilon       = float(cfg["norm_epsilon"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    all_trials:    list[np.ndarray] = []
    all_metas:     list[dict]       = []
    all_stds:      list[np.ndarray] = []   # one per session if available
    total_skipped_do_nothing = 0

    print(f"\n{'='*70}")
    print(f"Extracting & preprocessing single-character trials")
    print(f"Sessions: {sessions}")
    print(f"Dataset root: {dataset_root}")
    print(f"{'='*70}")

    offset = 0
    for sess in sessions:
        sess_dir = dataset_root / sess
        t, m, _, std, n_skip_dn = extract_and_preprocess_session(
            sess_dir, sess, offset, cfg
        )
        all_trials.extend(t)
        all_metas.extend(m)
        offset += len(m)
        total_skipped_do_nothing += int(n_skip_dn)
        if std is not None:
            all_stds.append(std)

    N = len(all_trials)
    print(f"\nTotal trials collected: {N}")
    if N == 0:
        print("ERROR: no trials found – check dataset_root in config.")
        sys.exit(1)

    # ── Deferred global normalization (if any session lacked stdAcrossAllData)
    needs_deferred_norm = any(not m.get("had_global_std", True) for m in all_metas)
    if needs_deferred_norm:
        print("\nComputing global channel std across all sessions (fallback) …",
              end=" ", flush=True)
        global_std = compute_global_std(all_trials, epsilon=epsilon)
        print(f"done  (mean std={global_std.mean():.4f})")
        # Apply normalization to trials that weren't normalized yet
        for i, m in enumerate(all_metas):
            if not m.get("had_global_std", True):
                all_trials[i] = all_trials[i] / (global_std[np.newaxis, :] + epsilon)
        print("  Deferred normalization applied.")
    else:
        print("\nAll sessions provided stdAcrossAllData – no deferred normalization needed.")
        global_std = None

    # ── Stack into a single (N, T_fixed, C) array ─────────────────────────
    T_fixed = all_trials[0].shape[0]
    C       = all_trials[0].shape[1]

    # Verify all trials have the same shape
    shapes = set(t.shape for t in all_trials)
    if len(shapes) > 1:
        print(f"  WARNING: inconsistent trial shapes detected: {shapes}")
        print("  Keeping only trials matching the most common shape …")
        from collections import Counter
        shape_counts = Counter(t.shape for t in all_trials)
        dominant = shape_counts.most_common(1)[0][0]
        keep = [i for i, t in enumerate(all_trials) if t.shape == dominant]
        all_trials = [all_trials[i] for i in keep]
        all_metas  = [all_metas[i]  for i in keep]
        N = len(all_trials)
        T_fixed, C = dominant

    trials_arr = np.stack(all_trials, axis=0).astype(np.float32)  # (N, T, C)
    print(f"\nFinal array shape: {trials_arr.shape}")
    print(f"  dtype={trials_arr.dtype}")
    print(f"  mean={trials_arr.mean():.4f}  std={trials_arr.std():.4f}")
    print(f"  min={trials_arr.min():.4f}   max={trials_arr.max():.4f}")

    labels       = np.array([m["label"]       for m in all_metas])
    session_ids  = np.array([m["session_id"]  for m in all_metas])
    block_ids    = np.array([m["block_id"]    for m in all_metas], dtype=np.int32)
    trial_ids    = np.array([m["trial_id"]    for m in all_metas], dtype=np.int32)

    # ── Save NPZ ──────────────────────────────────────────────────────────
    npz_path = processed_dir / "single_char_trials_preprocessed_same_as_notebook.npz"
    np.savez(
        str(npz_path),
        trials=trials_arr,
        labels=labels,
        session_ids=session_ids,
        block_ids=block_ids,
        trial_ids=trial_ids,
    )
    print(f"\nWrote: {npz_path}")

    # ── Save metadata CSV ─────────────────────────────────────────────────
    df = pd.DataFrame(all_metas)
    csv_path = metadata_dir / "single_char_trials_metadata.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"Wrote: {csv_path}")

    # ── Save preprocessing summary JSON ───────────────────────────────────
    from collections import Counter
    label_counts = Counter(labels.tolist())
    summary = {
        "n_sessions":         int(len(set(session_ids))),
        "sessions":           sorted(set(session_ids.tolist())),
        "n_trials":           int(N),
        "excluded_doNothing_trials": int(total_skipped_do_nothing),
        "T_fixed":            int(T_fixed),
        "n_channels":         int(C),
        "class_counts":       dict(sorted(label_counts.items())),
        "n_classes":          int(len(label_counts)),
        "preprocessing": {
            "steps": [
                "A2: block-mean subtraction (meansPerBlock from MATLAB dataset)",
                "A3: global channel normalization (stdAcrossAllData from MATLAB dataset)",
                "A4: Gaussian temporal smoothing (sigma_ms=20.0, dt_ms=10.0, sigma_bins=2.0)",
                "A6: movement-window extraction (bins [51, 151) → T_fixed=100)",
            ],
            "sigma_ms":       float(cfg["sigma_ms"]),
            "dt_ms":          float(cfg["dt_ms"]),
            "sigma_bins":     float(cfg["sigma_ms"]) / float(cfg["dt_ms"]),
            "go_bin":         int(cfg["go_bin"]),
            "move_win_start": int(cfg["move_win_start"]),
            "move_win_end":   int(cfg["move_win_end"]),
            "norm_epsilon":   float(cfg["norm_epsilon"]),
            "used_dataset_block_means":  True,
            "used_dataset_global_std":   not needs_deferred_norm,
        },
    }
    summary_path = metadata_dir / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {summary_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PREPROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"  Sessions:          {summary['n_sessions']}")
    print(f"  Total trials:      {N}")
    print(f"  Excluded doNothing:{total_skipped_do_nothing}")
    print(f"  Classes:           {summary['n_classes']}")
    print(f"  Trial shape:       ({T_fixed}, {C})")
    print(f"  Channels:          {C}")
    print(f"  Post-norm mean:    {trials_arr.mean():.4f}")
    print(f"  Post-norm std:     {trials_arr.std():.4f}")
    print(f"\nClass counts:")
    for char, count in sorted(label_counts.items()):
        print(f"  {char:20s}  {count:4d}")
    print("\nDone.\n")


if __name__ == "__main__":
    proj_root = Path(__file__).resolve().parent.parent
    os.chdir(proj_root)
    main()
