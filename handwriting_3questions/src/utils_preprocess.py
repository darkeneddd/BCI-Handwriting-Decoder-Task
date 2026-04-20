"""
utils_preprocess.py
────────────────────
Preprocessing functions that replicate the notebook's signal-conditioning
philosophy:

  1. Block-mean subtraction   – removes block-level baseline offsets / drift
  2. Global channel normalization – standardizes channel scales using a
                                    dataset-wide std estimate
  3. Gaussian smoothing        – along the time axis only
  4. Movement-window extraction – keeps only the movement-related epoch
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


# ─── A2. Block-mean subtraction ───────────────────────────────────────────────

def subtract_block_mean(trial: np.ndarray,
                        block_mean: np.ndarray) -> np.ndarray:
    """
    Subtract the block-level mean from a single trial.

    Parameters
    ----------
    trial      : (T, C) float32 – raw neural activity for one trial
    block_mean : (C,) float64   – mean activity of the block this trial belongs
                                  to (from meansPerBlock[block_index])

    Returns
    -------
    (T, C) float32 with block mean subtracted channel-wise.

    This removes slow baseline drifts that are common to all trials within a
    block, analogous to a local DC offset correction.
    """
    mean = block_mean.flatten().astype(np.float32)   # (C,)
    return trial.astype(np.float32) - mean[np.newaxis, :]  # broadcast over time


# ─── A3. Global channel normalization ────────────────────────────────────────

def normalize_channels(trial: np.ndarray,
                       global_std: np.ndarray,
                       epsilon: float = 1e-8) -> np.ndarray:
    """
    Divide each channel by a global (cross-session) standard deviation.

    Parameters
    ----------
    trial      : (T, C) float32 – block-mean-subtracted trial
    global_std : (C,) float64   – stdAcrossAllData from the MATLAB file, or an
                                  equivalent quantity computed once on all data
    epsilon    : small constant added to denominator to prevent division by zero

    Returns
    -------
    (T, C) float32 with channels standardized to unit-scale.

    Using a global std rather than a per-split std avoids any information
    leakage about the test-set distribution.
    """
    std = global_std.flatten().astype(np.float32)   # (C,)
    return trial / (std[np.newaxis, :] + epsilon)


# ─── A4. Gaussian temporal smoothing ─────────────────────────────────────────

def gaussian_smooth(trial: np.ndarray,
                    sigma_ms: float = 20.0,
                    dt_ms: float = 10.0) -> np.ndarray:
    """
    Apply Gaussian smoothing along the TIME axis only (axis 0).

    Parameters
    ----------
    trial    : (T, C) float32
    sigma_ms : smoothing kernel standard deviation in milliseconds
    dt_ms    : bin duration in milliseconds

    The kernel width sigma_bins = sigma_ms / dt_ms.  For the default values
    (sigma_ms=20, dt_ms=10) this gives sigma_bins=2.0.

    Each channel is smoothed independently; channels are never mixed.
    """
    if sigma_ms <= 0:
        return trial
    sigma_bins = sigma_ms / dt_ms
    smoothed = scipy.ndimage.gaussian_filter1d(
        trial.astype(np.float32), sigma=sigma_bins, axis=0
    )
    return smoothed.astype(np.float32)


# ─── A6. Movement-window extraction ──────────────────────────────────────────

def extract_movement_window(trial: np.ndarray,
                            win_start: int = 51,
                            win_end: int = 151) -> np.ndarray:
    """
    Slice the movement-related epoch from a pre-cut neural trial.

    Parameters
    ----------
    trial     : (T_raw, C) – full pre-cut trial aligned to go cue
    win_start : first bin to include (default 51, = go-cue aligned onset)
    win_end   : one-past-last bin (default 151)

    Returns
    -------
    (T_fixed, C) where T_fixed = win_end - win_start = 100 bins

    Bin 51 is defined as the movement-onset reference: bins 51..150 cover
    1000 ms of neural activity that spans the character writing movement.
    """
    return trial[win_start:win_end, :].astype(np.float32)


# ─── Full per-trial preprocessing pipeline ───────────────────────────────────

def preprocess_trial(raw_trial: np.ndarray,
                     block_mean: np.ndarray,
                     global_std: np.ndarray,
                     sigma_ms: float = 20.0,
                     dt_ms: float = 10.0,
                     win_start: int = 51,
                     win_end: int = 151,
                     epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply the full notebook-matching preprocessing pipeline to one trial.

    Steps (matching notebook logic):
      A2. Subtract block-mean              → removes block-level drift
      A3. Divide by global channel std     → normalizes channel scales
      A4. Gaussian smooth (time axis only) → reduces high-freq noise
      A6. Extract movement window [51:151] → focuses on writing epoch

    Parameters
    ----------
    raw_trial  : (T_raw, C) uint8/float32 – raw neural activity cube row
    block_mean : (C,)       float64       – meansPerBlock[block_index]
    global_std : (C,)       float64       – stdAcrossAllData
    sigma_ms   : Gaussian kernel width in ms (default 20.0)
    dt_ms      : bin size in ms           (default 10.0)
    win_start  : first bin of movement window (default 51)
    win_end    : end bin (exclusive)          (default 151)
    epsilon    : division epsilon for channel normalization

    Returns
    -------
    (T_fixed, C) float32  where T_fixed = win_end - win_start = 100
    """
    t = raw_trial.astype(np.float32)
    t = subtract_block_mean(t, block_mean)
    t = normalize_channels(t, global_std, epsilon)
    t = gaussian_smooth(t, sigma_ms, dt_ms)
    t = extract_movement_window(t, win_start, win_end)
    return t


# ─── Global std computation (fallback) ───────────────────────────────────────

def compute_global_std(trials_block_subtracted: list[np.ndarray],
                       epsilon: float = 1e-8) -> np.ndarray:
    """
    Compute a global per-channel standard deviation across all trials and time
    points.  Used as a fallback when stdAcrossAllData is not available in the
    MATLAB file.

    Parameters
    ----------
    trials_block_subtracted : list of (T, C) arrays already block-mean-subtracted

    Returns
    -------
    (C,) float64 global std per channel

    The std is computed over the concatenation of all timepoints from all
    trials, so each timepoint contributes equally (not each trial).  A small
    epsilon is added to prevent zero-division downstream.
    """
    stacked = np.vstack(trials_block_subtracted)  # (N*T, C)
    std = stacked.std(axis=0) + epsilon
    return std.astype(np.float64)
