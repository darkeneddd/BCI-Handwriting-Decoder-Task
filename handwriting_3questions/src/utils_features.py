"""
utils_features.py
──────────────────
Reusable feature-extraction functions that operate on the processed trial
array of shape (N, T_fixed, C).

Three representations are supported:

  flat         – flatten [T, C] → [T*C]                  (full spatiotemporal)
  temporal     – mean within 3 temporal windows → [3*C]  (coarse temporal)
  flat_pca     – flat features projected through PCA     (reduced-dimensional)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ─── D1. Flat features ────────────────────────────────────────────────────────

def build_flat(trials: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Flatten each (T, C) trial to a 1-D vector of length T*C.

    Parameters
    ----------
    trials : (N, T, C) float32
    idx    : indices of trials to include

    Returns
    -------
    (len(idx), T*C) float32
    """
    rows = [trials[i].flatten() for i in idx]
    return np.vstack(rows).astype(np.float32)


# ─── D2. Temporal features ────────────────────────────────────────────────────

# Default windows (bin boundaries within the 100-bin movement window)
TEMPORAL_WINDOWS_DEFAULT = {
    "early":  (0,  33),
    "middle": (33, 66),
    "late":   (66, 100),
}


def build_temporal(trials: np.ndarray,
                   idx: np.ndarray,
                   windows: dict | None = None) -> np.ndarray:
    """
    Summarize each trial as mean activity within 3 temporal windows.

    The movement window (100 bins) is divided into:
      early  = bins [0,  33)
      middle = bins [33, 66)
      late   = bins [66, 100)

    Each window is averaged over time per channel, then the three windows are
    concatenated → feature vector of length 3 * C.

    This tests whether broad temporal envelopes (early / middle / late activity)
    are sufficient for decoding, compared to the full spatiotemporal detail in
    the flat representation.

    Parameters
    ----------
    trials  : (N, T, C) float32
    idx     : indices of trials to include
    windows : optional dict {name: (start, end)} – overrides defaults

    Returns
    -------
    (len(idx), 3*C) float32
    """
    if windows is None:
        windows = TEMPORAL_WINDOWS_DEFAULT

    win_list = list(windows.values())   # preserve insertion order
    rows = []
    for i in idx:
        trial = trials[i]              # (T, C)
        parts = [trial[a:b].mean(axis=0) for a, b in win_list]
        rows.append(np.concatenate(parts))
    return np.vstack(rows).astype(np.float32)


def temporal_window_boundaries(cfg: dict) -> dict[str, tuple[int, int]]:
    """
    Read temporal window boundaries from config and return as dict.

    Expected config structure:
      temporal_windows:
        early:  [0,  33]
        middle: [33, 66]
        late:   [66, 100]
    """
    raw = cfg.get("temporal_windows", {})
    if not raw:
        return TEMPORAL_WINDOWS_DEFAULT
    return {name: tuple(bounds) for name, bounds in raw.items()}


# ─── D3. Flat-PCA features ────────────────────────────────────────────────────

class FlatPCATransformer:
    """
    Wraps StandardScaler → PCA applied to flat features.

    The scaler and PCA are ALWAYS fitted on training data only.  Validation
    and test data are only transformed, never used to update the fit.
    """

    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cumulative_variance_: np.ndarray | None = None

    def fit(self, X_train_flat: np.ndarray) -> "FlatPCATransformer":
        """
        Fit scaler and PCA on flat training features.

        Parameters
        ----------
        X_train_flat : (N_train, T*C) – flat features from training set only
        """
        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(X_train_flat)

        k = min(self.n_components, X_sc.shape[0], X_sc.shape[1])
        self.pca = PCA(n_components=k, random_state=self.random_state)
        self.pca.fit(X_sc)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_.copy()
        self.cumulative_variance_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Apply the fitted scaler → PCA to new data.

        Parameters
        ----------
        X_flat : (N, T*C) – flat features (train, val, or test)

        Returns
        -------
        (N, n_components) float32
        """
        assert self.scaler is not None and self.pca is not None, \
            "Must call fit() before transform()."
        X_sc = self.scaler.transform(X_flat)
        return self.pca.transform(X_sc).astype(np.float32)

    def fit_transform(self, X_train_flat: np.ndarray) -> np.ndarray:
        self.fit(X_train_flat)
        return self.transform(X_train_flat)


def build_flat_pca(trials: np.ndarray,
                   train_idx: np.ndarray,
                   target_idx: np.ndarray,
                   n_components: int,
                   random_state: int = 42) -> tuple[np.ndarray, FlatPCATransformer]:
    """
    Build flat-PCA features with no leakage.

    PCA is fitted exclusively on the training set.  The transformer is
    returned so it can be reused for other splits.

    Parameters
    ----------
    trials       : (N, T, C) float32
    train_idx    : indices for the training set (PCA fitted here)
    target_idx   : indices for the split you want projected features for
    n_components : number of PCA components

    Returns
    -------
    (X_projected, transformer)
    """
    transformer = FlatPCATransformer(n_components, random_state)
    X_train_flat = build_flat(trials, train_idx)
    transformer.fit(X_train_flat)
    X_target_flat = build_flat(trials, target_idx)
    return transformer.transform(X_target_flat), transformer
