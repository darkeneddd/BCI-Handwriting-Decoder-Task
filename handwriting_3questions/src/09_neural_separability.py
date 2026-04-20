"""
09_neural_separability.py
──────────────────────────
Mirror of neural_separability_analysis.ipynb
(Willett et al. 2021 — Extended Data Figure 1 reproduction)

Sections
────────
  1. Dataset structure — inspect singleLetters.mat
  2. Raw neural feature visualization
  3. Preprocessing comparison (raw vs z-score + smoothed)
  4. Trial-to-trial variability
  5. PCA neural trajectories (raw vs DTW time-aligned)
  6. Pen-tip velocity ↔ neural activity  (Ridge R²)
  7. Load all sessions  (flat movement-window features)
  8. t-SNE visualization
  9. k-NN (k=1 LOO) classification accuracy
 10. Summary figure

All figures and tables → results/part1_results/

Usage
─────
  cd /Users/yang/Desktop/py/3803/handwriting_3questions
  python 09_neural_separability.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# ── project root & sys.path ─────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJ_ROOT)
sys.path.insert(0, str(PROJ_ROOT / "src"))

from utils_data import load_config

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ── config & paths ───────────────────────────────────────────────────────────
cfg          = load_config()
DATASETS_DIR = Path(cfg["dataset_root"])                        # .../handwritingBCIData/Datasets
DATA_ROOT    = DATASETS_DIR.parent                              # .../handwritingBCIData
WARP_DIR     = DATA_ROOT / "RNNTrainingSteps" / "Step1_TimeWarping"
TMPL_PATH    = DATASETS_DIR / "computerMouseTemplates.mat"

OUT_DIR = PROJ_ROOT / "results" / "part1_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH    = OUT_DIR / "separability_summary.txt"
_log_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _log_lines.append(msg)


def save_log() -> None:
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_log_lines) + "\n")
    print(f"\nSummary written → {LOG_PATH}")


def savefig(name: str) -> None:
    stem = Path(name).stem
    for ext in ("png", "svg"):
        plt.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close()
    log(f"  [fig] {stem}.png / .svg")


def mat_load_simple(path: Path) -> dict:
    """Load .mat (v5/v6) with simplify_cells; h5py fallback for v7.3."""
    import h5py
    try:
        return sio.loadmat(str(path), simplify_cells=True)
    except Exception:
        import h5py
        out = {}
        with h5py.File(str(path), "r") as hf:
            for k in hf.keys():
                out[k] = hf[k][()]
        return out


# ── constants ────────────────────────────────────────────────────────────────
PRIMARY_SESSION = "t5.2019.05.08"
ALL_SESSIONS    = sorted(
    d for d in os.listdir(DATASETS_DIR)
    if d.startswith("t5.")
    and (DATASETS_DIR / d / "singleLetters.mat").exists()
)
LETTERS  = list("abcdefghijklmnopqrstuvwxyz")
GO_BIN   = cfg["go_bin"]          # 51
MOVE_WIN = (GO_BIN, GO_BIN + 100) # bins [51, 151) → 0–1000 ms after go cue

log("=" * 70)
log(f"Primary session : {PRIMARY_SESSION}")
log(f"All sessions    : {len(ALL_SESSIONS)}")
log(f"Go cue bin      : {GO_BIN}  (= {GO_BIN*10} ms)")
log(f"Movement window : bins [{MOVE_WIN[0]}, {MOVE_WIN[1]})  → 1000 ms")
log(f"Output dir      : {OUT_DIR}")
log("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset structure
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 1 — Dataset structure")
log("-" * 50)

sl_raw = mat_load_simple(DATASETS_DIR / PRIMARY_SESSION / "singleLetters.mat")

cube_keys  = [k for k in sl_raw if k.startswith("neuralActivityCube_")]
other_keys = [k for k in sl_raw if not k.startswith("_") and not k.startswith("neuralActivityCube_")]

ex = sl_raw["neuralActivityCube_a"]
log(f"Neural cubes   : {len(cube_keys)} letters")
log(f"  example shape: {ex.shape}  (trials, time_bins, channels)")
log(f"  dtype        : {ex.dtype}  range [{ex.min()}, {ex.max()}]  (raw spike counts)")
log("Other fields:")
for k in sorted(other_keys):
    v = sl_raw[k]
    if hasattr(v, "shape"):
        log(f"  {k:<30s} shape={v.shape}  dtype={v.dtype}")
    else:
        log(f"  {k:<30s} {type(v).__name__}: {str(v)[:80]}")

n_trials_primary, n_time, n_chan = ex.shape


# ══════════════════════════════════════════════════════════════════════════════
# 2. Raw neural feature visualization
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 2 — Raw neural feature visualization")
log("-" * 50)

# Most active channel (highest mean spike count across all letters & trials)
most_active_ch = int(np.array([
    sl_raw[f"neuralActivityCube_{l}"].mean(axis=(0, 1))
    for l in LETTERS if f"neuralActivityCube_{l}" in sl_raw
]).mean(axis=0).argmax())

t_bins      = np.arange(n_time) * 10   # ms
colors_l    = cm.tab20(np.linspace(0, 1, 26))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Raw neural features (not preprocessed)", fontsize=12)

for i, l in enumerate(LETTERS):
    key = f"neuralActivityCube_{l}"
    if key not in sl_raw:
        continue
    avg = sl_raw[key][:, :, most_active_ch].mean(axis=0)
    axes[0].plot(t_bins, avg, color=colors_l[i], alpha=0.6, linewidth=0.9, label=l)
axes[0].axvline(GO_BIN * 10, color="k", linestyle="--", linewidth=1.5, label=f"go cue (bin {GO_BIN})")
axes[0].set(xlabel="Time (ms)", ylabel="Spike count / 10 ms bin",
            title=f"All letters, trial mean (ch {most_active_ch}, most active)")
axes[0].legend(fontsize=6, ncol=4, loc="upper right")

nts_key = "neuralActivityTimeSeries"
if nts_key in sl_raw:
    ch_rates_hz = sl_raw[nts_key].astype(float).mean(axis=0) * 100
    axes[1].hist(ch_rates_hz, bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(2, color="red", linestyle="--", label="2 Hz threshold")
    axes[1].set(xlabel="Mean firing rate (Hz)", ylabel="# channels",
                title="192-channel firing rate distribution")
    axes[1].legend()
    log(f"Most active channel : {most_active_ch}")
    log(f"Channels ≥ 2 Hz     : {(ch_rates_hz >= 2).sum()} / {n_chan}")
    log(f"Mean firing rate    : {ch_rates_hz.mean():.1f} Hz")
else:
    axes[1].text(0.5, 0.5, "neuralActivityTimeSeries\nnot found",
                 ha="center", va="center", transform=axes[1].transAxes)

plt.tight_layout()
savefig("sep_fig1_explore_raw_data.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Preprocessing helpers + before/after comparison
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 3 — Preprocessing")
log("-" * 50)


def load_and_preprocess(session: str, sigma_ms: float = 20.0, dt_ms: float = 10.0) -> tuple:
    """
    Load one session's singleLetters.mat and return:
      cubes : dict  letter -> (trials, 201, 192) float32  (z-scored + Gaussian smoothed)
      sl    : raw mat-file dict
    """
    sl         = mat_load_simple(DATASETS_DIR / session / "singleLetters.mat")
    means      = np.array(sl["meansPerBlock"], dtype=np.float32)   # (n_blocks, 192)
    global_std = np.array(sl["stdAcrossAllData"], dtype=np.float32).flatten()
    global_std = np.where(global_std == 0, 1.0, global_std)
    block_nums = np.array(sl["blockNumsTimeSeries"]).flatten().astype(int)
    block_list = np.array(sl["blockList"]).flatten().astype(int)
    go_bins    = np.array(sl["goPeriodOnsetTimeBin"]).flatten().astype(int)
    char_cues  = np.array(sl["characterCues"]).flatten()
    sigma_bins = sigma_ms / dt_ms

    # block_id → row index in meansPerBlock
    bml = {int(b): means[i] for i, b in enumerate(block_list)}

    cubes = {}
    for letter in LETTERS:
        key = f"neuralActivityCube_{letter}"
        if key not in sl:
            continue
        trial_idx = np.where(char_cues == letter)[0]
        if len(trial_idx) == 0:
            continue
        tbm = np.array([
            bml.get(int(block_nums[min(int(go_bins[i]), len(block_nums) - 1)]),
                    means.mean(0))
            for i in trial_idx
        ], dtype=np.float32)   # (n_trials_letter, 192)
        cube = sl[key].astype(np.float32)                       # (trials, 201, 192)
        cube = (cube - tbm[:, None, :]) / global_std[None, None, :]
        cube = gaussian_filter1d(cube, sigma=sigma_bins, axis=1)
        cubes[letter] = cube
    return cubes, sl


def trial_avg(cubes: dict) -> dict:
    return {l: cubes[l].mean(axis=0) for l in cubes}


def movement_features(cube: np.ndarray, flatten: bool = True) -> np.ndarray:
    """Movement-window features: bins [GO_BIN, GO_BIN+100]."""
    w = cube[:, MOVE_WIN[0]:MOVE_WIN[1], :]   # (trials, 100, 192)
    return w.reshape(w.shape[0], -1) if flatten else w.mean(axis=1)


log(f"Loading & preprocessing {PRIMARY_SESSION} …")
cubes_primary, _ = load_and_preprocess(PRIMARY_SESSION)
avg_raw          = trial_avg(cubes_primary)

proc_ex = cubes_primary["a"]
log(f"Preprocessed cube shape: {proc_ex.shape}")
log(f"Value range (letter 'a'): [{proc_ex.min():.2f}, {proc_ex.max():.2f}]  "
    f"mean={proc_ex.mean():.4f}")

# Before / after comparison
t_ms      = np.arange(n_time) * 10
raw_t0    = sl_raw["neuralActivityCube_a"][0, :, most_active_ch].astype(float)
proc_t0   = cubes_primary["a"][0, :, most_active_ch]

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
fig.suptitle(f"Preprocessing comparison — letter 'a', channel {most_active_ch}", fontsize=11)
axes[0].plot(t_ms, raw_t0, color="steelblue", linewidth=1.2, label="raw spike count")
axes[0].axvline(GO_BIN * 10, color="red", linestyle="--", alpha=0.7, label="go cue")
axes[0].set(xlabel="Time (ms)", ylabel="Spike count / 10 ms", title="Raw (unprocessed)")
axes[0].legend()

axes[1].plot(t_ms, proc_t0, color="darkorange", linewidth=1.2, label="z-score + smooth")
axes[1].axvline(GO_BIN * 10, color="red", linestyle="--", alpha=0.7, label="go cue")
axes[1].axvspan(MOVE_WIN[0] * 10, MOVE_WIN[1] * 10, alpha=0.12, color="green",
                label="movement window")
axes[1].set(xlabel="Time (ms)", ylabel="Z-score",
            title="After preprocessing (z-score + 20 ms smooth)")
axes[1].legend()
plt.tight_layout()
savefig("sep_fig2_preprocess_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Trial-to-trial variability
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 4 — Trial-to-trial variability")
log("-" * 50)

n_trials_a = cubes_primary["a"].shape[0]
fig, axes  = plt.subplots(1, 2, figsize=(13, 3.5))
fig.suptitle("Trial-to-trial variability — letter 'a'", fontsize=11)

for i in range(n_trials_a):
    axes[0].plot(t_ms, cubes_primary["a"][i, :, most_active_ch],
                 color="steelblue", alpha=0.25, linewidth=0.7)
axes[0].plot(t_ms, avg_raw["a"][:, most_active_ch],
             color="navy", linewidth=2, label="trial mean")
axes[0].axvline(GO_BIN * 10, color="red", linestyle="--", alpha=0.7, label="go cue")
axes[0].axvspan(MOVE_WIN[0] * 10, MOVE_WIN[1] * 10, alpha=0.1, color="green")
axes[0].set(xlabel="Time (ms)", ylabel="Z-score",
            title=f"Channel {most_active_ch}: {n_trials_a} trials + mean")
axes[0].legend()

mov_avg = avg_raw["a"][MOVE_WIN[0]:MOVE_WIN[1], :]   # (100, 192)
im = axes[1].imshow(mov_avg.T, aspect="auto", origin="lower",
                    extent=[0, 1000, 0, n_chan], cmap="RdBu_r")
axes[1].set(xlabel="Time after go cue (ms)", ylabel="Channel",
            title="Trial-mean heatmap (movement window, all channels)")
plt.colorbar(im, ax=axes[1], label="Z-score")
plt.tight_layout()
savefig("sep_fig3_trial_variability.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PCA neural trajectories
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 5 — PCA neural trajectories")
log("-" * 50)


def fit_pca_trajectories(avg_dict: dict, letters: list, n_components: int = 3):
    X   = np.vstack([avg_dict[l] for l in letters if l in avg_dict])
    pca = PCA(n_components=n_components).fit(X)
    return pca, {l: pca.transform(avg_dict[l]) for l in letters if l in avg_dict}


pca_raw, proj_raw = fit_pca_trajectories(avg_raw, LETTERS)
log("Explained variance (raw):   " + "  ".join(
    f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca_raw.explained_variance_ratio_)))

# Time-aligned (DTW-warped) cubes
warp_path = WARP_DIR / f"{PRIMARY_SESSION}_warpedCubes.mat"
have_warp = warp_path.exists()
if have_warp:
    log(f"Loading warpedCubes from {warp_path.name} …")
    warp_mat     = mat_load_simple(warp_path)
    session_mean = sl_raw["meansPerBlock"].mean(axis=0)
    global_std_p = np.where(
        np.array(sl_raw["stdAcrossAllData"]).flatten() == 0, 1.0,
        np.array(sl_raw["stdAcrossAllData"]).flatten()
    )
    avg_warp = {}
    for l in LETTERS:
        if l not in warp_mat:
            continue
        w   = np.array(warp_mat[l], dtype=np.float64)
        avg = np.nanmean(w, axis=0)                             # (201, 192)
        avg = (avg - session_mean[None, :]) / global_std_p[None, :]
        avg = gaussian_filter1d(avg, sigma=2.0, axis=0)
        avg_warp[l] = avg
    pca_warp, proj_warp = fit_pca_trajectories(avg_warp, LETTERS)
    log("Explained variance (warped):" + "  ".join(
        f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca_warp.explained_variance_ratio_)))
    nan_frac = np.isnan(np.array(warp_mat["a"])).mean()
    log(f"warpedCubes NaN fraction ('a'): {nan_frac*100:.1f}%")
else:
    log("  warpedCubes not found — skipping time-aligned trajectories")
    avg_warp = {}
    pca_warp, proj_warp = pca_raw, proj_raw

SHOW          = ["a", "e", "i", "o", "u", "t"]
colors_show   = cm.tab10(np.linspace(0, 1, len(SHOW)))

# 3-D trajectories
fig = plt.figure(figsize=(14, 6))
fig.suptitle("PCA neural trajectories — trial-mean activity (first 3 PCs)", fontsize=13)
for col, (proj, title, pca_obj) in enumerate([
        (proj_raw,  "Raw (not time-aligned)",    pca_raw),
        (proj_warp, "Time-aligned (DTW warped)", pca_warp),
]):
    ax = fig.add_subplot(1, 2, col + 1, projection="3d")
    for i, l in enumerate(SHOW):
        if l not in proj:
            continue
        traj = proj[l]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=colors_show[i], label=l, linewidth=2, alpha=0.85)
        ax.scatter(*traj[0],      color=colors_show[i], s=40, zorder=5)
        ax.scatter(*traj[GO_BIN], color=colors_show[i], s=60,
                   marker="*", zorder=6, edgecolors="k", linewidths=0.5)
    v = pca_obj.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({v[0]*100:.1f}%)", labelpad=6)
    ax.set_ylabel(f"PC2 ({v[1]*100:.1f}%)", labelpad=6)
    ax.set_zlabel(f"PC3 ({v[2]*100:.1f}%)", labelpad=6)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.text2D(0.02, 0.02, "dot=start  star=go cue", transform=ax.transAxes, fontsize=7)

plt.tight_layout()
savefig("sep_fig4_pca_trajectories_3d.png")

# 2-D all-26-letter trajectories
colors_all26 = cm.tab20(np.linspace(0, 1, 26))
fig, axes    = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("All 26 letters: PCA trajectories (PC1 vs PC2)", fontsize=12)
for ax, proj, title in [
        (axes[0], proj_raw,  "Raw"),
        (axes[1], proj_warp, "Time-aligned"),
]:
    for i, l in enumerate(LETTERS):
        if l not in proj:
            continue
        traj = proj[l]
        ax.plot(traj[:, 0], traj[:, 1],
                color=colors_all26[i], linewidth=1.2, alpha=0.75, label=l)
        ax.scatter(traj[GO_BIN, 0], traj[GO_BIN, 1],
                   color=colors_all26[i], s=25, zorder=4)
    ax.set(xlabel="PC1", ylabel="PC2", title=title)
    ax.legend(fontsize=6, ncol=4, loc="lower right")
plt.tight_layout()
savefig("sep_fig5_pca_2d_all_letters.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Pen-tip velocity ↔ neural activity
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 6 — Pen velocity ↔ neural activity")
log("-" * 50)

tmpl     = mat_load_simple(TMPL_PATH)
tmpl_letters = [k for k in tmpl if not k.startswith("_") and k != "dataDescription"]
log(f"Velocity templates available: {sorted(tmpl_letters)}")


def resample_template(vel: np.ndarray, n_out: int) -> np.ndarray:
    f = interp1d(np.linspace(0, 1, vel.shape[0]), vel, axis=0)
    return f(np.linspace(0, 1, n_out))


# Plot a few example templates
fig, axes = plt.subplots(2, 3, figsize=(13, 5))
fig.suptitle("Pen-tip velocity templates (mouse drawing, 100 Hz)", fontsize=11)
for ax, l in zip(axes.flat, ["a", "b", "e", "m", "t", "z"]):
    if l not in tmpl:
        ax.set_visible(False)
        continue
    vel = np.array(tmpl[l])
    t   = np.arange(vel.shape[0]) * 10
    ax.plot(t, vel[:, 0], "b-", linewidth=1.4, label="vx")
    ax.plot(t, vel[:, 1], "r-", linewidth=1.4, label="vy")
    ax.set(title=f"Letter '{l}'", xlabel="ms", ylabel="velocity (a.u.)")
    ax.legend(fontsize=7)
plt.tight_layout()
savefig("sep_fig6_velocity_templates.png")

# Ridge: neural -> velocity
r2_per_letter: dict[str, float] = {}
vel_decoded: dict[str, tuple]   = {}
t_ax = np.arange(n_time) * 10

for letter in LETTERS:
    if letter not in avg_raw or letter not in tmpl:
        continue
    vel    = resample_template(np.array(tmpl[letter]), n_time)  # (201, 2)
    neural = avg_raw[letter]                                     # (201, 192)
    reg    = Ridge(alpha=1.0).fit(neural, vel)
    vel_p  = reg.predict(neural)
    r2_per_letter[letter] = r2_score(vel, vel_p, multioutput="variance_weighted")
    vel_decoded[letter]   = (vel, vel_p)

valid_letters = [l for l in LETTERS if l in r2_per_letter]
all_neural    = np.vstack([avg_raw[l] for l in valid_letters])
all_vel       = np.vstack([resample_template(np.array(tmpl[l]), n_time) for l in valid_letters])
total_r2      = r2_score(
    all_vel,
    Ridge(alpha=1.0).fit(all_neural, all_vel).predict(all_neural),
    multioutput="variance_weighted",
)
mean_r2 = float(np.mean(list(r2_per_letter.values())))
log(f"Mean per-letter R²  : {mean_r2:.3f}")
log(f"Total neural var. explained by pen velocity: {total_r2*100:.1f}%  (paper ~30%)")

letters_s = sorted(r2_per_letter)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Pen velocity ↔ neural activity", fontsize=12)

axes[0].bar(letters_s, [r2_per_letter[l] for l in letters_s],
            color="steelblue", alpha=0.8)
axes[0].axhline(mean_r2, color="red", linestyle="--",
                label=f"mean R²={mean_r2:.2f}")
axes[0].set(xlabel="Letter", ylabel="R²",
            title="Per-letter R² (neural → pen velocity)", ylim=(0, 1))
axes[0].legend()

for ax, letter in zip(axes[1:], ["a", "t"]):
    vel_t, vel_p = vel_decoded[letter]
    for vt, vp, col, lbl in zip(vel_t.T, vel_p.T, ["b", "r"], ["vx", "vy"]):
        ax.plot(t_ax, vt, f"{col}-",  linewidth=1.4, label=f"true {lbl}")
        ax.plot(t_ax, vp, f"{col}--", linewidth=1.4, alpha=0.8, label=f"decoded {lbl}")
    ax.axvline(GO_BIN * 10, color="k", linestyle=":", alpha=0.6)
    ax.axvspan(MOVE_WIN[0] * 10, MOVE_WIN[1] * 10, alpha=0.08, color="green")
    ax.set(xlabel="Time (ms)", ylabel="velocity (a.u.)",
           title=f"Decoded pen velocity — '{letter}'")
    ax.legend(fontsize=8)
plt.tight_layout()
savefig("sep_fig7_pen_velocity.png")

pd.DataFrame([
    {"letter": l, "r2": r2_per_letter[l]} for l in letters_s
]).to_csv(OUT_DIR / "velocity_r2_per_letter.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Load all sessions
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 7 — Load all sessions")
log("-" * 50)

X_all, y_all = [], []
sess_counts  = {}
for sess in ALL_SESSIONS:
    cubes_s, _ = load_and_preprocess(sess)
    n_sess     = 0
    for letter in LETTERS:
        if letter not in cubes_s:
            continue
        feats = movement_features(cubes_s[letter], flatten=True)  # (trials, 19200)
        X_all.append(feats)
        y_all.extend([letter] * feats.shape[0])
        n_sess += feats.shape[0]
    sess_counts[sess] = n_sess
    log(f"  {sess}: {n_sess} trials")

X_pool = np.vstack(X_all)
y_pool = np.array(y_all)
log(f"\nPooled: {len(y_pool)} trials × {X_pool.shape[1]} features")
log(f"Trials per letter: {dict(zip(*np.unique(y_pool, return_counts=True)))}")

pd.DataFrame(
    list(sess_counts.items()), columns=["session", "n_trials"]
).to_csv(OUT_DIR / "session_trial_counts.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 8. t-SNE
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 8 — t-SNE (PCA 30-D → t-SNE 2-D)")
log("-" * 50)

log("PCA 19200 → 30 …")
X_pca30 = PCA(n_components=30, random_state=42).fit_transform(X_pool)

log("Running t-SNE (may take ~1 min) …")
_tsne_kw = dict(n_components=2, perplexity=30, random_state=42,
                learning_rate="auto", init="pca")
try:
    tsne = TSNE(**_tsne_kw, max_iter=1000)
except TypeError:
    tsne = TSNE(**_tsne_kw, n_iter=1000)
Z = tsne.fit_transform(X_pca30)
log(f"t-SNE done.  KL divergence: {tsne.kl_divergence_:.4f}")

colors_26    = cm.tab20(np.linspace(0, 1, 26))
letter_to_i  = {l: i for i, l in enumerate(LETTERS)}
fig, axes    = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"t-SNE — single-trial neural activity ({len(ALL_SESSIONS)} sessions, "
             f"{len(y_pool)} trials)", fontsize=12)

for l in LETTERS:
    mask = y_pool == l
    axes[0].scatter(Z[mask, 0], Z[mask, 1],
                    color=colors_26[letter_to_i[l]], label=l, s=12, alpha=0.65)
axes[0].set(title="All 26 letters", xlabel="t-SNE dim 1", ylabel="t-SNE dim 2")
axes[0].legend(fontsize=6.5, ncol=5, loc="best", markerscale=1.8)

vowels = list("aeiou")
for i, l in enumerate(vowels):
    mask = y_pool == l
    axes[1].scatter(Z[mask, 0], Z[mask, 1],
                    color=cm.tab10(i / 10), label=l, s=25, alpha=0.8)
axes[1].set(title="Vowels only (a/e/i/o/u)", xlabel="t-SNE dim 1", ylabel="t-SNE dim 2")
axes[1].legend(fontsize=9, markerscale=1.8)
plt.tight_layout()
savefig("sep_fig8_tsne.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. k-NN (k=1, LOO)
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 9 — k-NN (k=1, leave-one-out)")
log("-" * 50)

log("PCA 19200 → 100 …")
X_pca100 = PCA(n_components=100, random_state=42).fit_transform(X_pool)

log(f"Running k-NN LOO ({len(y_pool)} trials) …")
knn     = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
correct = 0
for train_i, test_i in LeaveOneOut().split(X_pca100):
    knn.fit(X_pca100[train_i], y_pool[train_i])
    correct += int(knn.predict(X_pca100[test_i])[0] == y_pool[test_i[0]])

accuracy = correct / len(y_pool)
log(f"k-NN LOO accuracy: {accuracy*100:.1f}%  (paper: 94.1%)")

# Per-letter LOO accuracy
per_letter_acc: dict[str, float] = {}
for letter in LETTERS:
    mask       = y_pool == letter
    letter_idx = np.where(mask)[0]
    other_idx  = np.where(~mask)[0]
    correct_l  = 0
    for ti in letter_idx:
        train_idx = np.concatenate([other_idx, letter_idx[letter_idx != ti]])
        knn.fit(X_pca100[train_idx], y_pool[train_idx])
        correct_l += int(knn.predict(X_pca100[[ti]])[0] == letter)
    per_letter_acc[letter] = correct_l / len(letter_idx)

sorted_letters = sorted(per_letter_acc, key=per_letter_acc.get)
sorted_accs    = [per_letter_acc[l] for l in sorted_letters]

log("Lowest-accuracy letters:")
for l in sorted_letters[:5]:
    log(f"  '{l}': {per_letter_acc[l]*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle(f"k-NN accuracy (k=1 LOO, all sessions) — overall {accuracy*100:.1f}%", fontsize=12)

axes[0].bar(list(per_letter_acc),
            [per_letter_acc[l] for l in per_letter_acc],
            color=[cm.RdYlGn(per_letter_acc[l]) for l in per_letter_acc])
axes[0].axhline(accuracy, color="red", linestyle="--", linewidth=1.5,
                label=f"overall LOO = {accuracy*100:.1f}%")
axes[0].axhline(1 / 26, color="gray", linestyle=":", linewidth=1,
                label=f"chance = {100/26:.1f}%")
axes[0].set(xlabel="Letter", ylabel="accuracy",
            title="Per-letter accuracy", ylim=(0, 1.05))
axes[0].legend(fontsize=8)

axes[1].barh(sorted_letters, sorted_accs,
             color=[cm.RdYlGn(a) for a in sorted_accs])
axes[1].axvline(accuracy, color="red", linestyle="--", linewidth=1.5)
axes[1].axvline(1 / 26, color="gray", linestyle=":", linewidth=1)
axes[1].set(xlabel="accuracy", title="Sorted by accuracy (low → high)", xlim=(0, 1.05))
plt.tight_layout()
savefig("sep_fig9_knn_accuracy.png")

pd.DataFrame([
    {"letter": l, "accuracy": per_letter_acc[l]} for l in sorted(per_letter_acc)
]).to_csv(OUT_DIR / "knn_per_letter_accuracy.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Summary figure
# ══════════════════════════════════════════════════════════════════════════════
log("")
log("SECTION 10 — Summary figure")
log("-" * 50)

fig = plt.figure(figsize=(16, 10))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Neural separability analysis — Willett et al. 2021", fontsize=14)

# Panel 1 — PCA trajectory (warped PC1 vs PC2)
ax1 = fig.add_subplot(gs[0, 0])
for i, l in enumerate(SHOW):
    if l not in proj_warp:
        continue
    traj = proj_warp[l]
    ax1.plot(traj[:, 0], traj[:, 1], color=colors_show[i], label=l,
             linewidth=2, alpha=0.85)
    ax1.scatter(traj[GO_BIN, 0], traj[GO_BIN, 1],
                color=colors_show[i], s=40, marker="*", zorder=5)
v = pca_warp.explained_variance_ratio_
ax1.set(xlabel=f"PC1 ({v[0]*100:.1f}%)", ylabel=f"PC2 ({v[1]*100:.1f}%)",
        title="PCA trajectory (time-aligned)")
ax1.legend(fontsize=8)

# Panel 2 — t-SNE
ax2 = fig.add_subplot(gs[0, 1])
for l in LETTERS:
    mask = y_pool == l
    ax2.scatter(Z[mask, 0], Z[mask, 1],
                color=colors_26[letter_to_i[l]], label=l, s=8, alpha=0.6)
ax2.set(title="t-SNE (all sessions)", xlabel="dim 1", ylabel="dim 2")

# Panel 3 — k-NN bars
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(list(per_letter_acc),
        [per_letter_acc[l] for l in per_letter_acc],
        color=[cm.RdYlGn(per_letter_acc[l]) for l in per_letter_acc])
ax3.axhline(accuracy, color="red", linestyle="--",
            label=f"{accuracy*100:.1f}%")
ax3.set(title="k-NN accuracy", ylim=(0, 1.05))
ax3.legend(fontsize=9)

# Panel 4 — pen velocity R²
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(letters_s, [r2_per_letter[l] for l in letters_s],
        color="darkorange", alpha=0.8)
ax4.axhline(mean_r2, color="red", linestyle="--", label=f"mean={mean_r2:.2f}")
ax4.set(title="Pen velocity R² per letter", ylim=(0, 1))
ax4.legend(fontsize=8)

# Panel 5 — decoded velocity for 'a'
ax5 = fig.add_subplot(gs[1, 1])
if "a" in vel_decoded:
    for vt, vp, col, lbl in zip(vel_decoded["a"][0].T, vel_decoded["a"][1].T,
                                  ["b", "r"], ["vx", "vy"]):
        ax5.plot(t_ax, vt, f"{col}-",  linewidth=1.4, label=f"true {lbl}")
        ax5.plot(t_ax, vp, f"{col}--", linewidth=1.4, alpha=0.8, label=f"decoded {lbl}")
    ax5.axvline(GO_BIN * 10, color="k", linestyle=":", alpha=0.5)
    ax5.set(xlabel="Time (ms)", title="Decoded pen velocity — 'a'")
    ax5.legend(fontsize=7)

# Panel 6 — text summary
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
pca_w_str = "\n".join(
    f"  PC{i+1}: {v*100:.1f}%"
    for i, v in enumerate(pca_warp.explained_variance_ratio_)
)
summary_text = (
    f"Data summary\n"
    f"{'-'*28}\n"
    f"Sessions  : {len(ALL_SESSIONS)}\n"
    f"Trials    : {len(y_pool)}\n"
    f"Channels  : {n_chan}\n"
    f"Time bins : {n_time} × 10 ms\n"
    f"Go cue    : bin {GO_BIN} (={GO_BIN*10} ms)\n\n"
    f"Results\n"
    f"{'-'*28}\n"
    f"k-NN LOO  : {accuracy*100:.1f}%\n"
    f"           (paper: 94.1%)\n\n"
    f"Pen vel R²: {total_r2*100:.1f}% of total var.\n"
    f"           (paper: ~30%)\n\n"
    f"PCA (warped)\n"
    f"{pca_w_str}"
)
ax6.text(0.04, 0.97, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="#fffde7", alpha=0.9))

for _ext in ("png", "svg"):
    plt.savefig(OUT_DIR / f"sep_fig0_summary.{_ext}", dpi=150, bbox_inches="tight")
plt.close()
log("  [fig] sep_fig0_summary.png / .svg")

# ── final log ────────────────────────────────────────────────────────────────
log("")
log("=" * 70)
log(f"  k-NN LOO accuracy          : {accuracy*100:.1f}%  (paper: 94.1%)")
log(f"  Neural var. explained (vel): {total_r2*100:.1f}%  (paper: ~30%)")
log(f"  PCA explained var. (PC1-3) : " +
    " / ".join(f"{v*100:.1f}%" for v in pca_warp.explained_variance_ratio_))
log("=" * 70)
log(f"All outputs in: {OUT_DIR}")

save_log()
