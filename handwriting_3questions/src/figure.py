import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import os
from pathlib import Path

# ─── Real project parameters ──────────────────────────────────────────────────
# Dataset: 3,654 trials, 32 classes, 10 sessions
# block_aware:   train=1875 (51.3%), val=192 (5.3%), test=1587 (43.4%)
# random_trial:  train=1826 (50%),   val=914 (25%),  test=914  (25%)
#
# Most sessions have 2 blocks (1 train, 0 val, 1 test).
# Session t5.2019.05.08 has 9 blocks (5 train, 2 val, 2 test) — used as illustration.
#
# Preprocessed trial shape: (100 bins × 192 channels)
# Flat features:     100 × 192 = 19,200 dims
# Temporal features: 3 windows × 192 = 576 dims
#   early  = bins [0,  33)
#   middle = bins [33, 66)
#   late   = bins [66, 100)
# ──────────────────────────────────────────────────────────────────────────────

SAVE_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def generate_split_diagram():
    """
    Slide 9b: Block-aware vs Random-trial split diagram.

    Block-aware shows two rows:
      Row 1 (top):    t5.2019.05.08 — 9 blocks: 5 TRAIN / 2 VAL / 2 TEST
      Row 2 (bottom): typical session — 2 blocks: 1 TRAIN / 1 TEST
    Random-trial: 50% train / 25% val / 25% test.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('white')

    colors     = {'TRAIN': '#4472C4', 'VAL': '#ED7D31', 'TEST': '#C00000'}
    edge_color = '#FFFFFF'

    # ── Subplot 1: BLOCK-AWARE ────────────────────────────────────────────
    ax = axs[0]
    ax.set_title('BLOCK-AWARE\n(Chronological Block Hold-out)',
                 fontsize=15, pad=20, weight='bold')
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ── Row 1: 9-block session (t5.2019.05.08) ──
    ax.text(0.03, 0.75, 'Session 1  (9 blocks)',
            color='#444444', ha='left', va='center',
            transform=ax.transAxes, fontsize=9, style='italic')

    blocks9 = [f'B{i+1}' for i in range(9)]
    types9  = ['TRAIN'] * 5 + ['VAL'] * 2 + ['TEST'] * 2
    bw9     = 0.9 / 9
    x_pos   = 0.05
    for block, t in zip(blocks9, types9):
        ax.add_patch(patches.Rectangle(
            (x_pos, 0.64), bw9 - 0.006, 0.10,
            linewidth=1, edgecolor=edge_color, facecolor=colors[t]
        ))
        ax.text(x_pos + bw9 / 2, 0.69, block,
                color='white', ha='center', va='center', fontsize=8, weight='bold')
        x_pos += bw9

    # ── Row 2: typical 2-block session ──
    ax.text(0.03, 0.59, 'Sessions 2–10  (2 blocks each)',
            color='#444444', ha='left', va='center',
            transform=ax.transAxes, fontsize=9, style='italic')

    types2 = ['TRAIN', 'TEST']
    bw2    = 0.9 / 2
    x_pos  = 0.05
    for t in types2:
        ax.add_patch(patches.Rectangle(
            (x_pos, 0.50), bw2 - 0.006, 0.08,
            linewidth=1, edgecolor=edge_color, facecolor=colors[t]
        ))
        ax.text(x_pos + bw2 / 2, 0.54, t,
                color='white', ha='center', va='center', fontsize=9, weight='bold')
        x_pos += bw2

    # No-val note sits directly under row 2
    ax.text(0.5, 0.48,
            '(no val block — only 2 blocks per session)',
            color='#888888', ha='center', va='top',
            transform=ax.transAxes, fontsize=8, style='italic')

    # Time arrow
    ax.annotate('', xy=(0.95, 0.42), xytext=(0.05, 0.42),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#444444', lw=1.8))
    ax.text(0.5, 0.40, 'Time  →  (neural drift increases)',
            color='#444444', ha='center', va='top',
            transform=ax.transAxes, fontsize=10, style='italic')

    # Key message
    ax.text(0.05, 0.32,
            'Test = later blocks\n'
            'Distribution shifts over time\n'
            '→  HARDER / more realistic',
            color='#C00000', ha='left', va='top',
            transform=ax.transAxes, fontsize=11, weight='bold')

    # Stats box — placed to the right of key message
    stats_text = ('train=1,875 (51.3%)\n'
                  'val =  192  ( 5.3%)\n'
                  'test=1,587  (43.4%)')
    ax.text(0.97, 0.32, stats_text, color='#333333',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F2F2F2', edgecolor='#AAAAAA'))

    # Legend
    for i, (label, c) in enumerate(
            zip(['TRAIN', 'VAL', 'TEST'], [colors['TRAIN'], colors['VAL'], colors['TEST']])):
        ax.add_patch(patches.Rectangle(
            (0.05 + i * 0.2, 0.08), 0.04, 0.04,
            facecolor=c, transform=ax.transAxes))
        ax.text(0.10 + i * 0.2, 0.10, label, color='black',
                ha='left', va='center', transform=ax.transAxes, fontsize=10)

    # ── Subplot 2: RANDOM-TRIAL ───────────────────────────────────────────
    ax = axs[1]
    ax.set_title('RANDOM-TRIAL\n(Stratified Random Split)',
                 fontsize=15, pad=20, weight='bold')
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Pool box
    pool_rect = patches.Rectangle(
        (0.05, 0.58), 0.9, 0.28,
        linewidth=2, edgecolor='#7F7F7F', facecolor='#F2F2F2', linestyle='--'
    )
    ax.add_patch(pool_rect)
    ax.text(0.5, 0.90, 'All 3,654 Trials Pooled (all sessions)',
            color='#444444', ha='center', va='center',
            transform=ax.transAxes, fontsize=10, style='italic')

    # Shuffled strip — 50% train / 25% val / 25% test
    np.random.seed(42)
    trial_pool = ['TRAIN'] * 50 + ['VAL'] * 25 + ['TEST'] * 25
    np.random.shuffle(trial_pool)
    n_show = 60
    strip_x_start = 0.08
    strip_width   = 0.84 / n_show
    x_pos = strip_x_start
    for t in trial_pool[:n_show]:
        ax.add_patch(patches.Rectangle(
            (x_pos, 0.65), strip_width - 0.001, 0.12,
            linewidth=0, facecolor=colors[t]
        ))
        x_pos += strip_width

    # Arrow
    ax.annotate('Stratified shuffle & split',
                xy=(0.5, 0.62), xytext=(0.5, 0.52),
                xycoords='axes fraction',
                ha='center', va='center', fontsize=11, weight='bold',
                arrowprops=dict(arrowstyle='->', color='#444444', lw=2))

    ax.text(0.05, 0.46,
            'Test ≈ Train distribution\n'
            'Trials from the same time periods\n'
            'mixed across splits\n'
            '→  EASIER / upper-bound benchmark',
            color='#4472C4', ha='left', va='top',
            transform=ax.transAxes, fontsize=11, weight='bold')

    # Stats box
    stats_text2 = ('train=1,826 (50%)\n'
                   'val  =  914 (25%)\n'
                   'test =  914 (25%)')
    ax.text(0.97, 0.46, stats_text2, color='#333333',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F2F2F2', edgecolor='#AAAAAA'))

    # Legend
    for i, (label, c) in enumerate(
            zip(['TRAIN (50%)', 'VAL (25%)', 'TEST (25%)'],
                [colors['TRAIN'], colors['VAL'], colors['TEST']])):
        ax.add_patch(patches.Rectangle(
            (0.05 + i * 0.3, 0.08), 0.04, 0.04,
            facecolor=c, transform=ax.transAxes))
        ax.text(0.10 + i * 0.3, 0.10, label, color='black',
                ha='left', va='center', transform=ax.transAxes, fontsize=10)

    plt.suptitle('Two Evaluation Strategies', fontsize=17, weight='bold', y=1.01)
    plt.tight_layout()
    for ext in ['png', 'svg']:
        out = SAVE_DIR / f'split_diagram.{ext}'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")


def generate_feature_diagram():
    """
    Slide 10b: Flat vs Temporal feature diagram.

    Orientation: channels on x-axis (0–191) for ALL three panels.
                 y-axis = time bins (raw/flat) or windows (temporal).
    sharex between flat and temporal so channel axis is visually aligned.

    Layout:
      Left   : Input trial  (time bins on y, channels on x)  — Greys cmap
      Middle : arrows
      Right  : flat (100 × 192, Blues) stacked above temporal (3 × 192, Oranges)
               — both share the same x-axis (channels 0–191)
    """
    np.random.seed(99)

    # Simulate plausible neural activity (100 bins × 192 channels)
    t_arr = np.linspace(0, 2 * np.pi, 100)
    base  = np.outer(t_arr, np.ones(192))
    noise = np.random.normal(0, 0.4, (100, 192))
    trial = np.clip(np.sin(base) * 0.5 + noise * 0.5 + 0.5, 0, 1)  # (100, 192)

    # Temporal windows — mean over each third of time axis
    win_early  = trial[0:33 ].mean(axis=0, keepdims=True)   # (1, 192)
    win_middle = trial[33:66].mean(axis=0, keepdims=True)
    win_late   = trial[66:100].mean(axis=0, keepdims=True)
    temporal   = np.vstack([win_early, win_middle, win_late])  # (3, 192)

    colors = {'FLAT': '#4472C4', 'TEMP': '#ED7D31'}

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1.0, 0.12, 2.2],
        wspace=0.28,
        left=0.06, right=0.97, top=0.88, bottom=0.14,
    )

    # ── Left: raw trial  (time bins on y, channels on x) ─────────────────
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw.set_title('Input Trial\n(100 bins × 192 ch)',
                     fontsize=12, weight='bold', pad=10)
    # trial shape (100, 192) → imshow rows=time bins, cols=channels
    ax_raw.imshow(trial, cmap='Greys', aspect='auto',
                  interpolation='nearest', vmin=0, vmax=1)
    # Horizontal window dividers
    for y, label, ypos in [(33, 'early\n[0,33)',  16),
                            (66, 'mid\n[33,66)',   49),
                            (None, 'late\n[66,100)', 82)]:
        if y is not None:
            ax_raw.axhline(y=y, color='#ED7D31', lw=1.5, linestyle='--', alpha=0.8)
        ax_raw.text(196, ypos, label, color='#ED7D31',
                    ha='left', va='center', fontsize=8, weight='bold',
                    clip_on=False)
    ax_raw.set_ylabel('Time bins  (0 – 99)', fontsize=9)
    ax_raw.set_xlabel('Channels  (0 – 191)', fontsize=9)
    ax_raw.set_yticks([0, 33, 66, 99]); ax_raw.set_yticklabels(['0','33','66','99'], fontsize=8)
    ax_raw.set_xticks([0, 95, 191]);    ax_raw.set_xticklabels(['0','95','191'], fontsize=8)
    ax_raw.text(0.5, -0.09, '1 second of motor cortex activity',
                ha='center', va='top', transform=ax_raw.transAxes,
                fontsize=8, color='#555555', style='italic')

    # ── Middle: arrows ────────────────────────────────────────────────────
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_mid.axis('off')
    ax_mid.set_xlim(0, 1); ax_mid.set_ylim(0, 1)
    ax_mid.annotate('flatten\n→',
                    xy=(0.95, 0.74), xytext=(0.05, 0.74),
                    xycoords='data', ha='center', va='center',
                    fontsize=9, weight='bold', color=colors['FLAT'],
                    arrowprops=dict(arrowstyle='->', lw=2.2, color=colors['FLAT']))
    ax_mid.annotate('window\nmeans\n→',
                    xy=(0.95, 0.26), xytext=(0.05, 0.26),
                    xycoords='data', ha='center', va='center',
                    fontsize=9, weight='bold', color=colors['TEMP'],
                    arrowprops=dict(arrowstyle='->', lw=2.2, color=colors['TEMP']))

    # ── Right: flat (top) + temporal (bottom), sharex ─────────────────────
    # height_ratios: flat has 100 bins, temporal has 3 → ratio 6:1 looks clean
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs[0, 2],
        height_ratios=[6, 1],
        hspace=0.60,
    )
    ax_flat = fig.add_subplot(gs_right[0])
    ax_temp = fig.add_subplot(gs_right[1], sharex=ax_flat)

    # ── Flat: (100, 192) — time bins on y, channels on x ──
    ax_flat.set_title('FLAT FEATURES  —  100 × 192  =  19,200 dims',
                      fontsize=11, weight='bold', color=colors['FLAT'], pad=8)
    ax_flat.imshow(trial, cmap='Blues', aspect='auto',
                   interpolation='nearest', vmin=0, vmax=1)
    ax_flat.set_ylabel('Time bins\n(0 – 99)', fontsize=9, color=colors['FLAT'])
    ax_flat.set_yticks([0, 49, 99]); ax_flat.set_yticklabels(['0','49','99'], fontsize=8)
    ax_flat.tick_params(labelbottom=False)
    ax_flat.text(0.5, -0.06,
                 '✓ Preserves exact timing of every neural event'
                 '      ✗ 19,200 dims — high risk of overfitting',
                 ha='center', va='top', transform=ax_flat.transAxes,
                 fontsize=9, color='#333333')

    # ── Temporal: (3, 192) — windows on y, channels on x ──
    ax_temp.set_title('TEMPORAL FEATURES  —  3 × 192  =  576 dims',
                      fontsize=11, weight='bold', color=colors['TEMP'], pad=8)
    ax_temp.imshow(temporal, cmap='Oranges', aspect='auto',
                   interpolation='nearest', vmin=0, vmax=1)
    ax_temp.set_ylabel('Window', fontsize=9, color=colors['TEMP'])
    ax_temp.set_yticks([0, 1, 2])
    ax_temp.set_yticklabels(['early\n[0,33)', 'mid\n[33,66)', 'late\n[66,100)'], fontsize=8)
    ax_temp.set_xlabel('Channels  (0 – 191)  ←  same axis as flat above', fontsize=9)
    ax_temp.set_xticks([0, 47, 95, 143, 191])
    ax_temp.set_xticklabels(['0', '47', '95', '143', '191'], fontsize=8)
    ax_temp.text(0.5, -0.50,
                 '✓ Compact (576 dims) — easier to regularize'
                 '      ✗ Within-window timing is averaged out',
                 ha='center', va='top', transform=ax_temp.transAxes,
                 fontsize=9, color='#333333')

    fig.suptitle(
        'Two Feature Representations from the Same Preprocessed Trial  (100 bins × 192 ch)',
        fontsize=13, weight='bold',
    )
    for ext in ['png', 'svg']:
        out = SAVE_DIR / f'feature_diagram.{ext}'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")


def generate_pca_diagram():
    """
    Slide 11b: How PCA-for-classification works.

    Shows the full pipeline:
      Flat features (19,200) → StandardScaler → PCA(k) → Classifier → Label
    with a sweep annotation showing k ∈ {20, 50, 100, 200} and the raw-flat
    baseline (no PCA) as a reference condition.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # ── colour palette ────────────────────────────────────────────────────
    C_INPUT  = '#D9D9D9'
    C_SCALER = '#9DC3E6'
    C_PCA    = '#4472C4'
    C_CLF    = '#70AD47'
    C_OUT    = '#ED7D31'
    C_BASE   = '#FF0000'
    C_ARROW  = '#444444'

    box_h  = 0.9
    box_y  = 1.55   # vertical centre of main pipeline boxes

    def box(ax, x, y, w, h, fc, ec='#333333', lw=1.5, ls='-', alpha=1.0):
        ax.add_patch(patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.08', linewidth=lw,
            edgecolor=ec, facecolor=fc, linestyle=ls, alpha=alpha,
            zorder=3,
        ))

    def arrow(ax, x0, x1, y, color=C_ARROW, lw=2.0):
        ax.annotate('', xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=lw, mutation_scale=16),
                    zorder=4)

    def label(ax, x, y, text, fontsize=10, color='#111111',
              weight='normal', ha='center', va='center', style='normal'):
        ax.text(x, y, text, fontsize=fontsize, color=color,
                weight=weight, ha=ha, va=va, style=style, zorder=5)

    # ── Stage 0: one trial (input) ────────────────────────────────────────
    box(ax, 0.9, box_y, 1.2, box_h, C_INPUT)
    label(ax, 0.9, box_y + 0.18, 'Trial', fontsize=10, weight='bold')
    label(ax, 0.9, box_y - 0.18, '(100 × 192)', fontsize=8, color='#555555')
    label(ax, 0.9, box_y - 0.70, 'per-trial\nneural data', fontsize=7.5,
          color='#777777', style='italic')

    # flatten arrow + label
    arrow(ax, 1.52, 2.08, box_y)
    label(ax, 1.80, box_y + 0.28, 'flatten', fontsize=8,
          color=C_ARROW, style='italic')

    # ── Stage 1: flat features ────────────────────────────────────────────
    box(ax, 2.55, box_y, 1.2, box_h, C_INPUT)
    label(ax, 2.55, box_y + 0.18, 'Flat features', fontsize=10, weight='bold')
    label(ax, 2.55, box_y - 0.18, '19,200 dims', fontsize=8, color='#555555')
    label(ax, 2.55, box_y - 0.70, '100 × 192', fontsize=7.5,
          color='#777777', style='italic')

    arrow(ax, 3.18, 3.62, box_y)

    # ── Stage 2: StandardScaler ───────────────────────────────────────────
    box(ax, 4.05, box_y, 1.2, box_h, C_SCALER)
    label(ax, 4.05, box_y + 0.18, 'StandardScaler', fontsize=9, weight='bold')
    label(ax, 4.05, box_y - 0.18, 'fit on TRAIN', fontsize=8, color='#1F4E79')
    label(ax, 4.05, box_y - 0.70, 'zero mean,\nunit variance', fontsize=7.5,
          color='#777777', style='italic')

    arrow(ax, 4.68, 5.12, box_y)

    # ── Stage 3: PCA(k) ───────────────────────────────────────────────────
    box(ax, 5.60, box_y, 1.35, box_h, C_PCA, ec='#1F3864', lw=2.0)
    label(ax, 5.60, box_y + 0.20, 'PCA ( k )', fontsize=11,
          weight='bold', color='white')
    label(ax, 5.60, box_y - 0.20, 'fit on TRAIN only', fontsize=8,
          color='#BDD7EE')
    label(ax, 5.60, box_y - 0.72,
          'k ∈ {20, 50, 100, 200}\n← sweep on x-axis of decoder curve',
          fontsize=7.5, color=C_PCA, style='italic')

    # no-leakage badge
    ax.add_patch(patches.FancyBboxPatch(
        (5.05, box_y + 0.50), 1.10, 0.32,
        boxstyle='round,pad=0.06', linewidth=1,
        edgecolor='#C00000', facecolor='#FFE7E7', zorder=5,
    ))
    label(ax, 5.60, box_y + 0.66, '🔒 no leakage', fontsize=8,
          color='#C00000', weight='bold')

    arrow(ax, 6.30, 6.72, box_y)

    # ── Stage 4: Classifier ───────────────────────────────────────────────
    box(ax, 7.22, box_y, 1.30, box_h, C_CLF, ec='#375623', lw=1.5)
    label(ax, 7.22, box_y + 0.20, 'Classifier', fontsize=10,
          weight='bold', color='white')
    label(ax, 7.22, box_y - 0.20, 'LogReg / SVC / Ridge\nSVC-RBF / MLP', fontsize=7.5,
          color='#E2EFDA')
    label(ax, 7.22, box_y - 0.72, 'tuned on val only', fontsize=7.5,
          color='#777777', style='italic')

    arrow(ax, 7.90, 8.32, box_y)

    # ── Stage 5: Output ───────────────────────────────────────────────────
    box(ax, 8.82, box_y, 1.20, box_h, C_OUT, ec='#843C0C', lw=1.5)
    label(ax, 8.82, box_y + 0.18, 'Predicted label', fontsize=9,
          weight='bold', color='white')
    label(ax, 8.82, box_y - 0.18, 'Accuracy\nMacro-F1', fontsize=8,
          color='#FFF2CC')
    label(ax, 8.82, box_y - 0.70, 'evaluated on\ntest set only', fontsize=7.5,
          color='#777777', style='italic')

    # ── Baseline condition (no PCA) ───────────────────────────────────────
    base_y = 0.60
    ax.annotate('', xy=(6.72, base_y), xytext=(3.18, base_y),
                arrowprops=dict(arrowstyle='->', color=C_BASE,
                                lw=1.8, linestyle='dashed', mutation_scale=14),
                zorder=4)
    label(ax, 4.95, base_y + 0.22,
          'Baseline: skip PCA → use flat 19,200 dims directly',
          fontsize=8.5, color=C_BASE, weight='bold')
    label(ax, 4.95, base_y - 0.22,
          '(shown as dashed horizontal lines in decoder curve)',
          fontsize=7.5, color=C_BASE, style='italic')

    # ── k-sweep annotation ────────────────────────────────────────────────
    sweep_y = 3.55
    ax.annotate('', xy=(6.35, sweep_y), xytext=(4.82, sweep_y),
                arrowprops=dict(arrowstyle='<->', color=C_PCA,
                                lw=1.5, mutation_scale=12), zorder=4)
    label(ax, 5.58, sweep_y + 0.18,
          'sweep k ∈ {20, 50, 100, 200}  →  x-axis of decoder curve',
          fontsize=8.5, color=C_PCA, weight='bold')

    # Stage labels (top)
    for x, txt in [(0.9, '① Input'), (2.55, '② Flatten'),
                   (4.05, '③ Scale'), (5.60, '④ PCA'),
                   (7.22, '⑤ Classify'), (8.82, '⑥ Evaluate')]:
        label(ax, x, 3.20, txt, fontsize=8, color='#555555', weight='bold')

    ax.set_title('Q3 Pipeline: No PCA vs PCA before Classification',
                 fontsize=14, weight='bold', pad=14)

    plt.tight_layout()
    for ext in ['png', 'svg']:
        out = SAVE_DIR / f'pca_pipeline_diagram.{ext}'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()


# Run all
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    generate_split_diagram()
    generate_feature_diagram()
    generate_pca_diagram()
    print("Done.")