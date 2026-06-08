"""
qc_logs.py
==========
Analysis, reporting, and visualisation for TPCQ-Net experiments.

Provides:
  - Parameter and circuit count analysis for both variants
  - Training history plots
  - Side-by-side comparison between baseline and commuting
  - Shot efficiency visualisation

Public API
----------
  compute_analysis(cfg, runner_ctx)
  print_analysis(a, tag)
  plot_training_comparison(df_baseline, df_commuting)
  plot_shot_efficiency(counts_baseline, counts_commuting)
  plot_feature_contributions(df)
  summary_comparison_table(results_baseline, results_commuting)
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── dark theme ────────────────────────────────────────────────────────────────
C_BLUE   = '#5B8DEF'
C_TEAL   = '#3EC9A7'
C_PINK   = '#E8668A'
C_AMBER  = '#F5A623'
C_PURPLE = '#A78BFA'
C_GREEN  = '#4ADE80'

plt.rcParams.update({
    'figure.facecolor':  '#0F1117',
    'axes.facecolor':    '#1A1D27',
    'axes.edgecolor':    '#2E3147',
    'axes.labelcolor':   '#C8CAD4',
    'axes.titlecolor':   '#EAECF4',
    'axes.grid':         True,
    'axes.titlesize':    12,
    'axes.labelsize':    10,
    'grid.color':        '#252838',
    'grid.linewidth':    0.6,
    'xtick.color':       '#9799A6',
    'ytick.color':       '#9799A6',
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'text.color':        '#C8CAD4',
    'legend.facecolor':  '#1E2130',
    'legend.edgecolor':  '#3A3D4D',
    'legend.fontsize':   9,
    'figure.dpi':        120,
})


# ============================================================
#  ANALYSIS
# ============================================================

def compute_analysis(cfg, runner_ctx) -> dict:
    """Compute full parameter and circuit count analysis."""
    q   = cfg.N_QUBITS
    nl  = cfg.N_Q_LAYERS
    ng  = getattr(cfg, 'N_Q_LAYERS_GLOBAL', None)
    mw  = getattr(cfg, 'MAX_WEIGHT', 2)

    patch_pos        = (cfg.Q_RES - 1) // cfg.Q_STRIDE
    patches_per_sample = patch_pos * patch_pos
    q_out            = cfg.q_out()
    n_in             = cfg.n_inputs()

    # Parameter counts
    local_w  = nl * q
    global_w_gate    = runner_ctx.n_gate_params_g
    global_w_logical = runner_ctx.n_logical_params_g

    # Classical backbone
    def c2d(ic, oc, k): return oc*ic*k*k + oc
    def lin(i, o):       return i*o + o
    c1   = c2d(1,  16, 3)
    c2   = c2d(16, 32, 3)
    c3   = c2d(32, 64, 3)
    rd1  = c2d(32, 16, 1)
    rd2  = c2d(16,  q, 1)
    fc_in = 64 + q_out + q_out
    f1   = lin(fc_in, 64)
    f2   = lin(64, 10)
    classical_total = c1+c2+c3+rd1+rd2+f1+f2
    quantum_total   = local_w + global_w_gate
    model_total     = classical_total + quantum_total

    # Circuit counts per epoch
    n_train    = cfg.TRAIN_SAMPLES
    n_val      = cfg.VAL_SAMPLES
    n_obs      = cfg.N_MEASURE_BASES
    n_batches  = math.ceil(n_train / cfg.BATCH_SIZE)

    local_fwd_epoch  = patches_per_sample * n_train * n_obs
    global_fwd_epoch = n_train * n_obs
    local_val_epoch  = patches_per_sample * n_val * n_obs
    global_val_epoch = n_val * n_obs

    if runner_ctx.variant == 'baseline':
        local_bwd_epoch  = 2 * local_w   * n_obs * patches_per_sample * n_train
        global_bwd_epoch = 2 * global_w_gate * n_obs * n_train
        bwd_label = f'param-shift: 2×{global_w_gate} circuits/sample'
    else:
        local_bwd_epoch  = 2 * local_w * n_obs * patches_per_sample * n_train
        n_xy_obs         = max(0, n_obs - 1)
        global_bwd_epoch = (q * n_train +
                            2 * global_w_logical * n_xy_obs * n_train)
        bwd_label = f'parallel: {q} circuits/sample (+ {2*global_w_logical*n_xy_obs}/sample for X,Y)'

    fwd_total = (local_fwd_epoch + global_fwd_epoch +
                 local_val_epoch + global_val_epoch) * cfg.EPOCHS
    bwd_total = (local_bwd_epoch + global_bwd_epoch) * cfg.EPOCHS
    grand     = fwd_total + bwd_total

    return dict(
        variant=runner_ctx.variant,
        n_qubits=q, n_layers_local=nl, n_inputs=n_in,
        q_out=q_out, patches_per_sample=patches_per_sample,
        local_w=local_w, global_w_gate=global_w_gate,
        global_w_logical=global_w_logical,
        classical_total=classical_total, quantum_total=quantum_total,
        model_total=model_total,
        local_fwd_epoch=local_fwd_epoch, global_fwd_epoch=global_fwd_epoch,
        local_bwd_epoch=local_bwd_epoch, global_bwd_epoch=global_bwd_epoch,
        fwd_total=fwd_total, bwd_total=bwd_total, grand_total=grand,
        bwd_fwd_ratio=bwd_total/fwd_total if fwd_total > 0 else 0,
        bwd_label=bwd_label,
    )


def print_analysis(a: dict, tag: str = ''):
    label = tag or a['variant']
    W = 70
    print('═'*W)
    print(f"  TPCQ-Net Analysis — {label}".center(W))
    print('═'*W)
    print(f"  {'Variant':<35} {a['variant']}")
    print(f"  {'Qubits':<35} {a['n_qubits']}")
    print(f"  {'Local weight params':<35} {a['local_w']}")
    print(f"  {'Global gate params':<35} {a['global_w_gate']}")
    print(f"  {'Global logical params':<35} {a['global_w_logical']}")
    print(f"  {'Classical total params':<35} {a['classical_total']:,}")
    print(f"  {'Quantum total params':<35} {a['quantum_total']:,}")
    print(f"  {'Model total params':<35} {a['model_total']:,}")
    print('─'*W)
    print(f"  {'Backward method (global)':<35} {a['bwd_label']}")
    print(f"  {'Total forward circuits':<35} {a['fwd_total']:,}")
    print(f"  {'Total backward circuits':<35} {a['bwd_total']:,}")
    print(f"  {'Grand total circuits':<35} {a['grand_total']:,}")
    print(f"  {'Backward / forward ratio':<35} {a['bwd_fwd_ratio']:.1f}x")
    print('═'*W)


# ============================================================
#  PLOTS
# ============================================================

def _glow(ax, bar, col, ws=1.3, alpha=0.13, horizontal=False):
    if horizontal:
        ax.barh(bar.get_y()+bar.get_height()/2, bar.get_width(),
                height=bar.get_height()*ws, color=col, alpha=alpha, zorder=2)
    else:
        ax.bar(bar.get_x()+bar.get_width()/2, bar.get_height(),
               width=bar.get_width()*ws, color=col, alpha=alpha, zorder=2)


def plot_training_comparison(df_baseline: pd.DataFrame, df_commuting: pd.DataFrame,
                             save_path: str = None):
    """Side-by-side training curves: loss, accuracy, top-5."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Training Comparison: Baseline vs Commuting-Generator',
                 fontsize=13, fontweight='bold', color='#EAECF4', y=1.02)

    pairs = [
        ('loss',     'Loss',        'lower'),
        ('acc',      'Accuracy',    'upper'),
        ('top5',     'Top-5 Acc',   'upper'),
    ]
    for ax, (col, title, loc) in zip(axes, pairs):
        ax.plot(df_baseline['epoch'], df_baseline[col],
                color=C_PINK, linewidth=2, marker='o', markersize=3,
                label='Baseline (param-shift)')
        ax.plot(df_commuting['epoch'], df_commuting[col],
                color=C_TEAL, linewidth=2, marker='s', markersize=3,
                label='Commuting (parallel)')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(loc=loc)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
    return fig


def plot_shot_efficiency(counts_baseline: dict, counts_commuting: dict,
                         save_path: str = None):
    """Bar chart comparing total circuit executions."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Circuit Execution Count Comparison',
                 fontsize=13, fontweight='bold', color='#EAECF4', y=1.02)

    # --- Left: total breakdown per variant ---
    ax = axes[0]
    cats   = ['Forward', 'Backward (global)', 'Backward (local)']
    base_v = [counts_baseline['fwd_total'],
               counts_baseline['global_bwd'],
               counts_baseline['local_bwd']]
    comm_v = [counts_commuting['fwd_total'],
               counts_commuting['global_bwd'],
               counts_commuting['local_bwd']]
    x = np.arange(len(cats))
    w = 0.35
    bars_b = ax.bar(x - w/2, base_v, w, color=C_PINK,   label='Baseline',   edgecolor='none', zorder=3)
    bars_c = ax.bar(x + w/2, comm_v, w, color=C_TEAL,   label='Commuting',  edgecolor='none', zorder=3)
    for bar, col in [(b, C_PINK) for b in bars_b] + [(b, C_TEAL) for b in bars_c]:
        _glow(ax, bar, col)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel('Circuit executions'); ax.set_yscale('log')
    ax.set_title('Breakdown (log scale)', fontweight='bold')
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x', length=0)

    # --- Right: grand total comparison ---
    ax = axes[1]
    totals = [counts_baseline['grand_total'], counts_commuting['grand_total']]
    labels = ['Baseline', 'Commuting']
    colors = [C_PINK, C_TEAL]
    bars = ax.bar(labels, totals, color=colors, edgecolor='none', width=0.38, zorder=3)
    for bar, col, val in zip(bars, colors, totals):
        _glow(ax, bar, col)
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(totals)*0.02,
                f'{val:,}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#EAECF4')
    speedup = counts_baseline['grand_total'] / max(counts_commuting['grand_total'], 1)
    ax.set_title(f'Grand Total  ({speedup:.1f}× fewer circuits for commuting)',
                 fontweight='bold')
    ax.set_ylabel('Circuit executions')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x', length=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
    return fig


def plot_feature_contributions(df_baseline: pd.DataFrame,
                                df_commuting: pd.DataFrame,
                                save_path: str = None):
    """Stacked area chart of feature contributions per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    titles = ['Baseline', 'Commuting']
    for ax, df, title in zip(axes, [df_baseline, df_commuting], titles):
        ax.stackplot(df['epoch'],
                     df['feat_classical'], df['feat_local'], df['feat_global'],
                     labels=['Classical', 'Local Q', 'Global Q'],
                     colors=[C_BLUE, C_PINK, C_TEAL], alpha=0.8)
        ax.set_title(f'Feature Contributions — {title}', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Relative contribution')
        ax.legend(loc='upper right'); ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
    return fig


def plot_epoch_times(df_baseline: pd.DataFrame, df_commuting: pd.DataFrame,
                     save_path: str = None):
    """Epoch wall-clock times comparison."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_baseline['epoch'], df_baseline['epoch_time_s'],
            color=C_PINK, linewidth=2, marker='o', markersize=4, label='Baseline')
    ax.plot(df_commuting['epoch'], df_commuting['epoch_time_s'],
            color=C_TEAL, linewidth=2, marker='s', markersize=4, label='Commuting')
    ax.set_title('Epoch wall-clock time', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Seconds')
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
    return fig


def summary_comparison_table(results_baseline: dict, results_commuting: dict):
    """Print side-by-side summary of both runs."""
    cb = results_baseline['circuit_counts']
    cc = results_commuting['circuit_counts']

    speedup_global = cb['global_bwd'] / max(cc['global_bwd'], 1)
    speedup_total  = cb['grand_total'] / max(cc['grand_total'], 1)

    W = 75
    print('═'*W)
    print('  SUMMARY COMPARISON'.center(W))
    print('═'*W)
    fmt = '  {:<35} {:>16} {:>16}'
    print(fmt.format('Metric', 'Baseline', 'Commuting'))
    print('─'*W)
    print(fmt.format('Best val accuracy',
                     f"{results_baseline['best_acc']:.4f}",
                     f"{results_commuting['best_acc']:.4f}"))
    print(fmt.format('Global bwd circuits (total)',
                     f"{cb['global_bwd']:,}",
                     f"{cc['global_bwd']:,}"))
    print(fmt.format('Grand total circuits',
                     f"{cb['grand_total']:,}",
                     f"{cc['grand_total']:,}"))
    print(fmt.format('Global bwd speedup',
                     '1.0×',
                     f"{speedup_global:.1f}×"))
    print(fmt.format('Total circuit speedup',
                     '1.0×',
                     f"{speedup_total:.1f}×"))
    print('═'*W)

    try:
        df_b = pd.read_csv(results_baseline['csv_path'])
        df_c = pd.read_csv(results_commuting['csv_path'])
        avg_ep_b = df_b['epoch_time_s'].mean()
        avg_ep_c = df_c['epoch_time_s'].mean()
        print(fmt.format('Avg epoch time (s)',
                         f"{avg_ep_b:.1f}",
                         f"{avg_ep_c:.1f}"))
        print('─'*W)
    except Exception:
        pass
    print()
