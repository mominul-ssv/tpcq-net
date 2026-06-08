"""
qc_logs.py
==========
Analysis, parameter counting, report logging, and visualisation for the QC-CNN model.

No configuration is hardcoded here — all cfg objects come from the caller (notebook).
Style, palette, and all plot logic live here so the notebook stays clean.

Public API:
  compute_analysis(cfg)       ->  dict
  print_circuits(a)
  print_analysis(a, cfg)
  run_report(cfg)             ->  dict

  plot_gate_composition(a)
  plot_parameter_breakdown(a)
  plot_execution_counts(a)
  plot_fwd_bwd_ratio(a)
  summary_table(a)
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from qc_circuits import build_circuits


# ============================================================
#  STYLE & PALETTE  (applied on import)
# ============================================================

C_BLUE   = '#5B8DEF'
C_TEAL   = '#3EC9A7'
C_PINK   = '#E8668A'
C_AMBER  = '#F5A623'
C_PURPLE = '#A78BFA'

plt.rcParams.update({
    'figure.facecolor' : '#0F1117',
    'axes.facecolor'   : '#1A1D27',
    'axes.edgecolor'   : '#2E3147',
    'axes.labelcolor'  : '#C8CAD4',
    'axes.titlecolor'  : '#EAECF4',
    'axes.grid'        : True,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'grid.color'       : '#252838',
    'grid.linewidth'   : 0.6,
    'xtick.color'      : '#9799A6',
    'ytick.color'      : '#9799A6',
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'text.color'       : '#C8CAD4',
    'legend.facecolor' : '#1E2130',
    'legend.edgecolor' : '#3A3D4D',
    'legend.fontsize'  : 10,
    'figure.dpi'       : 130,
})


# ============================================================
#  CLASSICAL PARAM HELPERS
# ============================================================

def conv2d_params(in_ch: int, out_ch: int, kernel: int, bias: bool = True):
    """Return (weight_params, bias_params, total_params) for a Conv2d layer."""
    w = out_ch * in_ch * kernel * kernel
    b = out_ch if bias else 0
    return w, b, w + b


def linear_params(in_f: int, out_f: int, bias: bool = True):
    """Return (weight_params, bias_params, total_params) for a Linear layer."""
    w = in_f * out_f
    b = out_f if bias else 0
    return w, b, w + b


# ============================================================
#  ANALYSIS
# ============================================================

def compute_analysis(cfg) -> dict:
    """
    Full numerical analysis for a given cfg object.

    Builds both quantum circuits, counts all gates and parameters,
    computes the classical breakdown, patch geometry, and all circuit
    execution counts (train forward, val forward, parameter-shift gradients).

    Args:
        cfg : any object exposing the fields defined in CFG (notebook)

    Returns:
        dict containing every computed quantity plus the built circuits.
    """
    q       = cfg.N_QUBITS
    nl      = cfg.N_Q_LAYERS
    ng      = cfg.N_Q_LAYERS_GLOBAL
    n_in    = cfg.n_inputs()
    ipl     = cfg.inputs_per_layer()
    n_bases = cfg.N_MEASURE_BASES
    q_out   = q * n_bases

    # ── build circuits ────────────────────────────────────────
    c = build_circuits(cfg)
    qc_l, x_l, w_l = c['qc_local'],  c['x_local'],  c['w_local']
    qc_g, x_g, w_g = c['qc_global'], c['x_global'], c['w_global']

    # ── circuit intrinsics ────────────────────────────────────
    local_ops          = dict(qc_l.count_ops())
    global_ops         = dict(qc_g.count_ops())
    local_total_gates  = sum(local_ops.values())
    global_total_gates = sum(global_ops.values())
    local_depth        = qc_l.depth()
    global_depth       = qc_g.depth()

    # ── quantum parameter counts ──────────────────────────────
    local_w_params  = len(w_l)
    local_x_params  = len(x_l)
    global_w_params = len(w_g)
    global_x_params = len(x_g)

    # ── gate breakdown ────────────────────────────────────────
    local_ry_data    = nl * q
    local_ry_weight  = nl * q
    local_cnot       = nl * q
    global_ry_data   = n_in
    global_ry_weight = ng * q
    global_cnot      = ng * q

    # ── classical parameter breakdown ─────────────────────────
    c1_w,  c1_b,  c1_t  = conv2d_params(1,  16, 3)
    c2_w,  c2_b,  c2_t  = conv2d_params(16, 32, 3)
    c3_w,  c3_b,  c3_t  = conv2d_params(32, 64, 3)
    rd1_w, rd1_b, rd1_t = conv2d_params(32, 16, 1)
    rd2_w, rd2_b, rd2_t = conv2d_params(16,  q, 1)
    rd_t                 = rd1_t + rd2_t
    fc_in                = 64 + q_out + q_out
    f1_w,  f1_b,  f1_t  = linear_params(fc_in, 64)
    f2_w,  f2_b,  f2_t  = linear_params(64, 10)
    classical_total      = c1_t + c2_t + c3_t + rd_t + f1_t + f2_t
    quantum_total        = local_w_params + global_w_params
    model_total          = classical_total + quantum_total

    # ── patch geometry ────────────────────────────────────────
    patch_pos          = math.floor((cfg.Q_RES - 1) / cfg.Q_STRIDE)
    patches_per_sample = patch_pos * patch_pos

    # ── data loader ───────────────────────────────────────────
    train_batches = math.ceil(cfg.TRAIN_SAMPLES / cfg.BATCH_SIZE)
    val_batches   = math.ceil(cfg.VAL_SAMPLES   / cfg.BATCH_SIZE)

    # ── forward execution counts ──────────────────────────────
    l_fwd_batch    = patches_per_sample * cfg.BATCH_SIZE
    l_fwd_epoch    = l_fwd_batch * train_batches
    l_fwd_training = l_fwd_epoch * cfg.EPOCHS
    g_fwd_batch    = cfg.BATCH_SIZE
    g_fwd_epoch    = g_fwd_batch * train_batches
    g_fwd_training = g_fwd_epoch * cfg.EPOCHS
    fwd_total      = l_fwd_training + g_fwd_training

    # ── validation forward counts ─────────────────────────────
    l_val_epoch    = patches_per_sample * cfg.VAL_SAMPLES
    g_val_epoch    = cfg.VAL_SAMPLES
    l_val_training = l_val_epoch * cfg.EPOCHS
    g_val_training = g_val_epoch * cfg.EPOCHS
    val_total      = l_val_training + g_val_training

    # ── gradient counts (parameter-shift: 2 evals per weight) ─
    l_shift      = 2 * local_w_params
    g_shift      = 2 * global_w_params
    l_grad_epoch = l_shift * cfg.TRAIN_SAMPLES
    g_grad_epoch = g_shift * cfg.TRAIN_SAMPLES
    l_grad_train = l_grad_epoch * cfg.EPOCHS
    g_grad_train = g_grad_epoch * cfg.EPOCHS
    grad_total   = l_grad_train + g_grad_train

    l_total     = l_fwd_training + l_grad_train + l_val_training
    g_total     = g_fwd_training + g_grad_train + g_val_training
    grand_total = l_total + g_total

    return dict(
        qc_local=qc_l, qc_global=qc_g,
        n_qubits=q, n_layers_local=nl, n_layers_global=ng,
        n_inputs=n_in, ipl=ipl, n_bases=n_bases, q_out=q_out,
        local_ops=local_ops, local_total_gates=local_total_gates,
        local_depth=local_depth,
        local_w_params=local_w_params, local_x_params=local_x_params,
        local_ry_data=local_ry_data, local_ry_weight=local_ry_weight,
        local_cnot=local_cnot,
        global_ops=global_ops, global_total_gates=global_total_gates,
        global_depth=global_depth,
        global_w_params=global_w_params, global_x_params=global_x_params,
        global_ry_data=global_ry_data, global_ry_weight=global_ry_weight,
        global_cnot=global_cnot,
        c1_w=c1_w, c1_b=c1_b, c1_t=c1_t,
        c2_w=c2_w, c2_b=c2_b, c2_t=c2_t,
        c3_w=c3_w, c3_b=c3_b, c3_t=c3_t,
        rd1_w=rd1_w, rd1_b=rd1_b, rd1_t=rd1_t,
        rd2_w=rd2_w, rd2_b=rd2_b, rd2_t=rd2_t, rd_t=rd_t,
        fc_in=fc_in,
        f1_w=f1_w, f1_b=f1_b, f1_t=f1_t,
        f2_w=f2_w, f2_b=f2_b, f2_t=f2_t,
        classical_total=classical_total,
        quantum_total=quantum_total, model_total=model_total,
        patch_pos=patch_pos, patches_per_sample=patches_per_sample,
        train_batches=train_batches, val_batches=val_batches,
        l_fwd_batch=l_fwd_batch, l_fwd_epoch=l_fwd_epoch,
        l_fwd_training=l_fwd_training,
        g_fwd_batch=g_fwd_batch, g_fwd_epoch=g_fwd_epoch,
        g_fwd_training=g_fwd_training, fwd_total=fwd_total,
        l_val_epoch=l_val_epoch, g_val_epoch=g_val_epoch,
        l_val_training=l_val_training, g_val_training=g_val_training,
        val_total=val_total,
        l_shift=l_shift, g_shift=g_shift,
        l_grad_epoch=l_grad_epoch, g_grad_epoch=g_grad_epoch,
        l_grad_train=l_grad_train, g_grad_train=g_grad_train,
        grad_total=grad_total,
        l_total=l_total, g_total=g_total, grand_total=grand_total,
    )


# ============================================================
#  PRINTERS
# ============================================================

def print_circuits(a: dict) -> None:
    """Print both quantum circuits using Qiskit's text renderer."""
    print("\n" + "=" * 72)
    print(f"  LOCAL CIRCUIT  "
          f"({a['n_qubits']} qubits,  {a['n_layers_local']} layers,  "
          f"{a['local_x_params']} patch inputs)")
    print("=" * 72)
    print(a['qc_local'].draw(output='text', fold=100))

    print("\n" + "=" * 72)
    print(f"  GLOBAL CIRCUIT  "
          f"({a['n_qubits']} qubits,  {a['n_layers_global']} layers,  "
          f"{a['n_inputs']} inputs  /  {a['ipl']} per layer)")
    print("=" * 72)
    print(a['qc_global'].draw(output='text', fold=100))


def print_analysis(a: dict, cfg) -> None:
    """Print the full 8-section analysis report to stdout."""
    W = 140

    def top():   print("═" * W)
    def bot():   print("═" * W)
    def div():   print("═" * W)
    def sep():   print("─" * W)
    def blank(): print()

    def hdr(t):
        pad = W - len(t)
        print(" " * (pad // 2) + t)

    def row(label, value, indent=2):
        label = " " * indent + str(label)
        dots  = "." * max(1, W - len(label) - len(str(value)) - 2)
        print(f"{label}{dots}{value}")

    def calc(label, formula, result, indent=2):
        label = " " * indent + str(label)
        mid   = f"{formula}  =  {result}"
        dots  = "." * max(1, W - len(label) - len(mid) - 2)
        print(f"{label}{dots}{mid}")

    def note(text, indent=4):
        print(" " * indent + text)

    q  = a['n_qubits']
    nl = a['n_layers_local']
    ng = a['n_layers_global']

    top(); hdr("QC-CNN PARALLEL  —  CIRCUIT & PARAMETER ANALYSIS"); div()

    hdr("1.  CONFIGURATION"); sep()
    row("Qubits  (N_QUBITS)",            cfg.N_QUBITS)
    row("Local  circuit layers",          cfg.N_Q_LAYERS)
    row("Global circuit layers",          cfg.N_Q_LAYERS_GLOBAL)
    row("Feature map size  (Q_RES)",      f"{cfg.Q_RES} × {cfg.Q_RES}")
    row("Sliding stride    (Q_STRIDE)",   cfg.Q_STRIDE)
    row("n_inputs  (N_QUBITS × Q_RES²)",  f"{q} × {cfg.Q_RES}²  =  {a['n_inputs']}")
    row("Inputs per global layer",         f"ceil({a['n_inputs']} / {ng})  =  {a['ipl']}")
    row("Measurement bases",               f"{cfg.N_MEASURE_BASES}  (Z, X, Y)  →  {a['q_out']} outputs per branch")
    row("Train / Val samples",             f"{cfg.TRAIN_SAMPLES}  /  {cfg.VAL_SAMPLES}")
    row("Batch size  /  Epochs",           f"{cfg.BATCH_SIZE}  /  {cfg.EPOCHS}")
    row("Backend",                         "StatevectorEstimator  (shots=0, exact)")
    row("Gradient method",                 "parameter-shift rule  (2 evals / weight param)")

    div(); hdr("2.  LOCAL CIRCUIT  —  QUANTUM KERNEL"); sep()
    note(f"Quantum convolutional kernel over 2×2 patches — same {a['local_x_params']} values re-encoded every layer.")
    note(f"Per layer:  RY(x[i]) × {q}  →  RY(w[l×{q}+i]) × {q}  →  CX ring")
    sep(); hdr("gate counts"); sep()
    calc("RY data gates",    f"{nl} × {q}", f"{a['local_ry_data']}  (NOT trainable)")
    calc("RY weight gates",  f"{nl} × {q}", f"{a['local_ry_weight']}  (trainable)")
    calc("CNOT gates",       f"{nl} × {q}", f"{a['local_cnot']}  (no params)")
    calc("Total gates",      f"{a['local_ry_data']} + {a['local_ry_weight']} + {a['local_cnot']}", f"{a['local_total_gates']}")
    row("Qiskit ops dict",   a['local_ops'],   indent=4)
    row("Circuit depth",     a['local_depth'], indent=4)
    sep(); hdr("parameter counts"); sep()
    calc("Trainable weight params  len(w)", f"{nl} × {q}", f"{a['local_w_params']}")
    calc("Input params  len(x)  [NOT trained]", f"N_QUBITS = {q}", f"{a['local_x_params']}")
    sep()
    calc("Observables", f"{q} × {cfg.N_MEASURE_BASES}", f"{a['q_out']}  (Z, X, Y per qubit)")

    div(); hdr("3.  GLOBAL CIRCUIT  —  DATA RE-UPLOADING ENCODER"); sep()
    note(f"Encodes all {a['n_inputs']} inputs via re-uploading — {a['ipl']} fresh values per layer, cycling {q} qubits.")
    note(f"Per layer:  RY(x[idx++]) × {a['ipl']}  →  RY(w[l×{q}+i]) × {q}  →  CX ring")
    sep(); hdr("gate counts"); sep()
    calc("RY data gates  (all inputs)", "n_inputs", f"{a['global_ry_data']}  (NOT trainable)")
    calc("RY weight gates", f"{ng} × {q}", f"{a['global_ry_weight']}  (trainable)")
    calc("CNOT gates",      f"{ng} × {q}", f"{a['global_cnot']}  (no params)")
    calc("Total gates",     f"{a['global_ry_data']} + {a['global_ry_weight']} + {a['global_cnot']}", f"{a['global_total_gates']}")
    row("Qiskit ops dict",  a['global_ops'],   indent=4)
    row("Circuit depth",    a['global_depth'], indent=4)
    sep(); hdr("parameter counts"); sep()
    calc("Trainable weight params  len(w)", f"{ng} × {q}", f"{a['global_w_params']}")
    calc("Input params  len(x)  [NOT trained]", f"n_inputs = {a['n_inputs']}", f"{a['global_x_params']}")
    sep()
    calc("Observables", f"{q} × {cfg.N_MEASURE_BASES}", f"{a['q_out']}  (Z, X, Y per qubit)")

    div(); hdr("4.  CLASSICAL MODEL  —  TRAINABLE PARAMETER BREAKDOWN"); sep()
    note(f"{'Layer':<36}{'weights':>10}{'biases':>8}{'total':>10}"); sep()
    note(f"{'Conv2d(1→16,   k=3)   block1':<36}{a['c1_w']:>10,}{a['c1_b']:>8,}{a['c1_t']:>10,}")
    note(f"{'Conv2d(16→32,  k=3)   block2':<36}{a['c2_w']:>10,}{a['c2_b']:>8,}{a['c2_t']:>10,}")
    note(f"{'Conv2d(32→64,  k=3)   block3':<36}{a['c3_w']:>10,}{a['c3_b']:>8,}{a['c3_t']:>10,}")
    note(f"{'Conv2d(32→16,  k=1)   reduce[0]':<36}{a['rd1_w']:>10,}{a['rd1_b']:>8,}{a['rd1_t']:>10,}")
    note(f"{'Conv2d(16→'+str(q)+',   k=1)   reduce[1]':<36}{a['rd2_w']:>10,}{a['rd2_b']:>8,}{a['rd2_t']:>10,}")
    note(f"{'Linear('+str(a['fc_in'])+'→64)         fc[0]':<36}{a['f1_w']:>10,}{a['f1_b']:>8,}{a['f1_t']:>10,}")
    note(f"  fc_in = 64 classical + {a['q_out']} local-Q + {a['q_out']} global-Q = {a['fc_in']}")
    note(f"{'Linear(64→10)                fc[1]':<36}{a['f2_w']:>10,}{a['f2_b']:>8,}{a['f2_t']:>10,}"); sep()
    note(f"{'Classical subtotal':<36}{'':>18}{a['classical_total']:>10,}")
    note(f"{'Quantum  subtotal':<36}{'':>18}{a['quantum_total']:>10,}")
    note(f"  ({a['local_w_params']} local  +  {a['global_w_params']} global)"); sep()
    note(f"{'TOTAL MODEL PARAMETERS':<36}{'':>18}{a['model_total']:>10,}")
    note(f"  Classical  {a['classical_total']:,}  ({100*a['classical_total']/a['model_total']:.2f}%)")
    note(f"  Quantum    {a['quantum_total']:,}  ({100*a['quantum_total']/a['model_total']:.2f}%)")

    div(); hdr("5.  PATCH GEOMETRY  —  LOCAL CIRCUIT"); sep()
    note(f"Sliding 2×2 window over {cfg.Q_RES}×{cfg.Q_RES} feature map,  stride = {cfg.Q_STRIDE}"); sep()
    calc("Positions per axis",  f"floor(({cfg.Q_RES} - 1) / {cfg.Q_STRIDE})", f"{a['patch_pos']}")
    calc("Patches per sample",  f"{a['patch_pos']} × {a['patch_pos']}", f"{a['patches_per_sample']}")

    div(); hdr("6.  FORWARD PASS  —  CIRCUIT EXECUTION COUNTS"); sep()
    hdr("local  —  training"); sep()
    calc("Calls / batch",    f"{a['patches_per_sample']} patches × {cfg.BATCH_SIZE}", f"{a['l_fwd_batch']:,}")
    calc("Calls / epoch",    f"{a['l_fwd_batch']:,} × {a['train_batches']} batches",  f"{a['l_fwd_epoch']:,}")
    calc("Calls / training", f"{a['l_fwd_epoch']:,} × {cfg.EPOCHS} epochs",           f"{a['l_fwd_training']:,}")
    sep(); hdr("local  —  validation"); sep()
    calc("Calls / epoch",    f"{a['patches_per_sample']} × {cfg.VAL_SAMPLES}", f"{a['l_val_epoch']:,}")
    calc("Calls / training", f"{a['l_val_epoch']:,} × {cfg.EPOCHS} epochs",   f"{a['l_val_training']:,}")
    sep(); hdr("global  —  training"); sep()
    calc("Calls / batch",    f"1 × {cfg.BATCH_SIZE}", f"{a['g_fwd_batch']:,}")
    calc("Calls / epoch",    f"{a['g_fwd_batch']:,} × {a['train_batches']} batches", f"{a['g_fwd_epoch']:,}")
    calc("Calls / training", f"{a['g_fwd_epoch']:,} × {cfg.EPOCHS} epochs",          f"{a['g_fwd_training']:,}")
    sep(); hdr("global  —  validation"); sep()
    calc("Calls / epoch",    f"1 × {cfg.VAL_SAMPLES}", f"{a['g_val_epoch']:,}")
    calc("Calls / training", f"{a['g_val_epoch']:,} × {cfg.EPOCHS} epochs", f"{a['g_val_training']:,}")
    sep()
    calc("Forward total  (train + val,  both circuits)",
         f"{a['fwd_total']:,} + {a['val_total']:,}", f"{a['fwd_total'] + a['val_total']:,}")

    div(); hdr("7.  BACKWARD PASS  —  PARAMETER-SHIFT GRADIENT COUNTS"); sep()
    note("gradient(θ) = [ f(θ+π/2) − f(θ−π/2) ] / 2   →   2 evals per weight param per sample")
    note("x-params are NOT differentiated.  Validation has NO gradients."); sep()
    hdr("local circuit"); sep()
    calc("Shifts / sample",      f"2 × {a['local_w_params']}", f"{a['l_shift']}")
    calc("Grad calls / epoch",   f"{a['l_shift']} × {cfg.TRAIN_SAMPLES}", f"{a['l_grad_epoch']:,}")
    calc("Grad calls / training",f"{a['l_grad_epoch']:,} × {cfg.EPOCHS}", f"{a['l_grad_train']:,}")
    sep(); hdr("global circuit"); sep()
    calc("Shifts / sample",      f"2 × {a['global_w_params']}", f"{a['g_shift']}")
    calc("Grad calls / epoch",   f"{a['g_shift']} × {cfg.TRAIN_SAMPLES}", f"{a['g_grad_epoch']:,}")
    calc("Grad calls / training",f"{a['g_grad_epoch']:,} × {cfg.EPOCHS}", f"{a['g_grad_train']:,}")
    sep()
    calc("Gradient total  (both)", f"{a['l_grad_train']:,} + {a['g_grad_train']:,}", f"{a['grad_total']:,}")

    div(); hdr("8.  GRAND TOTAL  —  FULL TRAINING RUN"); sep()
    note(f"{'':4}{'':28}{'Local':>14}{'Global':>14}{'Both':>14}"); sep()
    note(f"{'':4}{'Train fwd / training':<28}{a['l_fwd_training']:>14,}{a['g_fwd_training']:>14,}{a['fwd_total']:>14,}")
    note(f"{'':4}{'Val fwd / training':<28}{a['l_val_training']:>14,}{a['g_val_training']:>14,}{a['val_total']:>14,}")
    note(f"{'':4}{'Bwd / training':<28}{a['l_grad_train']:>14,}{a['g_grad_train']:>14,}{a['grad_total']:>14,}")
    note(f"{'':4}{'Circuit total':<28}{a['l_total']:>14,}{a['g_total']:>14,}{a['grand_total']:>14,}"); sep()
    calc("GRAND TOTAL", f"{a['l_total']:,} + {a['g_total']:,}", f"{a['grand_total']:,}  circuit executions")
    blank()
    note(f"Ratio  bwd / fwd  ≈  {a['grad_total'] / (a['fwd_total'] + a['val_total']):.1f}×  "
         f"(parameter-shift backward dominates)")
    bot()


def run_report(cfg) -> dict:
    """Build circuits, compute analysis, and print full report. Returns analysis dict."""
    a = compute_analysis(cfg)
    print_circuits(a)
    print_analysis(a, cfg)
    return a


# ============================================================
#  PLOT HELPERS
# ============================================================

def _glow(ax, bar, col, width_scale=1.3, alpha=0.13, horizontal=False):
    """Draw a soft glow rectangle behind a bar."""
    if horizontal:
        ax.barh(bar.get_y() + bar.get_height() / 2, bar.get_width(),
                height=bar.get_height() * width_scale,
                color=col, alpha=alpha, zorder=2)
    else:
        ax.bar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
               width=bar.get_width() * width_scale,
               color=col, alpha=alpha, zorder=2)


def plot_gate_composition(a: dict) -> None:
    """Bar chart: gate composition for local and global circuits."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0F1117')
    gate_colors = [C_BLUE, C_TEAL, C_PINK]

    for ax, title, ry_d, ry_w, cnot in [
        (axes[0], 'Local Circuit',
         a['local_ry_data'],  a['local_ry_weight'],  a['local_cnot']),
        (axes[1], 'Global Circuit',
         a['global_ry_data'], a['global_ry_weight'], a['global_cnot']),
    ]:
        labels = ['RY data\n(not trained)', 'RY weight\n(trainable)', 'CNOT\n(no params)']
        values = [ry_d, ry_w, cnot]
        bars   = ax.bar(labels, values, color=gate_colors,
                        edgecolor='none', width=0.48, zorder=3)
        for bar, col in zip(bars, gate_colors):
            _glow(ax, bar, col)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.025,
                    str(val), ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='#EAECF4')
        ax.set_title(title, fontweight='bold', pad=12)
        ax.set_ylabel('Gate count')
        ax.set_ylim(0, max(values) * 1.28)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.tick_params(axis='x', length=0)

    fig.suptitle('Gate Composition per Circuit',
                 fontsize=15, fontweight='bold', color='#EAECF4', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_parameter_breakdown(a: dict) -> None:
    """Donut (classical vs quantum) + horizontal bar (layer breakdown)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0F1117')

    # ── donut ─────────────────────────────────────────────────
    ax = axes[0]
    sizes      = [a['classical_total'], a['quantum_total']]
    pie_labels = [f"Classical\n{a['classical_total']:,}",
                  f"Quantum\n{a['quantum_total']:,}"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=pie_labels, colors=[C_BLUE, C_AMBER],
        autopct='%1.2f%%', startangle=90, pctdistance=0.78,
        wedgeprops={'edgecolor': '#0F1117', 'linewidth': 3, 'width': 0.55},
        textprops={'fontsize': 11, 'color': '#C8CAD4'},
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_color('#EAECF4'); at.set_fontweight('bold')
    ax.text(0, 0, f"{a['model_total']:,}\nparams",
            ha='center', va='center', fontsize=10,
            color='#EAECF4', fontweight='bold')
    ax.set_title('Classical vs Quantum Parameters', fontweight='bold', pad=14)

    # ── horizontal bar ─────────────────────────────────────────
    ax = axes[1]
    layer_names = ['Conv1 (1→16)', 'Conv2 (16→32)', 'Conv3 (32→64)',
                   'Reduce (32→16→q)', 'FC1', 'FC2']
    layer_vals  = [a['c1_t'], a['c2_t'], a['c3_t'],
                   a['rd_t'], a['f1_t'], a['f2_t']]
    bar_colors  = [C_PURPLE, C_BLUE, C_TEAL, C_TEAL, C_PINK, C_AMBER]
    y_pos = np.arange(len(layer_names))

    hbars = ax.barh(y_pos, layer_vals, color=bar_colors,
                    edgecolor='none', height=0.52, zorder=3)
    for bar, col in zip(hbars, bar_colors):
        _glow(ax, bar, col, horizontal=True)
    for bar, val in zip(hbars, layer_vals):
        ax.text(bar.get_width() + max(layer_vals) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=9,
                color='#EAECF4', fontweight='bold')
    ax.set_yticks(y_pos); ax.set_yticklabels(layer_names)
    ax.set_xlabel('Parameters')
    ax.set_title('Classical Layer Breakdown', fontweight='bold', pad=12)
    ax.set_xlim(0, max(layer_vals) * 1.2)
    ax.invert_yaxis()
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    plt.show()


def plot_execution_counts(a: dict) -> None:
    """Bar chart: circuit execution counts over the full training run."""
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('#0F1117')

    categories = ['Local\nTrain Fwd', 'Local\nVal Fwd', 'Local\nBwd',
                  'Global\nTrain Fwd', 'Global\nVal Fwd', 'Global\nBwd']
    values     = [a['l_fwd_training'], a['l_val_training'], a['l_grad_train'],
                  a['g_fwd_training'], a['g_val_training'], a['g_grad_train']]
    bar_colors = [C_BLUE, C_BLUE, C_PINK, C_TEAL, C_TEAL, C_AMBER]
    alphas     = [1.0, 0.4, 1.0, 1.0, 0.4, 1.0]

    x    = np.arange(len(categories))
    bars = ax.bar(x, values, color=bar_colors, edgecolor='none', width=0.52, zorder=3)
    for bar, col, al, val in zip(bars, bar_colors, alphas, values):
        bar.set_alpha(al)
        _glow(ax, bar, col, alpha=al * 0.13)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.013,
                f'{val:,}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color='#EAECF4', rotation=12)

    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel('Circuit executions')
    ax.set_ylim(0, max(values) * 1.32)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.set_title(
        f'Circuit Execution Counts — Full Training Run\n'
        f'Grand total: {a["grand_total"]:,} executions',
        fontweight='bold', pad=12)
    ax.legend(handles=[
        mpatches.Patch(color=C_BLUE,  label='Local — forward'),
        mpatches.Patch(color=C_TEAL,  label='Global — forward'),
        mpatches.Patch(color=C_PINK,  label='Local — backward'),
        mpatches.Patch(color=C_AMBER, label='Global — backward'),
    ], loc='upper right', bbox_to_anchor=(1, 1.1))

    plt.tight_layout()
    plt.show()


def plot_fwd_bwd_ratio(a: dict) -> None:
    """Bar chart: total forward vs backward circuit executions."""
    total_fwd = a['fwd_total'] + a['val_total']
    total_bwd = a['grad_total']
    ratio     = total_bwd / total_fwd

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#0F1117')

    vals_fb = [total_fwd, total_bwd]
    cols_fb = [C_TEAL, C_PINK]
    bars = ax.bar(['Forward + Val', 'Backward (grad)'], vals_fb,
                  color=cols_fb, edgecolor='none', width=0.38, zorder=3)
    for bar, col, val in zip(bars, cols_fb, vals_fb):
        _glow(ax, bar, col)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals_fb) * 0.02,
                f'{val:,}', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#EAECF4')

    ax.set_ylim(0, max(vals_fb) * 1.28)
    ax.set_ylabel('Circuit executions')
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.set_title(
        f'Forward vs Backward Circuit Executions\n'
        f'Ratio  bwd / fwd  ≈  {ratio:.1f}×',
        fontweight='bold', pad=12)

    plt.tight_layout()
    plt.show()

    print(f"Forward + Val : {total_fwd:,}")
    print(f"Backward      : {total_bwd:,}")
    print(f"Ratio bwd/fwd : {ratio:.2f}×  (parameter-shift dominates)")


def summary_table(a: dict) -> None:
    """Display a styled summary table (pandas) or plain-text fallback."""
    from IPython.display import display as ipy_display

    data = {
        "Metric": [
            "Total model parameters",
            "  Classical",
            "  Quantum (local + global)",
            "Local circuit — weight params",
            "Global circuit — weight params",
            "Local circuit — depth",
            "Global circuit — depth",
            "Patches per sample",
            "Grand total circuit executions",
            "  of which: backward pass",
            "Bwd / Fwd ratio",
        ],
        "Value": [
            f"{a['model_total']:,}",
            f"{a['classical_total']:,}  ({100*a['classical_total']/a['model_total']:.2f}%)",
            f"{a['quantum_total']:,}  ({100*a['quantum_total']/a['model_total']:.2f}%)",
            f"{a['local_w_params']}",
            f"{a['global_w_params']}",
            f"{a['local_depth']}",
            f"{a['global_depth']}",
            f"{a['patches_per_sample']}",
            f"{a['grand_total']:,}",
            f"{a['grad_total']:,}  ({100*a['grad_total']/a['grand_total']:.1f}%)",
            f"{a['grad_total'] / (a['fwd_total'] + a['val_total']):.1f}×",
        ],
    }

    try:
        import pandas as pd
        df = pd.DataFrame(data)
        ipy_display(
            df.style
              .set_properties(**{'text-align': 'left', 'padding': '6px 18px',
                                 'color': '#C8CAD4', 'font-size': '13px'})
              .set_table_styles([
                  {'selector': 'th',
                   'props': [('background-color', '#1E2130'), ('color', '#A78BFA'),
                             ('font-weight', 'bold'), ('text-align', 'left'),
                             ('padding', '7px 18px'), ('font-size', '13px')]},
                  {'selector': 'tr:nth-child(even)',
                   'props': [('background-color', '#1A1D27')]},
                  {'selector': 'tr:nth-child(odd)',
                   'props': [('background-color', '#141620')]},
                  {'selector': 'table',
                   'props': [('border-collapse', 'collapse'),
                             ('border-radius', '8px'), ('overflow', 'hidden')]},
              ])
              .hide(axis='index')
        )
    except ImportError:
        col_w = max(len(m) for m in data['Metric']) + 2
        print(f"{'Metric':<{col_w}}  Value")
        print("-" * (col_w + 40))
        for m, v in zip(data['Metric'], data['Value']):
            print(f"{m:<{col_w}}  {v}")
