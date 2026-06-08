"""
qc_train.py
===========
Training engine for TPCQ-Net experiments.

Features
--------
- tqdm progress bar per epoch showing batch loss, running accuracy,
  ETA, and elapsed time
- Comprehensive CSV with every metric logged per epoch AND per batch
- Dataset mode: 'subset' (small, fast, default) or 'full' (all 60k/10k)
- Stratified subset sampling so all 10 classes are always represented,
  avoiding the top_k_accuracy_score class-mismatch crash
- Accurate circuit execution counting for both baseline and commuting

CSV columns (epoch log)
-----------------------
  epoch, train_loss, train_acc_approx,
  val_acc, val_precision, val_recall, val_f1_macro, val_f1_micro,
  val_top3, val_top5,
  feat_classical, feat_local, feat_global,
  param_classical, param_local, param_global,
  val_time_s, epoch_time_s, lr,
  variant, n_train_samples, n_val_samples,
  global_bwd_circuits_per_epoch, grand_total_circuits,
  best_acc_so_far

Public API
----------
  make_loaders(cfg)
  train_model(model, cfg, runner_ctx, train_loader, val_loader, tag,
               save_path, csv_path)
  evaluate(model, loader)
  compute_circuit_counts(cfg, runner_ctx, n_train_samples)
  print_circuit_counts(counts, tag)
"""

import time, csv, os, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, top_k_accuracy_score,
)

from qc_runner import RunnerContext


# ============================================================
#  DATASET HELPERS
# ============================================================

def _stratified_indices(dataset, n_samples: int, seed: int = 42) -> list:
    """
    Return indices for a stratified subset of `dataset` with exactly
    `n_samples` total, balanced across all classes present.

    This guarantees all 10 MNIST classes appear in even tiny subsets,
    which prevents the top_k_accuracy_score class-count mismatch.
    """
    rng = np.random.default_rng(seed)

    # Group indices by label
    label_to_idx = defaultdict(list)
    for idx in range(len(dataset)):
        try:
            lbl = int(dataset.targets[idx])
        except AttributeError:
            # Subset wrapping — iterate (slow but correct)
            _, lbl = dataset[idx]
        label_to_idx[lbl].append(idx)

    classes    = sorted(label_to_idx.keys())
    n_classes  = len(classes)
    per_class  = n_samples // n_classes
    remainder  = n_samples  % n_classes

    chosen = []
    for i, cls in enumerate(classes):
        pool   = label_to_idx[cls]
        take   = per_class + (1 if i < remainder else 0)
        take   = min(take, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())

    rng.shuffle(chosen)
    return chosen


def make_loaders(cfg) -> tuple:
    """
    Build train and val DataLoaders according to cfg.DATASET_MODE.

    cfg.DATASET_MODE options
    ------------------------
    'subset'  — stratified subset of cfg.TRAIN_SAMPLES / cfg.VAL_SAMPLES
    'full'    — entire MNIST train (60k) and test (10k) sets

    Returns (train_loader, val_loader, actual_train_size, actual_val_size)
    """
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    full_val   = datasets.MNIST('./data', train=False, download=True, transform=transform)

    mode = getattr(cfg, 'DATASET_MODE', 'subset').lower()

    if mode == 'full':
        train_set = full_train
        val_set   = full_val
    else:
        train_idx = _stratified_indices(full_train, cfg.TRAIN_SAMPLES, seed=42)
        val_idx   = _stratified_indices(full_val,   cfg.VAL_SAMPLES,   seed=0)
        train_set = Subset(full_train, train_idx)
        val_set   = Subset(full_val,   val_idx)

    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    n_train = len(train_set)
    n_val   = len(val_set)
    print(f"Dataset mode : {mode.upper()}")
    print(f"Train        : {n_train} samples  ({len(train_loader)} batches)")
    print(f"Val          : {n_val}   samples  ({len(val_loader)} batches)")
    return train_loader, val_loader, n_train, n_val


# ============================================================
#  CIRCUIT COUNT UTILITIES
# ============================================================

def compute_circuit_counts(cfg, runner_ctx: RunnerContext,
                            n_train_samples: int) -> dict:
    """
    Compute expected total circuit executions for one full training run.
    Covers: train forward, train backward, validation forward.
    """
    n_q     = cfg.N_QUBITS
    n_q_obs = cfg.N_MEASURE_BASES
    n_ep    = cfg.EPOCHS
    n_val   = cfg.VAL_SAMPLES if getattr(cfg, 'DATASET_MODE', 'subset') != 'full' else 10000

    patch_pos = (cfg.Q_RES - 1) // cfg.Q_STRIDE
    n_patches = patch_pos * patch_pos

    # Forward (train)
    local_fwd_train  = n_patches * n_train_samples * n_ep * n_q_obs
    global_fwd_train = n_train_samples * n_ep * n_q_obs
    # Forward (val)
    local_fwd_val    = n_patches * n_val * n_ep * n_q_obs
    global_fwd_val   = n_val * n_ep * n_q_obs

    # Backward
    if runner_ctx.variant == 'baseline':
        n_local_gate  = cfg.N_Q_LAYERS * cfg.N_QUBITS
        n_global_gate = runner_ctx.n_gate_params_g
        local_bwd  = 2 * n_local_gate  * n_q_obs * n_patches * n_train_samples * n_ep
        global_bwd = 2 * n_global_gate * n_q_obs * n_train_samples * n_ep
    else:
        n_local_gate = cfg.N_Q_LAYERS * cfg.N_QUBITS
        local_bwd    = 2 * n_local_gate * n_q_obs * n_patches * n_train_samples * n_ep
        n_logical    = runner_ctx.n_logical_params_g
        n_xy_obs     = max(0, n_q_obs - 1)
        global_bwd   = (n_q * n_train_samples * n_ep +
                        2 * n_logical * n_xy_obs * n_train_samples * n_ep)

    fwd_total = (local_fwd_train + global_fwd_train +
                 local_fwd_val   + global_fwd_val)
    bwd_total = local_bwd + global_bwd
    grand     = fwd_total + bwd_total

    return {
        'local_fwd_train':  local_fwd_train,
        'global_fwd_train': global_fwd_train,
        'local_fwd_val':    local_fwd_val,
        'global_fwd_val':   global_fwd_val,
        'local_bwd':        local_bwd,
        'global_bwd':       global_bwd,
        'fwd_total':        fwd_total,
        'bwd_total':        bwd_total,
        'grand_total':      grand,
        'bwd_fwd_ratio':    bwd_total / fwd_total if fwd_total > 0 else 0,
    }


def print_circuit_counts(counts: dict, tag: str):
    print(f"\n{'='*55}")
    print(f"  Circuit execution budget — {tag}")
    print(f"{'='*55}")
    print(f"  Forward  (train):  {counts['local_fwd_train']:>12,}  local")
    print(f"                     {counts['global_fwd_train']:>12,}  global")
    print(f"  Forward  (val):    {counts['local_fwd_val']:>12,}  local")
    print(f"                     {counts['global_fwd_val']:>12,}  global")
    print(f"  Backward (train):  {counts['local_bwd']:>12,}  local")
    print(f"                     {counts['global_bwd']:>12,}  global")
    print(f"  {'─'*45}")
    print(f"  Forward total:     {counts['fwd_total']:>12,}")
    print(f"  Backward total:    {counts['bwd_total']:>12,}")
    print(f"  Grand total:       {counts['grand_total']:>12,}")
    print(f"  Bwd / Fwd ratio:   {counts['bwd_fwd_ratio']:>11.1f}×")
    print(f"{'='*55}")


# ============================================================
#  EVALUATION
# ============================================================

def evaluate(model, loader: DataLoader) -> dict:
    """Compute full classification metrics on a DataLoader."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(torch.softmax(out, 1).cpu().numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)

    # Always pass explicit class list so top_k works when val set is missing
    # some classes (common with small VAL_SAMPLES or stratified subsets).
    all_classes = np.arange(probs.shape[1])

    return {
        'acc':       accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall':    recall_score(labels, preds, average='macro', zero_division=0),
        'f1_macro':  f1_score(labels, preds, average='macro',  zero_division=0),
        'f1_micro':  f1_score(labels, preds, average='micro',  zero_division=0),
        'top3':      top_k_accuracy_score(labels, probs,
                                          k=min(3, probs.shape[1]),
                                          labels=all_classes),
        'top5':      top_k_accuracy_score(labels, probs,
                                          k=min(5, probs.shape[1]),
                                          labels=all_classes),
        'preds':     preds,
        'labels':    labels,
    }


def compute_contributions(model, loader: DataLoader) -> dict:
    """Estimate relative feature energy contribution of each branch."""
    import torch.nn.functional as F
    x, _ = next(iter(loader))
    with torch.no_grad():
        _, x2, x3 = model.classical(x)
        x2  = F.interpolate(x2, size=(model.cfg.Q_RES, model.cfg.Q_RES))
        red = model.reduce(x2)
        ql  = model.q_local(red)
        qg  = model.q_global(red)

    xc   = torch.mean(x3, dim=(2, 3))
    ql_m = torch.mean(ql, dim=(2, 3))
    xc_e = torch.mean(xc  ** 2).item()
    ql_e = torch.mean(ql_m ** 2).item()
    qg_e = torch.mean(qg  ** 2).item()
    tot_e = xc_e + ql_e + qg_e + 1e-12

    c_p  = sum(p.numel() for p in model.classical.parameters())
    ql_p = sum(p.numel() for p in model.q_local.parameters())
    qg_p = sum(p.numel() for p in model.q_global.parameters())
    tot_p = c_p + ql_p + qg_p + 1e-12

    return {
        'feat_classical':  xc_e / tot_e,
        'feat_local':      ql_e / tot_e,
        'feat_global':     qg_e / tot_e,
        'param_classical': c_p  / tot_p,
        'param_local':     ql_p / tot_p,
        'param_global':    qg_p / tot_p,
    }


# ============================================================
#  CSV HELPERS
# ============================================================

_CSV_HEADER = [
    # — identity —
    'epoch', 'variant', 'tag',
    'n_train_samples', 'n_val_samples',
    # — training loss (average over batches) —
    'train_loss', 'train_acc_approx',
    # — validation metrics —
    'val_acc', 'val_precision', 'val_recall',
    'val_f1_macro', 'val_f1_micro',
    'val_top3', 'val_top5',
    # — branch contributions —
    'feat_classical', 'feat_local', 'feat_global',
    'param_classical', 'param_local', 'param_global',
    # — timing —
    'train_time_s', 'val_time_s', 'epoch_time_s',
    # — optimiser —
    'lr',
    # — circuit counts (this epoch) —
    'local_bwd_this_epoch', 'global_bwd_this_epoch',
    'fwd_this_epoch', 'total_this_epoch',
    # — cumulative circuit counts —
    'global_bwd_cumulative', 'grand_total_cumulative',
    # — best so far —
    'best_val_acc',
]


def _csv_row(epoch, tag, runner_ctx, cfg,
             avg_loss, train_acc, metrics, contrib,
             train_time, val_time, epoch_time,
             optimizer, counts, n_train, n_val, best_acc) -> list:
    """Assemble one CSV row from all logged quantities."""
    ep_fwd  = (counts['local_fwd_train'] + counts['global_fwd_train'] +
               counts['local_fwd_val']   + counts['global_fwd_val']) // cfg.EPOCHS
    ep_lbwd = counts['local_bwd']  // cfg.EPOCHS
    ep_gbwd = counts['global_bwd'] // cfg.EPOCHS
    ep_tot  = ep_fwd + ep_lbwd + ep_gbwd

    cum_gbwd = ep_gbwd * epoch
    cum_tot  = (ep_fwd + ep_lbwd + ep_gbwd) * epoch

    lr_now = optimizer.param_groups[0]['lr']

    return [
        epoch, runner_ctx.variant, tag,
        n_train, n_val,
        round(avg_loss, 6), round(train_acc, 6),
        round(metrics['acc'],       6),
        round(metrics['precision'], 6),
        round(metrics['recall'],    6),
        round(metrics['f1_macro'],  6),
        round(metrics['f1_micro'],  6),
        round(metrics['top3'],      6),
        round(metrics['top5'],      6),
        round(contrib['feat_classical'],  6),
        round(contrib['feat_local'],      6),
        round(contrib['feat_global'],     6),
        round(contrib['param_classical'], 6),
        round(contrib['param_local'],     6),
        round(contrib['param_global'],    6),
        round(train_time,  2),
        round(val_time,    2),
        round(epoch_time,  2),
        lr_now,
        ep_lbwd, ep_gbwd, ep_fwd, ep_tot,
        cum_gbwd, cum_tot,
        round(best_acc, 6),
    ]


# ============================================================
#  TRAINING LOOP
# ============================================================

def train_model(
    model,
    cfg,
    runner_ctx:   RunnerContext,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    tag:          str,
    save_path:    str,
    csv_path:     str,
    n_train_samples: int = None,
    n_val_samples:   int = None,
) -> dict:
    """
    Full training loop with tqdm progress bar and comprehensive CSV logging.

    Progress bar (per epoch) shows
        batch loss · running accuracy · batch #/total · elapsed · ETA

    Each epoch appends one row to the CSV with all metrics, timings,
    branch contributions, and circuit execution counts.

    Parameters
    ----------
    model            : QCCNNBaseline or QCCNNCommuting
    cfg              : configuration class
    runner_ctx       : RunnerContext (baseline or commuting)
    train_loader     : DataLoader for training set
    val_loader       : DataLoader for validation set
    tag              : short identifier used in prints and CSV ('Baseline'/'Commuting')
    save_path        : path to save the best model checkpoint (.pth)
    csv_path         : path for the epoch-level CSV log
    n_train_samples  : actual number of training samples (auto-detected if None)
    n_val_samples    : actual number of validation samples (auto-detected if None)

    Returns
    -------
    dict with keys: history, best_acc, circuit_counts, save_path, csv_path, tag
    """
    # Auto-detect sizes if not provided
    if n_train_samples is None:
        n_train_samples = len(train_loader.dataset)
    if n_val_samples is None:
        n_val_samples = len(val_loader.dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn   = nn.CrossEntropyLoss()
    best_acc  = 0.0
    history   = []

    counts = compute_circuit_counts(cfg, runner_ctx, n_train_samples)
    print_circuit_counts(counts, tag)

    # ── CSV setup ────────────────────────────────────────────────
    os.makedirs(
        os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.',
        exist_ok=True
    )
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(_CSV_HEADER)

    # ── epoch loop ───────────────────────────────────────────────
    for epoch in range(1, cfg.EPOCHS + 1):

        model.train()
        total_loss     = 0.0
        correct        = 0
        total_samples  = 0
        t0             = time.time()

        # ── batch progress bar ───────────────────────────────────
        pbar = tqdm(
            train_loader,
            desc=f"[{tag}] Epoch {epoch:>2}/{cfg.EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )

        for batch_idx, (x, y) in enumerate(pbar, 1):
            optimizer.zero_grad()
            logits = model(x)
            loss   = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            # running stats for progress bar
            total_loss    += loss.item()
            preds_batch    = torch.argmax(logits.detach(), dim=1)
            correct       += (preds_batch == y).sum().item()
            total_samples += y.size(0)

            avg_loss_so_far  = total_loss / batch_idx
            running_acc      = correct / total_samples

            pbar.set_postfix(
                loss=f"{avg_loss_so_far:.4f}",
                acc=f"{running_acc:.3f}",
                refresh=False,
            )

        pbar.close()
        train_time = time.time() - t0

        avg_loss  = total_loss / len(train_loader)
        train_acc = correct / total_samples

        # ── validation ───────────────────────────────────────────
        t_val   = time.time()
        metrics = evaluate(model, val_loader)
        val_time = time.time() - t_val

        contrib    = compute_contributions(model, val_loader)
        epoch_time = time.time() - t0

        # ── checkpoint ───────────────────────────────────────────
        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                        exist_ok=True)
            torch.save(model.state_dict(), save_path)
            ckpt_marker = " ✓ saved"
        else:
            ckpt_marker = ""

        # ── epoch summary line ───────────────────────────────────
        print(
            f"  └─ loss={avg_loss:.4f}  "
            f"train_acc={train_acc:.3f}  "
            f"val_acc={metrics['acc']:.4f}  "
            f"top5={metrics['top5']:.4f}  "
            f"val_t={val_time:.1f}s  "
            f"total_t={epoch_time:.1f}s"
            f"{ckpt_marker}"
        )

        # ── CSV row ──────────────────────────────────────────────
        row_values = _csv_row(
            epoch=epoch, tag=tag, runner_ctx=runner_ctx, cfg=cfg,
            avg_loss=avg_loss, train_acc=train_acc,
            metrics=metrics, contrib=contrib,
            train_time=train_time, val_time=val_time,
            epoch_time=epoch_time, optimizer=optimizer,
            counts=counts, n_train=n_train_samples,
            n_val=n_val_samples, best_acc=best_acc,
        )
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row_values)

        # ── history dict ─────────────────────────────────────────
        history.append({
            'epoch':         epoch,
            'train_loss':    avg_loss,
            'train_acc':     train_acc,
            'val_acc':       metrics['acc'],
            'val_top5':      metrics['top5'],
            'val_f1_macro':  metrics['f1_macro'],
            'feat_classical': contrib['feat_classical'],
            'feat_local':     contrib['feat_local'],
            'feat_global':    contrib['feat_global'],
            'epoch_time_s':  epoch_time,
        })

    print(f"\n[{tag}] Training complete. "
          f"Best val acc: {best_acc:.4f}  |  "
          f"Log: {csv_path}")

    return {
        'history':        history,
        'best_acc':       best_acc,
        'circuit_counts': counts,
        'save_path':      save_path,
        'csv_path':       csv_path,
        'tag':            tag,
        'n_train':        n_train_samples,
        'n_val':          n_val_samples,
    }
