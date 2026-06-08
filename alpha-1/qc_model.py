"""
qc_model.py
===========
PyTorch model definitions for TPCQ-Net (baseline and commuting variants).

Both variants share the same classical CNN backbone and FC head.
The only difference is the global quantum branch:
  - Baseline:  QuantumGlobalBaseline  (RY+CX, param-shift backward)
  - Commuting: QuantumGlobalCommuting (X-generators, parallel backward)

The local quantum branch is identical in both.

Public API
----------
  build_baseline_model(cfg, ctx)   → QCCNNBaseline
  build_commuting_model(cfg, ctx)  → QCCNNCommuting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from qc_runner import (
    RunnerContext,
    batch_forward_local,
    batch_forward_global,
    batch_backward_local_paramshift,
    batch_backward_global_paramshift,
    batch_backward_global_parallel_full,
)


# ============================================================
#  AUTOGRAD FUNCTIONS
# ============================================================

class _LocalFn(torch.autograd.Function):
    """Local circuit forward+backward (param-shift, same for both variants)."""

    @staticmethod
    def forward(ctx_t, x_batch_t, w_t, runner_ctx):
        x_np = x_batch_t.detach().numpy()
        w_np = w_t.detach().numpy().ravel()
        evs  = batch_forward_local(runner_ctx, x_np, w_np)
        ctx_t.save_for_backward(w_t)
        ctx_t.runner_ctx = runner_ctx
        ctx_t.x_np       = x_np
        ctx_t.w_shape    = w_t.shape
        return torch.as_tensor(evs, dtype=torch.float32)

    @staticmethod
    def backward(ctx_t, grad_out):
        w_np   = ctx_t.saved_tensors[0].detach().numpy().ravel()
        grad_w = batch_backward_local_paramshift(
            ctx_t.runner_ctx, ctx_t.x_np, w_np, grad_out.numpy()
        )
        return None, torch.as_tensor(
            grad_w.reshape(ctx_t.w_shape), dtype=torch.float32
        ), None


class _GlobalBaselineFn(torch.autograd.Function):
    """Global circuit with param-shift backward (baseline)."""

    @staticmethod
    def forward(ctx_t, x_batch_t, w_t, runner_ctx):
        x_np = x_batch_t.detach().numpy()
        w_np = w_t.detach().numpy().ravel()
        evs  = batch_forward_global(runner_ctx, x_np, w_np)
        ctx_t.save_for_backward(w_t)
        ctx_t.runner_ctx = runner_ctx
        ctx_t.x_np       = x_np
        ctx_t.w_shape    = w_t.shape
        return torch.as_tensor(evs, dtype=torch.float32)

    @staticmethod
    def backward(ctx_t, grad_out):
        w_np   = ctx_t.saved_tensors[0].detach().numpy().ravel()
        grad_w = batch_backward_global_paramshift(
            ctx_t.runner_ctx, ctx_t.x_np, w_np, grad_out.numpy()
        )
        return None, torch.as_tensor(
            grad_w.reshape(ctx_t.w_shape), dtype=torch.float32
        ), None


class _GlobalCommutingFn(torch.autograd.Function):
    """Global circuit with parallel gradient backward (commuting)."""

    @staticmethod
    def forward(ctx_t, x_batch_t, w_t, runner_ctx):
        x_np = x_batch_t.detach().numpy()
        w_np = w_t.detach().numpy().ravel()
        evs  = batch_forward_global(runner_ctx, x_np, w_np)
        ctx_t.save_for_backward(w_t)
        ctx_t.runner_ctx = runner_ctx
        ctx_t.x_np       = x_np
        ctx_t.w_shape    = w_t.shape
        return torch.as_tensor(evs, dtype=torch.float32)

    @staticmethod
    def backward(ctx_t, grad_out):
        w_np   = ctx_t.saved_tensors[0].detach().numpy().ravel()
        # Returns shape (n_logical,) — one entry per symmetry class
        grad_logical = batch_backward_global_parallel_full(
            ctx_t.runner_ctx, ctx_t.x_np, w_np, grad_out.numpy()
        )
        # Expand back to gate space: every gate in class j gets grad_logical[j].
        # Equivariant design — all gates in a class share the same update direction.
        gate_to_class = ctx_t.runner_ctx.gate_to_class
        n_gates       = len(gate_to_class)
        grad_gate     = np.array([grad_logical[gate_to_class[k]]
                                  for k in range(n_gates)], dtype=np.float32)
        return None, torch.as_tensor(
            grad_gate.reshape(ctx_t.w_shape), dtype=torch.float32
        ), None


# ============================================================
#  QUANTUM MODULES
# ============================================================

class QuantumLocal(nn.Module):
    """Local quantum conv (identical for baseline and commuting)."""

    def __init__(self, cfg, runner_ctx: RunnerContext):
        super().__init__()
        self.cfg    = cfg
        self.ctx    = runner_ctx
        n_params    = cfg.N_Q_LAYERS * cfg.N_QUBITS
        self.weights = nn.Parameter(torch.randn(n_params) * 0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.tanh(x)

        positions = [
            (i, j)
            for i in range(0, H - 1, self.cfg.Q_STRIDE)
            for j in range(0, W - 1, self.cfg.Q_STRIDE)
        ]
        n_patches = len(positions)
        grid      = int(math.sqrt(n_patches))
        assert grid * grid == n_patches, \
            f"Non-square patch grid: {n_patches}. Adjust Q_RES/Q_STRIDE."
        self.last_patch_count = n_patches

        all_patches = []
        for b in range(B):
            for (i, j) in positions:
                patch = x[b, :, i:i+2, j:j+2].reshape(-1)[:self.cfg.N_QUBITS]
                all_patches.append(patch)
        x_batch = torch.stack(all_patches)

        out = _LocalFn.apply(x_batch, self.weights, self.ctx)
        out = out.reshape(B, n_patches, self.cfg.q_out())
        out = out.permute(0, 2, 1)
        out = out.reshape(B, self.cfg.q_out(), grid, grid)
        return out


class QuantumGlobalBaseline(nn.Module):
    """Global quantum encoder — param-shift backward."""

    def __init__(self, cfg, runner_ctx: RunnerContext):
        super().__init__()
        self.cfg     = cfg
        self.ctx     = runner_ctx
        n_params     = cfg.N_Q_LAYERS_GLOBAL * cfg.N_QUBITS
        self.weights = nn.Parameter(torch.randn(n_params) * 0.01)

    def forward(self, x):
        B    = x.shape[0]
        x    = torch.tanh(x)
        xb   = x.reshape(B, -1)[:, :self.cfg.n_inputs()]
        return _GlobalBaselineFn.apply(xb, self.weights, self.ctx)


class QuantumGlobalCommuting(nn.Module):
    """Global quantum encoder — parallel gradient backward (Bowles et al.)."""

    def __init__(self, cfg, runner_ctx: RunnerContext):
        super().__init__()
        self.cfg     = cfg
        self.ctx     = runner_ctx
        n_gate_params = runner_ctx.n_gate_params_g
        self.weights  = nn.Parameter(torch.randn(n_gate_params) * 0.01)
        self.n_logical = runner_ctx.n_logical_params_g

    def forward(self, x):
        B  = x.shape[0]
        x  = torch.tanh(x)
        xb = x.reshape(B, -1)[:, :self.cfg.n_inputs()]
        return _GlobalCommutingFn.apply(xb, self.weights, self.ctx)


# ============================================================
#  CLASSICAL BACKBONE
# ============================================================

class ClassicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        return x1, x2, x3


# ============================================================
#  FULL MODELS
# ============================================================

class QCCNNBaseline(nn.Module):
    """TPCQ-Net baseline: RY+CX global circuit, param-shift gradients."""

    def __init__(self, cfg, runner_ctx: RunnerContext):
        super().__init__()
        self.cfg       = cfg
        self.classical = ClassicalCNN()
        self.reduce    = nn.Conv2d(32, cfg.N_QUBITS, 1)
        self.q_local   = QuantumLocal(cfg, runner_ctx)
        self.q_global  = QuantumGlobalBaseline(cfg, runner_ctx)
        fc_in = 64 + cfg.q_out() + cfg.q_out()
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        _, x2, x3 = self.classical(x)
        x2  = F.interpolate(x2, size=(self.cfg.Q_RES, self.cfg.Q_RES))
        red = self.reduce(x2)
        ql  = torch.mean(self.q_local(red), dim=(2, 3))
        qg  = self.q_global(red)
        xc  = torch.mean(x3, dim=(2, 3))
        return self.fc(torch.cat([xc, ql, qg], dim=1))


class QCCNNCommuting(nn.Module):
    """TPCQ-Net commuting: X-generator global circuit, parallel gradients."""

    def __init__(self, cfg, runner_ctx: RunnerContext):
        super().__init__()
        self.cfg       = cfg
        self.classical = ClassicalCNN()
        self.reduce    = nn.Conv2d(32, cfg.N_QUBITS, 1)
        self.q_local   = QuantumLocal(cfg, runner_ctx)
        self.q_global  = QuantumGlobalCommuting(cfg, runner_ctx)
        fc_in = 64 + cfg.q_out() + cfg.q_out()
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        _, x2, x3 = self.classical(x)
        x2  = F.interpolate(x2, size=(self.cfg.Q_RES, self.cfg.Q_RES))
        red = self.reduce(x2)
        ql  = torch.mean(self.q_local(red), dim=(2, 3))
        qg  = self.q_global(red)
        xc  = torch.mean(x3, dim=(2, 3))
        return self.fc(torch.cat([xc, ql, qg], dim=1))


# ============================================================
#  FACTORY FUNCTIONS
# ============================================================

def build_baseline_model(cfg, runner_ctx: RunnerContext) -> QCCNNBaseline:
    model = QCCNNBaseline(cfg, runner_ctx)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_total = (sum(p.numel() for p in model.q_local.parameters()) +
               sum(p.numel() for p in model.q_global.parameters()))
    print(f"[Baseline model] Total params: {total:,}  "
          f"(quantum: {q_total}, {100*q_total/total:.2f}%)")
    return model


def build_commuting_model(cfg, runner_ctx: RunnerContext) -> QCCNNCommuting:
    model = QCCNNCommuting(cfg, runner_ctx)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_total = (sum(p.numel() for p in model.q_local.parameters()) +
               sum(p.numel() for p in model.q_global.parameters()))
    print(f"[Commuting model] Total params: {total:,}  "
          f"(quantum: {q_total}, {100*q_total/total:.2f}%)")
    print(f"  Global circuit: {runner_ctx.n_gate_params_g} gate params, "
          f"{runner_ctx.n_logical_params_g} logical params")
    return model
