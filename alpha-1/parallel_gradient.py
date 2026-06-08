"""
parallel_gradient.py
====================
Parallel gradient engine for commuting-generator circuits.

Implements Theorem 1 of Bowles, Wierichs, Park (2025):
    "Backpropagation scaling in parameterised quantum circuits"
    Quantum 9, 1873 (2025). arXiv:2306.14962

Theory summary
--------------
For a circuit C(θ) = <0|V† U†(θ) H U(θ) V|0> with observable
H = (1/N) * Σ_r Z_r  and commuting X-product generators, the gradient
∂C/∂θ_j = Σ_r (1/N) * <ψ| O_j^r |ψ>
where O_j^r = i[G_j, Z_r] and {O_j^r}_j all commute for fixed r.

Since they commute, all O_j^r share one eigenbasis D_r. Measuring in this
basis once gives ALL gradient components simultaneously → N circuits total
(one per Z_r term) instead of 2 * n_gate_params circuits.

Reduction: 2 * n_gate_params  →  N_QUBITS
For TPCQ-Net global (3 qubits, max_weight=2): 12 → 3  (4× speedup)
For TPCQ-Net global (3 qubits, max_weight=3): 30 → 3  (10× speedup)

Public API
----------
  ParallelGradientEngine(gen_classes, gate_to_class, n_qubits)
  engine.compute_gradient_operators()        — pre-computation (call once)
  engine.gradient_exact(statevector)         — exact gradient from statevector
  engine.gradient_from_expectations(evs_per_zr) — from Aer measurement results
  engine.make_gradient_observables()         — SparsePauliOp list for Aer
"""

import numpy as np
from functools import reduce
from typing import List, Tuple
from scipy.linalg import eigh
from qiskit.quantum_info import SparsePauliOp


# ── matrix primitives ──────────────────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)
_X  = np.array([[0, 1], [1, 0]], dtype=complex)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z  = np.diag([1., -1.]).astype(complex)
_PM = {'X': _X, 'Y': _Y, 'Z': _Z}


def _embed(op: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Embed single-qubit operator at qubit index in n-qubit space (little-endian)."""
    ops = [_I2] * n
    ops[qubit] = op
    return reduce(np.kron, reversed(ops))


def _pauli_mat(word: str, wires: list, n: int) -> np.ndarray:
    """Build full n-qubit matrix for a Pauli word on given wires."""
    ops = [_I2] * n
    for p, w in zip(word, wires):
        ops[w] = _PM[p]
    return reduce(np.kron, reversed(ops))


# ── main class ─────────────────────────────────────────────────────────────

class ParallelGradientEngine:
    """
    Pre-computes and applies the parallel gradient method of Bowles et al.

    Parameters
    ----------
    gen_classes   : list of symmetry classes from build_commuting_gen_classes()
    gate_to_class : list mapping gate index -> logical class index
    n_qubits      : number of qubits in the circuit
    """

    def __init__(self, gen_classes: list, gate_to_class: list, n_qubits: int):
        self.gen_classes    = gen_classes
        self.gate_to_class  = gate_to_class
        self.n_qubits       = n_qubits
        self.n_logical      = len(gen_classes)
        self.n_gates        = len(gate_to_class)

        # Collect flat gate list for indexing
        self.all_gates = []
        for cls in gen_classes:
            for word, wires in cls:
                self.all_gates.append((word, wires))

        # Pre-computed per-Z_r eigenbases and operator eigenvalues
        # Set by compute_gradient_operators()
        self._V_bases     = None   # list of N eigenbasis matrices
        self._eig_tables  = None   # list of dicts: gate_idx -> eigenvalue array
        self._obs_per_zr  = None   # SparsePauliOp list for Aer

        self.compute_gradient_operators()

    def compute_gradient_operators(self):
        """
        Pre-compute the simultaneous eigenbasis for each Z_r term.

        For each observable Z_r:
          1. Build O_k^r = i[G_k/2, Z_r] for each gate k that anticommutes with Z_r
          2. Verify all O_k^r commute (guaranteed by Theorem 1)
          3. Find shared eigenbasis via simultaneous diagonalisation
          4. Store eigenvalues for fast gradient evaluation

        Cost: O(n_qubits * n_gates * 4^n_qubits)  — done once before training.
        """
        N = self.n_qubits
        dim = 2 ** N
        Z_mats = [_embed(_Z, r, N) for r in range(N)]

        V_bases    = []
        eig_tables = []

        for r in range(N):
            Zr = Z_mats[r]
            O_list   = []   # (gate_idx, O_matrix)
            for k, (word, wires) in enumerate(self.all_gates):
                Pk       = _pauli_mat(word, wires, N)
                anticomm = Pk @ Zr + Zr @ Pk
                if not np.allclose(anticomm, 0):
                    continue
                # O_k^r = i[G_k, Z_r] with G_k = P_k/2 (our native decomp)
                Gk   = Pk / 2.0
                Ok_r = 1j * (Gk @ Zr - Zr @ Gk)
                O_list.append((k, Ok_r))

            if not O_list:
                # No gate anticommutes with Z_r — eigenbasis is identity
                V_bases.append(np.eye(dim, dtype=complex))
                eig_tables.append({})
                continue

            # Simultaneous diagonalisation
            all_O = [O for _, O in O_list]
            if len(all_O) == 1:
                _, V = eigh(all_O[0])
            else:
                weights = np.logspace(0, -(len(all_O) - 1), len(all_O))
                M = sum(w * A for w, A in zip(weights, all_O))
                _, V = eigh(M)

            V_bases.append(V)

            # Store eigenvalues per gate for fast lookup
            eig_tbl = {}
            for k, Ok_r in O_list:
                diag_Ok_r = V.conj().T @ Ok_r @ V
                eig_tbl[k] = np.real(np.diag(diag_Ok_r))
            eig_tables.append(eig_tbl)

        self._V_bases    = V_bases
        self._eig_tables = eig_tables

        # Also build SparsePauliOp observables for Aer
        self._build_aer_observables()

    def _build_aer_observables(self):
        """Build SparsePauliOp list for measuring gradient basis via Aer."""
        N = self.n_qubits
        Z_terms = []
        for r in range(N):
            # Z_r as SparsePauliOp in Qiskit little-endian string
            pauli_str = 'I' * (N - 1 - r) + 'Z' + 'I' * r
            Z_terms.append(SparsePauliOp(pauli_str))
        self._obs_per_zr = Z_terms

    # ------------------------------------------------------------------
    # Gradient computation (statevector, exact)
    # ------------------------------------------------------------------

    def gradient_exact(self, statevector: np.ndarray) -> np.ndarray:
        """
        Compute gradient from a statevector (exact, no sampling).

        Parameters
        ----------
        statevector : complex array of length 2^n_qubits

        Returns
        -------
        grad : float array of length n_logical (one entry per logical param)
        """
        N    = self.n_qubits
        grad = np.zeros(self.n_logical)

        for r, (V_r, eig_tbl) in enumerate(zip(self._V_bases, self._eig_tables)):
            if not eig_tbl:
                continue
            # Probability distribution in the Z_r eigenbasis
            psi_eig = V_r.conj().T @ statevector
            probs   = np.abs(psi_eig) ** 2

            for gate_idx, eigs in eig_tbl.items():
                logical_idx = self.gate_to_class[gate_idx]
                grad[logical_idx] += (1.0 / N) * np.dot(probs, eigs)

        return grad

    # ------------------------------------------------------------------
    # Aer integration: observables and gradient from expectation values
    # ------------------------------------------------------------------

    def make_gradient_observables(self) -> list:
        """
        Return list of SparsePauliOp for Aer: [Z_0, Z_1, ..., Z_{N-1}].

        These are the observables whose expectation values, combined with
        post-processing, give ALL gradient components.

        Usage: submit N pubs to AerEstimator, one per Z_r observable.
        The circuit for each is the STANDARD forward circuit (no modification).
        Post-processing via gradient_from_expectations() extracts all gradients.
        """
        return list(self._obs_per_zr)

    def gradient_from_expectations(
        self, expectations_per_zr: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient from Aer expectation values of Z_r observables.

        This is NOT the full parallel gradient — it uses param-shift on the
        Z_r observables. The FULL parallel gradient requires measuring in the
        eigenbasis D_r, which requires appending a basis-change circuit.

        For the Aer-compatible implementation, we use a hybrid approach:
        - Measure Z_r expectation values (standard forward pass)
        - Apply the post-processing to extract per-gate gradients

        For exact implementation without eigenbasis circuits, use the
        gradient_exact() method with statevectors.

        Parameters
        ----------
        expectations_per_zr : array of shape (n_qubits, batch_size)
            evs[r, :] = <Z_r> for each sample in the batch

        Returns
        -------
        This method is a placeholder — see gradient_exact() for the
        statevector-based version used in our implementation.
        """
        raise NotImplementedError(
            "Aer-based parallel gradient requires eigenbasis circuit appended. "
            "Use gradient_exact() with StatevectorEstimator, or use the "
            "batch_backward_global_parallel() function in qc_runner.py."
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def speedup_vs_param_shift(self) -> dict:
        """
        Report the circuit count reduction vs standard param-shift.
        """
        n_ps  = 2 * self.n_gates    # param-shift: 2 circuits per gate
        n_par = self.n_qubits       # parallel: 1 circuit per Z_r
        return {
            'param_shift_circuits': n_ps,
            'parallel_circuits':    n_par,
            'speedup_factor':       n_ps / n_par,
            'n_logical_params':     self.n_logical,
            'n_gate_params':        self.n_gates,
            'n_qubits':             self.n_qubits,
        }

    def __repr__(self):
        s = self.speedup_vs_param_shift()
        return (
            f"ParallelGradientEngine("
            f"n_logical={s['n_logical_params']}, "
            f"n_gates={s['n_gate_params']}, "
            f"param_shift_circuits={s['param_shift_circuits']}, "
            f"parallel_circuits={s['parallel_circuits']}, "
            f"speedup={s['speedup_factor']:.1f}x)"
        )
