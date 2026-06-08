"""
qc_circuits.py
==============
Quantum circuit builders for the TPCQ-Net pipeline.

Provides two global circuit variants:
  - build_global_circuit()           Original RY+CX ring (unstructured)
  - build_global_circuit_commuting() Commuting X-generator ansatz (Bowles et al.)

The commuting circuit enables parallel gradient estimation: all gradients
can be computed from N_QUBITS circuits instead of 2 * n_gate_params circuits.

Public API
----------
  build_local_circuit(n_qubits, n_layers)
  build_global_circuit(n_qubits, n_layers, n_inputs)
  build_global_circuit_commuting(n_qubits, n_inputs, max_weight)
  build_circuits(cfg)
  build_circuits_commuting(cfg)
"""

import math
from itertools import combinations
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# ============================================================
#  LOCAL CIRCUIT  —  quantum convolutional kernel (unchanged)
# ============================================================

def build_local_circuit(n_qubits: int, n_layers: int):
    """
    Local quantum circuit — quantum convolutional kernel.

    Per layer:
        RY(x[i])               i in range(n_qubits)  — patch encoding (not trained)
        RY(w[l * n_qubits+i])  i in range(n_qubits)  — trainable weight
        CX(i, (i+1) % n_qubits)                       — ring entanglement

    Returns: qc, x (ParameterVector), w (ParameterVector)
    """
    x  = ParameterVector('x', length=n_qubits)
    w  = ParameterVector('w', length=n_layers * n_qubits)
    qc = QuantumCircuit(n_qubits)

    for l in range(n_layers):
        for i in range(n_qubits):
            qc.ry(x[i], i)
        for i in range(n_qubits):
            qc.ry(w[l * n_qubits + i], i)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)

    return qc, x, w


# ============================================================
#  GLOBAL CIRCUIT (BASELINE)  —  data re-uploading encoder
# ============================================================

def build_global_circuit(n_qubits: int, n_layers: int, n_inputs: int):
    """
    Global quantum circuit — data re-uploading encoder (original, unstructured).

    Per layer:
        RY(x[idx++])           for inputs_per_layer values  — fresh chunk (not trained)
        RY(w[l * n_qubits+i])  i in range(n_qubits)         — trainable weight
        CX(i, (i+1) % n_qubits)                             — ring entanglement

    Backward cost: 2 * n_layers * n_qubits circuit evaluations per sample.

    Returns: qc, x (ParameterVector), w (ParameterVector)
    """
    x                = ParameterVector('x', length=n_inputs)
    w                = ParameterVector('w', length=n_layers * n_qubits)
    qc               = QuantumCircuit(n_qubits)
    inputs_per_layer = math.ceil(n_inputs / n_layers)
    idx              = 0

    for l in range(n_layers):
        for _ in range(inputs_per_layer):
            if idx < n_inputs:
                qc.ry(x[idx], idx % n_qubits)
                idx += 1
        for i in range(n_qubits):
            qc.ry(w[l * n_qubits + i], i)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)

    return qc, x, w


# ============================================================
#  COMMUTING GENERATOR UTILITIES  (Bowles et al. 2025)
# ============================================================

def _cyclic_permutations(s: str) -> list:
    """All unique cyclic permutations of string s."""
    n = len(s)
    seen = set()
    result = []
    for shift in range(n):
        rotated = s[shift:] + s[:shift]
        if rotated not in seen:
            seen.add(rotated)
            result.append(rotated)
    return result


def build_commuting_gen_classes(n_qubits: int, max_weight: int) -> list:
    """
    Build equivariant symmetry classes of X-product Pauli generators.

    Each class contains all cyclic permutations of a seed Pauli string.
    Generators within and across classes all commute: [X...X, X...X] = 0
    because X operators always commute with each other.

    All generators anticommute with at least one Z_r in H = sum Z_r,
    ensuring non-zero gradients (the key condition from the paper).

    Returns
    -------
    list of classes, each class = list of (pauli_word_str, qubit_wires_list)
    E.g. for N=3, max_weight=2:
      Class 0: [('X',[2]), ('X',[1]), ('X',[0])]        — 1-body
      Class 1: [('XX',[1,2]), ('XX',[0,1]), ('XX',[0,2])] — 2-body
    """
    seen = set()
    classes = []
    for weight in range(1, max_weight + 1):
        for pos in combinations(range(n_qubits), weight):
            seed = ['I'] * n_qubits
            for p in pos:
                seed[p] = 'X'
            canonical = min(_cyclic_permutations(''.join(seed)))
            if canonical in seen:
                continue
            seen.add(canonical)
            cls = []
            for perm in _cyclic_permutations(canonical):
                word  = perm.replace('I', '')
                wires = [i for i, c in enumerate(perm) if c == 'X']
                if word:
                    cls.append((word, wires))
            if cls:
                classes.append(cls)
    return classes


def _add_pauli_rot_native(qc: QuantumCircuit, word: str, wires: list, param):
    """
    Append exp(-i param/2 * P) to qc using native CNOT + Rz decomposition.

    This is the standard basis-change → CNOT-ladder → Rz → undo construction.
    Compatible with Qiskit's parameter-shift rule (two distinct eigenvalues ±1
    of the Pauli generator, so shift = π/2).

    Parameters
    ----------
    word  : Pauli word string e.g. 'XX', 'XYZ'
    wires : list of qubit indices (same length as word)
    param : Qiskit Parameter or float
    """
    # Basis change in
    for p, q in zip(word, wires):
        if p == 'X':
            qc.h(q)
        elif p == 'Y':
            qc.rx(-math.pi / 2, q)
    # CNOT ladder
    for i in range(len(wires) - 1):
        qc.cx(wires[i], wires[i + 1])
    # Rz rotation
    qc.rz(param, wires[-1])
    # Undo CNOT ladder
    for i in reversed(range(len(wires) - 1)):
        qc.cx(wires[i], wires[i + 1])
    # Undo basis change
    for p, q in zip(word, wires):
        if p == 'X':
            qc.h(q)
        elif p == 'Y':
            qc.rx(math.pi / 2, q)


# ============================================================
#  GLOBAL CIRCUIT (COMMUTING)  —  Bowles et al. ansatz
# ============================================================

def build_global_circuit_commuting(n_qubits: int, n_inputs: int, max_weight: int = 2):
    """
    Global quantum circuit with MUTUALLY COMMUTING generators.

    Architecture
    ------------
    V(x):  data encoding via RY rotations (cycling through qubits)
    U(w):  PauliRot gates on symmetrized X-product generators (one param per GATE)
           All generators [G_k, G_l] = 0  →  parallel gradient estimation possible

    Observable (used externally): H = (Z_0 + Z_1 + ... + Z_{n-1}) / n_qubits
    Each Z_r anticommutes with every X-product generator touching qubit r,
    giving non-zero gradients.

    Gradient cost (parallel method): n_qubits circuits per sample
    Gradient cost (param-shift): 2 * n_gate_params circuits per sample

    Parameters
    ----------
    n_qubits   : number of qubits
    n_inputs   : number of data inputs to encode
    max_weight : max Pauli weight of generators (1=single-qubit, 2=two-body, ...)

    Returns
    -------
    qc          : QuantumCircuit
    x           : ParameterVector (data inputs, not trained)
    w           : ParameterVector (one entry per gate, all trained)
    gen_classes : list of symmetry classes (metadata for parallel gradient)
    gate_to_class: list mapping gate index -> logical class index
    """
    gen_classes   = build_commuting_gen_classes(n_qubits, max_weight)
    all_gates     = []
    gate_to_class = []
    for j, cls in enumerate(gen_classes):
        for word, wires in cls:
            all_gates.append((word, wires))
            gate_to_class.append(j)

    n_gate_params    = len(all_gates)
    n_logical_params = len(gen_classes)

    x  = ParameterVector('x', length=n_inputs)
    w  = ParameterVector('w', length=n_gate_params)
    qc = QuantumCircuit(n_qubits)

    # V: data encoding (RY, cycling through qubits)
    for idx in range(n_inputs):
        qc.ry(x[idx], idx % n_qubits)

    # U(w): one commuting PauliRot gate per circuit parameter
    for k, (word, wires) in enumerate(all_gates):
        _add_pauli_rot_native(qc, word, wires, w[k])

    return qc, x, w, gen_classes, gate_to_class


# ============================================================
#  CONVENIENCE BUILDERS
# ============================================================

def build_circuits(cfg) -> dict:
    """
    Build baseline circuits (local + original global) from a cfg object.

    Expects cfg to have: N_QUBITS, N_Q_LAYERS, N_Q_LAYERS_GLOBAL, n_inputs()
    """
    qc_l, x_l, w_l = build_local_circuit(cfg.N_QUBITS, cfg.N_Q_LAYERS)
    qc_g, x_g, w_g = build_global_circuit(
        cfg.N_QUBITS, cfg.N_Q_LAYERS_GLOBAL, cfg.n_inputs()
    )
    return dict(
        qc_local=qc_l, x_local=x_l, w_local=w_l,
        qc_global=qc_g, x_global=x_g, w_global=w_g,
    )


def build_circuits_commuting(cfg) -> dict:
    """
    Build circuits for the commuting-generator variant.

    Expects cfg to have: N_QUBITS, N_Q_LAYERS, n_inputs(), MAX_WEIGHT (optional)
    """
    max_w = getattr(cfg, 'MAX_WEIGHT', 2)
    qc_l, x_l, w_l = build_local_circuit(cfg.N_QUBITS, cfg.N_Q_LAYERS)
    qc_g, x_g, w_g, gen_classes, gate_to_class = build_global_circuit_commuting(
        cfg.N_QUBITS, cfg.n_inputs(), max_weight=max_w
    )
    return dict(
        qc_local=qc_l, x_local=x_l, w_local=w_l,
        qc_global=qc_g, x_global=x_g, w_global=w_g,
        gen_classes=gen_classes, gate_to_class=gate_to_class,
    )
