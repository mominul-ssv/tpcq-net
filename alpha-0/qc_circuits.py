"""
qc_circuits.py
==============
Generic quantum circuit builders for the QC-CNN parallel model.

No configuration is hardcoded here — all parameters are passed explicitly.
CFG lives in the notebook.

Public API:
  build_local_circuit(n_qubits, n_layers)              -> qc, x, w
  build_global_circuit(n_qubits, n_layers, n_inputs)   -> qc, x, w
  build_circuits(cfg)                                  -> dict
"""

import math
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# ============================================================
#  LOCAL CIRCUIT  —  quantum convolutional kernel
# ============================================================

def build_local_circuit(n_qubits: int, n_layers: int):
    """
    Local quantum circuit — quantum convolutional kernel.

    Encodes n_qubits patch values, re-encoding them identically each layer.
    Deeper layers extract increasingly non-linear features from the same patch.

    Per layer:
        RY(x[i])              i in range(n_qubits)  — patch encoding  (NOT trained)
        RY(w[l * n_qubits+i]) i in range(n_qubits)  — trainable weight
        CX(i, (i+1) % n_qubits)                      — ring entanglement

    Args:
        n_qubits : number of qubits (= patch size)
        n_layers : number of variational layers

    Returns:
        qc (QuantumCircuit), x (ParameterVector), w (ParameterVector)
    """
    x  = ParameterVector('x', length=n_qubits)
    w  = ParameterVector('w', length=n_layers * n_qubits)
    qc = QuantumCircuit(n_qubits)

    for l in range(n_layers):
        for i in range(n_qubits):
            qc.ry(x[i], i)                       # data encoding
        for i in range(n_qubits):
            qc.ry(w[l * n_qubits + i], i)        # trainable ansatz
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)         # ring entanglement

    return qc, x, w


# ============================================================
#  GLOBAL CIRCUIT  —  data re-uploading encoder
# ============================================================

def build_global_circuit(n_qubits: int, n_layers: int, n_inputs: int):
    """
    Global quantum circuit — data re-uploading encoder.

    n_layers controls depth only; ALL n_inputs are always encoded.
    Inputs are split into ceil(n_inputs / n_layers) per layer, cycling qubits.
    Each layer sees a genuinely fresh chunk of the feature map.

    Per layer:
        RY(x[idx++])          for inputs_per_layer values  — new chunk (NOT trained)
        RY(w[l * n_qubits+i]) i in range(n_qubits)         — trainable weight
        CX(i, (i+1) % n_qubits)                             — ring entanglement

    Args:
        n_qubits : number of qubits
        n_layers : number of variational layers (depth control)
        n_inputs : total number of input features to encode

    Returns:
        qc (QuantumCircuit), x (ParameterVector), w (ParameterVector)
    """
    x                = ParameterVector('x', length=n_inputs)
    w                = ParameterVector('w', length=n_layers * n_qubits)
    qc               = QuantumCircuit(n_qubits)
    inputs_per_layer = math.ceil(n_inputs / n_layers)
    idx              = 0

    for l in range(n_layers):
        for _ in range(inputs_per_layer):
            if idx < n_inputs:
                qc.ry(x[idx], idx % n_qubits)   # fresh chunk encoding
                idx += 1
        for i in range(n_qubits):
            qc.ry(w[l * n_qubits + i], i)        # trainable ansatz
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)         # ring entanglement

    return qc, x, w


# ============================================================
#  CONVENIENCE: build both circuits from a cfg object
# ============================================================

def build_circuits(cfg) -> dict:
    """
    Build both circuits using values from a cfg object.

    Expects cfg to expose:
        N_QUBITS, N_Q_LAYERS, N_Q_LAYERS_GLOBAL, n_inputs()

    Returns:
        dict with keys: qc_local, x_local, w_local,
                        qc_global, x_global, w_global
    """
    qc_l, x_l, w_l = build_local_circuit(cfg.N_QUBITS, cfg.N_Q_LAYERS)
    qc_g, x_g, w_g = build_global_circuit(
        cfg.N_QUBITS, cfg.N_Q_LAYERS_GLOBAL, cfg.n_inputs()
    )
    return dict(
        qc_local=qc_l,  x_local=x_l,  w_local=w_l,
        qc_global=qc_g, x_global=x_g, w_global=w_g,
    )
