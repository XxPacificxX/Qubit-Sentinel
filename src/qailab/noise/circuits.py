"""Random circuit generation for the fidelity dataset."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

ONE_QUBIT = ("h", "x", "sx", "s", "t", "rz", "ry")


def random_circuit(num_qubits: int, depth: int, rng: np.random.Generator,
                   two_qubit_prob: float = 0.35) -> QuantumCircuit:
    """Layered random circuit: each layer pairs up some qubits with CX, the rest get 1q gates."""
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        qubits = list(range(num_qubits))
        rng.shuffle(qubits)
        while qubits:
            if len(qubits) >= 2 and rng.random() < two_qubit_prob:
                qc.cx(qubits.pop(), qubits.pop())
            else:
                q = qubits.pop()
                g = str(rng.choice(ONE_QUBIT))
                if g in ("rz", "ry"):
                    getattr(qc, g)(float(rng.uniform(0, 2 * np.pi)), q)
                else:
                    getattr(qc, g)(q)
    return qc
