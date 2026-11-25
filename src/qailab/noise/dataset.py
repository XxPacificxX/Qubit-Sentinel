"""Build a (features, fidelity) dataset by running random circuits on a noisy device model."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector, hellinger_fidelity
from qiskit_aer import AerSimulator

from .circuits import random_circuit
from .features import circuit_features


def get_backend(name: str = "FakeManilaV2"):
    """A simulated IBM device (calibration data snapshot) from qiskit-ibm-runtime."""
    from qiskit_ibm_runtime import fake_provider

    try:
        return getattr(fake_provider, name)()
    except AttributeError:
        raise ValueError(f"unknown fake backend {name!r}; try FakeManilaV2, FakeLimaV2, FakeNairobiV2") from None


def build_dataset(num_circuits: int, backend_name: str = "FakeManilaV2", shots: int = 2000,
                  min_qubits: int = 2, max_qubits: int = 5, max_depth: int = 30,
                  target: str = "mirror", seed: int = 0, log=print) -> list[dict]:
    """``target``:

    * ``mirror``    - run C then C^-1 and record P(all zeros). The ideal answer is always
      |0...0>, so the score measures noise alone (mirror-circuit benchmarking).
    * ``hellinger`` - run C and compare its output distribution to the ideal one. More
      realistic, but noisier as a label: near-uniform ideal outputs hide noise.
    """
    if target not in ("mirror", "hellinger"):
        raise ValueError("target must be 'mirror' or 'hellinger'")
    backend = get_backend(backend_name)
    max_qubits = min(max_qubits, backend.num_qubits)
    sim = AerSimulator.from_backend(backend)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(num_circuits):
        n = int(rng.integers(min_qubits, max_qubits + 1))
        depth = int(rng.integers(1, max_depth + 1))
        body = random_circuit(n, depth, rng)
        qc = body
        if target == "mirror":
            qc = body.copy()
            qc.barrier()  # stops the transpiler from cancelling C against C^-1
            qc.compose(body.inverse(), inplace=True)
            ideal = {"0" * n: 1.0}
        else:
            ideal = Statevector(qc).probabilities_dict()
        tqc = transpile(qc.measure_all(inplace=False), backend, optimization_level=1,
                        seed_transpiler=seed + k)
        counts = sim.run(tqc, shots=shots, seed_simulator=seed + k).result().get_counts()
        row = circuit_features(tqc, backend.target, n)
        row["fidelity"] = hellinger_fidelity(ideal, counts)
        row["logical_depth"] = float(depth)
        rows.append(row)
        if (k + 1) % max(num_circuits // 10, 1) == 0:
            log(f"  simulated {k + 1}/{num_circuits} circuits")
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
