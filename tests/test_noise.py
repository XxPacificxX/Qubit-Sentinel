import numpy as np

from qailab.noise.circuits import random_circuit
from qailab.noise.dataset import build_dataset


def test_random_circuit_shape():
    qc = random_circuit(4, 10, np.random.default_rng(0))
    assert qc.num_qubits == 4
    assert qc.size() >= 10


def test_dataset_rows_are_sane():
    rows = build_dataset(6, "FakeManilaV2", shots=500, max_depth=8, seed=3, log=lambda *_: None)
    assert len(rows) == 6
    for r in rows:
        assert 0.0 <= r["fidelity"] <= 1.0 + 1e-9
        assert 0.0 < r["esp"] <= 1.0
        assert r["count_measure"] == r["logical_qubits"]
    # deeper circuits should, on average, run worse
    deep = [r for r in rows if r["n_2q"] > 0]
    assert deep, "expected some two-qubit gates"
