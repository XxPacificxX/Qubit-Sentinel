"""Structural and calibration-aware features of a transpiled circuit."""
from __future__ import annotations

import math
from collections import Counter

from qiskit import QuantumCircuit

SKIP = {"barrier", "delay"}


def circuit_features(tqc: QuantumCircuit, target, num_logical_qubits: int) -> dict[str, float]:
    """Features computed from the circuit *after* transpiling to the device.

    Includes the classic "expected success probability" (ESP = product of
    (1 - error) over every gate and readout), which is the usual heuristic the
    model has to beat.
    """
    counts: Counter[str] = Counter()
    gate_err_sum = gate_err_max = readout_err_sum = 0.0
    log_esp = 0.0
    n1 = n2 = 0
    clock: dict[int, float] = {}
    active: set[int] = set()

    for inst in tqc.data:
        name = inst.operation.name
        if name in SKIP:
            continue
        qargs = tuple(tqc.find_bit(q).index for q in inst.qubits)
        active.update(qargs)
        counts[name] += 1
        props = None
        if name in target.operation_names:
            try:
                props = target[name][qargs]
            except KeyError:
                props = None
        err = (props.error if props is not None and props.error is not None else 0.0)
        dur = (props.duration if props is not None and props.duration is not None else 0.0)
        log_esp += math.log(max(1.0 - err, 1e-12))
        if name == "measure":
            readout_err_sum += err
        else:
            gate_err_sum += err
            gate_err_max = max(gate_err_max, err)
            if len(qargs) == 1:
                n1 += 1
            elif len(qargs) >= 2:
                n2 += 1
        start = max((clock.get(q, 0.0) for q in qargs), default=0.0)
        for q in qargs:
            clock[q] = start + dur

    makespan = max(clock.values(), default=0.0)
    decoherence = 0.0
    qprops = target.qubit_properties
    for q in active:
        t2 = getattr(qprops[q], "t2", None) if qprops is not None else None
        if t2:
            decoherence += makespan / t2

    feats = {
        "logical_qubits": float(num_logical_qubits),
        "physical_qubits": float(len(active)),
        "depth": float(tqc.depth(lambda i: i.operation.name not in SKIP)),
        "two_qubit_depth": float(tqc.depth(lambda i: i.operation.num_qubits >= 2 and i.operation.name not in SKIP)),
        "n_1q": float(n1),
        "n_2q": float(n2),
        "gate_err_sum": gate_err_sum,
        "gate_err_max": gate_err_max,
        "readout_err_sum": readout_err_sum,
        "duration_us": makespan * 1e6,
        "decoherence_load": decoherence,
        "log_esp": log_esp,
        "esp": math.exp(log_esp),
    }
    for g in ("sx", "x", "rz", "cx", "ecr", "cz", "measure"):
        feats[f"count_{g}"] = float(counts.get(g, 0))
    return feats
