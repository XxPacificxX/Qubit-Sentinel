"""Rotated surface code memory experiments and their syndrome samplers.

Two backends share one interface (``SurfaceCodeTask``):

* ``stim``  - circuit-level noise from ``stim.Circuit.generated`` (the research standard).
* ``numpy`` - phenomenological noise (data-qubit bit flips each round plus syndrome
  measurement errors), simulated directly from the check matrix. Used when Stim is
  not installed or cannot load on this machine.

Both produce detection events of shape ``(shots, num_detectors)`` and a logical
observable flip of shape ``(shots, 1)``, plus a PyMatching decoder for that model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pymatching


def rotated_surface_code(d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check matrices of the distance-``d`` rotated surface code.

    Data qubits sit on a d x d grid, index ``i * d + j``. Returns
    ``(h_z, h_x, logical_z, z_centers)``: Z-type checks (detect X errors), X-type
    checks, a Z logical operator (one column of the grid) and the (row, col)
    center of each Z check, used as detector coordinates.
    """
    if d < 3 or d % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")
    n = d * d
    hz, hx, centers = [], [], []
    for i in range(-1, d):
        for j in range(-1, d):
            support = [a * d + b for a in (i, i + 1) for b in (j, j + 1) if 0 <= a < d and 0 <= b < d]
            z_type = (i + j) % 2 == 0
            bulk = 0 <= i <= d - 2 and 0 <= j <= d - 2
            if bulk:
                target = hz if z_type else hx
            elif len(support) == 2 and z_type and i in (-1, d - 1):
                target = hz  # weight-2 Z checks on the top and bottom edges
            elif len(support) == 2 and not z_type and j in (-1, d - 1):
                target = hx  # weight-2 X checks on the left and right edges
            else:
                continue
            row = np.zeros(n, dtype=np.uint8)
            row[support] = 1
            target.append(row)
            if target is hz:
                centers.append((i + 0.5, j + 0.5))
    logical_z = np.zeros(n, dtype=np.uint8)
    logical_z[[i * d for i in range(d)]] = 1  # Z on column 0
    return np.array(hz), np.array(hx), logical_z, np.array(centers)


@dataclass
class SurfaceCodeTask:
    distance: int
    rounds: int
    p: float
    backend: str
    num_detectors: int
    detector_coords: np.ndarray  # (num_detectors, 3): row, col, round
    _impl: object = field(repr=False, default=None)

    def sample(self, shots: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        return self._impl.sample(shots, seed)

    def matching(self) -> pymatching.Matching:
        return self._impl.matching()


class _NumpyPhenomenological:
    def __init__(self, d: int, rounds: int, p: float, q: float):
        self.hz, _, self.logical, self.centers = rotated_surface_code(d)
        self.rounds, self.p, self.q = rounds, p, q

    @property
    def num_detectors(self) -> int:
        return self.hz.shape[0] * self.rounds

    def coords(self) -> np.ndarray:
        nc = self.hz.shape[0]
        t = np.repeat(np.arange(self.rounds), nc)
        xy = np.tile(self.centers, (self.rounds, 1))
        return np.column_stack([xy, t]).astype(np.float32)

    def sample(self, shots: int, seed: int | None):
        rng = np.random.default_rng(seed)
        nq = self.hz.shape[1]
        flips = (rng.random((shots, self.rounds, nq)) < self.p).astype(np.uint8)
        errors = np.cumsum(flips, axis=1, dtype=np.uint8) % 2  # accumulated X errors per round
        syndrome = (errors.astype(np.int32) @ self.hz.T.astype(np.int32)) % 2
        meas = (rng.random(syndrome.shape) < self.q).astype(np.int32)
        meas[:, -1, :] = 0  # final round read out perfectly (data-qubit measurement)
        noisy = syndrome ^ meas
        det = noisy.copy()
        det[:, 1:, :] ^= noisy[:, :-1, :]  # detection events = change between rounds
        obs = (errors[:, -1, :].astype(np.int32) @ self.logical.astype(np.int32)) % 2
        # time-major flattening matches PyMatching's detector order for `repetitions`
        return det.reshape(shots, -1).astype(np.uint8), obs.reshape(shots, 1).astype(np.uint8)

    def matching(self) -> pymatching.Matching:
        w = np.log((1 - self.p) / self.p)
        tw = np.log((1 - self.q) / self.q)
        return pymatching.Matching.from_check_matrix(
            self.hz, weights=w, repetitions=self.rounds, timelike_weights=tw,
            faults_matrix=self.logical.reshape(1, -1),
        )


class _StimCircuit:
    def __init__(self, d: int, rounds: int, p: float):
        import stim

        self.circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=rounds,
            after_clifford_depolarization=p, before_round_data_depolarization=p,
            before_measure_flip_probability=p, after_reset_flip_probability=p,
        )

    @property
    def num_detectors(self) -> int:
        return self.circuit.num_detectors

    def coords(self) -> np.ndarray:
        c = self.circuit.get_detector_coordinates()
        return np.array([c[k][:3] for k in range(self.num_detectors)], dtype=np.float32)

    def sample(self, shots: int, seed: int | None):
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        det, obs = sampler.sample(shots, separate_observables=True)
        return det.astype(np.uint8), obs.astype(np.uint8)

    def matching(self) -> pymatching.Matching:
        dem = self.circuit.detector_error_model(decompose_errors=True)
        return pymatching.Matching.from_detector_error_model(dem)


def stim_available() -> bool:
    try:
        import stim  # noqa: F401
    except Exception:  # ImportError, or a DLL blocked by OS policy
        return False
    return True


def make_task(distance: int = 3, rounds: int | None = None, p: float = 0.03,
              backend: str = "auto", measurement_p: float | None = None) -> SurfaceCodeTask:
    """Build a memory experiment. ``rounds`` defaults to ``distance``."""
    rounds = rounds or distance
    if backend == "auto":
        backend = "stim" if stim_available() else "numpy"
    if backend == "stim":
        impl = _StimCircuit(distance, rounds, p)
    elif backend == "numpy":
        impl = _NumpyPhenomenological(distance, rounds, p, measurement_p if measurement_p is not None else p)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return SurfaceCodeTask(distance, rounds, p, backend, impl.num_detectors, impl.coords(), impl)
