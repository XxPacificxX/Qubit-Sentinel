import numpy as np
import pytest
import torch

from qailab.qec.codes import make_task, rotated_surface_code
from qailab.qec.models import build_model


def _gf2_rank(m: np.ndarray) -> int:
    m = m.copy() % 2
    rank = 0
    for col in range(m.shape[1]):
        pivot = next((r for r in range(rank, m.shape[0]) if m[r, col]), None)
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(m.shape[0]):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
    return rank


@pytest.mark.parametrize("d", [3, 5, 7])
def test_surface_code_structure(d):
    hz, hx, lz, centers = rotated_surface_code(d)
    assert hz.shape == ((d * d - 1) // 2, d * d)
    assert hx.shape == ((d * d - 1) // 2, d * d)
    assert len(centers) == hz.shape[0]
    assert not ((hz.astype(int) @ hx.T.astype(int)) % 2).any(), "Z and X checks must commute"
    assert not ((hx.astype(int) @ lz.astype(int)) % 2).any(), "logical Z must commute with X checks"
    assert _gf2_rank(np.vstack([hz, lz])) == _gf2_rank(hz) + 1, "logical Z must not be a stabilizer"
    # every single-qubit X error is detected
    assert hz.any(axis=0).all()


def test_numpy_sampler_matches_pymatching_order():
    task = make_task(3, 3, 0.05, backend="numpy")
    det, obs = task.sample(500, seed=1)
    assert det.shape == (500, task.num_detectors) and obs.shape == (500, 1)
    m = task.matching()
    assert m.num_detectors == task.num_detectors
    batch = m.decode_batch(det)
    # per-shot 2D decoding (PyMatching's documented (checks, rounds) layout) must agree with our flattening
    for k in range(20):
        two_d = det[k].reshape(task.rounds, -1).T
        assert (m.decode(two_d) == batch[k]).all()


def test_pymatching_beats_no_decoding_and_improves_with_distance():
    p = 0.02
    rates = []
    for d in (3, 5):
        task = make_task(d, d, p, backend="numpy")
        det, obs = task.sample(20000, seed=2)
        pred = task.matching().decode_batch(det)
        rates.append((pred != obs).mean())
        assert rates[-1] < obs.mean()
    assert rates[1] < rates[0], "below threshold, a larger code should have a lower logical error rate"


@pytest.mark.parametrize("kind", ["transformer", "mlp"])
def test_models_forward(kind):
    task = make_task(3, 3, 0.03, backend="numpy")
    det, _ = task.sample(8, seed=0)
    model = build_model(kind, task.detector_coords)
    out = model(torch.from_numpy(det))
    assert out.shape == (8,)
