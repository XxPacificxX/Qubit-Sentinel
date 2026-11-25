"""Train a neural decoder and benchmark it against PyMatching (minimum-weight perfect matching)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .codes import make_task
from .models import build_model


@dataclass
class QECConfig:
    distance: int = 3
    rounds: int | None = None
    p: float = 0.03
    backend: str = "auto"
    model: str = "transformer"
    train_shots: int = 200_000
    test_shots: int = 50_000
    epochs: int = 6
    batch_size: int = 512
    lr: float = 1e-3
    eval_ps: tuple[float, ...] = ()
    seed: int = 0
    out_dir: str = "results/qec"


def _predict(model: nn.Module, det: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(det), batch):
            out.append((model(torch.from_numpy(det[i:i + batch])) > 0).numpy())
    return np.concatenate(out).astype(np.uint8)


def _wilson(k: int, n: int) -> float:
    """Half-width of a 95% Wilson interval, for error bars on logical error rates."""
    if n == 0:
        return 0.0
    z, ph = 1.96, k / n
    return z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def benchmark(model: nn.Module, task, shots: int, seed: int) -> dict:
    det, obs = task.sample(shots, seed=seed)
    obs = obs[:, 0]

    matcher = task.matching()
    t0 = time.perf_counter()
    mwpm = matcher.decode_batch(det)[:, 0]
    t_mwpm = time.perf_counter() - t0

    t0 = time.perf_counter()
    nn_pred = _predict(model, det)
    t_nn = time.perf_counter() - t0

    def row(pred, seconds=None):
        k = int((pred != obs).sum())
        r = {"logical_error_rate": k / shots, "ci95": _wilson(k, shots)}
        if seconds is not None:
            r["us_per_shot"] = 1e6 * seconds / shots
        return r

    return {
        "p": task.p,
        "shots": shots,
        "no_decoding": row(np.zeros_like(obs)),
        "pymatching": row(mwpm, t_mwpm),
        "neural": row(nn_pred, t_nn),
    }


def train(cfg: QECConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    task = make_task(cfg.distance, cfg.rounds, cfg.p, cfg.backend)
    log(f"[qec] d={task.distance} rounds={task.rounds} p={task.p} backend={task.backend} "
        f"detectors={task.num_detectors}")

    det, obs = task.sample(cfg.train_shots, seed=cfg.seed)
    n_val = max(cfg.train_shots // 20, 1000)
    x_tr, y_tr = torch.from_numpy(det[n_val:]), torch.from_numpy(obs[n_val:, 0]).float()
    x_va, y_va = det[:n_val], obs[:n_val, 0]

    model = build_model(cfg.model, task.detector_coords)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = cfg.epochs * math.ceil(len(x_tr) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    loss_fn = nn.BCEWithLogitsLoss()
    log(f"[qec] {cfg.model}: {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"{len(x_tr):,} training shots")

    history = []
    t_start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        total = 0.0
        for i in range(0, len(x_tr), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        val_ler = float((_predict(model, x_va) != y_va).mean())
        history.append({"epoch": epoch + 1, "loss": total / len(x_tr), "val_ler": val_ler})
        log(f"  epoch {epoch + 1}/{cfg.epochs}  loss {total / len(x_tr):.4f}  val LER {val_ler:.4f}")
    train_seconds = time.perf_counter() - t_start

    evals = [benchmark(model, task, cfg.test_shots, seed=cfg.seed + 1)]
    for p in cfg.eval_ps:
        if p != cfg.p:
            other = make_task(cfg.distance, cfg.rounds, p, task.backend)
            evals.append(benchmark(model, other, cfg.test_shots, seed=cfg.seed + 2))

    log("\n  p        no-decode   PyMatching          neural")
    for e in evals:
        log(f"  {e['p']:<8} {e['no_decoding']['logical_error_rate']:.4f}      "
            f"{e['pymatching']['logical_error_rate']:.4f}±{e['pymatching']['ci95']:.4f}    "
            f"{e['neural']['logical_error_rate']:.4f}±{e['neural']['ci95']:.4f}")
    log(f"  decode time: PyMatching {evals[0]['pymatching']['us_per_shot']:.1f} µs/shot, "
        f"neural (CPU, batched) {evals[0]['neural']['us_per_shot']:.1f} µs/shot")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"d{task.distance}_r{task.rounds}_p{task.p}_{cfg.model}_{task.backend}"
    torch.save(model.state_dict(), out / f"{tag}.pt")
    result = {"config": asdict(cfg), "backend": task.backend, "rounds": task.rounds,
              "num_detectors": task.num_detectors, "train_seconds": train_seconds,
              "history": history, "evaluations": evals}
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    log(f"[qec] saved {out / tag}.json")
    return result
