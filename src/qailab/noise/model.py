"""Fidelity predictor vs. the expected-success-probability (ESP) heuristic."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .dataset import build_dataset, load_csv, save_csv

TARGET = "fidelity"


@dataclass
class NoiseConfig:
    num_circuits: int = 600
    backend: str = "FakeManilaV2"
    shots: int = 2000
    max_depth: int = 30
    target: str = "mirror"
    seed: int = 0
    reuse_dataset: bool = False
    out_dir: str = "results/noise"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)),
            "spearman": _spearman(y, pred)}


def run(cfg: NoiseConfig, log=print) -> dict:
    out = Path(cfg.out_dir)
    tag = f"{cfg.backend}_{cfg.target}"
    data_path = out / f"dataset_{tag}.csv"
    if cfg.reuse_dataset and data_path.exists():
        rows = load_csv(data_path)
        log(f"[noise] loaded {len(rows)} circuits from {data_path}")
    else:
        log(f"[noise] simulating {cfg.num_circuits} random {cfg.target} circuits on {cfg.backend} "
            f"({cfg.shots} shots each)")
        rows = build_dataset(cfg.num_circuits, cfg.backend, cfg.shots, max_depth=cfg.max_depth,
                             target=cfg.target, seed=cfg.seed, log=log)
        save_csv(rows, data_path)

    names = [k for k in rows[0] if k != TARGET]
    X = np.array([[r[k] for k in names] for r in rows])
    y = np.array([r[TARGET] for r in rows])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=cfg.seed)
    esp_col = names.index("esp")

    # Baseline: ESP, calibrated onto the fidelity scale with a monotone fit on the training split
    iso = IsotonicRegression(out_of_bounds="clip").fit(X_tr[:, esp_col], y_tr)
    esp_raw = X_te[:, esp_col]
    esp_cal = iso.predict(esp_raw)

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=cfg.seed)
    model.fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 0, 1)

    imp = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=cfg.seed)
    top = sorted(zip(names, imp.importances_mean), key=lambda t: -t[1])[:8]

    result = {
        "config": asdict(cfg),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "fidelity_range": [float(y.min()), float(y.max())],
        "esp_raw": _metrics(y_te, esp_raw),
        "esp_calibrated": _metrics(y_te, esp_cal),
        "gradient_boosting": _metrics(y_te, pred),
        "top_features": [{"feature": n, "importance": float(v)} for n, v in top],
    }

    log(f"\n  model                 MAE      R²       Spearman   (test set, {len(y_te)} circuits)")
    for key, label in (("esp_raw", "ESP (raw)"), ("esp_calibrated", "ESP (calibrated)"),
                       ("gradient_boosting", "gradient boosting")):
        m = result[key]
        log(f"  {label:<20}  {m['mae']:.4f}   {m['r2']:+.3f}   {m['spearman']:.3f}")
    log("  most useful features: " + ", ".join(f"{n} ({v:.3f})" for n, v in top[:5]))

    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{tag}.json").write_text(json.dumps(result, indent=2))
    log(f"[noise] saved {out / f'results_{tag}.json'}")
    return result
