"""Neural decoders: detection events -> probability that the logical observable flipped."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TransformerDecoder(nn.Module):
    """Treats every detector as a token.

    A token's embedding = its value (fired / quiet) + a learned per-detector
    embedding + an MLP of its (row, col, round) coordinates. A [CLS] token reads
    out the logical flip.
    """

    def __init__(self, coords: np.ndarray, d_model: int = 64, heads: int = 4, layers: int = 3):
        super().__init__()
        c = torch.as_tensor(coords, dtype=torch.float32)
        c = (c - c.mean(0)) / (c.std(0) + 1e-6)
        self.register_buffer("coords", c)
        self.value = nn.Embedding(2, d_model)
        self.slot = nn.Embedding(c.shape[0], d_model)
        self.pos = nn.Sequential(nn.Linear(c.shape[1], d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, det: torch.Tensor) -> torch.Tensor:
        b = det.shape[0]
        idx = torch.arange(self.coords.shape[0], device=det.device)
        x = self.value(det.long()) + (self.slot(idx) + self.pos(self.coords))[None]
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        return self.head(self.encoder(x)[:, 0]).squeeze(-1)


class MLPDecoder(nn.Module):
    """Small fully connected baseline."""

    def __init__(self, num_detectors: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        dims = [num_detectors] + [hidden] * layers
        blocks = []
        for a, b in zip(dims, dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*blocks, nn.Linear(hidden, 1))

    def forward(self, det: torch.Tensor) -> torch.Tensor:
        return self.net(det.float()).squeeze(-1)


def build_model(kind: str, coords: np.ndarray) -> nn.Module:
    if kind == "transformer":
        return TransformerDecoder(coords)
    if kind == "mlp":
        return MLPDecoder(coords.shape[0])
    raise ValueError(f"unknown model {kind!r}")
