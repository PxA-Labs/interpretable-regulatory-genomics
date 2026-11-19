"""Standalone definitions of the IRG CNN baselines (identical to kaggle/kernels/07-cnn-tuned).

Input: DNA string(s) of any length (trained on 1000 bp). Output: probability that the window is a candidate
regulatory element of the biosample the weights were trained on. Predictions average the forward and the
reverse-complement strand, as in training evaluation.

    from safetensors.torch import load_file
    model = build("shallow_cnn"); model.load_state_dict(load_file("weights/k562/shallow_cnn/seed0.safetensors"))
    p = predict(model, ["ACGT..."])
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_LUT = np.full(256, 4, dtype=np.uint8)  # A0 C1 G2 T3, anything else (N) = 4
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i


class ShallowCNN(nn.Module):
    def __init__(self, n_filters: int = 128, k: int = 15):
        super().__init__()
        self.conv = nn.Conv1d(4, n_filters, k, padding=k // 2)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x):
        return self.fc(self.drop(F.relu(self.conv(x)).amax(-1))).squeeze(-1)


class AttnCNN(nn.Module):
    def __init__(self, n_filters: int = 128, k: int = 15, heads: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(4, n_filters, k, padding=k // 2)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_filters, n_filters, 7, padding=3)
        self.attn = nn.MultiheadAttention(n_filters, heads, batch_first=True)
        self.norm = nn.LayerNorm(n_filters)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters * 2, 1)

    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))
        h = F.relu(self.conv2(h)).permute(0, 2, 1)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + a)
        return self.fc(self.drop(torch.cat([h.mean(1), h.amax(1)], -1))).squeeze(-1)


MODELS = {"shallow_cnn": ShallowCNN, "attn_cnn": AttnCNN}


def build(name: str) -> nn.Module:
    return MODELS[name]()


def encode(seqs: list[str]) -> torch.Tensor:
    """(B, L) long tensor, 4 = N. All sequences must have equal length."""
    return torch.from_numpy(np.stack([_LUT[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs])).long()


def onehot(idx: torch.Tensor) -> torch.Tensor:
    return F.one_hot(idx.clamp(max=4), 5)[..., :4].float().permute(0, 2, 1)


def revcomp(idx: torch.Tensor) -> torch.Tensor:
    return torch.where(idx < 4, 3 - idx, idx).flip(-1)


@torch.no_grad()
def predict(model: nn.Module, seqs: list[str], batch_size: int = 1024) -> np.ndarray:
    model.eval()
    X, out = encode(seqs), []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size]
        out.append((torch.sigmoid(model(onehot(xb))) + torch.sigmoid(model(onehot(revcomp(xb))))) / 2)
    return torch.cat(out).numpy()
