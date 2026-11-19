"""k-mer frequency features used by the IRG XGBoost baselines (identical to kaggle/kernels/09-xgb-tuned).

    from xgb_features import kmer_features
    X = kmer_features(["ACGT" * 250], k=6)          # (n, 4096) float32, each row sums to 1 (windows containing N are skipped)
"""
from __future__ import annotations

import numpy as np

_LUT = np.full(256, -1, dtype=np.int64)
for _i, _c in enumerate("ACGT"):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i


def kmer_features(seqs: list[str], k: int = 6) -> np.ndarray:
    X = np.zeros((len(seqs), 4**k), dtype=np.float32)
    for r, s in enumerate(seqs):
        a = _LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
        n = len(a) - k + 1
        idx, ok = np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool)
        for j in range(k):
            seg = a[j : j + n]
            ok &= seg >= 0
            idx = idx * 4 + np.where(seg >= 0, seg, 0)
        c = np.bincount(idx[ok], minlength=4**k).astype(np.float32)
        X[r] = c / max(c.sum(), 1.0)
    return X
