"""Re-score exported XGBoost models on the real test rows and compare with the AUROC recorded during the run.

  python tools/hf/verify_xgb_models.py --pkg OUT --tables DIR --raw results/xgb_raw.csv --tag wide

DIR holds <biosample>_gc_matched.parquet (PxALabs/encode-regulatory-elements v2.1). Prints one line per model; exits non-zero if any
reproduced AUROC differs from the recorded one by more than 5e-4. Run with `python -u`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from xgb_features import kmer_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for a in ("pkg", "tables", "raw", "tag"):
        ap.add_argument(f"--{a}", required=True)
    a = ap.parse_args()
    man = json.load(open(Path(a.pkg) / "manifest.json"))["models"]
    raw = pd.read_csv(a.raw)
    feats, bad = {}, 0
    for m in man:
        bio, seed = m["biosample"], m["seed"]
        if bio not in feats:
            d = pd.read_parquet(Path(a.tables) / f"{bio}_gc_matched.parquet")
            d = d[d.split == "test"]
            feats[bio] = (kmer_features(d.sequence.tolist(), 6), d.label_binary.values.astype(int))
        X, y = feats[bio]
        clf = xgb.XGBClassifier()
        clf.load_model(str(Path(a.pkg) / m["file"]))
        t = time.time()
        auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
        rec = float(raw[(raw.biosample == bio) & (raw.model == f"xgb_k6_{a.tag}") & (raw.seed == seed)].auroc_full.iloc[0])
        ok = abs(auc - rec) < 5e-4
        bad += not ok
        print(f"{bio:8s} seed{seed} n={len(y)} reproduced {auc:.4f} recorded {rec:.4f} diff {auc - rec:+.5f} {'OK' if ok else 'MISMATCH'} ({time.time() - t:.0f}s)")
    print("ALL REPRODUCED" if not bad else f"{bad} MISMATCHES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
