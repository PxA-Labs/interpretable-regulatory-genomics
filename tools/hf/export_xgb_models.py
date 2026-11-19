"""Arrange notebook-09 / notebook-11 XGBoost models for publication, verify they load, and write manifests.

  python tools/hf/export_xgb_models.py --models DIR/models --raw results/xgb_raw.csv --protocol results/protocol.json \
      --out OUT --kernel purvanshj009/irg-09-xgb-tuned --kernel-version 2 --git-sha <sha> [--tag tuned|wide]

Models are XGBoost's native binary format (.ubj, no pickle). Each file is reloaded and its best_iteration compared with the run log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import xgboost as xgb


def main():
    ap = argparse.ArgumentParser()
    for a in ("models", "raw", "protocol", "out", "kernel", "kernel-version", "git-sha"):
        ap.add_argument(f"--{a}", required=True)
    ap.add_argument("--tag", default="tuned", help="model-name suffix: xgb_k6_<tag> (tuned = notebook 09, wide = notebook 11)")
    a = ap.parse_args()
    out = Path(a.out)
    raw = pd.read_csv(a.raw)
    proto = json.load(open(a.protocol))
    files = sorted(Path(a.models).glob("*.ubj"))
    assert len(files) == 15, f"expected 15 models, got {len(files)}"
    manifest = []
    for f in files:
        bio, seed = re.fullmatch(rf"(tcell|k562|gm12878)_xgb_k6_{a.tag}_s(\d)\.ubj", f.name).groups()
        seed = int(seed)
        dst = out / "weights" / bio / f"seed{seed}.ubj"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)
        clf = xgb.XGBClassifier()
        clf.load_model(str(dst))
        r = raw[(raw.biosample == bio) & (raw.model == f"xgb_k6_{a.tag}") & (raw.seed == seed)]
        assert len(r) == 1, f.name
        r = r.iloc[0]
        assert int(clf.best_iteration) == int(r.best_iteration), f"best_iteration mismatch {f.name}"
        manifest.append({"file": str(dst.relative_to(out)), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "biosample": bio, "seed": seed,
                         "bytes": dst.stat().st_size, "best_iteration": int(r.best_iteration), "auroc_val": round(float(r.auroc_val), 4),
                         "auroc_test": round(float(r.auroc_full), 4), "auroc_gc_cpg_matched": round(float(r.auroc_gc_cpg_matched), 4),
                         "params": (proto["selected_params"][bio] if "selected_params" in proto else proto["selected"][bio]["params"])})
    (out / "manifest.json").write_text(json.dumps({"source_kernel": a.kernel, "kernel_version": a.kernel_version, "git_sha": a.git_sha,
                                                  "features": "6-mer frequencies, 4096 columns, tools/hf/xgb_features.py", "models": manifest}, indent=1))
    print("exported and verified", len(manifest), "models;", round(sum(m["bytes"] for m in manifest) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
