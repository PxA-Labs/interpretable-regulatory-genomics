"""Convert notebook-07 `.pt` state dicts to safetensors, verify, and write manifests.

  python tools/hf/export_cnn_weights.py --pt DIR/models --raw results/cnn_raw.csv --out OUT \
      --kernel purvanshj009/irg-07-cnn-tuned --kernel-version 2 --git-sha <sha>

Verification per file: tensors reload bit-identical, and the safetensors model gives bit-identical predictions to the
original `.pt` model on a fixed random batch. Loads `.pt` with weights_only=True (no pickle code execution).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
import model_def  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--kernel-version", required=True)
    ap.add_argument("--git-sha", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    raw = pd.read_csv(a.raw)
    files = sorted(Path(a.pt).glob("*.pt"))
    assert len(files) == len(raw) == 30, f"expected 30 weights and 30 rows, got {len(files)} / {len(raw)}"
    g = torch.Generator().manual_seed(0)
    probe = ["".join(np.random.default_rng(i).choice(list("ACGT"), 1000)) for i in range(8)]
    manifest = []
    for f in files:
        bio, name, seed = re.fullmatch(r"(tcell|k562|gm12878)_(shallow_cnn|attn_cnn)_s(\d)\.pt", f.name).groups()
        seed = int(seed)
        sd = torch.load(f, map_location="cpu", weights_only=True)
        dst = out / "weights" / bio / name / f"seed{seed}.safetensors"
        dst.parent.mkdir(parents=True, exist_ok=True)
        row = raw[(raw.biosample == bio) & (raw.model == name) & (raw.seed == seed)]
        assert len(row) == 1, f.name
        r = row.iloc[0]
        meta = {"biosample": bio, "model": name, "seed": str(seed), "lr": str(r.lr), "source_kernel": a.kernel,
                "source_kernel_version": a.kernel_version, "git_sha": a.git_sha, "format": "state_dict of tools/hf/model_def.py"}
        save_file({k: v.contiguous() for k, v in sd.items()}, str(dst), metadata=meta)
        # verify: tensors identical and predictions identical
        back = load_file(str(dst))
        assert set(back) == set(sd) and all(torch.equal(back[k], sd[k]) for k in sd), f"tensor mismatch {f.name}"
        m1, m2 = model_def.build(name), model_def.build(name)
        m1.load_state_dict(sd)
        m2.load_state_dict(back)
        assert np.array_equal(model_def.predict(m1, probe), model_def.predict(m2, probe)), f"prediction mismatch {f.name}"
        manifest.append({"file": str(dst.relative_to(out)), "sha256": sha256(dst), "biosample": bio, "model": name, "seed": seed,
                         "lr": float(r.lr), "epochs": int(r.epochs), "n_params": int(r.n_params), "auroc_val": round(float(r.auroc_val), 4),
                         "auroc_test": round(float(r.auroc_test), 4), "auroc_matched": round(float(r.auroc_matched), 4)})
    (out / "manifest.json").write_text(json.dumps({"source_kernel": a.kernel, "kernel_version": a.kernel_version, "git_sha": a.git_sha,
                                                  "models": manifest}, indent=1))
    print("verified and exported", len(manifest), "files")


if __name__ == "__main__":
    main()
