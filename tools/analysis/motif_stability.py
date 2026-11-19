"""Recompute cross-seed motif stability and between-biosample Jaccard from motif_pairs_q05.csv (notebook 10 output).

Notebook 10 v2 divided by the number of seeds *that had a hit* instead of all seeds; this script uses all seeds (default 5).
  python tools/analysis/motif_stability.py --pairs motif_pairs_q05.csv --out DIR [--n-seeds 5]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BIOS = ["tcell", "k562", "gm12878"]
ARCHS = ["shallow_cnn", "attn_cnn"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    a = ap.parse_args()
    p = pd.read_csv(a.pairs)
    p["tf"] = p.motif_name.map(lambda n: re.split(r"::|\(", n)[0].upper())
    rows = [{"biosample": b, "model": m, "tf": tf, "n_seeds_significant": int(gg.seed.nunique()), "n_seeds": a.n_seeds,
             "n_filters": int(gg["filter"].nunique())}
            for (b, m), g in p.groupby(["biosample", "model"]) for tf, gg in g.groupby("tf")]
    st = pd.DataFrame(rows).sort_values(["biosample", "model", "n_seeds_significant"], ascending=[True, True, False])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st.to_csv(out / "motif_stability.csv", index=False)
    stable = st[st.n_seeds_significant >= int(np.ceil(0.6 * a.n_seeds))]
    sets = {k: set(g.tf) for k, g in stable.groupby(["biosample", "model"])}
    jac = []
    for m in ARCHS:
        for i, b1 in enumerate(BIOS):
            for b2 in BIOS[i + 1:]:
                s1, s2 = sets.get((b1, m), set()), sets.get((b2, m), set())
                jac.append({"model": m, "pair": f"{b1}-{b2}", "n1": len(s1), "n2": len(s2), "shared": len(s1 & s2),
                            "jaccard": len(s1 & s2) / max(1, len(s1 | s2))})
    pd.DataFrame(jac).to_csv(out / "motif_jaccard_between_biosamples.csv", index=False)
    print(st[st.n_seeds_significant >= 2].to_string(index=False))


if __name__ == "__main__":
    main()
