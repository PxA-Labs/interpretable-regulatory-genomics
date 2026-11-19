"""Per-row repeat coverage from UCSC RepeatMasker (hg38 rmsk) for the v2.1 tables.

Why: notebook 08 asks how much of the classification signal is repeat content (Alu / L1 / low complexity), which the
GC and CpG matching in notebooks 03-06 does not control. The v2.1 parquet tables carry no repeat annotation (the Ensembl
genome used is unmasked), so coverage is computed here from the public rmsk track.

Usage:
  python tools/data/repeat_annotation.py --rmsk rmsk.txt.gz --tables DIR --out repeat_fractions.parquet

Output columns: id, biosample, negative_strategy, rep_frac, alu_frac, l1_frac, simple_frac (fractions of the row's bp
covered by the union of the corresponding rmsk intervals; 0-based half-open coordinates, same as the tables).
Only reads the small columns it needs from the tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RMSK_COLS = ["bin", "swScore", "milliDiv", "milliDel", "milliIns", "genoName", "genoStart", "genoEnd", "genoLeft",
             "strand", "repName", "repClass", "repFamily", "repStart", "repEnd", "repLeft", "id"]
CATEGORIES = {
    "rep_frac": lambda d: np.ones(len(d), dtype=bool),
    "alu_frac": lambda d: (d.repFamily == "Alu").values,
    "l1_frac": lambda d: (d.repFamily == "L1").values,
    "simple_frac": lambda d: d.repClass.isin(["Simple_repeat", "Low_complexity"]).values,
}


def merge_intervals(s: np.ndarray, e: np.ndarray):
    o = np.argsort(s, kind="stable")
    s, e = s[o], e[o]
    ends = np.maximum.accumulate(e)
    new = np.ones(len(s), dtype=bool)
    new[1:] = s[1:] > ends[:-1]  # strictly separated -> new merged block
    grp = np.cumsum(new) - 1
    ms = s[new]
    me = np.zeros(len(ms), dtype=np.int64)
    np.maximum.at(me, grp, e)
    return ms, me


def covered_before(ms, me, prefix, x):
    """Total merged length lying left of coordinate x."""
    i = np.searchsorted(ms, x, side="right") - 1
    out = np.zeros(len(x), dtype=np.int64)
    ok = i >= 0
    ii = i[ok]
    out[ok] = prefix[ii] + np.clip(x[ok] - ms[ii], 0, me[ii] - ms[ii])
    return out


def overlap_bp(ms, me, a, b):
    if len(ms) == 0:
        return np.zeros(len(a), dtype=np.int64)
    prefix = np.concatenate([[0], np.cumsum(me - ms)])[:-1]
    return covered_before(ms, me, prefix, b) - covered_before(ms, me, prefix, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmsk", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="*_gc_matched.parquet")
    args = ap.parse_args()

    rm = pd.read_csv(args.rmsk, sep="\t", header=None, names=RMSK_COLS,
                     usecols=["genoName", "genoStart", "genoEnd", "repFamily", "repClass"],
                     dtype={"genoName": str, "repFamily": str, "repClass": str})
    print("rmsk rows", len(rm), "chroms", rm.genoName.nunique())
    merged = {}
    for cat, sel in CATEGORIES.items():
        d = rm[sel(rm)]
        for chrom, g in d.groupby("genoName"):
            merged[(cat, chrom)] = merge_intervals(g.genoStart.values.astype(np.int64), g.genoEnd.values.astype(np.int64))

    parts = []
    for f in sorted(Path(args.tables).glob(args.pattern)):
        t = pd.read_parquet(f, columns=["id", "biosample", "negative_strategy", "chrom", "start", "end"])
        t["start"], t["end"] = t.start.astype(np.int64), t.end.astype(np.int64)
        length = (t.end - t.start).values
        assert (length > 0).all(), f"non-positive interval length in {f.name}"
        miss = set(t.chrom.unique()) - set(rm.genoName.unique())
        assert not miss, f"chrom naming mismatch in {f.name}: {sorted(miss)[:5]}"
        for cat in CATEGORIES:
            col = np.zeros(len(t), dtype=np.float64)
            for chrom, idx in t.groupby("chrom").indices.items():
                ms, me = merged.get((cat, chrom), (np.array([], dtype=np.int64),) * 2)
                col[idx] = overlap_bp(ms, me, t.start.values[idx], t.end.values[idx])
            t[cat] = col / length
        print(f.name, len(t), "mean rep_frac %.3f alu %.3f l1 %.3f simple %.3f" %
              tuple(t[c].mean() for c in CATEGORIES))
        parts.append(t[["id", "biosample", "negative_strategy", *CATEGORIES]])
    out = pd.concat(parts, ignore_index=True)
    assert out[list(CATEGORIES)].apply(lambda s: s.between(0, 1)).all().all()
    out.to_parquet(args.out, index=False)
    print("wrote", args.out, out.shape)


if __name__ == "__main__":
    main()
