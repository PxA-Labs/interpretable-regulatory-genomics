"""Wave 1 analysis (W1.1 uncertainty, W1.2 negative-contamination probe) from per-window predictions.

  python tools/analysis/wave1_uncertainty.py --preds DIR --tables DIR --out OUT [--B 1000]

Inputs: DIR/preds_<biosample>.parquet from wave1_predictions.py; the v2.1 `<biosample>_gc_matched.parquet` tables (coordinates for W1.2).
Hypotheses and pass/fail rules are in docs/research/novelty_and_next_experiments.md (W1.1, W1.2); they were fixed before this ran.

W1.1  For every biosample and test set (full, non-homologous, GC+CpG matched, GC+CpG+repeat matched) and for each model *family*. Point estimates are
      reported both as the mean of the 5 per-seed AUROCs and as the AUROC of the seed-averaged probability (the *ensemble*); all intervals use the
      ensemble, which is cheap enough to bootstrap 1,000 times:
      * per-chromosome AUROC;
      * paired differences between families with three intervals: (a) window bootstrap stratified by chromosome (sampling noise within the test
        chromosomes), (b) chromosome-cluster bootstrap (resample the 4 test chromosomes with replacement, then windows within; only 4 units, so
        treat as indicative), (c) leave-one-chromosome-out range of the difference.
W1.2  Negatives of biosample A that overlap a positive (cCRE) of biosample B or C. Compares the model's scores on overlapping vs clean negatives
      with a within-(GC, CpG)-cell permutation test so that composition cannot explain a difference; reports AUROC/AUPRC before and after removing them.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

BIOS = ["tcell", "k562", "gm12878"]
FAMILIES = ["xgb_wide", "xgb_tuned", "shallow_cnn", "attn_cnn", "comp3_lr"]
PAIRS = [("shallow_cnn", "xgb_wide"), ("attn_cnn", "xgb_wide"), ("xgb_wide", "xgb_tuned"), ("xgb_wide", "comp3_lr")]
SETS = {"full": None, "non_homologous": "nh", "gc_cpg_matched": "m_gc_cpg", "gc_cpg_rep_matched": "m_gc_cpg_rep"}


def fast_auc(y, p):
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def family_cols(P, fam):
    return [c for c in P.columns if c == fam or (c.startswith(fam + "_s") and c[len(fam) + 2:].isdigit())]


def fam_auc(P, idx, fam):
    y = P.y.values[idx]
    return float(np.mean([fast_auc(y, P[c].values[idx]) for c in family_cols(P, fam)]))


def ens(P, fam):
    """Seed-averaged probability of a model family."""
    return P[family_cols(P, fam)].mean(axis=1).values


def select(P, key):
    if key is None:
        return np.arange(len(P))
    if key == "nh":
        return np.where(P.homology_flag.values == 0)[0]
    return np.where(P[key].values)[0]


def strat_boot(P, idx, rng):
    """Window bootstrap, resampling inside each chromosome."""
    ch = P.chrom.values[idx]
    out = []
    for c in np.unique(ch):
        ii = idx[ch == c]
        out.append(rng.choice(ii, len(ii), replace=True))
    return np.concatenate(out)


def cluster_boot(P, idx, rng):
    """Resample chromosomes with replacement, then windows within each drawn chromosome."""
    ch = P.chrom.values[idx]
    chroms = np.unique(ch)
    out = []
    for c in rng.choice(chroms, len(chroms), replace=True):
        ii = idx[ch == c]
        out.append(rng.choice(ii, len(ii), replace=True))
    return np.concatenate(out)


def ci(v):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def w11(preds, out, B, seed):
    rng = np.random.default_rng(seed)
    per_chr, gaps, points = [], [], []
    for b in BIOS:
        P = pd.read_parquet(Path(preds) / f"preds_{b}.parquet")
        E = {f: ens(P, f) for f in FAMILIES}
        y = P.y.values
        for sname, key in SETS.items():
            idx = select(P, key)
            for fam in FAMILIES:
                points.append({"biosample": b, "set": sname, "family": fam, "n": len(idx), "auroc_seed_mean": fam_auc(P, idx, fam),
                               "auroc_ensemble": float(fast_auc(y[idx], E[fam][idx]))})
                for c in np.unique(P.chrom.values[idx]):
                    ii = idx[P.chrom.values[idx] == c]
                    per_chr.append({"biosample": b, "set": sname, "family": fam, "chrom": c, "n": len(ii), "pos_frac": float(y[ii].mean()),
                                    "auroc_ensemble": float(fast_auc(y[ii], E[fam][ii]))})
            point = {f: float(fast_auc(y[idx], E[f][idx])) for f in FAMILIES}
            reps = {"window": [], "chrom": []}
            for kind, fn in (("window", strat_boot), ("chrom", cluster_boot)):
                for _ in range(B):
                    j = fn(P, idx, rng)
                    yj = y[j]
                    reps[kind].append({f: fast_auc(yj, E[f][j]) for f in FAMILIES})
            chs = np.unique(P.chrom.values[idx])
            for a, c in PAIRS:
                d_s = [r[a] - r[c] for r in reps["window"]]
                d_c = [r[a] - r[c] for r in reps["chrom"]]
                loco = []
                for ch in chs:
                    ii = idx[P.chrom.values[idx] != ch]
                    loco.append(fast_auc(y[ii], E[a][ii]) - fast_auc(y[ii], E[c][ii]))
                lo_s, hi_s = ci(d_s)
                lo_c, hi_c = ci(d_c)
                gaps.append({"biosample": b, "set": sname, "a": a, "b": c, "auroc_a": point[a], "auroc_b": point[c], "diff": point[a] - point[c],
                             "ci_window_lo": lo_s, "ci_window_hi": hi_s, "ci_chrom_lo": lo_c, "ci_chrom_hi": hi_c,
                             "loco_min": min(loco), "loco_max": max(loco), "n": len(idx),
                             "excludes0_window": bool(lo_s > 0 or hi_s < 0), "excludes0_chrom": bool(lo_c > 0 or hi_c < 0)})
            print(f"W1.1 {b} {sname} done", flush=True)
    pd.DataFrame(points).to_csv(Path(out) / "point_estimates.csv", index=False)
    pd.DataFrame(per_chr).to_csv(Path(out) / "per_chromosome_auroc.csv", index=False)
    pd.DataFrame(gaps).to_csv(Path(out) / "paired_gaps.csv", index=False)


def overlap_flags(neg, pos, min_frac):
    """neg, pos: DataFrames with chrom,start,end. Returns bool per neg row: overlaps some pos by >= min_frac of the negative window (any bp if 0)."""
    flag = np.zeros(len(neg), dtype=bool)
    for c, g in neg.groupby("chrom"):
        p = pos[pos.chrom == c].sort_values("start")
        if p.empty:
            continue
        ps, pe = p.start.values, p.end.values
        cummax = np.maximum.accumulate(pe)
        for i, s, e in zip(g.index.values, g.start.values, g.end.values):
            hi = np.searchsorted(ps, e, side="left")  # candidates start before e
            best = 0
            k = hi - 1
            while k >= 0 and cummax[k] > s:
                best = max(best, min(e, pe[k]) - max(s, ps[k]))
                k -= 1
            flag[neg.index.get_loc(i)] = best > 0 and best >= min_frac * (e - s)
    return flag


def w12(preds, tables, out, B, seed):
    rng = np.random.default_rng(seed)
    T = {b: pd.read_parquet(Path(tables) / f"{b}_gc_matched.parquet", columns=["id", "chrom", "start", "end", "label_binary", "split"]) for b in BIOS}
    rows = []
    for a in BIOS:
        P = pd.read_parquet(Path(preds) / f"preds_{a}.parquet")
        t = T[a].set_index("id").loc[P.id.values].reset_index()
        others = [o for o in BIOS if o != a]
        pos_other = pd.concat([T[o][T[o].label_binary == 1][["chrom", "start", "end"]] for o in others], ignore_index=True)
        for frac in (0.0, 0.5):
            neg_mask = (P.y.values == 0)
            neg = t[neg_mask][["chrom", "start", "end"]].reset_index(drop=True)
            flag_neg = overlap_flags(neg, pos_other, frac)
            flag = np.zeros(len(P), dtype=bool)
            flag[np.where(neg_mask)[0]] = flag_neg
            for fam in ("xgb_wide", "shallow_cnn", "attn_cnn", "comp3_lr"):
                cols = family_cols(P, fam)
                s = P[cols].mean(axis=1).values
                y = P.y.values
                full = fam_auc(P, np.arange(len(P)), fam)
                keep = ~flag
                after = float(np.mean([fast_auc(y[keep], P[c].values[keep]) for c in cols]))
                ap_full = float(np.mean([average_precision_score(y, P[c].values) for c in cols]))
                ap_after = float(np.mean([average_precision_score(y[keep], P[c].values[keep]) for c in cols]))
                # contaminated vs clean negatives, within (GC, CpG) 8x8 cells
                gc_b = np.digitize(P.gc.values, np.quantile(P.gc.values, np.linspace(0, 1, 9)[1:-1]))
                oe_b = np.digitize(P.cpg_oe.values, np.quantile(P.cpg_oe.values, np.linspace(0, 1, 9)[1:-1]))
                cell = gc_b * 8 + oe_b
                nm = neg_mask
                obs = s[flag & nm].mean() - s[~flag & nm].mean() if (flag & nm).any() else np.nan

                def stat(fl):
                    num = den = 0.0
                    for u in np.unique(cell[nm]):
                        m = nm & (cell == u)
                        n1, n0 = (m & fl).sum(), (m & ~fl).sum()
                        if n1 and n0:
                            num += (n1 + n0) * (s[m & fl].mean() - s[m & ~fl].mean())
                            den += n1 + n0
                    return num / den if den else np.nan

                stratified = stat(flag)
                null = []
                for _ in range(B):
                    perm = flag.copy()
                    for u in np.unique(cell[nm]):
                        m = np.where(nm & (cell == u))[0]
                        perm[m] = rng.permutation(flag[m])
                    null.append(stat(perm))
                null = np.array(null)
                p = float((1 + (np.abs(null) >= abs(stratified)).sum()) / (1 + B)) if not np.isnan(stratified) else np.nan
                rows.append({"biosample": a, "min_overlap_frac": frac, "family": fam, "n_neg": int(nm.sum()), "n_contaminated_neg": int((flag & nm).sum()),
                             "frac_contaminated": float((flag & nm).sum() / nm.sum()), "mean_score_contaminated_minus_clean": float(obs),
                             "mean_diff_within_cells": float(stratified), "perm_p": p, "auroc_all": full, "auroc_without_contaminated": after,
                             "auprc_all": ap_full, "auprc_without_contaminated": ap_after})
            print(f"W1.2 {a} overlap>={frac} done", flush=True)
    pd.DataFrame(rows).to_csv(Path(out) / "negative_contamination.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    for a in ("preds", "tables", "out"):
        ap.add_argument(f"--{a}", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="both", choices=["both", "w11", "w12"])
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.only in ("both", "w11"):
        w11(a.preds, a.out, a.B, a.seed)
    if a.only in ("both", "w12"):
        w12(a.preds, a.tables, a.out, a.B, a.seed)


if __name__ == "__main__":
    main()
