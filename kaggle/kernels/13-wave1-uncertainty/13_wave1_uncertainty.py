# %% [markdown]
# # 13 - Wave 1: do our gaps survive proper uncertainty, and are the negatives clean? (W1.1, W1.2)
#
# Plan and precommitted rules: `docs/research/novelty_and_next_experiments.md`. No model is trained here: all 75 saved models
# (XGBoost nb09 tuned and nb11 wide, 5 seeds each; ShallowCNN and AttnCNN from nb07, 5 seeds each) plus a three-feature composition
# logistic regression (trained on the train split) are re-scored on the test chromosomes (chr19-22), and the per-window predictions are
# saved. W1.1: paired differences between model families with window-bootstrap, chromosome-cluster bootstrap and leave-one-chromosome-out
# ranges. W1.2: how many GC-matched negatives of one biosample are cCREs in the other two, and whether the models score them higher,
# within (GC, CpG) cells. Matched sets use exactly notebook 08's procedure and seeds.

# %%
import glob, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
SMOKE = os.environ.get("IRG_SMOKE") == "1"
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
B_BOOT = 20 if SMOKE else 1000
BIOS = ["tcell", "k562", "gm12878"]
OUT = Path("/kaggle/working/results")
OUT.mkdir(parents=True, exist_ok=True)
m = glob.glob("/kaggle/input/**/tcell_gc_matched.parquet", recursive=True)
assert m, os.listdir("/kaggle/input")
DATA = Path(m[0]).parent
rf = glob.glob("/kaggle/input/**/repeat_fractions.parquet", recursive=True)
assert rf, "attach dataset irg-repeat-coverage"
REP = pd.read_parquet(rf[0])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEV, "smoke", SMOKE)

# %% [markdown]
# ## Features, matching and CNNs (identical to notebook 08)

# %%
LUT = np.full(256, -1, dtype=np.int64)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i


def kmer_features(seqs, k=6):
    X = np.zeros((len(seqs), 4**k), dtype=np.float32)
    for r, s in enumerate(seqs):
        a = LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
        n = len(a) - k + 1
        idx, ok = np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool)
        for j in range(k):
            seg = a[j : j + n]
            ok &= seg >= 0
            idx = idx * 4 + np.where(seg >= 0, seg, 0)
        c = np.bincount(idx[ok], minlength=4**k).astype(np.float32)
        X[r] = c / max(c.sum(), 1.0)
    return X


def cpg_oe(seqs):
    out = np.zeros(len(seqs), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = len(s)
        c, g, cg = s.count("C"), s.count("G"), s.count("CG")
        out[i] = (cg / L) / (max(c, 1) / L * max(g, 1) / L)
    return out


def matched_mask(cols, y, nb, seed=0):
    rng = np.random.default_rng(seed)
    bins = [np.digitize(c, np.quantile(c, np.linspace(0, 1, nb + 1)[1:-1])) for c in cols]
    cell = np.ravel_multi_index(bins, (nb,) * len(cols))
    keep = np.zeros(len(y), dtype=bool)
    for u in np.unique(cell):
        pi, ni = np.where((cell == u) & (y == 1))[0], np.where((cell == u) & (y == 0))[0]
        n = min(len(pi), len(ni))
        if n:
            keep[rng.choice(pi, n, replace=False)] = True
            keep[rng.choice(ni, n, replace=False)] = True
    return keep


def onehot_batch(idx):
    return F.one_hot(idx.clamp(max=4), 5)[..., :4].float().permute(0, 2, 1)


def revcomp(idx):
    return torch.where(idx < 4, 3 - idx, idx).flip(-1)


class ShallowCNN(nn.Module):
    def __init__(self, n_filters=128, k=15):
        super().__init__()
        self.conv = nn.Conv1d(4, n_filters, k, padding=k // 2)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x):
        return self.fc(self.drop(F.relu(self.conv(x)).amax(-1))).squeeze(-1)


class AttnCNN(nn.Module):
    def __init__(self, n_filters=128, k=15, heads=4):
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


CNNS = {"shallow_cnn": ShallowCNN, "attn_cnn": AttnCNN}
ENC = np.full(256, 4, dtype=np.uint8)
for i, c in enumerate("ACGT"):
    ENC[ord(c)] = i


@torch.no_grad()
def cnn_predict(model, seqs):
    X = torch.from_numpy(np.stack([ENC[np.frombuffer(s.encode(), dtype=np.uint8)] for s in seqs]))
    model.eval()
    ps = []
    for i in range(0, len(X), 1024):
        xb = X[i : i + 1024].to(DEV).long()
        ps.append(((torch.sigmoid(model(onehot_batch(xb))) + torch.sigmoid(model(onehot_batch(revcomp(xb))))) / 2).float().cpu())
    return torch.cat(ps).numpy()


# %% [markdown]
# ## Step 1: per-window test predictions of all saved models

# %%
FOUND = 0
for b in BIOS:
    d = pd.read_parquet(DATA / f"{b}_gc_matched.parquet")
    d = d.merge(REP[REP.biosample == b][["id", "rep_frac"]], on="id", how="left", validate="1:1")
    assert d.rep_frac.notna().all(), b
    if SMOKE:
        d = d.groupby("split", group_keys=False).apply(lambda g: g.sample(min(len(g), 1500), random_state=0)).reset_index(drop=True)
    seqs = np.array(d.sequence.tolist(), dtype=object)
    oe = cpg_oe(seqs.tolist())
    gc = d.gc.values.astype(np.float64)
    tr, te = (d.split == "train").values, (d.split == "test").values
    y = d.label_binary.values.astype(int)
    yt = y[te]
    P = pd.DataFrame({"id": d.id.values[te], "chrom": d.chrom.values[te], "y": yt, "homology_flag": d.homology_flag.values[te].astype(int),
                      "gc": gc[te], "cpg_oe": oe[te], "rep_frac": d.rep_frac.values[te],
                      "m_gc_cpg": matched_mask([gc[te], oe[te]], yt, 8),
                      "m_gc_cpg_rep": matched_mask([gc[te], oe[te], d.rep_frac.values[te]], yt, 6)})
    Z = np.column_stack([gc, oe, d.rep_frac.values])
    sc = StandardScaler().fit(Z[tr])
    P["comp3_lr"] = LogisticRegression(max_iter=1000).fit(sc.transform(Z[tr]), y[tr]).predict_proba(sc.transform(Z[te]))[:, 1]
    Xte = kmer_features(seqs[te].tolist(), 6)
    for tag, pat in (("xgb_tuned", "xgb_k6_tuned"), ("xgb_wide", "xgb_k6_wide")):
        for s in SEEDS:
            w = glob.glob(f"/kaggle/input/**/{b}_{pat}_s{s}.ubj", recursive=True)
            assert w, f"missing {b} {pat} s{s}"
            clf = xgb.XGBClassifier()
            clf.load_model(w[0])
            P[f"{tag}_s{s}"] = clf.predict_proba(Xte)[:, 1]
            FOUND += 1
    for name, cls in CNNS.items():
        for s in SEEDS:
            w = glob.glob(f"/kaggle/input/**/{b}_{name}_s{s}.pt", recursive=True)
            assert w, f"missing {b} {name} s{s}"
            model = cls().to(DEV)
            model.load_state_dict(torch.load(w[0], map_location=DEV))
            P[f"{name}_s{s}"] = cnn_predict(model, seqs[te].tolist())
            FOUND += 1
            del model
    P.to_parquet(OUT / f"preds_{b}.parquet", index=False)
    print(f"[{time.time()-T0:.0f}s] {b}: test {te.sum()}", {c: round(float(roc_auc_score(yt, P[c])), 4) for c in P.columns if c.endswith("_s0") or c == "comp3_lr"}, flush=True)
assert FOUND == len(BIOS) * 4 * len(SEEDS), FOUND

# %% [markdown]
# ## Step 2: analysis (embedded verbatim from tools/analysis/wave1_uncertainty.py)

# %%

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



# %%
w11(str(OUT), str(OUT), B_BOOT, 0)
w12(str(OUT), str(DATA), str(OUT), B_BOOT, 0)
print("done", round(time.time() - T0), "s")
