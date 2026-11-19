# %% [markdown]
# # 05 - Is the surviving shortcut CpG content? (issue #90)
#
# Notebook 03: GC-matching removes the GC shortcut (GC-only AUROC 0.51) but 16 dinucleotide frequencies still reach 0.74-0.80.
# Hypothesis: that signal is mostly CpG (depleted in bulk genome, enriched at promoters / CpG islands).
#
# Tests (GC-matched tables, canonical split, XGBoost k-mer models, seeds 0-2):
#   A  XGBoost on all k-mers (k=4 and k=6)
#   B  same, with every k-mer containing the dinucleotide CG removed
#   D  logistic regression on tiny feature sets: CpG obs/exp alone, GC + CpG obs/exp, dinucleotides with/without CG
#   E  evaluation on a test subset **matched on GC and CpG obs/exp** (equal positives and negatives per bin): what is left
#      when CpG is balanced?
# No language models.

# %%
import glob, json, os, time, warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
SEEDS = [0, 1, 2]
BIOS = ["tcell", "k562", "gm12878"]
OUT = Path("/kaggle/working/results")
OUT.mkdir(parents=True, exist_ok=True)
m = glob.glob("/kaggle/input/**/tcell_gc_matched.parquet", recursive=True)
assert m, os.listdir("/kaggle/input")
DATA = Path(m[0]).parent


def _device():
    try:
        Xs = np.random.rand(200, 8).astype("float32")
        xgb.XGBClassifier(n_estimators=2, device="cuda", tree_method="hist").fit(Xs, (Xs[:, 0] > .5).astype(int))
        return "cuda"
    except Exception as e:  # noqa: BLE001
        print("GPU unavailable, CPU:", str(e)[:100])
        return "cpu"


DEV = _device()
tables = {b: pd.read_parquet(DATA / f"{b}_gc_matched.parquet") for b in BIOS}
print("device", DEV, {b: len(t) for b, t in tables.items()})

# %% [markdown]
# ## Features

# %%
LUT = np.full(256, -1, dtype=np.int64)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i


def kmer_features(seqs, k):
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


def kmers(k):
    return ["".join(p) for p in product("ACGT", repeat=k)]


def keep_no_cg(k):  # columns whose k-mer does NOT contain "CG"
    return np.array(["CG" not in km for km in kmers(k)])


def cpg_oe(seqs):  # observed/expected CpG = (CG/L) / ((C/L)*(G/L))
    out = np.zeros(len(seqs), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = len(s)
        c, g, cg = s.count("C"), s.count("G"), s.count("CG")
        out[i] = (cg / L) / (max(c, 1) / L * max(g, 1) / L)
    return out


feats = {}
for b, d in tables.items():
    seqs = d.sequence.tolist()
    feats[b] = {"k4": kmer_features(seqs, 4), "k6": kmer_features(seqs, 6), "k2": kmer_features(seqs, 2), "oe": cpg_oe(seqs)}
    print(f"[{time.time()-T0:.0f}s] features {b}")
print("fraction of k-mers removed (contain CG): k2", 1 - keep_no_cg(2).mean(), "k4", round(1 - keep_no_cg(4).mean(), 3), "k6", round(1 - keep_no_cg(6).mean(), 3))

# %%
ROWS = []


def xgb_fit(Xtr, ytr, Xva, yva, seed):
    clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", device=DEV,
                            n_jobs=-1, random_state=seed, early_stopping_rounds=30, eval_metric="logloss")
    return clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


def lr_fit(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def boot(y, p, n=300):
    rng, v = np.random.default_rng(0), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


# matched evaluation subset: bin GC and CpG o/e (quantile bins on the test set), keep min(pos,neg) per bin
def matched_mask(gc, oe, y, nb=8, seed=0):
    rng = np.random.default_rng(seed)
    gb = np.digitize(gc, np.quantile(gc, np.linspace(0, 1, nb + 1)[1:-1]))
    ob = np.digitize(oe, np.quantile(oe, np.linspace(0, 1, nb + 1)[1:-1]))
    keep = np.zeros(len(y), dtype=bool)
    for g in range(nb):
        for o in range(nb):
            cell = (gb == g) & (ob == o)
            pi, ni = np.where(cell & (y == 1))[0], np.where(cell & (y == 0))[0]
            n = min(len(pi), len(ni))
            if n:
                keep[rng.choice(pi, n, replace=False)] = True
                keep[rng.choice(ni, n, replace=False)] = True
    return keep


for b in BIOS:
    d, f = tables[b], feats[b]
    tr, va, te = (d.split == "train").values, (d.split == "val").values, (d.split == "test").values
    y = d.label_binary.values
    gc = d.gc.values
    mk = matched_mask(gc[te], f["oe"][te], y[te])
    print(f"[{time.time()-T0:.0f}s] {b}: matched test subset {mk.sum()} of {te.sum()} rows; pos rate {y[te][mk].mean():.2f}; "
          f"CpG o/e mean pos {f['oe'][te][mk][y[te][mk]==1].mean():.3f} neg {f['oe'][te][mk][y[te][mk]==0].mean():.3f}")

    # tiny LR feature sets
    sets = {"cpg_oe_only": f["oe"][:, None], "gc+cpg_oe": np.c_[gc, f["oe"]], "dinuc16": f["k2"], "dinuc15_noCG": f["k2"][:, keep_no_cg(2)]}
    for name, X in sets.items():
        pr = lr_fit(X[tr], y[tr])(X[te])
        ROWS.append({"biosample": b, "exp": "D_lr_small", "features": name, "seed": 0, "auroc_test": roc_auc_score(y[te], pr),
                     "auroc_matched": roc_auc_score(y[te][mk], pr[mk])})

    for k in ["k4", "k6"]:
        for variant, cols in {"all": np.ones(f[k].shape[1], dtype=bool), "noCG": keep_no_cg(int(k[1]))}.items():
            X = f[k][:, cols]
            for seed in SEEDS:
                clf = xgb_fit(X[tr], y[tr], X[va], y[va], seed)
                p = clf.predict_proba(X[te])[:, 1]
                r = {"biosample": b, "exp": "AB_xgb", "features": f"{k}_{variant}", "n_features": int(cols.sum()), "seed": seed,
                     "auroc_test": roc_auc_score(y[te], p), "auroc_matched": roc_auc_score(y[te][mk], p[mk])}
                if seed == 0:
                    r["matched_ci_lo"], r["matched_ci_hi"] = boot(y[te][mk], p[mk])
                ROWS.append(r)
        print(f"[{time.time()-T0:.0f}s] {b} {k} done")

# %% [markdown]
# ## Summary

# %%
res = pd.DataFrame(ROWS)
res.to_csv(OUT / "cpg_ablation_raw.csv", index=False)
agg = (res.groupby(["biosample", "exp", "features"])
       .agg(n_features=("n_features", "first"), auroc_test=("auroc_test", "mean"), sd=("auroc_test", "std"),
            auroc_matched=("auroc_matched", "mean"), matched_ci_lo=("matched_ci_lo", "first"), matched_ci_hi=("matched_ci_hi", "first"))
       .reset_index())
agg.to_csv(OUT / "cpg_ablation.csv", index=False)
(OUT / "protocol.json").write_text(json.dumps({"seeds": SEEDS, "device": DEV, "runtime_s": round(time.time() - T0)}, indent=1))
pd.set_option("display.width", 220, "display.max_columns", 20)
print(agg.round(3).to_string(index=False))
print(f"done in {time.time()-T0:.0f}s")
