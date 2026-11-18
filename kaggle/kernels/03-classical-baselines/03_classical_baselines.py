# %% [markdown]
# # 03 - Classical baselines under one protocol (issue #89 / #90)
#
# Re-runs the classical models on the corrected, validated v2 tables with ONE protocol:
# canonical chromosome split (train chr1-14,X,Y / val chr15-18 / test chr19-22), fixed hyper-parameters, seeds, bootstrap CIs.
#
# Experiments
# * E1/E7  k=4 LR / RF / XGBoost on every biosample x negative strategy (+ trivial GC-only and dinucleotide baselines)
# * E3     k-mer sweep k=3..6 (GC-matched)
# * E7b    cross-strategy matrix (train strategy A -> test strategy B)
# * E4     cross-biosample transfer matrix with a **shared-element control**: cCRE accessions are registry-wide, so a test
#          positive that is also a training-biosample positive can be "memorised". We report AUROC on shared vs specific positives.
# * Every test metric is reported on all rows AND excluding homology-flagged rows.
#
# No language models are used. Hyper-parameters are fixed (no test-set tuning); XGBoost early-stops on the validation split only.

# %%
import glob, json, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
QUICK = False  # True = tiny smoke test (5k rows/table, 1 seed)
T0 = time.time()
SEEDS = [0] if QUICK else [0, 1, 2, 3, 4]


def _xgb_device():
    try:  # use the GPU for XGBoost when a CUDA device is usable, else CPU
        Xs = np.random.rand(200, 8).astype("float32")
        xgb.XGBClassifier(n_estimators=2, device="cuda", tree_method="hist").fit(Xs, (Xs[:, 0] > .5).astype(int))
        return "cuda"
    except Exception as e:  # noqa: BLE001
        print("GPU unavailable for XGBoost, using CPU:", str(e)[:120])
        return "cpu"

BIOS, STRAT = ["tcell", "k562", "gm12878"], ["gc_matched", "random_unmatched", "flanking"]
OUT = Path("/kaggle/working/results")
OUT.mkdir(parents=True, exist_ok=True)

m = glob.glob("/kaggle/input/**/tcell_gc_matched.parquet", recursive=True)
assert m, f"packaged tables not found: {os.listdir('/kaggle/input')}"
DATA = Path(m[0]).parent
XGB_DEVICE = _xgb_device()
print("data:", DATA, "| QUICK =", QUICK, "| xgb device =", XGB_DEVICE)

# %%
tables = {}
for b in BIOS:
    for s in STRAT:
        d = pd.read_parquet(DATA / f"{b}_{s}.parquet")
        if QUICK:
            d = d.groupby("split", group_keys=False).apply(lambda g: g.sample(min(len(g), 1700), random_state=0)).reset_index(drop=True)
        tables[(b, s)] = d
print({k: len(v) for k, v in tables.items()})
pos_acc = {b: set(tables[(b, "gc_matched")].query("label_binary==1").accession) for b in BIOS}

# %% [markdown]
# ## Features and metrics

# %%
LUT = np.full(256, -1, dtype=np.int64)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i
_cache = {}


def kmer_features(df, key, k):
    ck = (key, k)
    if ck in _cache:
        return _cache[ck]
    X = np.zeros((len(df), 4**k), dtype=np.float32)
    for r, s in enumerate(df.sequence.values):
        a = LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
        n = len(a) - k + 1
        idx, ok = np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool)
        for j in range(k):
            seg = a[j : j + n]
            ok &= seg >= 0
            idx = idx * 4 + np.where(seg >= 0, seg, 0)
        c = np.bincount(idx[ok], minlength=4**k).astype(np.float32)
        X[r] = c / max(c.sum(), 1.0)
    _cache[ck] = X
    return X


def ece(y, p, bins=10):
    e, n = 0.0, len(y)
    for lo in np.linspace(0, 1, bins, endpoint=False):
        mk = (p > lo) & (p <= lo + 1 / bins)
        if mk.any():
            e += mk.sum() / n * abs(y[mk].mean() - p[mk].mean())
    return float(e)


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "f1": f1_score(y, p >= 0.5),
            "logloss": log_loss(y, p), "ece": ece(y, p)}


def boot_ci(y, p, n=300, seed=0):
    rng, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fit_predict(model, Xtr, ytr, Xva, yva, Xte_list, seed):
    if model == "lr":
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(Xtr), ytr)
        return [clf.predict_proba(sc.transform(X))[:, 1] for X in Xte_list]
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=60 if QUICK else 200, min_samples_leaf=2, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        return [clf.predict_proba(X)[:, 1] for X in Xte_list]
    clf = xgb.XGBClassifier(n_estimators=60 if QUICK else 600, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", device=XGB_DEVICE, n_jobs=-1, random_state=seed, early_stopping_rounds=30, eval_metric="logloss")
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return [clf.predict_proba(X)[:, 1] for X in Xte_list]


def split(df, name):
    return df[df.split == name]


ROWS = []


def record(exp, b, s, feat, model, seed, y, p, flag, extra=None):
    r = {"exp": exp, "biosample": b, "strategy": s, "features": feat, "model": model, "seed": seed, "n_test": len(y)}
    r.update({f"test_{k}": v for k, v in metrics(y, p).items()})
    keep = ~flag
    r["n_test_noflag"] = int(keep.sum())
    r["test_auroc_noflag"] = roc_auc_score(y[keep], p[keep])
    if extra:
        r.update(extra)
    ROWS.append(r)


def run_table(exp, b, s, k, models, seeds, ci=True, feat_name=None):
    df = tables[(b, s)]
    X = kmer_features(df, (b, s), k)
    tr, va, te = (df.split == "train").values, (df.split == "val").values, (df.split == "test").values
    y = df.label_binary.values
    for model in models:
        for seed in (seeds if model != "lr" else [0]):
            p = fit_predict(model, X[tr], y[tr], X[va], y[va], [X[te]], seed)[0]
            extra = {}
            if ci and seed == 0:
                extra["auroc_ci_lo"], extra["auroc_ci_hi"] = boot_ci(y[te], p)
            record(exp, b, s, feat_name or f"{k}mer", model, seed, y[te], p, df.homology_flag.values[te], extra)
    print(f"[{time.time()-T0:6.0f}s] {exp} {b}/{s} k={k} done")


# %% [markdown]
# ## E1/E7: k=4 on every table, three models; trivial baselines

# %%
for b in BIOS:
    for s in STRAT:
        run_table("E1_E7", b, s, 4, ["lr", "xgb"] + (["rf"] if s == "gc_matched" else []), SEEDS)
        run_table("trivial_dinuc", b, s, 2, ["lr"], [0], ci=False, feat_name="2mer")
        df = tables[(b, s)]  # GC-only
        tr, te = (df.split == "train").values, (df.split == "test").values
        g = df[["gc"]].values
        clf = LogisticRegression().fit(g[tr], df.label_binary.values[tr])
        record("trivial_gc", b, s, "gc", "lr", 0, df.label_binary.values[te], clf.predict_proba(g[te])[:, 1], df.homology_flag.values[te])

# %% [markdown]
# ## E3: k-mer sweep (GC-matched)

# %%
for b in BIOS:
    for k in [3, 5, 6] if not QUICK else [3]:
        run_table("E3", b, "gc_matched", k, ["lr", "xgb"], [0], ci=False)

# %% [markdown]
# ## E7b: cross-strategy matrix (XGBoost, k=4, seed 0)

# %%
for b in BIOS:
    mods = {}
    for sa in STRAT:
        da = tables[(b, sa)]
        Xa = kmer_features(da, (b, sa), 4)
        tr, va = (da.split == "train").values, (da.split == "val").values
        ya = da.label_binary.values
        for sb in STRAT:
            db = tables[(b, sb)]
            Xb = kmer_features(db, (b, sb), 4)
            te = (db.split == "test").values
            p = fit_predict("xgb", Xa[tr], ya[tr], Xa[va], ya[va], [Xb[te]], 0)[0]
            record("E7b_cross_strategy", b, f"{sa}->{sb}", "4mer", "xgb", 0, db.label_binary.values[te], p, db.homology_flag.values[te])
    print(f"[{time.time()-T0:6.0f}s] E7b {b} done")

# %% [markdown]
# ## E4: cross-biosample transfer with shared-element control

# %%
for ba in BIOS:
    da = tables[(ba, "gc_matched")]
    Xa = kmer_features(da, (ba, "gc_matched"), 4)
    tr, va = (da.split == "train").values, (da.split == "val").values
    ya = da.label_binary.values
    for bb in BIOS:
        db = tables[(bb, "gc_matched")]
        Xb = kmer_features(db, (bb, "gc_matched"), 4)
        te = (db.split == "test").values
        yb = db.label_binary.values[te]
        p = fit_predict("xgb", Xa[tr], ya[tr], Xa[va], ya[va], [Xb[te]], 0)[0]
        acc = db.accession.values[te]
        shared_pos = (yb == 1) & np.isin(acc, list(pos_acc[ba]))
        spec_pos = (yb == 1) & ~shared_pos
        neg = yb == 0
        ex = {"n_pos_shared": int(shared_pos.sum()), "n_pos_specific": int(spec_pos.sum())}
        if shared_pos.sum() > 20 and spec_pos.sum() > 20:
            ex["auroc_shared_pos"] = roc_auc_score(np.r_[np.ones(shared_pos.sum()), np.zeros(neg.sum())], np.r_[p[shared_pos], p[neg]])
            ex["auroc_specific_pos"] = roc_auc_score(np.r_[np.ones(spec_pos.sum()), np.zeros(neg.sum())], np.r_[p[spec_pos], p[neg]])
        record("E4_transfer", bb, f"train={ba}", "4mer", "xgb", 0, yb, p, db.homology_flag.values[te], ex)
    print(f"[{time.time()-T0:6.0f}s] E4 train={ba} done")

# %% [markdown]
# ## Save

# %%
res = pd.DataFrame(ROWS)
res.to_csv(OUT / "results.csv", index=False)
agg = (res.groupby(["exp", "biosample", "strategy", "features", "model"])
       .agg(n_seeds=("seed", "nunique"), auroc=("test_auroc", "mean"), auroc_sd=("test_auroc", "std"), auroc_noflag=("test_auroc_noflag", "mean"),
            auprc=("test_auprc", "mean"), f1=("test_f1", "mean"), logloss=("test_logloss", "mean"), ece=("test_ece", "mean"))
       .reset_index())
agg.to_csv(OUT / "summary.csv", index=False)
(OUT / "protocol.json").write_text(json.dumps({"seeds": SEEDS, "quick": QUICK, "split": "train chr1-14,X,Y / val 15-18 / test 19-22",
                                               "xgb": "600 trees, lr .1, depth 6, early stop on val", "rf": "200 trees", "lr": "C=1, standardised",
                                               "xgb_device": XGB_DEVICE, "runtime_s": round(time.time() - T0)}, indent=1))
pd.set_option("display.width", 220, "display.max_columns", 20)
print(agg[agg.exp == "E1_E7"].round(4).to_string(index=False))
print(f"done in {time.time()-T0:.0f}s")
