# %% [markdown]
# # 11 - Was the tuned XGBoost of notebook 09 still under-tuned? (issues #13, #90)
#
# Notebook 09: the best configurations all used 1,466-1,499 of the 1,500-tree cap, with the lowest learning rate and column
# sampling on the grid, so validation AUC may still have been rising. This notebook (a) retrains each biosample's three best
# notebook-09 configurations with the tree cap lifted to 8,000 (stage A) and (b) tries 8 new configurations that move off the grid
# edges: colsample {0.1, 0.2, 0.3}, lambda {20, 50, 100}, depth {6, 8, 10}, lr {0.05, 0.08}, cap 4,000 (stage B).
# Selection is on VALIDATION AUC only (best over stages A and B per biosample); then 5 seeds at the selected configuration
# (cap 8,000, patience 100) evaluated on the same four test sets as notebooks 07-09: full, non-homologous, GC+CpG matched,
# GC+CpG+repeat matched. Test chromosomes are never used for selection. k=6 only. No pretrained language models.

# %%
import glob, json, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
T0 = time.time()
SMOKE = os.environ.get("IRG_SMOKE") == "1"
N_B = 2 if SMOKE else 8
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
CAP_A, CAP_B, CAP_FINAL, ES = (30, 30, 30, 5) if SMOKE else (8000, 4000, 8000, 100)
BIOS = ["tcell", "k562", "gm12878"]
OUT = Path("/kaggle/working/results")
OUT.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("/kaggle/working/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
m = glob.glob("/kaggle/input/**/tcell_gc_matched.parquet", recursive=True)
assert m, os.listdir("/kaggle/input")
DATA = Path(m[0]).parent
rf = glob.glob("/kaggle/input/**/repeat_fractions.parquet", recursive=True)
assert rf, "attach dataset irg-repeat-coverage"
REP = pd.read_parquet(rf[0])


def _device():
    try:
        Xs = np.random.rand(200, 8).astype("float32")
        xgb.XGBClassifier(n_estimators=2, device="cuda", tree_method="hist").fit(Xs, (Xs[:, 0] > .5).astype(int))
        return "cuda"
    except Exception as e:  # noqa: BLE001
        print("GPU unavailable, CPU:", str(e)[:100])
        return "cpu"


DEV = _device()
print("device", DEV, "xgboost", xgb.__version__)

# %% [markdown]
# ## Configurations

# %%
BANK = {  # notebook-09 configurations (ids as in its xgb_search.csv)
    1: {"max_depth": 8, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.3, "min_child_weight": 1, "reg_lambda": 20.0},
    20: {"max_depth": 8, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.5, "min_child_weight": 5, "reg_lambda": 5.0},
    21: {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.6, "colsample_bytree": 0.3, "min_child_weight": 10, "reg_lambda": 1.0},
    23: {"max_depth": 6, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.3, "min_child_weight": 5, "reg_lambda": 20.0},
}
TOP3 = {"tcell": [21, 1, 23], "k562": [23, 1, 20], "gm12878": [1, 21, 23]}  # best three per biosample in notebook 09 (by validation AUC)
SPACE_B = {
    "max_depth": [6, 8, 10],
    "learning_rate": [0.05, 0.08],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.1, 0.2, 0.3],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [20.0, 50.0, 100.0],
}
_rng = np.random.default_rng(777)
CFG_B = []
while len(CFG_B) < N_B:
    c = {k: v[int(_rng.integers(len(v)))] for k, v in SPACE_B.items()}
    if c not in CFG_B:
        CFG_B.append(c)
print("stage B configurations:", CFG_B)

# %% [markdown]
# ## Features and matching (identical procedure to notebooks 05-08)

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


def auc(y, p, mask=None):
    if mask is not None:
        y, p = y[mask], p[mask]
    return float(roc_auc_score(y, p)) if len(y) > 50 and y.min() != y.max() else float("nan")


def fit(cfg, seed, Xtr, ytr, Xva, yva, cap):
    clf = xgb.XGBClassifier(n_estimators=cap, tree_method="hist", device=DEV, n_jobs=-1, random_state=seed,
                            early_stopping_rounds=ES, eval_metric="auc", **cfg)
    return clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


# %% [markdown]
# ## Stage A + B on validation, then final seeds

# %%
SEARCH, RAW = [], []
BEST = {}
for b in BIOS:
    d = pd.read_parquet(DATA / f"{b}_gc_matched.parquet")
    d = d.merge(REP[REP.biosample == b][["id", "rep_frac"]], on="id", how="left", validate="1:1")
    assert d.rep_frac.notna().all()
    if SMOKE:
        d = d.groupby("split", group_keys=False).apply(lambda g: g.sample(min(len(g), 1500), random_state=0)).reset_index(drop=True)
    seqs = d.sequence.tolist()
    y = d.label_binary.values.astype(int)
    tr, va, te = [(d.split == s).values for s in ("train", "val", "test")]
    X = kmer_features(seqs, 6)
    oe, gc, rep = cpg_oe(seqs), d.gc.values.astype(np.float64), d.rep_frac.values
    yt = y[te]
    sets = {"full": None, "nonhomologous": ~d.homology_flag.values[te],
            "gc_cpg_matched": matched_mask([gc[te], oe[te]], yt, 8), "gc_cpg_rep_matched": matched_mask([gc[te], oe[te], rep[te]], yt, 6)}
    print(f"[{time.time()-T0:.0f}s] {b}: features {X.shape}, train {tr.sum()} val {va.sum()} test {te.sum()}")
    cands = [("A", f"nb09-{i}", BANK[i], CAP_A) for i in TOP3[b]] + [("B", f"B{j}", c, CAP_B) for j, c in enumerate(CFG_B)]
    if SMOKE:
        cands = cands[:2] + cands[-1:]
    for stage, cid, cfg, cap in cands:
        t1 = time.time()
        clf = fit(cfg, 0, X[tr], y[tr], X[va], y[va], cap)
        SEARCH.append({"biosample": b, "stage": stage, "config": cid, **cfg, "cap": cap, "auroc_val": float(clf.best_score),
                       "best_iteration": int(clf.best_iteration), "hit_cap": int(clf.best_iteration) >= cap - 5, "seconds": round(time.time() - t1)})
        pd.DataFrame(SEARCH).to_csv(OUT / "xgb_wide_search.csv", index=False)
        print(f"[{time.time()-T0:.0f}s] {b} {stage}/{cid}: val {clf.best_score:.4f} iters {clf.best_iteration}/{cap} ({time.time()-t1:.0f}s) {cfg}")
        del clf
    s = pd.DataFrame([r for r in SEARCH if r["biosample"] == b])
    k = s.auroc_val.idxmax()
    BEST[b] = {"stage": s.loc[k, "stage"], "config": s.loc[k, "config"], "params": {p: (s.loc[k, p].item() if hasattr(s.loc[k, p], "item") else s.loc[k, p]) for p in SPACE_B}}
    print(f"[{time.time()-T0:.0f}s] {b}: selected {BEST[b]['stage']}/{BEST[b]['config']} (val {s.auroc_val.max():.4f}; best stage-A {s[s.stage=='A'].auroc_val.max():.4f}; best stage-B {s[s.stage=='B'].auroc_val.max():.4f})")
    for seed in SEEDS:
        clf = fit(BEST[b]["params"], seed, X[tr], y[tr], X[va], y[va], CAP_FINAL)
        p = clf.predict_proba(X[te])[:, 1]
        row = {"biosample": b, "model": "xgb_k6_wide", "seed": seed, "config": f"{BEST[b]['stage']}/{BEST[b]['config']}", "best_iteration": int(clf.best_iteration),
               "hit_cap": int(clf.best_iteration) >= CAP_FINAL - 5, "auroc_val": float(clf.best_score)}
        row.update({f"auroc_{k2}": auc(yt, p, mk) for k2, mk in sets.items()})
        RAW.append(row)
        pd.DataFrame(RAW).to_csv(OUT / "xgb_wide_raw.csv", index=False)
        clf.save_model(str(MODELS_DIR / f"{b}_xgb_k6_wide_s{seed}.ubj"))
        print(f"[{time.time()-T0:.0f}s] {b} final s{seed}: full {row['auroc_full']:.4f} matched {row['auroc_gc_cpg_matched']:.4f} +rep {row['auroc_gc_cpg_rep_matched']:.4f} iters {clf.best_iteration}")
        del clf
    del X

# %% [markdown]
# ## Summary

# %%
res = pd.DataFrame(RAW)
mc = [c for c in res.columns if c.startswith("auroc_") and c != "auroc_val"]
agg = res.groupby(["biosample", "model"])[mc + ["auroc_val", "best_iteration"]].mean().reset_index()
sd = res.groupby(["biosample", "model"])[mc].std().add_suffix("_sd").reset_index()
agg = agg.merge(sd, on=["biosample", "model"])
agg.to_csv(OUT / "xgb_wide.csv", index=False)
sizes = {p.name: p.stat().st_size for p in sorted(MODELS_DIR.glob("*.ubj"))}
(OUT / "protocol.json").write_text(json.dumps({"stage_a": {"configs": {b: TOP3[b] for b in BIOS}, "cap": CAP_A}, "stage_b": {"space": SPACE_B, "configs": CFG_B, "cap": CAP_B, "space_seed": 777},
                                               "final_cap": CAP_FINAL, "early_stopping_rounds": ES, "selected": {b: {"stage": BEST[b]["stage"], "config": str(BEST[b]["config"]), "params": BEST[b]["params"]} for b in BIOS},
                                               "seeds": SEEDS, "smoke": SMOKE, "device": DEV, "xgboost": xgb.__version__,
                                               "model_bytes_total": sum(sizes.values()), "runtime_s": round(time.time() - T0)}, indent=1))
pd.set_option("display.width", 250, "display.max_columns", 30)
print(agg[["biosample", "model", "auroc_val", "best_iteration"] + mc].round(4).to_string(index=False))
print("model files", len(sizes), "total MB", round(sum(sizes.values()) / 1e6, 1))
print(f"done in {time.time()-T0:.0f}s")
