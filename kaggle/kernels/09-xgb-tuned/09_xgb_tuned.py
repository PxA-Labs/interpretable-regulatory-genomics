# %% [markdown]
# # 09 - Is the CNN advantage real once XGBoost is tuned too? (issues #13, #90)
#
# Notebook 07: a tuned ShallowCNN beats an untuned k=6 XGBoost by 0.005-0.013 AUROC (full test) and 0.011-0.028 (GC+CpG
# matched). The CNNs got a learning-rate sweep, early stopping and decay; XGBoost got none. This notebook gives XGBoost a
# comparable (larger) search effort.
#
# Protocol: k=6 k-mer frequencies (4,096 features); random search of 24 configurations on the VALIDATION set only (seed 0,
# early stopping on validation AUC; config 0 is the notebook-05/08 default, so the search cannot do worse than the default on
# validation); then 5 seeds at the selected configuration and 5 seeds at the default, evaluated on the same four test sets as
# notebooks 07/08: full, without homology-flagged rows, GC+CpG matched, GC+CpG+repeat matched. Test chromosomes are never used
# for selection. Models are saved in XGBoost's native binary format. No pretrained language models.

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
N_CONF = 3 if SMOKE else 24
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
MAX_TREES, ES = (30, 5) if SMOKE else (1500, 50)
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
# ## Search space (fixed in advance; config 0 = previous default)

# %%
SPACE = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.8],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [1.0, 5.0, 20.0],
}
DEFAULT = {"max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0}
_rng = np.random.default_rng(12345)
CONFIGS = [DEFAULT]
while len(CONFIGS) < N_CONF:
    c = {k: v[int(_rng.integers(len(v)))] for k, v in SPACE.items()}
    if c not in CONFIGS:
        CONFIGS.append(c)
print(len(CONFIGS), "configurations; first:", CONFIGS[:2])

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


def fit(cfg, seed, Xtr, ytr, Xva, yva):
    clf = xgb.XGBClassifier(n_estimators=MAX_TREES, tree_method="hist", device=DEV, n_jobs=-1, random_state=seed,
                            early_stopping_rounds=ES, eval_metric="auc", **cfg)
    return clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


# %% [markdown]
# ## Search on validation, then final seeds

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
    for ci, cfg in enumerate(CONFIGS):
        t1 = time.time()
        clf = fit(cfg, 0, X[tr], y[tr], X[va], y[va])
        SEARCH.append({"biosample": b, "config": ci, **cfg, "auroc_val": float(clf.best_score), "best_iteration": int(clf.best_iteration),
                       "seconds": round(time.time() - t1)})
        pd.DataFrame(SEARCH).to_csv(OUT / "xgb_search.csv", index=False)
        print(f"[{time.time()-T0:.0f}s] {b} cfg{ci}: val {clf.best_score:.4f} iters {clf.best_iteration} ({time.time()-t1:.0f}s) {cfg}")
        del clf
    s = pd.DataFrame([r for r in SEARCH if r["biosample"] == b])
    BEST[b] = int(s.loc[s.auroc_val.idxmax(), "config"])
    print(f"[{time.time()-T0:.0f}s] {b}: selected config {BEST[b]} (val {s.auroc_val.max():.4f}; default val {s.auroc_val.iloc[0]:.4f})")
    for tag, ci in (("xgb_k6_default", 0), ("xgb_k6_tuned", BEST[b])):
        if tag == "xgb_k6_tuned" and ci == 0:
            continue  # tuned == default; reported as default
        for seed in SEEDS:
            clf = fit(CONFIGS[ci], seed, X[tr], y[tr], X[va], y[va])
            p = clf.predict_proba(X[te])[:, 1]
            row = {"biosample": b, "model": tag, "seed": seed, "config": ci, "best_iteration": int(clf.best_iteration), "auroc_val": float(clf.best_score)}
            row.update({f"auroc_{k}": auc(yt, p, mk) for k, mk in sets.items()})
            RAW.append(row)
            pd.DataFrame(RAW).to_csv(OUT / "xgb_raw.csv", index=False)
            if tag == "xgb_k6_tuned":
                clf.save_model(str(MODELS_DIR / f"{b}_xgb_k6_tuned_s{seed}.ubj"))
            print(f"[{time.time()-T0:.0f}s] {b} {tag} s{seed}: full {row['auroc_full']:.4f} matched {row['auroc_gc_cpg_matched']:.4f} +rep {row['auroc_gc_cpg_rep_matched']:.4f}")
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
agg.to_csv(OUT / "xgb_tuned.csv", index=False)
sizes = {p.name: p.stat().st_size for p in sorted(MODELS_DIR.glob("*.ubj"))}
(OUT / "protocol.json").write_text(json.dumps({"n_configs": len(CONFIGS), "space": SPACE, "default": DEFAULT, "space_seed": 12345, "selected_config": BEST,
                                               "selected_params": {b: CONFIGS[i] for b, i in BEST.items()}, "seeds": SEEDS, "max_trees": MAX_TREES,
                                               "early_stopping_rounds": ES, "smoke": SMOKE, "device": DEV, "xgboost": xgb.__version__,
                                               "model_bytes_total": sum(sizes.values()), "runtime_s": round(time.time() - T0)}, indent=1))
pd.set_option("display.width", 250, "display.max_columns", 30)
print(agg[["biosample", "model", "auroc_val", "best_iteration"] + mc].round(4).to_string(index=False))
print("model files", len(sizes), "total MB", round(sum(sizes.values()) / 1e6, 1))
print(f"done in {time.time()-T0:.0f}s")
