# %% [markdown]
# # 12 - What distinguishes promoters from proximal and distal enhancers? Multiclass XGBoost + SHAP (issues #13, #90)
#
# Four classes per biosample from the `classification` column of the GC-matched v2.1 tables: background (non-regulatory), PLS
# (promoter-like), pELS (proximal enhancer-like), dELS (distal enhancer-like). The negatives were GC-matched to the *pooled*
# positives, not per class, so a promoter-vs-background classifier can still use GC/CpG (promoters are GC-rich). Therefore:
#   * per-class AUROC against background is reported on the full test set AND on a test subset matched on GC and CpG o/e
#     (equal class-c and background counts per bin, notebook 05/06 procedure), and a two-feature [GC, CpG o/e] model shows
#     how much of each class is explained by composition alone;
#   * k=4 (256 features) keeps SHAP readable; SHAP is XGBoost's built-in TreeSHAP (pred_contribs) on a stratified test sample;
#   * for each class: top features by mean |SHAP|, the share of |SHAP| mass on CG-containing 4-mers versus the share of such
#     features (expected under no preference), feature direction (Spearman of value vs SHAP), stability of the top-20 across
#     3 seeds, and overlap of the top-20 between biosamples.
# XGBoost uses fixed, untuned parameters (earlier notebooks' default); seeds 0-2; validation-based early stopping only.
# Test chromosomes are never used for selection. No pretrained language models.

# %%
import glob, json, os, time, warnings
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
T0 = time.time()
SMOKE = os.environ.get("IRG_SMOKE") == "1"
ROOT = os.environ.get("IRG_INPUT", "/kaggle/input")
OUT = Path(os.environ.get("IRG_OUT", "/kaggle/working/results"))
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0] if SMOKE else [0, 1, 2]
N_TREES, ES = (30, 5) if SMOKE else (1500, 50)
N_SHAP = 400 if SMOKE else 4000  # stratified test rows for SHAP
BIOS = ["tcell", "k562", "gm12878"]
CLASSES = ["background", "PLS", "pELS", "dELS"]
CMAP = {"non-regulatory": 0, "PLS": 1, "pELS": 2, "dELS": 3}
m = glob.glob(f"{ROOT}/**/tcell_gc_matched.parquet", recursive=True)
assert m, os.listdir(ROOT)
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
print("device", DEV, "xgboost", xgb.__version__)

# %% [markdown]
# ## Features and matching

# %%
LUT = np.full(256, -1, dtype=np.int64)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i
KM = ["".join(p) for p in product("ACGT", repeat=4)]
HAS_CG = np.array(["CG" in k for k in KM])


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


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(y) > 50 and y.min() != y.max() else float("nan")


# %% [markdown]
# ## Run

# %%
METRICS, SHAPROWS, CONF = [], [], {}
TOP = {}  # (bio, seed, class) -> top-20 feature set
for b in BIOS:
    d = pd.read_parquet(DATA / f"{b}_gc_matched.parquet", columns=["sequence", "classification", "split", "gc"])
    d["cls"] = d.classification.map(CMAP)
    assert d.cls.notna().all(), d.classification.unique()
    if SMOKE:
        d = d.groupby("split", group_keys=False).apply(lambda g: g.sample(min(len(g), 1500), random_state=0)).reset_index(drop=True)
    seqs = d.sequence.tolist()
    cls = d.cls.values.astype(int)
    tr, va, te = [(d.split == s).values for s in ("train", "val", "test")]
    X = kmer_features(seqs, 4)
    oe, gc = cpg_oe(seqs), d.gc.values.astype(np.float64)
    ct = cls[te]
    # per-class evaluation subsets: class c vs background; full and GC+CpG matched
    vs_bg, matched = {}, {}
    for c in (1, 2, 3):
        sub = (ct == c) | (ct == 0)
        keep = np.zeros(len(ct), dtype=bool)
        idx = np.where(sub)[0]
        keep[idx[matched_mask([gc[te][sub], oe[te][sub]], (ct[sub] == c).astype(int), 8)]] = True
        vs_bg[c], matched[c] = sub, keep
    print(f"[{time.time()-T0:.0f}s] {b}: features {X.shape}; test class counts {np.bincount(ct, minlength=4).tolist()}; matched sizes {[int(matched[c].sum()) for c in (1,2,3)]}")
    # composition-only model: how much of each class does [GC, CpG o/e] explain?
    F2 = np.column_stack([gc, oe])
    sc = StandardScaler().fit(F2[tr])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(F2[tr]), cls[tr])
    P2 = lr.predict_proba(sc.transform(F2[te]))
    for c in (1, 2, 3):
        METRICS.append({"biosample": b, "model": "gc_cpg_lr", "seed": 0, "class": CLASSES[c], "auroc_ovr_full": auc((ct == c).astype(int), P2[:, c]),
                        "auroc_vs_bg_full": auc((ct[vs_bg[c]] == c).astype(int), P2[vs_bg[c], c]),
                        "auroc_vs_bg_matched": auc((ct[matched[c]] == c).astype(int), P2[matched[c], c])})
    for seed in SEEDS:
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=4, n_estimators=N_TREES, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                tree_method="hist", device=DEV, n_jobs=-1, random_state=seed, early_stopping_rounds=ES, eval_metric="mlogloss")
        clf.fit(X[tr], cls[tr], eval_set=[(X[va], cls[va])], verbose=False)
        P = clf.predict_proba(X[te])
        pred = P.argmax(1)
        f1s = f1_score(ct, pred, average=None, labels=[0, 1, 2, 3])
        row0 = {"biosample": b, "model": "xgb_k4_multiclass", "seed": seed, "accuracy": accuracy_score(ct, pred), "macro_f1": float(f1s.mean()),
                "best_iteration": int(clf.best_iteration)}
        for c in (1, 2, 3):
            METRICS.append({**row0, "class": CLASSES[c], "f1": float(f1s[c]), "auroc_ovr_full": auc((ct == c).astype(int), P[:, c]),
                            "auroc_vs_bg_full": auc((ct[vs_bg[c]] == c).astype(int), P[vs_bg[c], c]),
                            "auroc_vs_bg_matched": auc((ct[matched[c]] == c).astype(int), P[matched[c], c])})
        if seed == SEEDS[0]:
            CONF[b] = confusion_matrix(ct, pred, labels=[0, 1, 2, 3]).tolist()
        pd.DataFrame(METRICS).to_csv(OUT / "multiclass_metrics_raw.csv", index=False)
        print(f"[{time.time()-T0:.0f}s] {b} s{seed}: acc {row0['accuracy']:.3f} macroF1 {row0['macro_f1']:.3f} iters {clf.best_iteration}")
        # SHAP (built-in TreeSHAP) on a stratified test sample
        rng = np.random.default_rng(seed)
        per = N_SHAP // 4
        si = np.concatenate([rng.choice(np.where(ct == c)[0], min(per, (ct == c).sum()), replace=False) for c in range(4)])
        booster = clf.get_booster()
        booster.set_param({"device": DEV})
        contrib = booster.predict(xgb.DMatrix(X[te][si]), pred_contribs=True)  # (n, 4, 257)
        assert contrib.shape[1] == 4 and contrib.shape[2] == X.shape[1] + 1, contrib.shape
        for c in (1, 2, 3):
            sv = contrib[:, c, :-1]
            mabs = np.abs(sv).mean(0)
            share_cg = float(mabs[HAS_CG].sum() / mabs.sum())
            order = np.argsort(-mabs)[:20]
            TOP[(b, seed, c)] = set(order.tolist())
            for f in order:
                rho = spearmanr(X[te][si][:, f], sv[:, f]).correlation if X[te][si][:, f].std() > 0 else 0.0
                SHAPROWS.append({"biosample": b, "seed": seed, "class": CLASSES[c], "rank": int(np.where(order == f)[0][0]) + 1, "kmer": KM[f], "has_CG": bool(HAS_CG[f]),
                                 "mean_abs_shap": float(mabs[f]), "direction_spearman": float(rho)})
            SHAPROWS.append({"biosample": b, "seed": seed, "class": CLASSES[c], "rank": 0, "kmer": "__summary__", "has_CG": True, "mean_abs_shap": float(mabs.sum()),
                             "direction_spearman": share_cg})  # rank 0 row: total mass and CG share (in direction column)
        del clf
        pd.DataFrame(SHAPROWS).to_csv(OUT / "shap_top_features.csv", index=False)

# %% [markdown]
# ## Summary

# %%
M = pd.DataFrame(METRICS)
agg = M.groupby(["biosample", "model", "class"])[["auroc_ovr_full", "auroc_vs_bg_full", "auroc_vs_bg_matched"] + (["f1"] if "f1" in M else [])].mean().reset_index()
sd = M[M.model == "xgb_k4_multiclass"].groupby(["biosample", "class"])[["auroc_vs_bg_full", "auroc_vs_bg_matched"]].std().add_suffix("_sd").reset_index()
agg = agg.merge(sd, on=["biosample", "class"], how="left")
overall = M[M.model == "xgb_k4_multiclass"].groupby(["biosample", "seed"])[["accuracy", "macro_f1"]].first().groupby("biosample").agg(["mean", "std"])
agg.to_csv(OUT / "multiclass_metrics.csv", index=False)
overall.to_csv(OUT / "multiclass_overall.csv")
json.dump(CONF, open(OUT / "confusion_matrices.json", "w"))
S = pd.DataFrame(SHAPROWS)
summ = S[S["rank"] == 0].rename(columns={"direction_spearman": "cg_share_of_abs_shap", "mean_abs_shap": "total_mean_abs_shap"}).drop(columns=["rank", "kmer", "has_CG"])
summ["cg_share_expected_if_no_preference"] = float(HAS_CG.mean())
summ.to_csv(OUT / "shap_cg_share.csv", index=False)
stab = []
for b in BIOS:
    for c in (1, 2, 3):
        ss = [TOP[(b, s, c)] for s in SEEDS if (b, s, c) in TOP]
        j = [len(x & y) / len(x | y) for x, y in combinations(ss, 2)]
        stab.append({"biosample": b, "class": CLASSES[c], "n_seeds": len(ss), "mean_jaccard_top20_across_seeds": float(np.mean(j)) if j else float("nan")})
pd.DataFrame(stab).to_csv(OUT / "shap_seed_stability.csv", index=False)
ov = []
for c in (1, 2, 3):
    for b1, b2 in combinations(BIOS, 2):
        x, y = TOP[(b1, SEEDS[0], c)], TOP[(b2, SEEDS[0], c)]
        ov.append({"class": CLASSES[c], "pair": f"{b1}-{b2}", "jaccard_top20": len(x & y) / len(x | y), "shared": len(x & y)})
pd.DataFrame(ov).to_csv(OUT / "shap_biosample_overlap.csv", index=False)
(OUT / "protocol.json").write_text(json.dumps({"seeds": SEEDS, "n_trees_cap": N_TREES, "early_stopping": ES, "n_shap_rows": N_SHAP, "k": 4, "classes": CLASSES,
                                               "xgb_params": {"learning_rate": 0.1, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8}, "smoke": SMOKE, "device": DEV,
                                               "xgboost": xgb.__version__, "runtime_s": round(time.time() - T0)}, indent=1))
pd.set_option("display.width", 250, "display.max_columns", 30)
print(overall.round(3).to_string())
print(agg.round(3).to_string(index=False))
print(summ.round(3).to_string(index=False))
print(pd.DataFrame(stab).round(3).to_string(index=False))
print(pd.DataFrame(ov).round(3).to_string(index=False))
print(f"done in {time.time()-T0:.0f}s")
