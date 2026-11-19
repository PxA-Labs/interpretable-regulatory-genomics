# %% [markdown]
# # 04 - Cross-biosample transfer, stratified by element class (issue #90)
#
# Notebook 03 found that cross-biosample AUROC is much higher on test positives that are ALSO positives in the training
# biosample ("shared") than on biosample-specific ones. Shared elements are probably enriched for promoters (PLS), which are
# easier, so "shared vs specific" is confounded with "promoter vs enhancer". Here we stratify by element class
# (PLS / pELS / dELS) and by shared vs specific, against the same GC-matched negatives.
#
# Protocol identical to notebook 03: canonical chromosome split, XGBoost k=4 (600 trees, early stop on val), seeds 0-2.
# No language models. Positives of each class are scored against ALL test negatives of the same biosample.

# %%
import glob, json, os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
T0 = time.time()
SEEDS = [0, 1, 2]
BIOS = ["tcell", "k562", "gm12878"]
CLASSES = ["PLS", "pELS", "dELS"]
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
pos_acc = {b: set(tables[b].query("label_binary==1").accession) for b in BIOS}
print("data:", DATA, "| device:", DEV, {b: len(t) for b, t in tables.items()})

# %%
LUT = np.full(256, -1, dtype=np.int64)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i


def kmer_features(df, k=4):
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
    return X


F = {b: kmer_features(tables[b]) for b in BIOS}
print(f"[{time.time()-T0:.0f}s] features done")


def boot_ci(y, p, n=300, seed=0):
    rng, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def auc(pos_p, neg_p, ci=True):
    if len(pos_p) < 30:
        return np.nan, np.nan, np.nan
    y = np.r_[np.ones(len(pos_p)), np.zeros(len(neg_p))]
    p = np.r_[pos_p, neg_p]
    lo, hi = boot_ci(y, p) if ci else (np.nan, np.nan)
    return roc_auc_score(y, p), lo, hi


# %% [markdown]
# ## Train once per (train biosample, seed); score every test biosample

# %%
rows = []
for ba in BIOS:
    da = tables[ba]
    tr, va = (da.split == "train").values, (da.split == "val").values
    ya = da.label_binary.values
    for seed in SEEDS:
        clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                device=DEV, n_jobs=-1, random_state=seed, early_stopping_rounds=30, eval_metric="logloss")
        clf.fit(F[ba][tr], ya[tr], eval_set=[(F[ba][va], ya[va])], verbose=False)
        for bb in BIOS:
            db = tables[bb]
            te = (db.split == "test").values
            p = clf.predict_proba(F[bb][te])[:, 1]
            y = db.label_binary.values[te]
            cls = db.classification.values[te]
            acc = db.accession.values[te]
            neg = p[y == 0]
            shared = np.isin(acc, list(pos_acc[ba])) & (y == 1)
            for c in CLASSES + ["ALL"]:
                is_c = (y == 1) & ((cls == c) if c != "ALL" else True)
                groups = {"all": is_c}
                if ba != bb:
                    groups["shared"], groups["specific"] = is_c & shared, is_c & ~shared
                for g, mk in groups.items():
                    a, lo, hi = auc(p[mk], neg, ci=(seed == 0))
                    rows.append({"train": ba, "test": bb, "seed": seed, "class": c, "group": g, "n_pos": int(mk.sum()), "n_neg": len(neg),
                                 "auroc": a, "ci_lo": lo, "ci_hi": hi})
    print(f"[{time.time()-T0:.0f}s] train={ba} done")

res = pd.DataFrame(rows)
res.to_csv(OUT / "transfer_by_class_raw.csv", index=False)

# %% [markdown]
# ## Summary

# %%
agg = (res.groupby(["train", "test", "class", "group"])
       .agg(n_pos=("n_pos", "first"), auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), ci_lo=("ci_lo", "first"), ci_hi=("ci_hi", "first"))
       .reset_index())
agg.to_csv(OUT / "transfer_by_class.csv", index=False)

comp = {}
for ba in BIOS:
    for bb in BIOS:
        if ba == bb:
            continue
        pos = tables[bb][(tables[bb].split == "test") & (tables[bb].label_binary == 1)]
        sh = pos.accession.isin(pos_acc[ba])
        comp[f"{ba}->{bb}"] = {
            "shared_class_mix": pos[sh].classification.value_counts(normalize=True).round(3).to_dict(),
            "specific_class_mix": pos[~sh].classification.value_counts(normalize=True).round(3).to_dict(),
        }
(OUT / "class_composition.json").write_text(json.dumps(comp, indent=1))
(OUT / "protocol.json").write_text(json.dumps({"seeds": SEEDS, "device": DEV, "runtime_s": round(time.time() - T0)}, indent=1))

pd.set_option("display.width", 220, "display.max_columns", 20)
cross = agg[(agg.train != agg.test) & (agg.group != "all")]
print(cross.pivot_table(index=["train", "test", "class"], columns="group", values="auroc").round(3).to_string())
print(json.dumps(comp, indent=1)[:1500])
print(f"done in {time.time()-T0:.0f}s")
