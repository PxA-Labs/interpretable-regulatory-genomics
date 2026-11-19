# %% [markdown]
# # 08 - Is the surviving signal repeat content? (issue #90)
#
# Notebooks 03-06: GC matching removes the GC shortcut and a GC+CpG-matched test subset lowers AUROC by 0.09-0.17. Repeats
# (Alu, L1, simple/low-complexity) were never controlled; cCREs are repeat-depleted (rep_frac AUROC about 0.55-0.57 inverted).
# Repeat coverage comes from UCSC RepeatMasker (hg38 rmsk), computed with tools/data/repeat_annotation.py and attached as the
# dataset irg-repeat-coverage.
#
# Models (trained on all training rows, evaluation-only masking): XGBoost k=4 and k=6 (seeds 0-2), ShallowCNN and AttnCNN
# (weights from notebook 07, seeds 0-4), and a six-feature baseline [GC, CpG o/e, rep, Alu, L1, simple].
# Test sets: full; GC+CpG matched (as notebook 05/06, 8x8 bins); GC+CpG+repeat matched (6x6x6 bins, equal positives and
# negatives per cell); and repeat strata (rep_frac < 0.1, 0.1-0.4, >= 0.4). No pretrained language models.

# %%
import glob, json, os, time, warnings
from itertools import product
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
XGB_SEEDS = [0] if SMOKE else [0, 1, 2]
CNN_SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
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
XDEV = "cuda" if DEV == "cuda" else "cpu"
print("device", DEV, "repeat rows", len(REP))

# %% [markdown]
# ## Features and matching

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
    """Equal positives and negatives in every joint quantile cell of `cols` (same procedure as notebooks 05/06 for 2 columns)."""
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


# %% [markdown]
# ## CNNs (identical architecture to notebooks 06/07; weights loaded, not retrained)

# %%
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
# ## Run

# %%
RAW = OUT / "repeat_control_raw.csv"
ROWS, SIZES = [], []
WEIGHTS_FOUND = 0
for b in BIOS:
    d = pd.read_parquet(DATA / f"{b}_gc_matched.parquet")
    d = d.merge(REP[REP.biosample == b][["id", "rep_frac", "alu_frac", "l1_frac", "simple_frac"]], on="id", how="left", validate="1:1")
    assert d.rep_frac.notna().all(), f"{b}: rows without repeat annotation"
    if SMOKE:
        d = d.groupby("split", group_keys=False).apply(lambda g: g.sample(min(len(g), 1500), random_state=0)).reset_index(drop=True)
    seqs = np.array(d.sequence.tolist(), dtype=object)
    y = d.label_binary.values.astype(int)
    tr, va, te = [(d.split == s).values for s in ("train", "val", "test")]
    oe, gc = cpg_oe(seqs.tolist()), d.gc.values.astype(np.float64)
    rep = d.rep_frac.values
    yt = y[te]
    sets = {
        "full": None,
        "gc_cpg_matched": matched_mask([gc[te], oe[te]], yt, 8),
        "gc_cpg_rep_matched": matched_mask([gc[te], oe[te], rep[te]], yt, 6),
        "rep_low": rep[te] < 0.1,
        "rep_mid": (rep[te] >= 0.1) & (rep[te] < 0.4),
        "rep_high": rep[te] >= 0.4,
    }
    for nm, mk in sets.items():
        n = len(yt) if mk is None else int(mk.sum())
        SIZES.append({"biosample": b, "set": nm, "n": n, "pos_frac": float(yt.mean() if mk is None else yt[mk].mean())})
    print(f"[{time.time()-T0:.0f}s] {b}: test {te.sum()}", {k: (int(v.sum()) if v is not None else int(te.sum())) for k, v in sets.items()})

    def record(model_name, seed, p):
        row = {"biosample": b, "model": model_name, "seed": seed}
        for nm, mk in sets.items():
            row[f"auroc_{nm}"] = auc(yt, p, mk)
        ROWS.append(row)
        pd.DataFrame(ROWS).to_csv(RAW, index=False)
        print(f"[{time.time()-T0:.0f}s] {b} {model_name} s{seed}: full {row['auroc_full']:.3f} gc_cpg {row['auroc_gc_cpg_matched']:.3f} +rep {row['auroc_gc_cpg_rep_matched']:.3f}")

    # six-feature baseline: does matching remove everything these features can say?
    F6 = np.column_stack([gc, oe, rep, d.alu_frac.values, d.l1_frac.values, d.simple_frac.values])
    sc = StandardScaler().fit(F6[tr])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(F6[tr]), y[tr])
    record("feat6_lr", 0, lr.predict_proba(sc.transform(F6[te]))[:, 1])

    # XGBoost k-mers
    for k in (4, 6):
        Xk = kmer_features(seqs.tolist(), k)
        for seed in XGB_SEEDS:
            clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                    device=XDEV, n_jobs=-1, random_state=seed, early_stopping_rounds=30, eval_metric="logloss")
            clf.fit(Xk[tr], y[tr], eval_set=[(Xk[va], y[va])], verbose=False)
            record(f"xgb_k{k}", seed, clf.predict_proba(Xk[te])[:, 1])
        del Xk

    # CNNs from notebook 07 weights
    for name, cls in CNNS.items():
        for seed in CNN_SEEDS:
            w = glob.glob(f"/kaggle/input/**/{b}_{name}_s{seed}.pt", recursive=True)
            if not w:
                continue
            WEIGHTS_FOUND += 1
            model = cls().to(DEV)
            model.load_state_dict(torch.load(w[0], map_location=DEV))
            record(name, seed, cnn_predict(model, seqs[te].tolist()))
            del model
            torch.cuda.empty_cache()
if not SMOKE:
    assert WEIGHTS_FOUND == len(BIOS) * len(CNNS) * len(CNN_SEEDS), f"CNN weights found: {WEIGHTS_FOUND}"

# %% [markdown]
# ## Summary

# %%
res = pd.DataFrame(ROWS)
metric_cols = [c for c in res.columns if c.startswith("auroc_")]
agg = res.groupby(["biosample", "model"])[metric_cols].mean().reset_index()
sd = res.groupby(["biosample", "model"])[metric_cols].std().add_suffix("_sd").reset_index()
agg = agg.merge(sd, on=["biosample", "model"])
agg.to_csv(OUT / "repeat_control.csv", index=False)
pd.DataFrame(SIZES).to_csv(OUT / "test_set_sizes.csv", index=False)
(OUT / "protocol.json").write_text(json.dumps({"xgb_seeds": XGB_SEEDS, "cnn_seeds": CNN_SEEDS, "smoke": SMOKE, "device": DEV,
                                               "cnn_weights_found": WEIGHTS_FOUND, "torch": torch.__version__, "runtime_s": round(time.time() - T0)}, indent=1))
pd.set_option("display.width", 250, "display.max_columns", 30)
print(agg[["biosample", "model"] + metric_cols].round(3).to_string(index=False))
print(pd.DataFrame(SIZES).to_string(index=False))
print(f"done in {time.time()-T0:.0f}s")
