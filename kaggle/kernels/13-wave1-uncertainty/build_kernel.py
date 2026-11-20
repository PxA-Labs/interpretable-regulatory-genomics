"""Generate 13_wave1_uncertainty.py: header + prediction cell + tools/analysis/wave1_uncertainty.py embedded verbatim (without its CLI main).

  python kaggle/kernels/13-wave1-uncertainty/build_kernel.py
"""
from pathlib import Path

here = Path(__file__).parent
analysis = (here.parents[2] / "tools" / "analysis" / "wave1_uncertainty.py").read_text()
analysis = analysis[analysis.index("from __future__") : analysis.index("def main():")]
analysis = "\n".join(l for l in analysis.splitlines() if l.strip() != "from __future__ import annotations")

HEAD = '''# %% [markdown]
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
'''

TAIL = '''

# %%
w11(str(OUT), str(OUT), B_BOOT, 0)
w12(str(OUT), str(DATA), str(OUT), B_BOOT, 0)
print("done", round(time.time() - T0), "s")
'''
(here / "13_wave1_uncertainty.py").write_text(HEAD + analysis + TAIL)
print("wrote", here / "13_wave1_uncertainty.py")
