# %% [markdown]
# # 14 - Shuffle ladder: is the GM12878 CNN edge higher-order structure or composition? (W1.4)
#
# Notebook 13 showed that the CNN-over-XGBoost edge on GM12878 is real (matched sets: ShallowCNN +0.012, AttnCNN +0.020 ensemble AUROC).
# Here the *saved* models are scored on test windows whose sequence has been destroyed in a controlled way (no training):
#   * k-let shuffles (Altschul-Erickson / uShuffle, Euler paths on the (k-1)-mer de Bruijn graph): every k-mer count is preserved exactly
#     for k = 6, 5, 4, 3, 2; k = 1 is a plain base shuffle (floor). 
#   * block shuffles: the window is cut into blocks of 100 or 250 bp and the blocks are permuted (local content kept, long-range order broken).
# Each level also reports how much of the original window survives (fraction of identical positions; overlap of the 6-mer frequency
# spectrum). A unit test on random 1 kb sequences gave about 25-26% identical positions at every k, i.e. chance level: k-let shuffles
# rebuild the window by a different path through the same k-mers, so long-range order is destroyed while the k-mer composition is kept.
#
# PRECOMMITTED READING (written before the run; refinement of the W1.4 rule in docs/research/novelty_and_next_experiments.md):
#  * By construction wide XGBoost (6-mer frequencies) is unchanged at the 6-let level; the run asserts this (max |score difference| < 1e-4).
#  * The CNN edge is called "carried by structure beyond 6-mer composition" if, on the GM12878 GC+CpG-matched set, the CNN-minus-XGBoost
#    ensemble AUROC gap stays above 0 with the 95% window-bootstrap interval excluding 0 at the 5-let level (6-mers broken, 5-mers kept)
#    AND at the 4-let level, for the same architecture.
#  * It is called "explained by low-order composition" if the gap's interval includes 0 (or is negative) at the 4-let level.
#  * Anything between (positive at 5-let, not at 4-let) is reported as "partly", with the level at which the gap closes.
#  * All of this is read per architecture and per biosample; T cell and K562 are reported but have no precommitted prediction.
#  * Shuffled windows are off-distribution for the CNNs, so the absolute AUROCs are robustness measurements, not performance claims.
# No pretrained language models; no training.

# %%
import glob, os, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
T0 = time.time()
SMOKE = os.environ.get("IRG_SMOKE") == "1"
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
B_BOOT = 20 if SMOKE else 500
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
# ## Shuffles (with tests)

# %%
def kshuffle(s, k, rng):
    """Random sequence with exactly the same k-mer counts (and the same first/last (k-1)-mer) as s. k=1: permutation of the bases."""
    if k <= 1:
        a = list(s)
        rng.shuffle(a)
        return "".join(a)
    mm = k - 1
    L = len(s)
    adj = defaultdict(list)
    for i in range(L - mm):
        adj[s[i : i + mm]].append(s[i + 1 : i + mm + 1])
    start, end = s[:mm], s[L - mm :]
    nxt, in_tree = {}, {end}
    nodes = list(adj.keys())
    rng.shuffle(nodes)
    for u in nodes:  # Wilson's algorithm: loop-erased random walks give the last-exit tree
        cur = u
        while cur not in in_tree:
            nxt[cur] = int(rng.integers(len(adj[cur])))
            cur = adj[cur][nxt[cur]]
        cur = u
        while cur not in in_tree:
            in_tree.add(cur)
            cur = adj[cur][nxt[cur]]
    order = {}
    for u, outs in adj.items():
        idx = list(range(len(outs)))
        if u in nxt and u != end:
            last = nxt[u]
            idx.remove(last)
            rng.shuffle(idx)
            idx.append(last)
        else:
            rng.shuffle(idx)
        order[u] = [outs[i] for i in idx]
    ptr = defaultdict(int)
    out, cur = [start], start
    for _ in range(L - mm):
        nx = order[cur][ptr[cur]]
        ptr[cur] += 1
        out.append(nx[-1])
        cur = nx
    return "".join(out)


def block_shuffle(s, blk, rng):
    parts = [s[i : i + blk] for i in range(0, len(s), blk)]
    rng.shuffle(parts)
    return "".join(parts)


def kmer_counts(s, k):
    c = defaultdict(int)
    for i in range(len(s) - k + 1):
        c[s[i : i + k]] += 1
    return dict(c)


_rng = np.random.default_rng(0)
_s = "".join(_rng.choice(list("ACGT"), 1000))
for k in (2, 3, 4, 5, 6):
    t = kshuffle(_s, k, _rng)
    assert len(t) == len(_s) and kmer_counts(t, k) == kmer_counts(_s, k), f"k-let shuffle broke {k}-mer counts"
assert sorted(block_shuffle(_s, 100, _rng)) == sorted(_s)
print("shuffle tests passed; 5-let shuffle keeps", round(np.mean([a == b for a, b in zip(kshuffle(_s, 5, _rng), _s)]), 3), "of positions")

# %% [markdown]
# ## Features, matching, models (identical to notebooks 08 and 13)

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


def fast_auc(y, p):
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# %% [markdown]
# ## Load models once

# %%
XGB, CNN = {}, {}
for b in BIOS:
    for s in SEEDS:
        w = glob.glob(f"/kaggle/input/**/{b}_xgb_k6_wide_s{s}.ubj", recursive=True)
        assert w, f"missing {b} xgb wide s{s}"
        c = xgb.XGBClassifier()
        c.load_model(w[0])
        XGB[(b, s)] = c
        for name, cls in CNNS.items():
            w = glob.glob(f"/kaggle/input/**/{b}_{name}_s{s}.pt", recursive=True)
            assert w, f"missing {b} {name} s{s}"
            mod = cls().to(DEV)
            mod.load_state_dict(torch.load(w[0], map_location=DEV))
            CNN[(b, name, s)] = mod
print("loaded", len(XGB), "xgb and", len(CNN), "cnn models")

# %% [markdown]
# ## Ladder

# %%
LEVELS = [("original", None), ("kmer6", 6), ("kmer5", 5), ("kmer4", 4), ("kmer3", 3), ("kmer2", 2), ("kmer1", 1), ("block250", "b250"), ("block100", "b100")]
FAMS = ["xgb_wide", "shallow_cnn", "attn_cnn"]
LADDER, DIAG, PRED, GAPS = [], [], {}, []


def make_level(seqs, tag, rng):
    if tag is None:
        return list(seqs)
    if isinstance(tag, str):
        return [block_shuffle(s, int(tag[1:]), rng) for s in seqs]
    return [kshuffle(s, tag, rng) for s in seqs]


for b in BIOS:
    d = pd.read_parquet(DATA / f"{b}_gc_matched.parquet")
    d = d.merge(REP[REP.biosample == b][["id", "rep_frac"]], on="id", how="left", validate="1:1")
    if SMOKE:
        d = d.groupby("split", group_keys=False).apply(lambda g: g.sample(min(len(g), 1500), random_state=0)).reset_index(drop=True)
    seqs = np.array(d.sequence.tolist(), dtype=object)
    oe = cpg_oe(seqs.tolist())
    gc = d.gc.values.astype(np.float64)
    te = (d.split == "test").values
    y = d.label_binary.values.astype(int)[te]
    sets = {"full": np.ones(len(y), dtype=bool), "gc_cpg_matched": matched_mask([gc[te], oe[te]], y, 8)}
    orig = seqs[te].tolist()
    orig6 = kmer_features(orig, 6)
    rng = np.random.default_rng(1234)
    for lname, tag in LEVELS:
        shuf = make_level(orig, tag, rng)
        if tag is not None:
            same = float(np.mean([np.mean(np.frombuffer(a.encode(), np.uint8) == np.frombuffer(c.encode(), np.uint8)) for a, c in zip(orig[:500], shuf[:500])]))
            Xs = kmer_features(shuf, 6)
            keep6 = float(np.mean(np.minimum(Xs, orig6).sum(1)))  # overlap of 6-mer frequency spectra (1 = identical)
        else:
            same, keep6, Xs = 1.0, 1.0, orig6
        DIAG.append({"biosample": b, "level": lname, "frac_identical_positions": same, "spectrum6_overlap": keep6})
        P = {}
        for s in SEEDS:
            P[("xgb_wide", s)] = XGB[(b, s)].predict_proba(Xs)[:, 1]
            for name in CNNS:
                P[(name, s)] = cnn_predict(CNN[(b, name, s)], shuf)
        if lname == "kmer6":
            ref = np.mean([XGB[(b, s)].predict_proba(orig6)[:, 1] for s in SEEDS], axis=0)
            diff = float(np.abs(np.mean([P[("xgb_wide", s)] for s in SEEDS], axis=0) - ref).max())
            DIAG[-1]["xgb_max_abs_diff_vs_original"] = diff
            assert diff < 1e-4, f"6-let shuffle changed 6-mer frequencies: {diff}"
        ens = {f: np.mean([P[(f, s)] for s in SEEDS], axis=0) for f in FAMS}
        for f in FAMS:
            PRED[(b, lname, f)] = ens[f].astype(np.float32)
            for sname, mk in sets.items():
                LADDER.append({"biosample": b, "level": lname, "set": sname, "family": f, "n": int(mk.sum()),
                               "auroc_seed_mean": float(np.mean([fast_auc(y[mk], P[(f, s)][mk]) for s in SEEDS])),
                               "auroc_ensemble": float(fast_auc(y[mk], ens[f][mk]))})
        for sname, mk in sets.items():
            idx = np.where(mk)[0]
            brng = np.random.default_rng(7)
            for cnn in ("shallow_cnn", "attn_cnn"):
                dd = []
                for _ in range(B_BOOT):
                    j = brng.choice(idx, len(idx), replace=True)
                    dd.append(fast_auc(y[j], ens[cnn][j]) - fast_auc(y[j], ens["xgb_wide"][j]))
                pt = fast_auc(y[idx], ens[cnn][idx]) - fast_auc(y[idx], ens["xgb_wide"][idx])
                GAPS.append({"biosample": b, "level": lname, "set": sname, "cnn": cnn, "gap_cnn_minus_xgb": float(pt),
                             "ci_lo": float(np.percentile(dd, 2.5)), "ci_hi": float(np.percentile(dd, 97.5)), "excludes0": bool(np.percentile(dd, 2.5) > 0 or np.percentile(dd, 97.5) < 0)})
        print(f"[{time.time()-T0:.0f}s] {b} {lname}: identical pos {same:.2f}, 6-mer spectrum overlap {keep6:.2f}, full AUROC "
              + ", ".join(f"{f} {fast_auc(y, ens[f]):.3f}" for f in FAMS), flush=True)
        pd.DataFrame(LADDER).to_csv(OUT / "ladder.csv", index=False)
        pd.DataFrame(GAPS).to_csv(OUT / "ladder_gaps.csv", index=False)
        pd.DataFrame(DIAG).to_csv(OUT / "shuffle_diagnostics.csv", index=False)
    pd.DataFrame({"id": d.id.values[te], "y": y, **{f"{lv}__{f}": PRED[(b, lv, f)] for lv, _ in LEVELS for f in FAMS}}).to_parquet(OUT / f"ladder_preds_{b}.parquet", index=False)
print("done", round(time.time() - T0), "s")
