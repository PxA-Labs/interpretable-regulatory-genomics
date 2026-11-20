# %% [markdown]
# # 15 - Does XGBoost with longer or spaced k-mers close the GM12878 CNN edge? (follow-up to notebook 14)
#
# Notebook 13: on GM12878 the CNN ensembles beat the wide 6-mer XGBoost ensemble on the GC+CpG matched set (ShallowCNN +0.012, AttnCNN +0.020).
# Notebook 14: that edge survives block shuffles and not k-let shuffles, i.e. it comes from local (~7-100 bp) context beyond 6-mer counts.
# Open question: is that context available to a *non-neural* model given longer or gapped k-mers? Here XGBoost is trained from scratch on
#   * k8      : 8-mer frequencies (65,536 features, sparse),
#   * spaced  : three weight-6 gapped k-mers (offsets 0,1,3,4,6,7 / 0,1,4,6,7,8 / 0,2,3,5,6,8; 3 x 4,096 features, sparse),
#   * k6_ctrl : the 6-mer features in the same sparse pipeline (seed 0 only; control for the sparse/max_bin=64 pipeline against notebook 11).
# Same data, split and four test sets as notebooks 07-14. Hyper-parameters: per biosample the notebook-11 selected configuration (C0) and the
# same with colsample_bytree halved (C1, since there are more features); choice on VALIDATION AUROC; final models 3 seeds (0-2), cap 6,000, patience 100,
# max_bin 64 (memory). Comparisons use the SAME seeds 0-2 of the saved notebook-07 CNNs and notebook-11 XGBoost, ensemble AUROC (seed-averaged probability).
#
# PRECOMMITTED READING (written before any run):
#  H1 (the CNN edge is longer k-mer content): on the GM12878 GC+CpG-matched set the validation-selected feature set (k8 or spaced) gives an ensemble
#     AUROC gap (CNN - this XGBoost), 95% window bootstrap, whose interval INCLUDES 0 or is negative for BOTH CNNs.
#  H0 (the edge is not reproduced by these k-mer features): the gap stays positive with the interval excluding 0 for BOTH CNNs.
#  Mixed otherwise (one CNN, or the gap shrinks by >50% but stays positive): reported as "partly", with the sizes.
#  Also reported (no rule): gain of each new feature set over the notebook-11 6-mer XGBoost in every biosample; the same ladder on shuffled test windows
#  (levels original, k-let 8, 6, 4, 250 bp blocks). By construction k8 features are unchanged at the 8-let level and k6 at the 6-let level (asserted, 1e-4).
#  A failure of H1 does not identify motifs, and a pass does not show the CNN uses k-mers: both statements are about what a k-mer model can reach.
#  Limitations stated in advance: small search (2 configs per feature set), 3 seeds, max_bin 64, four test chromosomes, training-seed variance not in intervals.
# No pretrained language models.

# %%
import glob, json, os, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
T0 = time.time()
SMOKE = os.environ.get("IRG_SMOKE") == "1"
SEEDS = [0] if SMOKE else [0, 1, 2]
B_BOOT = 20 if SMOKE else 500
CAP, ES = (30, 5) if SMOKE else (6000, 100)
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
DEV = "cuda" if torch.cuda.is_available() else "cpu"
XDEV = "cuda" if DEV == "cuda" else "cpu"
print("device", DEV, "smoke", SMOKE, "xgboost", xgb.__version__)

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


LUT = np.full(256, -1, dtype=np.int64)
for i, c in enumerate("ACGT"):
    LUT[ord(c)] = i


# %% [markdown]
# ## Features, matching, CNN models

# %%
MASKS = {"k8": [list(range(8))], "spaced": [[0, 1, 3, 4, 6, 7], [0, 1, 4, 6, 7, 8], [0, 2, 3, 5, 6, 8]], "k6_ctrl": [list(range(6))]}


def sparse_feats(seqs, masks):
    """Frequency features of (possibly gapped) k-mers as CSR; one block of 4**w columns per mask. Zero frequency = absent."""
    w = len(masks[0])
    width = 4**w
    rows, ptr, ind, dat = [], [0], [], []
    for s in seqs:
        a = LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
        cols, wts = [], []
        for mi, offs in enumerate(masks):
            span = offs[-1] + 1
            n = len(a) - span + 1
            idx, ok = np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool)
            for o in offs:
                seg = a[o : o + n]
                ok &= seg >= 0
                idx = idx * 4 + np.where(seg >= 0, seg, 0)
            u, c = np.unique(idx[ok], return_counts=True)
            cols.append(u + mi * width)
            wts.append(c / max(c.sum(), 1))
        cols, wts = np.concatenate(cols), np.concatenate(wts)
        ind.append(cols.astype(np.int32))
        dat.append(wts.astype(np.float32))
        ptr.append(ptr[-1] + len(cols))
    return sp.csr_matrix((np.concatenate(dat), np.concatenate(ind), np.array(ptr)), shape=(len(seqs), width * len(masks)))


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
# ## Load saved notebook-07 CNNs and notebook-11 XGBoost (seeds 0-2)

# %%
XGB6, CNN = {}, {}
for b in BIOS:
    for s in SEEDS:
        w = glob.glob(f"/kaggle/input/**/{b}_xgb_k6_wide_s{s}.ubj", recursive=True)
        assert w, f"missing {b} xgb wide s{s}"
        c = xgb.XGBClassifier()
        c.load_model(w[0])
        XGB6[(b, s)] = c
        for name, cls in CNNS.items():
            w = glob.glob(f"/kaggle/input/**/{b}_{name}_s{s}.pt", recursive=True)
            assert w, f"missing {b} {name} s{s}"
            mod = cls().to(DEV)
            mod.load_state_dict(torch.load(w[0], map_location=DEV))
            CNN[(b, name, s)] = mod
print("loaded", len(XGB6), "xgb6 and", len(CNN), "cnn models")
PROTO = glob.glob("/kaggle/input/**/protocol.json", recursive=True)
assert PROTO
SEL = {}
for p in PROTO:
    j = json.load(open(p))
    if "selected" in j:
        SEL = {b: j["selected"][b]["params"] for b in BIOS}
assert SEL, "notebook-11 protocol.json not found"
print("notebook-11 selected configs:", SEL)

# %% [markdown]
# ## Train (search on validation, then final seeds), then ladder on shuffled test windows

# %%
def fit(cfg, seed, Xtr, ytr, Xva, yva):
    clf = xgb.XGBClassifier(n_estimators=CAP, tree_method="hist", device=XDEV, n_jobs=-1, random_state=seed, max_bin=64,
                            early_stopping_rounds=ES, eval_metric="auc", **cfg)
    return clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)


LEVELS = [("original", None), ("kmer8", 8), ("kmer6", 6), ("kmer4", 4), ("block250", "b250")]
FAMS = ["xgb_k6_wide", "xgb_k8", "xgb_spaced", "shallow_cnn", "attn_cnn"]
SEARCH, FINAL, LADDER, DIAG, GAPS, PRED = [], [], [], [], [], {}


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
    seqs = d.sequence.tolist()
    y_all = d.label_binary.values.astype(int)
    tr, va, te = [(d.split == s).values for s in ("train", "val", "test")]
    gc = d.gc.values.astype(np.float64)
    oe = cpg_oe(seqs)
    yt = y_all[te]
    sets = {"full": np.ones(te.sum(), dtype=bool), "gc_cpg_matched": matched_mask([gc[te], oe[te]], yt, 8)}
    arr = np.array(seqs, dtype=object)
    S_te = arr[te].tolist()
    base = SEL[b]
    cfgs = {"C0": base, "C1": {**base, "colsample_bytree": max(0.05, base["colsample_bytree"] / 2)}}
    fs = {"k8": "k8", "spaced": "spaced", "k6_ctrl": "k6_ctrl"}
    MODEL = {}
    best_cfg = {}
    for fname in fs:
        Xall = sparse_feats(seqs, MASKS[fname])
        Xtr, Xva = Xall[tr], Xall[va]
        print(f"[{time.time()-T0:.0f}s] {b} {fname}: {Xall.shape}, nnz/row {Xall.nnz/Xall.shape[0]:.0f}", flush=True)
        cand = {"C0": cfgs["C0"]} if fname == "k6_ctrl" else cfgs
        for cid, cfg in cand.items():
            t1 = time.time()
            clf = fit(cfg, 0, Xtr, y_all[tr], Xva, y_all[va])
            SEARCH.append({"biosample": b, "features": fname, "config": cid, **cfg, "auroc_val": float(clf.best_score), "best_iteration": int(clf.best_iteration),
                           "hit_cap": int(clf.best_iteration) >= CAP - 5, "seconds": round(time.time() - t1)})
            pd.DataFrame(SEARCH).to_csv(OUT / "longk_search.csv", index=False)
            print(f"[{time.time()-T0:.0f}s] {b} {fname}/{cid}: val {clf.best_score:.4f} iters {clf.best_iteration} ({time.time()-t1:.0f}s)", flush=True)
            del clf
        s = pd.DataFrame([r for r in SEARCH if r["biosample"] == b and r["features"] == fname])
        best_cfg[fname] = s.loc[s.auroc_val.idxmax(), "config"]
        for seed in ([0] if fname == "k6_ctrl" else SEEDS):
            clf = fit(cfgs[best_cfg[fname]], seed, Xtr, y_all[tr], Xva, y_all[va])
            MODEL[(fname, seed)] = clf
            clf.save_model(str(MODELS_DIR / f"{b}_xgb_{fname}_s{seed}.ubj"))
            FINAL.append({"biosample": b, "features": fname, "seed": seed, "config": best_cfg[fname], "auroc_val": float(clf.best_score), "best_iteration": int(clf.best_iteration),
                          "hit_cap": int(clf.best_iteration) >= CAP - 5})
        del Xall, Xtr, Xva
    selected = max(["k8", "spaced"], key=lambda f: max(r["auroc_val"] for r in SEARCH if r["biosample"] == b and r["features"] == f))
    print(f"[{time.time()-T0:.0f}s] {b}: validation-selected feature set = {selected}", flush=True)
    SELECTED = globals().setdefault("SELECTED", {})
    SELECTED[b] = selected
    pd.DataFrame(FINAL).to_csv(OUT / "longk_final.csv", index=False)
    # ladder
    rng = np.random.default_rng(1234)
    for lname, tag in LEVELS:
        shuf = make_level(S_te, tag, rng)
        feats = {"k8": sparse_feats(shuf, MASKS["k8"]), "spaced": sparse_feats(shuf, MASKS["spaced"]), "k6": kmer_features(shuf, 6)}
        same = 1.0 if tag is None else float(np.mean([np.mean(np.frombuffer(a.encode(), np.uint8) == np.frombuffer(c.encode(), np.uint8)) for a, c in zip(S_te[:300], shuf[:300])]))
        P = {}
        for s in SEEDS:
            P[("xgb_k6_wide", s)] = XGB6[(b, s)].predict_proba(feats["k6"])[:, 1]
            P[("xgb_k8", s)] = MODEL[("k8", s)].predict_proba(feats["k8"])[:, 1]
            P[("xgb_spaced", s)] = MODEL[("spaced", s)].predict_proba(feats["spaced"])[:, 1]
            for name in CNNS:
                P[(name, s)] = cnn_predict(CNN[(b, name, s)], shuf)
        if lname == "kmer8":
            ref = np.mean([MODEL[("k8", s)].predict_proba(sparse_feats(S_te, MASKS["k8"]))[:, 1] for s in SEEDS], axis=0)
            diff = float(np.abs(np.mean([P[("xgb_k8", s)] for s in SEEDS], axis=0) - ref).max())
            assert diff < 1e-4, f"8-let shuffle changed 8-mer frequencies: {diff}"
            DIAG.append({"biosample": b, "check": "k8 xgb unchanged at 8-let", "max_abs_diff": diff})
        if lname == "kmer6":
            ref = np.mean([XGB6[(b, s)].predict_proba(kmer_features(S_te, 6))[:, 1] for s in SEEDS], axis=0)
            diff = float(np.abs(np.mean([P[("xgb_k6_wide", s)] for s in SEEDS], axis=0) - ref).max())
            assert diff < 1e-4, f"6-let shuffle changed 6-mer frequencies: {diff}"
            DIAG.append({"biosample": b, "check": "k6 xgb unchanged at 6-let", "max_abs_diff": diff})
        ens = {f: np.mean([P[(f, s)] for s in SEEDS], axis=0) for f in FAMS}
        for f in FAMS:
            PRED[(b, lname, f)] = ens[f].astype(np.float32)
            for sname, mk in sets.items():
                LADDER.append({"biosample": b, "level": lname, "set": sname, "family": f, "n": int(mk.sum()), "frac_identical_positions": same,
                               "auroc_seed_mean": float(np.mean([fast_auc(yt[mk], P[(f, s)][mk]) for s in SEEDS])), "auroc_ensemble": float(fast_auc(yt[mk], ens[f][mk]))})
        for sname, mk in sets.items():
            idx = np.where(mk)[0]
            brng = np.random.default_rng(7)
            pairs = [(c, x) for c in ("shallow_cnn", "attn_cnn") for x in ("xgb_k6_wide", "xgb_k8", "xgb_spaced")] + [("xgb_k8", "xgb_k6_wide"), ("xgb_spaced", "xgb_k6_wide")]
            boots = {p: [] for p in pairs}
            for _ in range(B_BOOT):
                j = brng.choice(idx, len(idx), replace=True)
                a = {f: fast_auc(yt[j], ens[f][j]) for f in FAMS}
                for p in pairs:
                    boots[p].append(a[p[0]] - a[p[1]])
            for p in pairs:
                pt = fast_auc(yt[idx], ens[p[0]][idx]) - fast_auc(yt[idx], ens[p[1]][idx])
                lo, hi = np.percentile(boots[p], [2.5, 97.5])
                GAPS.append({"biosample": b, "level": lname, "set": sname, "a": p[0], "b": p[1], "gap": float(pt), "ci_lo": float(lo), "ci_hi": float(hi),
                             "excludes0": bool(lo > 0 or hi < 0), "selected_feature_set": selected})
        print(f"[{time.time()-T0:.0f}s] {b} {lname}: full AUROC " + ", ".join(f"{f} {fast_auc(yt, ens[f]):.3f}" for f in FAMS), flush=True)
        pd.DataFrame(LADDER).to_csv(OUT / "longk_ladder.csv", index=False)
        pd.DataFrame(GAPS).to_csv(OUT / "longk_gaps.csv", index=False)
        pd.DataFrame(DIAG).to_csv(OUT / "longk_diagnostics.csv", index=False)
    # control: sparse k6 pipeline vs notebook-11 dense model (seed 0), original windows
    p_ctrl = MODEL[("k6_ctrl", 0)].predict_proba(sparse_feats(S_te, MASKS["k6_ctrl"]))[:, 1]
    p_ref = XGB6[(b, 0)].predict_proba(kmer_features(S_te, 6))[:, 1]
    DIAG.append({"biosample": b, "check": "sparse k6 control vs nb11 dense (seed 0) full-test AUROC", "auroc_ctrl": float(fast_auc(yt, p_ctrl)), "auroc_nb11": float(fast_auc(yt, p_ref))})
    pd.DataFrame(DIAG).to_csv(OUT / "longk_diagnostics.csv", index=False)
    pd.DataFrame({"id": d.id.values[te], "y": yt, **{f"{lv}__{f}": PRED[(b, lv, f)] for lv, _ in LEVELS for f in FAMS}}).to_parquet(OUT / f"longk_preds_{b}.parquet", index=False)
    del MODEL

json.dump({"selected_feature_set": SELECTED, "configs": {b: {"C0": SEL[b], "C1_colsample": max(0.05, SEL[b]["colsample_bytree"] / 2)} for b in BIOS}, "cap": CAP, "early_stopping": ES,
           "max_bin": 64, "seeds": SEEDS, "smoke": SMOKE, "xgboost": xgb.__version__, "runtime_s": round(time.time() - T0)}, open(OUT / "protocol.json", "w"), indent=1)
print("done", round(time.time() - T0), "s")
