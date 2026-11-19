# %% [markdown]
# # 10 - Do the CNN filters match known transcription-factor motifs, with FDR control? (issues #11, #13, #90)
#
# Discussion #11 claimed the CNN "definitively proves" real TF binding sites on the basis of Tomtom E-values < 0.5 without
# multiple-testing correction. This notebook redoes the question with an explicit null and Benjamini-Hochberg FDR.
# **It is not Tomtom** (no MEME suite offline); it is a documented column-correlation matcher with a shuffle null.
#
# 1. For each of the 30 notebook-07 CNNs (3 biosamples x {ShallowCNN, AttnCNN} x 5 seeds), build a position-frequency matrix
#    per first-layer filter from the windows that activate it above 0.7 x its maximum over the VALIDATION sequences
#    (test chromosomes are not used). Filters with fewer than 30 windows are dropped.
# 2. Score every filter against all 879 JASPAR 2024 CORE vertebrate non-redundant motifs: mean Pearson correlation over
#    overlapping columns (overlap >= min(6, motif width)), best over both strands and all offsets.
# 3. Null: each motif's columns are randomly permuted (200 shuffles per motif, so composition is kept and column order is
#    destroyed); scores of the shuffled motifs against the same filters are pooled per motif-width bin. p = (1 + #null >= obs)/(1 + N).
# 4. Benjamini-Hochberg over all filter x motif tests of a model; report q < 0.05 and q < 0.01.
# 5. Stability across the 5 seeds, specificity across biosamples (Jaccard), and for ShallowCNN the validation-AUROC drop when a
#    filter is switched off (forward strand only), to see whether significant filters are functionally used.
# No pretrained language models.

# %%
import glob, json, os, re, time, warnings, zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
T0 = time.time()
SMOKE = os.environ.get("IRG_SMOKE") == "1"
ROOT = os.environ.get("IRG_INPUT", "/kaggle/input")
OUT = Path(os.environ.get("IRG_OUT", "/kaggle/working/results"))
OUT.mkdir(parents=True, exist_ok=True)
BIOS = ["tcell", "k562", "gm12878"]
ARCHS = ["shallow_cnn", "attn_cnn"]
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
K_SHUF = 3 if SMOKE else 200
THR_FRAC, MIN_WIN, PSEUDO = 0.7, 30, 1.0
WIDTH_EDGES = [0, 8, 10, 12, 15, 20, 1000]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def find(name):
    m = glob.glob(f"{ROOT}/**/{name}", recursive=True)
    assert m, f"{name} not found under {ROOT}"
    return m[0]


DATA = Path(find("tcell_gc_matched.parquet")).parent
print("device", DEV, "data", DATA)

# %% [markdown]
# ## JASPAR

# %%
def read_meme(path):
    motifs, cur = [], None
    for line in open(path):
        if line.startswith("MOTIF"):
            parts = line.split()
            cur = {"id": parts[1], "name": " ".join(parts[2:]), "rows": []}
            motifs.append(cur)
        elif cur is not None and line.strip() and line[0] in " 0123456789." and not line.startswith("letter"):
            vals = line.split()
            if len(vals) == 4:
                cur["rows"].append([float(v) for v in vals])
    for m in motifs:
        P = np.array(m.pop("rows"))
        P = (P + 1e-3) / (P + 1e-3).sum(1, keepdims=True)
        m["P"] = P
    return motifs


JASPAR = read_meme(find("jaspar2024_core_vert_nr.meme"))
if SMOKE:
    JASPAR = JASPAR[:120]
print(len(JASPAR), "motifs; widths", min(len(m["P"]) for m in JASPAR), "-", max(len(m["P"]) for m in JASPAR))

# %% [markdown]
# ## CNNs (same definitions as notebook 07) and filter PFMs

# %%
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
        return self.fc(self.drop(torch.cat([self.norm(h + a).mean(1), self.norm(h + a).amax(1)], -1))).squeeze(-1)


CLS = {"shallow_cnn": ShallowCNN, "attn_cnn": AttnCNN}
ENC = np.full(256, 4, dtype=np.uint8)
for i, c in enumerate("ACGT"):
    ENC[ord(c)] = i


def onehot(idx):
    return F.one_hot(idx.clamp(max=4), 5)[..., :4].float().permute(0, 2, 1)


@torch.no_grad()
def filter_pfms(conv, X, bs=256):
    """Position-frequency matrices (F, k, 4) and window counts from windows with activation > THR_FRAC * max (per filter)."""
    k, nf = conv.kernel_size[0], conv.out_channels
    mx = torch.zeros(nf, device=DEV)
    for i in range(0, len(X), bs):
        mx = torch.maximum(mx, F.relu(conv(onehot(X[i : i + bs].to(DEV)))).amax((0, 2)))
    thr = THR_FRAC * mx
    counts = torch.zeros(nf, 4, k, device=DEV)
    nwin = torch.zeros(nf, device=DEV)
    for i in range(0, len(X), bs):
        oh = onehot(X[i : i + bs].to(DEV))
        mask = ((F.relu(conv(oh)) > thr[None, :, None]) & (mx > 0)[None, :, None]).float()
        win = F.pad(oh, (k // 2, k // 2)).unfold(2, k, 1)  # (B, 4, L, k)
        counts += torch.einsum("bfl,bclk->fck", mask, win)
        nwin += mask.sum((0, 2))
    c = counts.permute(0, 2, 1).cpu().numpy()  # (F, k, 4)
    n = nwin.cpu().numpy()
    return c, n


def pfm_to_prob(c):
    return (c + PSEUDO / 4) / (c.sum(-1, keepdims=True) + PSEUDO)


def info_content(P):
    return (2 + (P * np.log2(P)).sum(-1)).sum(-1)


# %% [markdown]
# ## Matching statistic and null

# %%
def std_cols(P):
    v = P - P.mean(-1, keepdims=True)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-8, v / np.maximum(n, 1e-8), 0.0)


def offsets(k, w):
    mo = min(6, w, k)
    for o in range(-(w - mo), k - mo + 1):
        fs, ms = max(0, o), max(0, -o)
        n = min(k - fs, w - ms)
        if n >= mo:
            yield fs, ms, n


def best_scores(Vf, motif_Vs):
    """Vf (F,k,4) standardized filter columns; motif_Vs list of (w,4) standardized. Returns (F, M) best mean column PCC."""
    nf, k, _ = Vf.shape
    out = np.full((nf, len(motif_Vs)), -1.0, dtype=np.float32)
    cache = {}
    for j, Vm in enumerate(motif_Vs):
        w = len(Vm)
        offs = cache.setdefault(w, list(offsets(k, w)))
        for Vs in (Vm, Vm[::-1, ::-1]):  # reverse complement: reverse columns, A<->T, C<->G (standardisation is channel-permutation invariant)
            C = np.einsum("fka,wa->fkw", Vf, Vs)
            for fs, ms, n in offs:
                s = np.einsum("fnn->f", C[:, fs : fs + n, ms : ms + n]) / n
                np.maximum(out[:, j], s, out=out[:, j])
    return out


def bh(p):
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


MOTIF_V = [std_cols(m["P"]) for m in JASPAR]
MOTIF_W = np.array([len(m["P"]) for m in JASPAR])
WBIN = np.digitize(MOTIF_W, WIDTH_EDGES[1:-1])


def match_one(key, C, nwin):
    """key=(bio,arch,seed); C (F,k,4) counts; returns (hits_df, summary_dict, pair_df)."""
    rng = np.random.default_rng(zlib.crc32(str(key).encode()))
    keep = np.where(nwin >= MIN_WIN)[0]
    P = pfm_to_prob(C[keep])
    ic = info_content(P)
    Vf = std_cols(P)
    obs = best_scores(Vf, MOTIF_V)  # (Fk, M)
    null_bins = [[] for _ in range(len(WIDTH_EDGES) - 1)]
    shuf_Vs, shuf_bin = [], []
    for j, Vm in enumerate(MOTIF_V):
        for _ in range(K_SHUF):
            shuf_Vs.append(Vm[rng.permutation(len(Vm))])
            shuf_bin.append(WBIN[j])
    nul = best_scores(Vf, shuf_Vs)  # (Fk, M*K)
    shuf_bin = np.array(shuf_bin)
    pv = np.ones_like(obs, dtype=np.float64)
    for b in np.unique(WBIN):
        pool = np.sort(nul[:, shuf_bin == b].ravel())
        cols = np.where(WBIN == b)[0]
        # number of null scores >= obs
        ge = len(pool) - np.searchsorted(pool, obs[:, cols], side="left")
        pv[:, cols] = (1 + ge) / (1 + len(pool))
    q = bh(pv.ravel()).reshape(pv.shape)
    best = pv.argmin(1)
    rows = []
    for fi, f in enumerate(keep):
        j = int(best[fi])
        rows.append({"biosample": key[0], "model": key[1], "seed": key[2], "filter": int(f), "n_windows": int(nwin[f]), "ic": float(ic[fi]),
                     "motif_id": JASPAR[j]["id"], "motif_name": JASPAR[j]["name"], "score": float(obs[fi, j]), "p": float(pv[fi, j]), "q": float(q[fi, j]),
                     "min_q_filter": float(q[fi].min())})
    sig = np.argwhere(q < 0.05)
    pairs = [{"biosample": key[0], "model": key[1], "seed": key[2], "filter": int(keep[fi]), "motif_id": JASPAR[j]["id"], "motif_name": JASPAR[j]["name"],
              "score": float(obs[fi, j]), "q": float(q[fi, j])} for fi, j in sig]
    summ = {"biosample": key[0], "model": key[1], "seed": key[2], "n_filters": len(keep), "n_tests": int(pv.size), "null_size_total": int(nul.size),
            "filters_q05": int((q.min(1) < 0.05).sum()), "filters_q01": int((q.min(1) < 0.01).sum()), "pairs_q05": int((q < 0.05).sum()),
            "median_ic": float(np.median(ic)), "min_p": float(pv.min())}
    return pd.DataFrame(rows), summ, pd.DataFrame(pairs)


# %% [markdown]
# ## Run: extract PFMs on the validation sequences, then match (parallel on CPU)

# %%
tables = {}
PF = {}
IMPORT = []
WEIGHTS = {}
for b in BIOS:
    d = pd.read_parquet(DATA / f"{b}_gc_matched.parquet", columns=["sequence", "label_binary", "split"])
    d = d[d.split == "val"].reset_index(drop=True)
    if SMOKE:
        d = d.sample(min(len(d), 1500), random_state=0).reset_index(drop=True)
    X = torch.from_numpy(np.stack([ENC[np.frombuffer(s.encode(), dtype=np.uint8)] for s in d.sequence])).long()
    yv = d.label_binary.values.astype(int)
    tables[b] = (X, yv)
    for arch in ARCHS:
        for seed in SEEDS:
            w = glob.glob(f"{ROOT}/**/{b}_{arch}_s{seed}.pt", recursive=True)
            assert w, f"missing weights {b} {arch} {seed}"
            model = CLS[arch]().to(DEV)
            model.load_state_dict(torch.load(w[0], map_location=DEV, weights_only=True))
            model.eval()
            conv = model.conv if arch == "shallow_cnn" else model.conv1
            PF[(b, arch, seed)] = filter_pfms(conv, X)
            if arch == "shallow_cnn":  # functional importance: validation AUROC drop when one filter's pooled feature is zeroed (forward strand)
                with torch.no_grad():
                    pooled = torch.cat([F.relu(conv(onehot(X[i : i + 256].to(DEV)))).amax(-1).cpu() for i in range(0, len(X), 256)])
                    fcw, fcb = model.fc.weight.cpu()[0], model.fc.bias.cpu()
                    base = roc_auc_score(yv, (pooled @ fcw + fcb).numpy())
                    for f in range(pooled.shape[1]):
                        z = pooled.clone()
                        z[:, f] = 0
                        IMPORT.append({"biosample": b, "model": arch, "seed": seed, "filter": f, "auroc_val_base": float(base),
                                       "delta_auroc_val": float(base - roc_auc_score(yv, (z @ fcw + fcb).numpy()))})
            del model
    print(f"[{time.time()-T0:.0f}s] extracted {b}: {len(X)} validation sequences")
pd.DataFrame(IMPORT).to_csv(OUT / "filter_importance.csv", index=False)
np.savez_compressed(OUT / "filter_pfm_counts.npz", **{f"{k[0]}__{k[1]}__s{k[2]}__counts": v[0] for k, v in PF.items()},
                    **{f"{k[0]}__{k[1]}__s{k[2]}__nwin": v[1] for k, v in PF.items()})

# %%
res = Parallel(n_jobs=-1, verbose=0)(delayed(match_one)(k, v[0], v[1]) for k, v in PF.items())
HITS = pd.concat([r[0] for r in res], ignore_index=True)
SUMM = pd.DataFrame([r[1] for r in res])
PAIRS = pd.concat([r[2] for r in res], ignore_index=True) if any(len(r[2]) for r in res) else pd.DataFrame()
HITS.to_csv(OUT / "motif_hits_best_per_filter.csv", index=False)
SUMM.to_csv(OUT / "motif_match_summary.csv", index=False)
PAIRS.to_csv(OUT / "motif_pairs_q05.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(f"[{time.time()-T0:.0f}s] matching done")
print(SUMM.groupby(["biosample", "model"])[["n_filters", "filters_q05", "filters_q01"]].mean().round(1))

# %% [markdown]
# ## Stability across seeds, specificity across biosamples, functional importance

# %%
def base_name(n):  # family-ish key: first TF of a heterodimer name, upper-case
    return re.split(r"::|\(", n)[0].upper()


if len(PAIRS):
    PAIRS["tf"] = PAIRS.motif_name.map(base_name)
    per_seed = PAIRS.groupby(["biosample", "model", "seed"]).tf.agg(lambda s: set(s)).reset_index()
    stab = []
    for (b, a), g in per_seed.groupby(["biosample", "model"]):
        from collections import Counter
        cnt = Counter(tf for s in g.tf for tf in s)
        for tf, n in cnt.items():
            stab.append({"biosample": b, "model": a, "tf": tf, "n_seeds_significant": n, "n_seeds": len(g)})
    STAB = pd.DataFrame(stab).sort_values(["biosample", "model", "n_seeds_significant"], ascending=[True, True, False])
    STAB.to_csv(OUT / "motif_stability.csv", index=False)
    stable = STAB[STAB.n_seeds_significant >= max(2, int(np.ceil(0.6 * STAB.n_seeds.max())))]
    sets = {(b, a): set(g.tf) for (b, a), g in stable.groupby(["biosample", "model"])}
    jac = []
    for a in ARCHS:
        for i, b1 in enumerate(BIOS):
            for b2 in BIOS[i + 1 :]:
                s1, s2 = sets.get((b1, a), set()), sets.get((b2, a), set())
                jac.append({"model": a, "pair": f"{b1}-{b2}", "n1": len(s1), "n2": len(s2), "shared": len(s1 & s2),
                            "jaccard": len(s1 & s2) / max(1, len(s1 | s2))})
    pd.DataFrame(jac).to_csv(OUT / "motif_jaccard_between_biosamples.csv", index=False)
    print(STAB.groupby(["biosample", "model"]).head(8).to_string(index=False))
    print(pd.DataFrame(jac).round(3).to_string(index=False))

IMP = pd.read_csv(OUT / "filter_importance.csv").merge(
    HITS[HITS.model == "shallow_cnn"][["biosample", "seed", "filter", "min_q_filter", "motif_name"]], on=["biosample", "seed", "filter"], how="left")
IMP["sig_q05"] = IMP.min_q_filter < 0.05
IMP.to_csv(OUT / "filter_importance_with_hits.csv", index=False)
print(IMP.groupby(["biosample", "sig_q05"]).delta_auroc_val.agg(["count", "mean", "median"]).round(4))
(OUT / "protocol.json").write_text(json.dumps({"k_shuffles": K_SHUF, "thr_frac": THR_FRAC, "min_windows": MIN_WIN, "pseudocount": PSEUDO, "width_edges": WIDTH_EDGES,
                                               "n_motifs": len(JASPAR), "seeds": SEEDS, "smoke": SMOKE, "device": DEV, "torch": torch.__version__,
                                               "runtime_s": round(time.time() - T0)}, indent=1))
print(f"done in {time.time()-T0:.0f}s")
