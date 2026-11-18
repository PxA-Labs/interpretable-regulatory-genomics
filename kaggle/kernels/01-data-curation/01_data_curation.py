# %% [markdown]
# # 01 - Data curation (v2)
#
# Rebuilds the regulatory-element tables for **three correctly identified ENCODE biosamples** and validates them.
#
# | key | ENCODE file | biosample (from ENCODE metadata) |
# |---|---|---|
# | `tcell` | ENCFF726XBE | naive CD8+ alpha-beta T cell (the legacy "k562_*" tables came from here) |
# | `k562` | ENCFF464BRU | K562 |
# | `gm12878` | ENCFF590IMH | GM12878 |
#
# Outputs (`/kaggle/working/curated/`): per biosample x negative strategy parquet tables with canonical chromosome split,
# near-duplicate (homology) flags, 3-class labels; a validation report; SLM-ready JSONL; SHA-256 manifest.
#
# Canonical split: train = chr1-14, X, Y; validation = chr15-18; test = chr19-22.

# %%
import glob, gzip, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 42
T0 = time.time()
WORK = Path("/kaggle/working")
OUT = WORK / "curated"
OUT.mkdir(parents=True, exist_ok=True)
(WORK / "data/raw").mkdir(parents=True, exist_ok=True)
(WORK / "data/processed").mkdir(parents=True, exist_ok=True)
(WORK / "data/reference").mkdir(parents=True, exist_ok=True)


def find_one(name, root="/kaggle/input"):
    hits = glob.glob(f"{root}/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}; attached inputs: {os.listdir(root)}")
    return hits[0]


def find_raw_bed(acc, root="/kaggle/input"):
    """Kaggle auto-extracts uploaded .gz archives, so accept .bed.gz, <acc>.bed/<anything>.bed or <acc>.bed."""
    for pat in (f"{acc}.bed.gz", f"{acc}.bed/*.bed", f"{acc}.bed"):
        hits = [h for h in glob.glob(f"{root}/**/{pat}", recursive=True) if os.path.isfile(h)]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"raw cCRE file for {acc} not found under {root}")


# --- source code snapshot (pinned) ---
src_marker = find_one("negative_sampling.py")
SRC_ROOT = str(Path(src_marker).parents[2])  # .../<root> containing src/
sys.path.insert(0, SRC_ROOT)
GIT_SHA = Path(find_one("GIT_SHA.txt")).read_text().strip()
from src.data.negative_sampling import build_negative_dataset  # noqa: E402
from src.data.parse_encode import parse_and_resize_ccres  # noqa: E402
from src.data.sequence_extractor import extract_sequences  # noqa: E402

print("source snapshot:", SRC_ROOT, "sha", GIT_SHA)
print("python", platform.python_version(), "pandas", pd.__version__, "numpy", np.__version__)

# %% [markdown]
# ## 1. Mount reference genome (GRCh38) and raw cCRE files

# %%
genome_dir = Path(find_one("Homo_sapiens.GRCh38.dna.chromosome.1.fa")).parent
ref = WORK / "data/reference"
for c in [*map(str, range(1, 23)), "X", "Y"]:
    dst = ref / f"chr{c}.fa"
    if not dst.exists():
        os.symlink(genome_dir / f"Homo_sapiens.GRCh38.dna.chromosome.{c}.fa", dst)
print("genome:", genome_dir, "->", len(list(ref.glob('chr*.fa'))), "chromosomes linked")

BIOSAMPLES = {
    "tcell": ("ENCFF726XBE", "naive CD8+ alpha-beta T cell (CL:0000900)"),
    "k562": ("ENCFF464BRU", "K562 (EFO:0002067)"),
    "gm12878": ("ENCFF590IMH", "GM12878 (EFO:0002784)"),
}
STRATEGIES = ["gc_matched", "random_unmatched", "flanking"]
ELEMENT_TYPES = ["PLS", "pELS", "dELS"]

# %% [markdown]
# ## 2. Positives -> sequences -> negatives (three strategies)

# %%
tables = {}
for key, (acc, desc) in BIOSAMPLES.items():
    raw = find_raw_bed(acc)
    print(key, "raw file:", raw)
    bed = str(WORK / f"data/processed/{key}.bed")
    pos_tsv = str(WORK / f"data/processed/{key}_pos.tsv")
    parse_and_resize_ccres(raw, bed, element_types=ELEMENT_TYPES, target_length=1000)
    extract_sequences(bed, str(ref), pos_tsv, max_n_fraction=0.10)
    for strat in STRATEGIES:
        out_tsv = str(WORK / f"data/processed/{key}_{strat}.tsv")
        build_negative_dataset(pos_tsv, str(ref), out_tsv, strategy=strat, multiclass=False, random_seed=SEED)
        df = pd.read_csv(out_tsv, sep="\t")
        df["biosample"] = key
        df["negative_strategy"] = strat
        tables[(key, strat)] = df
        print(f"[{key}/{strat}] rows={len(df):,} pos={(df.label==1).sum():,} neg={(df.label==0).sum():,}")

# %% [markdown]
# ## 3. Canonical split, 3-class label, stable ids

# %%
TRAIN = {f"chr{i}" for i in range(1, 15)} | {"chrX", "chrY"}
VAL = {f"chr{i}" for i in range(15, 19)}
TEST = {f"chr{i}" for i in range(19, 23)}


def split_of(c):
    return "train" if c in TRAIN else "val" if c in VAL else "test" if c in TEST else "unassigned"


def label3(cls, lab):
    if lab == 0:
        return 0
    u = str(cls).upper()
    return 1 if "PLS" in u else 2


for k, df in tables.items():
    df["split"] = df.chrom.map(split_of)
    df["label_binary"] = df.label.astype(int)
    df["label_3class"] = [label3(c, l) for c, l in zip(df.classification, df.label)]
    df["id"] = (df.biosample + "|" + df.negative_strategy + "|" + df.chrom + ":" + df.start.astype(str) + "-"
                + df.end.astype(str) + "|" + df.label_binary.astype(str))
    assert (df.split != "unassigned").all(), "chromosome outside canonical split"

# %% [markdown]
# ## 4. Homology / near-duplicate flagging (8-mer Jaccard, test vs train)
# Chromosome hold-out does not remove repeat-derived or paralogous sequence shared across splits. We flag test/val rows
# whose maximum 8-mer Jaccard similarity to any **training** sequence exceeds a threshold, so downstream evaluation can
# report metrics with and without them.

# %%
K = 8
LUT = np.full(256, -1, dtype=np.int64)
for i, b in enumerate("ACGT"):
    LUT[ord(b)] = i


def kmer_matrix(seqs, k=K):
    rows, cols = [], []
    for r, s in enumerate(seqs):
        a = LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
        if len(a) < k:
            continue
        idx = np.zeros(len(a) - k + 1, dtype=np.int64)
        valid = np.ones(len(a) - k + 1, dtype=bool)
        for j in range(k):
            seg = a[j : len(a) - k + 1 + j]
            valid &= seg >= 0
            idx = idx * 4 + np.where(seg >= 0, seg, 0)
        u = np.unique(idx[valid])
        rows.append(np.full(len(u), r))
        cols.append(u)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(seqs), 4**k))


def max_jaccard(query, ref, chunk=400):
    nq, nr = np.asarray(query.sum(1)).ravel(), np.asarray(ref.sum(1)).ravel()
    refT = ref.T.tocsc()
    best = np.zeros(query.shape[0], dtype=np.float32)
    for s in range(0, query.shape[0], chunk):
        inter = (query[s : s + chunk] @ refT).toarray()
        jac = inter / (nq[s : s + chunk, None] + nr[None, :] - inter + 1e-9)
        best[s : s + chunk] = jac.max(1)
    return best


HOMOLOGY_THRESHOLD = 0.30
for (key, strat), df in tables.items():
    X = kmer_matrix(df.sequence.tolist())
    tr = (df.split == "train").to_numpy()
    ev = ~tr
    mj = np.zeros(len(df), dtype=np.float32)
    mj[ev] = max_jaccard(X[ev], X[tr])
    df["max_train_jaccard"] = mj
    df["homology_flag"] = (df.split != "train") & (df.max_train_jaccard >= HOMOLOGY_THRESHOLD)
    print(f"[{key}/{strat}] flagged eval rows: {int(df.homology_flag.sum()):,} / {int(ev.sum()):,} "
          f"(p99 jaccard={np.percentile(mj[ev], 99):.3f})")

# %% [markdown]
# ## 5. Validation report

# %%
report = {"git_sha": GIT_SHA, "seed": SEED, "homology_threshold_8mer_jaccard": HOMOLOGY_THRESHOLD, "tables": {}}
ALPHA = set("ACGTN")
for (key, strat), df in tables.items():
    seqs = df.sequence
    nfrac = seqs.str.count("N") / seqs.str.len()
    dup_seq = int(seqs.duplicated().sum())
    pos = df[df.label_binary == 1]
    r = {
        "rows": len(df),
        "positives": int((df.label_binary == 1).sum()),
        "negatives": int((df.label_binary == 0).sum()),
        "all_len_1000": bool((seqs.str.len() == 1000).all()),
        "alphabet_ok": bool(seqs.map(lambda s: set(s) <= ALPHA).all()),
        "max_n_fraction": float(nfrac.max()),
        "duplicate_sequences": dup_seq,
        "duplicate_ids": int(df.id.duplicated().sum()),
        "pos_class_counts": pos.classification.value_counts().to_dict(),
        "rows_per_split": df.split.value_counts().to_dict(),
        "pos_rate_per_split": df.groupby("split").label_binary.mean().round(4).to_dict(),
        "gc_mean_pos": float(pos.gc.mean()),
        "gc_mean_neg": float(df[df.label_binary == 0].gc.mean()),
        "homology_flagged": int(df.homology_flag.sum()),
        "splits_disjoint_by_chrom": bool(df.groupby("chrom").split.nunique().max() == 1),
    }
    assert r["all_len_1000"] and r["alphabet_ok"] and r["splits_disjoint_by_chrom"], (key, strat, r)
    report["tables"][f"{key}/{strat}"] = r

# cross-biosample sanity: legacy "k562" tables came from the T-cell file
legacy = None
try:
    legacy = pd.read_csv(find_one("k562_combined_gc_matched.tsv"), sep="\t", usecols=["accession", "label"])
    lp = set(legacy[legacy.label == 1].accession)
    for key in BIOSAMPLES:
        ps = set(tables[(key, "gc_matched")].query("label_binary==1").accession)
        report.setdefault("legacy_k562_table_overlap", {})[key] = len(lp & ps)
except FileNotFoundError:
    pass
print(json.dumps(report["legacy_k562_table_overlap"] if legacy is not None else "legacy table not attached", indent=1))

# %% [markdown]
# ## 6. Write outputs (parquet + SLM-ready JSONL + manifest)

# %%
COLS = ["id", "biosample", "negative_strategy", "chrom", "start", "end", "accession", "classification", "sequence", "gc",
        "label_binary", "label_3class", "split", "max_train_jaccard", "homology_flag"]
for (key, strat), df in tables.items():
    df[COLS].to_parquet(OUT / f"{key}_{strat}.parquet", index=False)

slm = OUT / "slm_ready"
slm.mkdir(exist_ok=True)
for key in BIOSAMPLES:  # SLM-ready: GC-matched negatives only (least confounded)
    d = tables[(key, "gc_matched")]
    with gzip.open(slm / f"{key}_gc_matched.jsonl.gz", "wt") as fh:
        for r in d[["id", "sequence", "label_binary", "label_3class", "split", "biosample", "homology_flag"]].itertuples(index=False):
            fh.write(json.dumps(dict(id=r.id, sequence=r.sequence, label=int(r.label_binary), label_3class=int(r.label_3class),
                                     split=r.split, biosample=r.biosample, homology_flag=bool(r.homology_flag))) + "\n")

(OUT / "splits.json").write_text(json.dumps({"train": sorted(TRAIN), "val": sorted(VAL), "test": sorted(TEST)}, indent=1))
report["biosamples"] = {k: {"encode_file": a, "biosample": d} for k, (a, d) in BIOSAMPLES.items()}
report["runtime_s"] = round(time.time() - T0, 1)
(OUT / "validation_report.json").write_text(json.dumps(report, indent=1, default=str))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
(OUT / "MANIFEST.sha256").write_text("\n".join(f"{sha(p)}  {p.relative_to(OUT)}" for p in files))
print(f"wrote {len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.1f} MB in {time.time()-T0:.0f}s")
