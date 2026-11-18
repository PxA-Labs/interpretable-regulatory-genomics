# %% [markdown]
# # 02 - Verify, deduplicate and package the curated dataset (v2.1)
#
# Input: output of `irg-01-data-curation` (attached as a kernel source).
# Output: `/kaggle/working/irg_curated/` - a self-contained dataset folder ready for Kaggle ("Output -> New Dataset")
# or Hugging Face upload. No credentials are used in this notebook.
#
# Steps: (1) verify SHA-256 of every input file against the manifest, (2) drop duplicated ids / sequences,
# (3) re-validate, (4) regenerate SLM-ready JSONL from the deduplicated GC-matched tables, (5) write data card,
# dataset metadata and a new manifest.

# %%
import glob, gzip, hashlib, json, os, time
from pathlib import Path

import pandas as pd

T0 = time.time()
manifest_path = glob.glob("/kaggle/input/**/curated/MANIFEST.sha256", recursive=True)
assert manifest_path, f"MANIFEST.sha256 not found under /kaggle/input: {os.listdir('/kaggle/input')}"
SRC = Path(manifest_path[0]).parent
OUT = Path("/kaggle/working/irg_curated")
OUT.mkdir(parents=True, exist_ok=True)
print("input:", SRC)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# %% [markdown]
# ## 1. Verify input integrity

# %%
bad, n = [], 0
for line in SRC.joinpath("MANIFEST.sha256").read_text().splitlines():
    digest, rel = line.split("  ", 1)
    p = SRC / rel
    n += 1
    if not p.exists() or sha256(p) != digest:
        bad.append(rel)
print(f"verified {n - len(bad)}/{n} files")
assert not bad, f"hash mismatch or missing: {bad}"

# %% [markdown]
# ## 2. Deduplicate and re-validate

# %%
BIOS = ["tcell", "k562", "gm12878"]
STRAT = ["gc_matched", "random_unmatched", "flanking"]
report = {"source_run": "irg-01-data-curation", "tables": {}}
tables = {}
for b in BIOS:
    for s in STRAT:
        df = pd.read_parquet(SRC / f"{b}_{s}.parquet")
        n0 = len(df)
        df = df.drop_duplicates("id", keep="first")
        df = df.drop_duplicates("sequence", keep="first").reset_index(drop=True)
        r = {
            "rows_in": n0,
            "rows_out": len(df),
            "dropped": n0 - len(df),
            "positives": int((df.label_binary == 1).sum()),
            "negatives": int((df.label_binary == 0).sum()),
            "all_len_1000": bool((df.sequence.str.len() == 1000).all()),
            "duplicate_ids": int(df.id.duplicated().sum()),
            "duplicate_sequences": int(df.sequence.duplicated().sum()),
            "splits_disjoint_by_chrom": bool(df.groupby("chrom").split.nunique().max() == 1),
            "rows_per_split": df.split.value_counts().to_dict(),
            "homology_flagged": int(df.homology_flag.sum()),
            "gc_mean_pos": float(df[df.label_binary == 1].gc.mean()),
            "gc_mean_neg": float(df[df.label_binary == 0].gc.mean()),
        }
        assert r["all_len_1000"] and r["splits_disjoint_by_chrom"] and r["duplicate_ids"] == 0 and r["duplicate_sequences"] == 0
        report["tables"][f"{b}/{s}"] = r
        tables[(b, s)] = df
        df.to_parquet(OUT / f"{b}_{s}.parquet", index=False)
        print(f"{b}/{s}: {n0:,} -> {len(df):,} (dropped {n0 - len(df)})")

# %% [markdown]
# ## 3. SLM-ready JSONL (GC-matched only, deduplicated)

# %%
slm = OUT / "slm_ready"
slm.mkdir(exist_ok=True)
for b in BIOS:
    d = tables[(b, "gc_matched")]
    with gzip.open(slm / f"{b}_gc_matched.jsonl.gz", "wt") as fh:
        for r in d[["id", "sequence", "label_binary", "label_3class", "split", "biosample", "homology_flag"]].itertuples(index=False):
            fh.write(json.dumps({"id": r.id, "sequence": r.sequence, "label": int(r.label_binary), "label_3class": int(r.label_3class),
                                 "split": r.split, "biosample": r.biosample, "homology_flag": bool(r.homology_flag)}) + "\n")

# %% [markdown]
# ## 4. Data card, metadata, manifest

# %%
(OUT / "splits.json").write_text((SRC / "splits.json").read_text())
upstream = json.loads((SRC / "validation_report.json").read_text())
report["upstream"] = {k: upstream[k] for k in ("git_sha", "seed", "homology_threshold_8mer_jaccard", "legacy_k562_table_overlap", "biosamples")}
(OUT / "validation_report.json").write_text(json.dumps(report, indent=1))

rows = "\n".join(
    f"| {k} | {v['rows_out']:,} | {v['positives']:,} | {v['negatives']:,} | {v['dropped']} | {v['homology_flagged']} |"
    for k, v in report["tables"].items()
)
CARD = f"""---
license: other
task_categories: [text-classification]
tags: [genomics, regulatory-elements, ENCODE, cCRE, dna]
pretty_name: ENCODE regulatory elements (T cell, K562, GM12878), v2.1
---
# ENCODE regulatory elements - confound-aware tables (v2.1)

1000 bp sequences centred on ENCODE SCREEN V4 candidate cis-Regulatory Elements (PLS, pELS, dELS), GRCh38, for three
biosamples, each with three negative strategies. Built for the Interpretable Regulatory Genomics project
(github.com/PxA-Labs/interpretable-regulatory-genomics).

## Biosamples (identities checked against ENCODE file metadata)
| key | ENCODE file | biosample |
|---|---|---|
| tcell | ENCFF726XBE | naive CD8+ alpha-beta T cell (CL:0000900) |
| k562 | ENCFF464BRU | K562 (EFO:0002067) |
| gm12878 | ENCFF590IMH | GM12878 (EFO:0002784) |

Earlier project tables named `k562_*` were built from ENCFF726XBE, i.e. **T cell**, not K562.

## Files
`<biosample>_<strategy>.parquet` with strategy in `gc_matched`, `random_unmatched`, `flanking`; `slm_ready/*.jsonl.gz`
(GC-matched only); `splits.json`; `validation_report.json`; `MANIFEST.sha256`.

## Columns
`id, biosample, negative_strategy, chrom, start, end, accession, classification, sequence, gc, label_binary,
label_3class (0 background, 1 PLS, 2 pELS/dELS), split, max_train_jaccard, homology_flag`.

## Splits (by chromosome)
train: chr1-14, X, Y; val: chr15-18; test: chr19-22. `homology_flag` marks val/test rows whose maximum 8-mer Jaccard
similarity to any training sequence is >= {upstream['homology_threshold_8mer_jaccard']} (report metrics with and without them).

## Tables
| table | rows | positives | negatives | dropped duplicates | homology-flagged |
|---|---|---|---|---|---|
{rows}

## Known limitations
- Negatives are genomic regions outside the biosample's cCREs; they can still be regulatory in other cell types.
- 3 K562 GC-matched negatives (chr19: 1, chr20: 2) were filled with random coordinates after a sampler timeout.
- CTCF-bound and CTCF-only classes are not included; chrY has no K562/GM12878 positives.
- Homology flags use 8-mer Jaccard, a heuristic, not alignment-based homology.

## Sources and licences
ENCODE (open data, https://www.encodeproject.org/), reference GRCh38. Check ENCODE data-use terms before redistribution.
Source snapshot: git SHA {upstream['git_sha']}, seed {upstream['seed']}.
"""
(OUT / "README.md").write_text(CARD)
(OUT / "dataset-metadata.json").write_text(json.dumps({
    "title": "ENCODE Regulatory Elements v2.1 (T cell, K562, GM12878)",
    "id": "purvanshj009/encode-regulatory-elements-v2",
    "licenses": [{"name": "other"}],
    "subtitle": "Confound-aware 1 kb cCRE tables with chromosome splits and homology flags",
    "description": "See README.md (data card).",
}, indent=1))

files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
(OUT / "MANIFEST.sha256").write_text("\n".join(f"{sha256(p)}  {p.relative_to(OUT)}" for p in files))
print(f"packaged {len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.0f} MB in {time.time()-T0:.0f}s")
