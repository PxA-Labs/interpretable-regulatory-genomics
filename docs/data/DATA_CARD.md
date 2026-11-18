# Data Card: ENCODE K562 / GM12878 regulatory-element tables

Status: **v1 (raw inputs, not yet leakage-audited).** See "Known issues" before using for any claim.

## Provenance

| Item | Source | Notes |
|---|---|---|
| Positive regions | ENCODE SCREEN Registry V4 candidate cis-Regulatory Elements (cCREs), GRCh38 | Accessions `ENCFF726XBE` (registry), `ENCFF464BRU` (K562), `ENCFF590IMH` (GM12878) |
| Reference genome | GRCh38 / hg38 | Not redistributed here. On Kaggle attach `aliabedimadiseh/grch38-human-genome-dna` |
| Motif database | JASPAR 2026 CORE vertebrates non-redundant PFMs | CC BY 4.0 |

Positives are `PLS` (promoter-like) and `dELS` (distal enhancer-like) elements, resized to a centred 1000 bp window.
`pELS` and CTCF-only classes are **not** in the current tables.

## Local layout (git-ignored, not tracked)

```
data/
├── raw/                      # cCRE BEDs (K562, GM12878)
├── reference/                # chr22 FASTA (demo only)
├── processed/                # 200-row chr22 demo subsets (K562, GM12878)
├── phase_1_data/
│   ├── raw/  reference/      # registry BED, JASPAR PFMs
│   └── processed/            # FULL K562 tables, 3 negative strategies
└── phase_3_data/             # chr19-only subset (test-mode)
```

The paths above are hard-coded in `scripts/phase1_analysis/*.py` and `src/data/download.py`, so the local layout is intentionally unchanged.
The reproducible copy lives on Kaggle: `purvanshj009/encode-regulatory-switches-data`.

## Table schema (`*_combined_*.tsv`)

| Column | Type | Meaning |
|---|---|---|
| `chrom`, `start`, `end` | str/int/int | GRCh38 coordinates (1000 bp window) |
| `accession` | str | ENCODE cCRE accession (`EH38E…`); synthetic for negatives |
| `classification` | str | `PLS`, `dELS`, or the negative-strategy tag |
| `sequence` | str | Uppercase DNA, length 1000 |
| `gc` | float | GC fraction |
| `label` | int | `1` active regulatory, `0` background (binary tables) |

## Row counts (full K562 tables, measured 2026-11-18)

| Table | Rows | Positives | Negatives | Negative strategy |
|---|---|---|---|---|
| `k562_combined_gc_matched.tsv` | 52,724 | 26,362 | 26,362 | GC-matched (5 % tolerance) |
| `k562_combined_random_unmatched.tsv` | 52,724 | 26,362 | 26,362 | uniform non-coding |
| `k562_combined_flanking.tsv` | 52,724 | 26,362 | 26,362 | +/- 5 kb shift |
| `phase_3_data/k562_combined_dataset.tsv` | 4,032 | 2,016 | 2,016 | chr19 only |

## Known issues (tracked in the audit issue)

1. **Count mismatch.** Discussion #16 reports 26,364 positives; the stored table has 26,362.
2. **Possibly two different protocols (unconfirmed).** The notebooks fall back to a chr19-only test mode, and `phase_3_data` is chr19-only.
   Headline numbers in discussions #1 and #5 may come from that mode, while #16 states full mode with chr19-22 as test.
   Until the re-run in the audit issue, treat them as not directly comparable.
3. **Logged split != executed split.** `src/tracking/logger.py` records train / val (chr15-18) / test (chr19-22);
   the notebooks train on everything except chr19-22 and have no validation fold.
4. **No homology filtering.** Chromosome hold-out does not remove paralogous or repeat-derived sequence shared
   across train and test (e.g. Alu / L1). Leakage is untested.
5. **Negatives are not proven inactive.** Non-cCRE regions can still be regulatory in other cell types.
6. **Multiclass and GM12878 tables** referenced in discussion #18 (2,558 / 3,414 sequences) are not in the repository.

## Intended use / non-use

Intended: benchmarking interpretable sequence classifiers of regulatory elements under confound controls.
Not intended: clinical or variant-level interpretation; claims about cell types other than K562 / GM12878.
