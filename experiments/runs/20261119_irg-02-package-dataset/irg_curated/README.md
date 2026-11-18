---
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
similarity to any training sequence is >= 0.3 (report metrics with and without them).

## Tables
| table | rows | positives | negatives | dropped duplicates | homology-flagged |
|---|---|---|---|---|---|
| tcell/gc_matched | 85,180 | 42,595 | 42,585 | 10 | 557 |
| tcell/random_unmatched | 85,189 | 42,595 | 42,594 | 1 | 527 |
| tcell/flanking | 85,189 | 42,595 | 42,594 | 1 | 102 |
| k562/gc_matched | 114,480 | 57,246 | 57,234 | 12 | 664 |
| k562/random_unmatched | 114,489 | 57,246 | 57,243 | 3 | 708 |
| k562/flanking | 114,490 | 57,246 | 57,244 | 2 | 125 |
| gm12878/gc_matched | 128,845 | 64,426 | 64,419 | 7 | 567 |
| gm12878/random_unmatched | 128,849 | 64,426 | 64,423 | 3 | 755 |
| gm12878/flanking | 128,848 | 64,426 | 64,422 | 4 | 123 |

## Known limitations
- Negatives are genomic regions outside the biosample's cCREs; they can still be regulatory in other cell types.
- 3 K562 GC-matched negatives (chr19: 1, chr20: 2) were filled with random coordinates after a sampler timeout.
- CTCF-bound and CTCF-only classes are not included; chrY has no K562/GM12878 positives.
- Homology flags use 8-mer Jaccard, a heuristic, not alignment-based homology.

## Sources and licences
ENCODE (open data, https://www.encodeproject.org/), reference GRCh38. Check ENCODE data-use terms before redistribution.
Source snapshot: git SHA ac3e4156be59a868885099c707400ee107b0ce53, seed 42.
