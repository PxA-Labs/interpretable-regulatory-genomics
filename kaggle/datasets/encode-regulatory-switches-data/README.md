# ENCODE regulatory switches: v1 inputs (raw tables, NOT yet leakage-audited)

Inputs for the interpretable-regulatory-genomics project (github.com/PxA-Labs/interpretable-regulatory-genomics).

- `phase1_processed/` K562 active cCREs (PLS + dELS), 1000 bp, with three negative sets: GC-matched, random unmatched, flanking. Columns: chrom,start,end,accession,classification,sequence,gc,label. GRCh38.
- `phase3_processed/` chr19 subset used for the embedding experiments.
- `chr22_demo/` 200-row chr22 demo subsets (K562, GM12878) used by unit-style scripts.
- `raw/` ENCODE SCREEN registry V4 (ENCFF726XBE), K562 (ENCFF464BRU), GM12878 (ENCFF590IMH) cCRE beds.
- `reference/` JASPAR 2026 CORE vertebrates non-redundant PFMs.

The GRCh38 genome is NOT included: attach Kaggle dataset aliabedimadiseh/grch38-human-genome-dna.
Sources: ENCODE (public), JASPAR (CC BY 4.0).
