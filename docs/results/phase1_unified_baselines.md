# Phase 1 re-run: classical baselines under one protocol

Run `20261119_irg-03-classical-baselines` (Kaggle, v2; XGBoost on GPU, others CPU; 1,501 s). Data: packaged v2.1 tables (T cell, K562, GM12878).
Protocol: train chr1-14,X,Y / val chr15-18 / test chr19-22; fixed hyper-parameters; 5 seeds for XGBoost/RF; XGBoost early-stops on val only.
Uncertainty: seed SD of XGBoost AUROC <= 0.0011; bootstrap 95 % CI half-width about 0.0045. Differences below about 0.01 should not be read as real.

> These numbers replace the conflicting ones in discussions #1, #5, #16 for the **v2 tables** (which now include `pELS`). They are not directly comparable to the legacy tables.

## 1. k=4 baselines (test AUROC)
|                                 |    lr |      rf |   xgb |   xgb_sd |   trivial_dinuc |   trivial_gc |
|:--------------------------------|------:|--------:|------:|---------:|----------------:|-------------:|
| ('gm12878', 'flanking')         | 0.864 | nan     | 0.871 |   0.0011 |           0.78  |        0.733 |
| ('gm12878', 'gc_matched')       | 0.792 |   0.802 | 0.823 |   0.001  |           0.737 |        0.509 |
| ('gm12878', 'random_unmatched') | 0.88  | nan     | 0.893 |   0.0009 |           0.843 |        0.815 |
| ('k562', 'flanking')            | 0.9   | nan     | 0.908 |   0.0005 |           0.841 |        0.79  |
| ('k562', 'gc_matched')          | 0.822 |   0.828 | 0.849 |   0.0011 |           0.769 |        0.511 |
| ('k562', 'random_unmatched')    | 0.921 | nan     | 0.93  |   0.0003 |           0.897 |        0.868 |
| ('tcell', 'flanking')           | 0.933 | nan     | 0.939 |   0.001  |           0.891 |        0.838 |
| ('tcell', 'gc_matched')         | 0.843 |   0.855 | 0.87  |   0.0009 |           0.803 |        0.513 |
| ('tcell', 'random_unmatched')   | 0.949 | nan     | 0.955 |   0.0005 |           0.931 |        0.902 |

## 2. Negative-set sensitivity (the "GC trap")
| biosample   |   flanking |   gc_matched |   random_unmatched |   random - gc_matched |
|:------------|-----------:|-------------:|-------------------:|----------------------:|
| gm12878     |      0.871 |        0.823 |              0.893 |                 0.071 |
| k562        |      0.908 |        0.849 |              0.93  |                 0.081 |
| tcell       |      0.939 |        0.87  |              0.955 |                 0.086 |

The earlier gate (|random - gc_matched| < 0.05) **fails for all three biosamples** (0.07-0.09). Discussion #16 reported 0.0187.
GC fraction alone scores 0.51 on GC-matched negatives (matching works) but 0.82-0.90 on random negatives, so random-negative AUROCs overstate sequence learning.
Dinucleotide frequencies alone (16 features) reach 0.74-0.80 on GC-matched negatives, i.e. roughly 74-82 % of the above-chance (AUROC - 0.5) signal of k=4 XGBoost; this is not removed by GC matching. A follow-up (notebook 05, `cpg_ablation.md`) tested the hypothesis that it is mostly CpG: removing CG-containing k-mers costs only 0.007-0.021 AUROC, but matching the test set on GC and CpG o/e removes 0.09-0.17, so the compositional signal is distributed over many k-mers rather than concentrated in CpG.

## 3. k-mer sweep (GC-matched, test AUROC)
| biosample   |   lr k=3 |   lr k=4 |   lr k=5 |   lr k=6 |   xgb k=3 |   xgb k=4 |   xgb k=5 |   xgb k=6 |
|:------------|---------:|---------:|---------:|---------:|----------:|----------:|----------:|----------:|
| gm12878     |    0.761 |    0.792 |    0.815 |    0.82  |     0.812 |     0.823 |     0.829 |     0.835 |
| k562        |    0.794 |    0.822 |    0.841 |    0.847 |     0.841 |     0.849 |     0.858 |     0.862 |
| tcell       |    0.819 |    0.843 |    0.861 |    0.863 |     0.862 |     0.87  |     0.879 |     0.882 |

AUROC rises monotonically up to k=6 for both models, with diminishing returns (k=5 -> 6: +0.003 to +0.004 for XGBoost). k=4 is not optimal.

## 4. Cross-strategy matrix (XGBoost, k=4)
**tcell** (rows: train strategy, cols: test strategy)

| tr               |   flanking |   gc_matched |   random_unmatched |
|:-----------------|-----------:|-------------:|-------------------:|
| flanking         |      0.937 |        0.809 |              0.941 |
| gc_matched       |      0.876 |        0.87  |              0.868 |
| random_unmatched |      0.932 |        0.787 |              0.955 |

**k562** (rows: train strategy, cols: test strategy)

| tr               |   flanking |   gc_matched |   random_unmatched |
|:-----------------|-----------:|-------------:|-------------------:|
| flanking         |      0.908 |        0.793 |              0.913 |
| gc_matched       |      0.844 |        0.849 |              0.846 |
| random_unmatched |      0.895 |        0.775 |              0.93  |

**gm12878** (rows: train strategy, cols: test strategy)

| tr               |   flanking |   gc_matched |   random_unmatched |
|:-----------------|-----------:|-------------:|-------------------:|
| flanking         |      0.872 |        0.765 |              0.855 |
| gc_matched       |      0.811 |        0.822 |              0.815 |
| random_unmatched |      0.858 |        0.749 |              0.893 |

Models trained on random negatives lose 0.14-0.17 AUROC on GC-matched test sets, and flanking-trained models lose 0.11-0.13; models trained on GC-matched negatives transfer to the other test sets at 0.81-0.88.

## 5. Cross-biosample transfer (XGBoost, k=4, GC-matched; rows train, cols test)
| train   |   gm12878 |   k562 |   tcell |
|:--------|----------:|-------:|--------:|
| gm12878 |     0.822 |  0.832 |   0.866 |
| k562    |     0.802 |  0.849 |   0.856 |
| tcell   |     0.806 |  0.822 |   0.87  |

Compared with in-domain scores, transfer costs 0.004-0.027 AUROC: sequence signal is largely shared across these three biosamples.

### Shared-element control
cCRE accessions are registry-wide, so some test positives are also positives in the training biosample.
| train   | test    |   test_auroc |   n_pos_shared |   n_pos_specific |   auroc_shared_pos |   auroc_specific_pos |
|:--------|:--------|-------------:|---------------:|-----------------:|-------------------:|---------------------:|
| tcell   | k562    |        0.822 |            537 |             7395 |              0.93  |                0.814 |
| tcell   | gm12878 |        0.806 |            581 |             7719 |              0.905 |                0.798 |
| k562    | tcell   |        0.856 |            537 |             5674 |              0.926 |                0.85  |
| k562    | gm12878 |        0.802 |           4112 |             4188 |              0.909 |                0.697 |
| gm12878 | tcell   |        0.866 |            581 |             5630 |              0.911 |                0.862 |
| gm12878 | k562    |        0.832 |           4112 |             3820 |              0.906 |                0.753 |

On elements that are positive in **both** biosamples AUROC is 0.90-0.93; on elements **specific** to the test biosample it falls to 0.70-0.86 (largest gap: K562 -> GM12878, 0.909 vs 0.697).
**Caveat:** shared elements are probably enriched for promoter-like (PLS) elements, which are easier. This control is not yet stratified by element class, so "shared vs specific" and "promoter vs enhancer" are confounded. Next step: report per class.

## 6. Homology leakage
Excluding homology-flagged evaluation rows (8-mer Jaccard >= 0.3 to a training sequence) changes XGBoost AUROC by -0.0025 on average (range -0.0060 to -0.0001). Near-duplicate leakage at this threshold is small.

## What this does and does not show
- Shows: random negatives inflate AUROC; GC matching removes the GC shortcut but not a dinucleotide (likely CpG) shortcut; sequence signal transfers across these cell types; transfer to cell-type-specific elements is much weaker.
- Does not show: biological interpretability, motif validity, or anything about pretrained language models (not run).
- Limits: one task definition (balanced 1:1, 1 kb), fixed untuned hyper-parameters, 3 biosamples, k-mer heuristic for homology.
