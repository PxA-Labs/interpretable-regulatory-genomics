# Is the surviving shortcut CpG content? (notebook 05)

Run `20261119_irg-05-cpg-ablation` (Kaggle v1, XGBoost on GPU, 705 s). GC-matched tables, canonical split, seeds 0-2 (CIs on the matched subset: seed 0, bootstrap).

**Hypothesis (from the unified baselines):** the signal that survives GC matching is mostly CpG. **Verdict: not supported by the feature ablation; supported only in a weaker, distributional sense.**

## 1. Removing every CG-containing k-mer (XGBoost, all test rows)
| biosample   |   k4_all |   k4_noCG |   k6_all |   k6_noCG |
|:------------|---------:|----------:|---------:|----------:|
| gm12878     |    0.823 |     0.816 |    0.834 |     0.819 |
| k562        |    0.85  |     0.841 |    0.863 |     0.842 |
| tcell       |    0.87  |     0.862 |    0.881 |     0.862 |

Dropping explicit CG k-mers costs only 0.007-0.008 AUROC at k=4 and 0.015-0.021 at k=6. Most of the signal is **not** carried by CG-containing k-mers.

## 2. Test subset matched on GC and CpG observed/expected (8 x 8 quantile bins, equal positives and negatives per bin)
| biosample   |   k4_all |   k4_noCG |   k6_all |   k6_noCG |
|:------------|---------:|----------:|---------:|----------:|
| gm12878     |    0.718 |     0.714 |    0.743 |     0.73  |
| k562        |    0.733 |     0.728 |    0.759 |     0.738 |
| tcell       |    0.699 |     0.694 |    0.726 |     0.706 |

Matching works (CpG o/e mean for positives vs negatives: 0.479/0.477, 0.438/0.433, 0.392/0.387; logistic regression on GC + CpG o/e alone: 0.51-0.52 AUROC). On this subset AUROC falls by 0.09-0.17 relative to the full test set (GM12878 / K562 / T cell; k=4: 0.105 / 0.117 / 0.170; k=6: 0.091 / 0.104 / 0.155). k=6 still beats k=4 by about 0.025-0.027, so higher-order k-mers carry information beyond GC and CpG.

## 3. Tiny models (sanity rows only)
| biosample   |   cpg_oe_only (matched) |   gc+cpg_oe (matched) |   cpg_oe_only (test) |   gc+cpg_oe (test) |
|:------------|------------------------:|----------------------:|---------------------:|-------------------:|
| gm12878     |                   0.505 |                 0.514 |                0.666 |              0.714 |
| k562        |                   0.505 |                 0.516 |                0.703 |              0.746 |
| tcell       |                   0.505 |                 0.518 |                0.76  |              0.805 |

CpG o/e alone is predictive on the full test set (0.67-0.76) but is at chance (0.505) on the matched subset, as intended.

## Interpretation
- The compositional signal (GC and CpG) is **distributed over many k-mers**: removing CG k-mers barely hurts because GC-rich context stays in the other features, yet balancing the test set on GC and CpG o/e removes 0.09-0.17 AUROC.
- What remains on the matched subset (about 0.70-0.76) is the better estimate of sequence information beyond these two composition variables, but see limits.
- The earlier statement "probably CpG-driven" in the unified-baselines document is therefore too specific and has been revised there.

## Limits (read before citing)
- **The 16 vs 15 dinucleotide logistic-regression comparison is uninformative and is not reported.** Dinucleotide frequencies sum to 1, so dropping CG loses no information for a linear model; identical AUROCs were expected and mean nothing.
- Removing CG k-mers removes *explicit* CpG features only. CpG content can partly be inferred from the remaining features, so the drop in section 1 is a lower bound on how much the model uses CpG.
- The matched subset is a sub-population (52-68 % of test rows: the overlap of the GC and CpG distributions). Its lower AUROC partly reflects removing easy cases at the extremes, so it is not the AUROC of the full task.
- Only GC and CpG o/e are matched. Repeat content (Alu/L1) and other dinucleotide skews are not controlled.
- Fixed untuned hyper-parameters; 3 biosamples; no language models.
