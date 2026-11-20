# Wave 1: do the model gaps survive proper uncertainty, and are the negatives clean? (notebook 13)

Run `20261120_irg-13-wave1-uncertainty` (Kaggle v2, GPU, 629 s). No model was trained: all 75 saved models (XGBoost notebook 09 "tuned" and notebook 11 "wide", ShallowCNN and AttnCNN from notebook 07, 5 seeds each) plus a three-feature composition logistic regression `comp3_lr` (GC, CpG obs/exp, repeat fraction; trained on the train split) were re-scored on the test chromosomes chr19-22. Seed-0 AUROCs reproduce the earlier verification values exactly (for example wide XGBoost 0.8923 / 0.8719 / 0.8453 and ShallowCNN 0.8877 / 0.8671 / 0.8464 for T cell / K562 / GM12878). Matched sets use notebook 08's procedure and seeds. Hypotheses and pass/fail rules were fixed beforehand in `docs/research/novelty_and_next_experiments.md` (W1.1, W1.2).

**Read this first.** All intervals below are for *ensembles*: the AUROC of the seed-averaged probability of a model family. Ensembling helps high-variance models more: on average it adds +0.014 AUROC to AttnCNN, +0.007 to ShallowCNN, +0.003 to XGBoost (`point_estimates.csv`, `auroc_ensemble` vs `auroc_seed_mean`). The intervals capture test-set sampling noise only, **not** training-seed variance (AttnCNN seed SD 0.005-0.012 in `cnn_tuned.md`). So "CNN ensemble ahead of XGBoost ensemble" is a different statement from "a single trained CNN is ahead of a single trained XGBoost", and the earlier single-model tables (`xgb_wide.md`) remain the right comparison for the latter.

## W1.1 Paired gaps with three kinds of interval
Gap = family A minus family B, ensemble AUROC, 1,000 resamples. "Window" = windows resampled within each test chromosome; "chrom" = the 4 test chromosomes resampled with replacement, then windows within (only 4 units, so indicative); "LOCO" = range when each test chromosome is left out in turn. Full table: `paired_gaps.csv`.

CNN ensemble minus wide-XGBoost ensemble (95% chromosome-cluster interval in brackets; **bold** = excludes 0 under both the window and chromosome intervals):

| biosample | CNN | full | GC+CpG matched | + repeat matched |
|:--|:--|--:|--:|--:|
| T cell | ShallowCNN | -0.003 [-0.010, +0.001] | -0.006 [-0.027, +0.005] | -0.008 [-0.027, +0.003] (window interval excludes 0, chrom does not) |
| T cell | AttnCNN | **+0.009** [+0.005, +0.014] | **+0.015** [+0.002, +0.024] | **+0.017** [+0.007, +0.027] |
| K562 | ShallowCNN | +0.001 [-0.008, +0.006] | +0.000 [-0.018, +0.010] | +0.001 [-0.014, +0.011] |
| K562 | AttnCNN | -0.003 [-0.007, +0.003] | **-0.012** [-0.021, -0.005] | **-0.012** [-0.023, -0.004] |
| GM12878 | ShallowCNN | **+0.006** [+0.002, +0.014] | **+0.012** [+0.005, +0.022] | **+0.017** [+0.010, +0.028] |
| GM12878 | AttnCNN | **+0.013** [+0.008, +0.020] | **+0.020** [+0.014, +0.030] | **+0.022** [+0.016, +0.033] |

Across the 24 CNN-versus-wide-XGBoost cells (3 biosamples x 4 test sets x 2 CNNs; includes the non-homologous set) the window interval excludes 0 in 15 and the chromosome interval in 14. Wide XGBoost minus the notebook-09 XGBoost: the window interval excludes 0 in 8 of 12 cells (all in K562 and GM12878; none in T cell).

**Rule check.** The precommitted expectation was that the CNN-minus-XGBoost intervals would include 0 in most cells; it is **falsified for the ensemble comparison**. The precommitted failure condition, intervals excluding 0 for the GM12878 matched sets, **is met**: ShallowCNN +0.012 and AttnCNN +0.020, with chromosome intervals clear of 0 and every leave-one-chromosome-out value positive (ShallowCNN +0.010 to +0.015). The GM12878 CNN edge is therefore not a test-sampling artefact, and the shuffle-ladder experiment (W1.4) is worth running. The edge is not uniform across architectures and biosamples: ShallowCNN is level elsewhere, AttnCNN is ahead on T cell but behind on K562.

**Per-chromosome spread (full test set, ensemble AUROC; `per_chromosome_auroc.csv`).** Across chr19-22 the spread is 0.031-0.049 for the XGBoost and CNN families, larger than any model gap above (largest 0.022). chr21 is the lowest chromosome for wide XGBoost in all three biosamples and for every model family in GM12878 except `comp3_lr` (lowest on chr22); I did not tabulate the other cells. Even the composition-only `comp3_lr` varies by 0.056 across chromosomes in GM12878 and 0.051 in T cell. A single pooled test number is one draw from a heterogeneous set. This is only a four-chromosome picture: a full leave-one-chromosome-out study over all 22 chromosomes needs retraining and was not done, so the claim that chr19-22 are atypical remains untested.

## W1.2 How many "negatives" are cCREs of another biosample
Negatives are GC-matched windows outside the biosample's own cCREs. For each biosample's test negatives, overlap with positives (cCREs) of the other two biosamples (any base overlap, or at least half of the window; `negative_contamination.csv`):

| biosample | negatives | overlapping any bp | overlapping >= 50% |
|:--|--:|--:|--:|
| T cell | 6,211 | 622 (10.0%) | 438 (7.1%) |
| K562 | 7,931 | 574 (7.2%) | 387 (4.9%) |
| GM12878 | 8,300 | 720 (8.7%) | 498 (6.0%) |

**Do the models score them higher, beyond composition?** Within (GC, CpG obs/exp) 8x8 cells, mean wide-XGBoost score on overlapping minus clean negatives is +0.079 (T cell), +0.089 (K562), +0.121 (GM12878) for any-overlap, larger for >= 50% overlap (+0.116, +0.112, +0.164); permutation p = 0.001 in every case (floor of 1,000 permutations). The composition-only `comp3_lr` shows a much smaller within-cell difference (+0.008 to +0.025; p > 0.05 in 2 of the 6 comparisons), so composition (including repeat fraction) does not explain the elevated scores. Whatever the model sees is in the k-mer content.

**Effect on metrics.** Removing the overlapping negatives raises full-test AUROC by +0.005 to +0.012 for every model (XGBoost +0.007 to +0.012, ShallowCNN +0.007 to +0.012, AttnCNN +0.007 to +0.010, `comp3_lr` +0.005 to +0.010) and wide-XGBoost AUPRC by +0.015 to +0.023 (any-overlap). The gain is similar across models, so rankings are unchanged; absolute AUROCs are modestly understated if these windows are regarded as non-regulatory-like.

**Rule check.** Supported: a non-trivial fraction (5-10%) of negatives are cCREs elsewhere and the sequence models score them higher, not explained by GC, CpG or repeats. The precommitted alternative reading **cannot be excluded**: these windows may be genuinely biosample-specific elements, in which case the model is correctly detecting regulatory-like sequence and the "false positives" are a specificity question, not label noise. That is a reason to run the specificity experiment (W2.1), not a reason to drop these negatives from the benchmark.

## What this does not show
- **Training-seed variance is outside every interval.** Single-model comparisons need the seed SDs of `cnn_tuned.md`; these intervals do not replace them.
- **Four test chromosomes.** The chromosome-cluster interval is only indicative; the LOCO range uses the same four chromosomes.
- **Overlap is a coordinate test.** It uses 1 kb windows from the v2.1 tables; it says nothing about whether the other biosample's call is correct.
- **The two interpretations of the contamination result are not separated** (label noise versus genuine shared regulatory sequence).
- **Ensemble results should not be quoted as single-model performance.**
