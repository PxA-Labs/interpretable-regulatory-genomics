# Is the surviving signal repeat content? (notebook 08)

Run `20261119_irg-08-repeat-control` (Kaggle v2, 481 s). Repeat coverage per 1,000 bp window from UCSC RepeatMasker (hg38 `rmsk`; Alu, L1, simple/low-complexity, all repeats; `tools/data/repeat_annotation.py`). GC-matched v2.1 tables, canonical split, models trained once on all training rows (evaluation-only masking). XGBoost k=4 and k=6 (3 seeds), ShallowCNN and AttnCNN (5 seeds, weights from notebook 07), and a six-feature logistic regression `feat6` [GC, CpG o/e, repeat fraction, Alu, L1, simple].

Test sets: **full**; **GC+CpG matched** (8x8 bins, as notebooks 05-07); **GC+CpG+repeat matched** (6x6x6 quantile bins, equal positives and negatives per cell; 6,164-10,362 rows).

## Test AUROC
| biosample | model | full | GC+CpG matched | GC+CpG+repeat matched | change from adding repeat matching |
|:--|:--|--:|--:|--:|--:|
| tcell | XGB k=4 | 0.870 | 0.699 | 0.686 | -0.013 |
| tcell | XGB k=6 | 0.881 | 0.726 | 0.713 | -0.013 |
| tcell | ShallowCNN | 0.886 | 0.737 | 0.724 | -0.013 |
| tcell | AttnCNN | 0.891 | 0.749 | 0.739 | -0.010 |
| tcell | feat6 (6 features) | 0.824 | 0.588 | 0.547 | -0.041 |
| k562 | XGB k=4 | 0.850 | 0.733 | 0.711 | -0.022 |
| k562 | XGB k=6 | 0.863 | 0.759 | 0.740 | -0.019 |
| k562 | ShallowCNN | 0.868 | 0.768 | 0.751 | -0.017 |
| k562 | AttnCNN | 0.858 | 0.749 | 0.730 | -0.019 |
| k562 | feat6 | 0.770 | 0.580 | 0.537 | -0.043 |
| gm12878 | XGB k=4 | 0.823 | 0.718 | 0.695 | -0.023 |
| gm12878 | XGB k=6 | 0.834 | 0.743 | 0.718 | -0.025 |
| gm12878 | ShallowCNN | 0.847 | 0.764 | 0.746 | -0.018 |
| gm12878 | AttnCNN | 0.848 | 0.766 | 0.745 | -0.021 |
| gm12878 | feat6 | 0.743 | 0.598 | 0.533 | -0.065 |

Seed SDs (matched sets): XGB 0.001-0.006, ShallowCNN 0.002-0.004, AttnCNN 0.008-0.028. `feat6` is deterministic (no SD).

## What this shows
- **Repeat content explains only a small part of the signal.** Adding repeat fraction to the matching lowers every sequence model by 0.010-0.025 AUROC; 0.69-0.75 remains. The k-mer and CNN models are not mainly detecting repeat depletion.
- **The model ranking is essentially unchanged by repeat matching.** ShallowCNN stays above XGBoost k=6 by 0.011 (tcell), 0.011 (k562) and 0.028 (gm12878). AttnCNN is above k=6 on gm12878 (+0.027, SD 0.008) and tcell (+0.026, but SD 0.028), and below it on k562 (-0.010). So the small CNN advantage in `cnn_tuned.md` is not a repeat artefact.
- **Performance depends on repeat content.** All models do worse on repeat-rich windows (repeat fraction >= 0.4: XGB k=6 0.78-0.84) than repeat-poor ones (< 0.1: 0.85-0.89). Positives are repeat-depleted, and the positive share differs by stratum (32-38% in the high stratum, 51-59% elsewhere), so compare AUROC within a stratum, not across.
- **The six simple features keep a lot of signal on the full test (0.74-0.82)** and fall to 0.53-0.55 after full matching, which is the intended behaviour, but see below.

## What this does not show
- **Matching is coarse, not exact.** After GC+CpG+repeat matching, `feat6` still scores 0.53-0.55 rather than 0.50, so within-bin residual information remains (quantile bins, not exact matching). Treat the matched AUROCs as upper bounds on signal that is independent of these features, and finer bins would lower them further.
- **Evaluation-only masking.** Models were trained on all training rows, repeats included. Whether they would learn something different if repeat content were balanced or masked during training is untested.
- **Alu and L1 are included in "repeat" but not separately matched.** Only total repeat fraction was matched; the sub-type fractions enter `feat6` only. Alu-specific effects (Alu are GC- and CpG-rich) are not isolated.
- **Annotation is by coordinate overlap** with the 2022 UCSC rmsk track; recent or unannotated repeats are not captured.
- **Composition is controlled, not provenance.** Positives and negatives can still differ in other ways, such as conservation, mappability or chromatin state outside the model's input.
- XGBoost was untuned in this notebook too; see `cnn_tuned.md` for the tuning caveat.
