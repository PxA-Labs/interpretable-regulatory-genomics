# Is the CNN edge higher-order structure or composition? A shuffle ladder on the saved models (notebook 14)

Run `20261120_irg-14-shuffle-ladder` (Kaggle v2, GPU, 1,284 s). No training: the saved wide XGBoost (notebook 11) and ShallowCNN / AttnCNN (notebook 07) models, 5 seeds each, were scored on chr19-22 windows destroyed in controlled ways. Levels: **k-let shuffles** (exact k-mer counts preserved, Euler paths on the (k-1)-mer graph; k = 6, 5, 4, 3, 2, and k = 1 = plain base shuffle), and **block shuffles** (window cut into 250 bp or 100 bp blocks, blocks permuted). The unshuffled "original" AUROCs match notebook 13's ensemble values. All numbers are ensemble AUROCs (seed-averaged probability); intervals are 95% window bootstraps (500 replicates) and cover test-sampling noise only. Full tables: `ladder.csv`, `ladder_gaps.csv`, `shuffle_diagnostics.csv`.

Diagnostics. Every k-let level leaves only 27-29% of positions identical to the original (chance is 25%), so k-let shuffles rebuild each window along a different path through the same k-mers: long-range arrangement is destroyed, k-mer composition is kept. The 6-mer frequency spectrum overlap with the original is 1.00 (k = 6), 0.68 (5), 0.42 (4), 0.32 (3), 0.28 (2), 0.22 (1); block shuffles keep 0.97-0.99. As asserted in the run, wide XGBoost (6-mer frequencies) is unchanged at k = 6 (max score difference < 1e-4).

## Matched-set AUROC (GC+CpG matched), ensemble, GM12878
| level | AttnCNN | ShallowCNN | wide XGBoost | AttnCNN - XGB | ShallowCNN - XGB |
|:--|--:|--:|--:|--:|--:|
| original | 0.780 | 0.773 | 0.760 | **+0.020** [+0.014, +0.026] | **+0.012** [+0.007, +0.018] |
| 6-let | 0.712 | 0.712 | 0.760 | -0.048 [-0.054, -0.041] | -0.049 [-0.055, -0.042] |
| 5-let | 0.666 | 0.657 | 0.731 | -0.065 [-0.071, -0.058] | -0.074 [-0.081, -0.067] |
| 4-let | 0.633 | 0.614 | 0.666 | -0.033 [-0.039, -0.026] | -0.052 [-0.058, -0.045] |
| 3-let | 0.605 | 0.579 | 0.615 | -0.010 [-0.016, -0.004] | -0.036 [-0.043, -0.030] |
| 2-let | 0.582 | 0.563 | 0.579 | +0.003 [-0.004, +0.009] | -0.016 [-0.023, -0.010] |
| bases only | 0.506 | 0.500 | 0.500 | +0.006 [-0.003, +0.014] | -0.000 [-0.009, +0.009] |
| 250 bp blocks | 0.778 | 0.771 | 0.759 | **+0.019** [+0.013, +0.024] | **+0.012** [+0.006, +0.018] |
| 100 bp blocks | 0.774 | 0.766 | 0.758 | **+0.016** [+0.011, +0.022] | **+0.009** [+0.003, +0.015] |

T cell and K562 show the same pattern (full-set tables in `ladder_gaps.csv`): the CNN-minus-XGBoost gap collapses and turns negative at 6-, 5- and 4-let in every biosample for both architectures, is within about 0.016 of 0 by 2-let, and is essentially unchanged by block shuffles. Retained signal, (AUROC - 0.5) relative to the original, on the matched sets: at 6-let the CNNs keep 0.76-0.84 while XGBoost keeps 1.00; at 4-let CNNs 0.41-0.53, XGBoost 0.62-0.69; block shuffles 0.96-0.99 for all models.

## Reading against the precommitted rule
The rule written into the notebook header before the run said: the GM12878 CNN edge is "carried by structure beyond 6-mer composition" if the CNN-minus-XGBoost gap stays above 0 with an interval excluding 0 at the 5-let and 4-let levels, and "explained by low-order composition" if the interval includes 0 or is negative at 4-let.

**Literal outcome: the gap is significantly negative at 5-let and 4-let for both architectures, so the rule returns "explained by low-order composition".** I do not think that is a valid reading, and the rule was badly designed:
- XGBoost on 6-mer frequencies is exactly invariant to a 6-let shuffle, while a CNN is not a 6-mer-frequency model. The shuffled windows are off-distribution for the CNNs (trained on real sequence), so a CNN falling below XGBoost on them measures robustness to the shuffle, not whether its original edge was composition.
- What the ladder does show is the reverse of "composition only": at 6-let, where the 6-mer composition is identical, the CNNs still lose 16-24% of their above-chance signal and XGBoost loses none. The CNNs therefore use information that 6-mer counts do not carry: arrangement of content on scales longer than 6 bp.
- The block shuffles bound that scale. Permuting 100 or 250 bp blocks, which keeps local arrangement inside each block, costs every model at most 0.010 AUROC and leaves the GM12878 gap essentially unchanged (+0.016/+0.019 for AttnCNN against +0.020 original). AttnCNN's attention does not exploit arrangement beyond about 100 bp; whatever the CNN edge is, it sits at scales between 7 bp and about 100 bp.

## Conclusions that the evidence supports
1. The GM12878 CNN edge survives block shuffles and does not survive k-let shuffles that break local arrangement; so it comes from local (roughly 7-100 bp) sequence context, not from long-range order and not from 6-mer composition alone.
2. No model in this set uses long-range (above about 100 bp) order to a measurable degree (at most 0.010 AUROC lost under 100 bp block shuffles).
3. This is **consistent with**, not proof of, the CNNs using longer or degenerate motifs than 6-mers can express. It does not identify which motifs.

## What this does not show
- **The k-let shuffle is not a controlled intervention for the CNNs.** Off-distribution inputs can lower a CNN's score for reasons unrelated to what it uses on real windows; the absolute AUROCs are robustness measurements, not performance claims.
- **The precommitted rule's literal verdict is reported, not silently replaced.** A rule that distinguishes the readings properly needs an equal-footing comparison (for example XGBoost with k > 6 or spaced k-mer features on the same shuffles) which was not run.
- **No claim about motif identity or grammar.** A planted-grammar benchmark (W2.3) and a k-mer baseline with longer or spaced k-mers are the natural next tests: if XGBoost with 8-mer or spaced features closes the GM12878 edge, the edge is longer k-mer content, not CNN-specific structure.
- Ensemble results; training-seed variance is not in the intervals; four test chromosomes.
