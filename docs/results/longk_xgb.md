# Can XGBoost on longer or spaced k-mers reach the GM12878 CNN edge? (notebook 15)

Run `20261120_irg-15-longk-xgb` (Kaggle v2, GPU, 8,186 s). Follow-up to notebooks 13-14: the GM12878 CNN edge over 6-mer XGBoost is real (nb13) and comes from local context beyond 6-mer counts (nb14). Here XGBoost is **trained from scratch** on (a) 8-mer frequencies (65,536 sparse features), (b) three gapped 6-of-8/9 k-mers (12,288 sparse features), and (c) 6-mers through the same sparse pipeline (control, seed 0). Same data, split and test sets as notebooks 07-14. Search: 2 configurations per feature set (the notebook-11 selected one, and colsample halved), chosen on validation AUROC; cap 6,000 trees, patience 100, `max_bin` 64; final models 3 seeds. Comparisons use the same seeds 0-2 of the saved notebook-07 CNNs and notebook-11 XGBoost, ensemble AUROC (seed-averaged probability), 95% window bootstrap (500 replicates; test-sampling noise only). Hypotheses and the rule were fixed in the notebook header before the run. Models: private HF repo `PxALabs/irg-xgboost-longk` (21 `.ubj`, pushed from inside Kaggle).

## Precommitted rule applied
H1 (edge is longer k-mer content): gap CNN minus validation-selected XGBoost on the GM12878 GC+CpG-matched set includes 0 or is negative for both CNNs. H0: gap positive with interval excluding 0 for both.

GM12878 matched set; validation selected **k8** (val 0.8430 vs spaced 0.8398):

| comparison | gap | 95% CI |
|:--|--:|--:|
| ShallowCNN - XGB k8 | +0.018 | [+0.012, +0.024] |
| AttnCNN - XGB k8 | +0.025 | [+0.019, +0.031] |

**Outcome: H0.** The edge is not reproduced by 8-mer features. It is larger against k8 than against the 6-mer XGBoost (+0.011 and +0.018 in the same seed 0-2 ensembles).

## The new feature sets are worse than 6-mers, in every biosample
Ensemble AUROC on the GC+CpG-matched test set (original windows):

| biosample | 6-mer (nb11) | k8 | spaced | ShallowCNN | AttnCNN |
|:--|--:|--:|--:|--:|--:|
| tcell | 0.749 | 0.739 | 0.714 | 0.740 | 0.771 |
| k562 | 0.776 | 0.762 | 0.740 | 0.776 | 0.767 |
| gm12878 | 0.760 | 0.753 | 0.739 | 0.770 | 0.777 |

k8 minus 6-mer is -0.010 / -0.014 / -0.007 and spaced minus 6-mer -0.035 / -0.036 / -0.020 (T cell / K562 / GM12878; all intervals exclude 0). Validation AUROC gives the same ordering (T cell 0.895 for 6-mers vs 0.886 for k8 and spaced).

## Shuffle ladder (GM12878 matched set)
Ensemble AUROC: original 6-mer 0.760, k8 0.753, spaced 0.739, ShallowCNN 0.770, AttnCNN 0.777. Under 250 bp block shuffles every model is within 0.005 of its original value. Assertions passed: k8 XGBoost is unchanged at the 8-let shuffle and 6-mer XGBoost at the 6-let shuffle (max difference 0). At 6-let the CNNs fall below k8 (gap about -0.013); at 4-let the gaps are 0.000 (ShallowCNN) and +0.020 (AttnCNN). Per notebook 14, shuffled windows are off-distribution for the CNNs, so these are robustness measurements, not a test of the edge. Full tables: `longk_ladder.csv`, `longk_gaps.csv`.

## What this shows
- Making the k-mer representation longer or gapped does not give XGBoost the CNN edge: with sparse 8-mer or spaced features it gets worse, not better. The "local context beyond 6-mers" that notebook 14 localised is not recovered by simply enlarging k at this data size.
- A plausible reason is dilution: a 1 kb window holds about 1,000 8-mers out of 65,536 possible, so most columns are rarely non-zero. This is a hypothesis; it was not tested (for example with a learning curve or hashed 7-mers).

## What this does not show
- **It does not show that no k-mer model can match the CNN.** Two representations and a 2-configuration search were tried; the 6-mer model had a larger search (3 + 8 configurations, cap 8,000).
- **Sparse-pipeline handicap.** The control (sparse 6-mers, `max_bin` 64, cap 6,000) scores 0.0015-0.0047 below the notebook-11 dense model on the full test set (seed 0: T cell 0.8876 vs 0.8923, K562 0.8681 vs 0.8719, GM12878 0.8438 vs 0.8453), so part of the new models' deficit is pipeline, not features. This is smaller than the k8 deficits on the matched sets but not negligible.
- Three seeds, four test chromosomes, training-seed variance outside the intervals; one spaced-mask choice (arbitrary, fixed beforehand).
- No claim about which motifs the CNNs use.
