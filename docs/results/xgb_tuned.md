# Is the CNN advantage real once XGBoost is tuned too? (notebook 09)

Run `20261119_irg-09-xgb-tuned` (Kaggle v2, GPU, 6,248 s). k=6 k-mer frequencies (4,096 features), GC-matched v2.1 tables, canonical split, same four test sets as notebooks 07-08. A random search of 24 XGBoost configurations (config 0 is the previous default) was run on the **validation** set only (seed 0, early stopping on validation AUC, up to 1,500 trees); then 5 seeds at the selected configuration and 5 seeds at the default. Test chromosomes were never used for selection.

## Test AUROC, mean over 5 seeds (SD in brackets, full and matched sets)
"Matched" is GC+CpG matched; "rep-matched" adds repeat fraction (notebook 08).

| biosample | model | full | matched | rep-matched |
|:--|:--|--:|--:|--:|
| tcell | XGB default | 0.8845 (0.0015) | 0.7347 (0.0031) | 0.7217 |
| tcell | **XGB tuned** | **0.8894 (0.0003)** | **0.7431 (0.0012)** | **0.7311** |
| k562 | XGB default | 0.8636 (0.0014) | 0.7614 (0.0022) | 0.7418 |
| k562 | **XGB tuned** | **0.8695 (0.0004)** | **0.7686 (0.0011)** | **0.7496** |
| gm12878 | XGB default | 0.8350 (0.0010) | 0.7452 (0.0015) | 0.7210 |
| gm12878 | **XGB tuned** | **0.8429 (0.0008)** | **0.7526 (0.0011)** | **0.7289** |

Selected configurations (all with learning rate 0.05, colsample 0.3): tcell depth 6, subsample 0.6, min_child_weight 10, lambda 1; k562 depth 6, subsample 1.0, min_child_weight 5, lambda 20; gm12878 depth 8, subsample 1.0, min_child_weight 1, lambda 20. Mean best iteration 1,369-1,486 trees.

## ShallowCNN / AttnCNN (notebook 07) minus tuned XGBoost (positive = CNN ahead)
| biosample | CNN | full | matched | rep-matched |
|:--|:--|--:|--:|--:|
| tcell | ShallowCNN | -0.004 | -0.006 | -0.007 |
| tcell | AttnCNN | +0.001 | +0.005 | +0.008 |
| k562 | ShallowCNN | -0.002 | -0.000 | +0.001 |
| k562 | AttnCNN | -0.011 | -0.020 | -0.019 |
| gm12878 | ShallowCNN | +0.004 | +0.012 | +0.017 |
| gm12878 | AttnCNN | +0.005 | +0.013 | +0.016 |

## What this shows
- **Tuning helps XGBoost consistently:** +0.005 to +0.008 AUROC on the full test set, +0.007 to +0.008 on the matched set and +0.008 to +0.009 on the repeat-matched set, in all three biosamples (seed SD 0.0003-0.003, so the gain is real). On validation the best of 24 configurations beat the default by 0.005-0.007.
- **The CNN advantage reported in `cnn_tuned.md` was largely a tuning-effort asymmetry.** Against a tuned XGBoost, ShallowCNN is level on T cell and K562 (-0.004 and -0.002 full; within seed noise on K562) and AttnCNN is level or worse (K562 -0.011 full, -0.020 matched).
- **GM12878 is the only biosample where the CNNs stay ahead,** modestly: +0.004/+0.005 on the full test but +0.012/+0.013 on the matched subset and +0.016/+0.017 on the repeat-matched subset. Both ShallowCNN (SD 0.002) and tuned XGBoost (SD 0.001) are stable enough that this is beyond seed noise. This is the one place a learned-motif model might add something beyond k-mer composition, and it is a small effect.
- **Practical reading:** a tuned 6-mer gradient-boosted model is a strong baseline that matches or beats these CNNs on two of three biosamples. Any claim that a CNN or a language model "adds value" must be measured against it, not against an untuned XGBoost.

## What this does not show
- **Both searches are truncated.** The best XGBoost configuration sits on the edge of the search space (learning rate 0.05 is the lowest value tried, colsample 0.3 the lowest, lambda 20 the highest in two biosamples) and used 1,369-1,486 of the 1,500-tree cap, so the cap was nearly binding and the tuned XGBoost is probably still underestimated. The CNN learning-rate grid had the same edge problem (`cnn_tuned.md`).
- **The tuning effort is comparable, not identical.** XGBoost got a 24-configuration search on one seed; the CNNs got a 3-value learning-rate sweep on one seed plus early stopping and decay, but no architecture search (filters, kernel size, depth). A better CNN is untested.
- **Only k=6 was tuned.** k=5 and k=7 were not searched.
- **Single-seed search:** configuration differences on validation are small (0.005-0.007 across 24 configurations), so the selected configuration is not clearly better than its neighbours.
- **Composition shortcuts are matched, not eliminated.** Matched AUROCs remain upper bounds on shortcut-independent signal (see `repeat_control.md`).
