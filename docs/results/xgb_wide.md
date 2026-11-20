# Was the tuned XGBoost still under-tuned? Wider search, lifted tree cap (notebook 11)

Run `20261120_irg-11-xgb-wide` (Kaggle v2, GPU, 5,725 s). Same data, split, features (k=6, 4,096 features) and four test sets as notebooks 07-09. Notebook 09 left two open problems: its best configurations sat on the edge of the search space, and used 1,369-1,486 of a 1,500-tree cap. This run addresses both.

## Protocol (all selection on validation AUROC only; test chromosomes never used for selection)
- **Stage A:** the 3 best notebook-09 configurations per biosample were re-run with the cap lifted to 8,000 trees (early stopping 100 rounds). Config choice per biosample is fixed in `protocol.json`.
- **Stage B:** 8 new configurations (random search, space seed 777) in a space shifted toward the edges notebook 09 hit: depth {6, 8, 10}, learning rate {0.05, 0.08}, subsample {0.6, 0.8, 1.0}, colsample {0.1, 0.2, 0.3}, min_child_weight {1, 5, 10}, lambda {20, 50, 100}; cap 4,000.
- **Final:** the best validation configuration per biosample, 5 seeds, cap 8,000.
- No configuration hit its cap (`hit_cap` is False for all 33 search rows and all 15 final models), so the cap is no longer binding.

## Test AUROC, mean over 5 seeds (SD in brackets)
| biosample | model | full | matched (GC+CpG) | rep-matched |
|:--|:--|--:|--:|--:|
| tcell | XGB tuned (nb09) | 0.8894 (0.0003) | 0.7431 (0.0012) | 0.7311 |
| tcell | **XGB wide (nb11)** | **0.8911 (0.0009)** | **0.7458 (0.0033)** | **0.7345 (0.0030)** |
| k562 | XGB tuned (nb09) | 0.8695 (0.0004) | 0.7686 (0.0011) | 0.7496 |
| k562 | **XGB wide (nb11)** | **0.8710 (0.0008)** | **0.7724 (0.0019)** | **0.7536 (0.0021)** |
| gm12878 | XGB tuned (nb09) | 0.8429 (0.0008) | 0.7526 (0.0011) | 0.7289 |
| gm12878 | **XGB wide (nb11)** | **0.8456 (0.0004)** | **0.7575 (0.0006)** | **0.7344 (0.0015)** |

Selected: tcell and k562 stage-B config B5 (depth 8, lr 0.05, subsample 0.6, colsample 0.1, min_child_weight 5, lambda 100); gm12878 stage-A config nb09-1 (depth 8, lr 0.05, subsample 1.0, colsample 0.3, min_child_weight 1, lambda 20). Mean best iteration 1,780-3,096.

Gain over notebook 09 (full / matched / rep-matched): tcell +0.0017 / +0.0027 / +0.0034; k562 +0.0014 / +0.0038 / +0.0040; gm12878 +0.0026 / +0.0049 / +0.0055. Validation AUROC gain +0.0015 to +0.0030.

## Five-seed CNN mean minus wide XGBoost (positive = CNN ahead)
CNN numbers are the 5-seed means from `cnn_tuned.md` / `repeat_control.md` (not the single re-scored models).

| biosample | CNN (seed SD, full) | full | matched | rep-matched |
|:--|:--|--:|--:|--:|
| tcell | ShallowCNN (0.0016) | -0.005 | -0.008 | -0.010 |
| tcell | AttnCNN (0.0116) | -0.001 | +0.003 | +0.005 |
| k562 | ShallowCNN (0.0024) | -0.004 | -0.004 | -0.003 |
| k562 | AttnCNN (0.0069) | -0.013 | -0.024 | -0.024 |
| gm12878 | ShallowCNN (0.0013) | +0.001 | +0.007 | +0.012 |
| gm12878 | AttnCNN (0.0055) | +0.002 | +0.008 | +0.011 |

## What this shows
- **The cap and the search edge were limiting, but only slightly.** Widening the search and removing the cap improved every metric in every biosample, by 0.001-0.003 full AUROC and 0.003-0.006 on the matched sets. The earlier tuned baseline was under-tuned by that much, not by a lot.
- **XGBoost is now level with or ahead of the CNNs on the full test set in tcell and k562**, and within 0.002 in gm12878. The one exception is that AttnCNN's tcell mean is within 0.001 of XGBoost with a seed SD of 0.012, so that comparison is uninformative.
- **GM12878 remains the only biosample where both CNNs are ahead on the controlled sets** (+0.007 to +0.012, against CNN seed SDs of 0.001-0.009). It is a small effect and it is not present in the full-set comparison.
- **Net:** against a reasonably tuned k-mer XGBoost, neither CNN shows a robust advantage. The earlier claim that CNNs beat k-mer baselines was mostly a tuning-effort asymmetry (`xgb_tuned.md`), and this run narrows the remaining gap further.

## What this does not show
- **Still a truncated search:** 3 + 8 configurations per biosample, one seed for selection; the best configurations are again near the edge of the new space (lr 0.05 is its lowest value, colsample 0.1 its lowest, lambda 100 its highest). Another round would likely gain a further ~0.001, not change the conclusion, but that is a guess, not a result.
- **CNN tuning is still smaller than XGBoost tuning** (no architecture search), so "no CNN advantage" is conditional on these CNN architectures.
- **Matched AUROCs are shortcut-controlled upper bounds, not shortcut-free** (`repeat_control.md`).
- **Cross-version note:** the models were trained with XGBoost 3.4.1 on Kaggle GPU. All 15 `.ubj` files load under XGBoost 2.1.4 on CPU and, re-scored on the real test rows (`tools/hf/verify_xgb_models.py`), reproduce the AUROC recorded during the run to four decimals (largest difference below 5e-6).
