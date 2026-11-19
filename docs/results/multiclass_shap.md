# What distinguishes promoters from proximal and distal enhancers? (notebook 12)

Run `20261119_irg-12-multiclass-shap` (Kaggle v2, GPU, 115 s). Four classes per biosample from the `classification` column of the GC-matched v2.1 tables: background (non-regulatory), PLS (promoter-like), pELS (proximal enhancer-like), dELS (distal enhancer-like). XGBoost on 4-mer frequencies (256 features), fixed untuned parameters, 3 seeds, canonical chromosome split. SHAP is XGBoost's built-in TreeSHAP on a stratified test sample (4,000 rows). The old 3-class results in Discussions #4 and #18 used the legacy (T-cell) tables and different class definitions, so they are not comparable.

The negatives were GC-matched to the *pooled* positives, not per class, so a class-vs-background model can still use GC and CpG. Each class is therefore also evaluated against background on a test subset **matched on GC and CpG observed/expected** (equal class and background counts per bin), and a two-feature [GC, CpG o/e] model shows how much composition alone explains.

## Overall (test chromosomes, 3 seeds)
| biosample | accuracy | macro-F1 | PLS / pELS / dELS F1 |
|:--|--:|--:|:--|
| tcell | 0.636 | 0.473 | 0.549 / 0.327 / 0.217 |
| k562 | 0.630 | 0.484 | 0.421 / 0.501 / 0.234 |
| gm12878 | 0.622 | 0.498 | 0.423 / 0.464 / 0.340 |

(seed SD 0.001-0.004). Argmax F1 depends on class priors; the AUROCs below are the main comparison.

## AUROC of each class against background
| biosample | class | XGB full | XGB GC+CpG matched | [GC, CpG] model full | [GC, CpG] model matched |
|:--|:--|--:|--:|--:|--:|
| tcell | PLS | 0.938 | 0.717 | 0.899 | 0.530 |
| tcell | pELS | 0.876 | 0.685 | 0.832 | 0.529 |
| tcell | dELS | 0.743 | 0.658 | 0.656 | 0.501 |
| k562 | PLS | 0.930 | 0.701 | 0.882 | 0.519 |
| k562 | pELS | 0.871 | 0.728 | 0.788 | 0.520 |
| k562 | dELS | 0.745 | 0.667 | 0.637 | 0.505 |
| gm12878 | PLS | 0.926 | 0.697 | 0.877 | 0.527 |
| gm12878 | pELS | 0.848 | 0.715 | 0.763 | 0.521 |
| gm12878 | dELS | 0.765 | 0.672 | 0.668 | 0.503 |

## What this shows
- **Composition explains most of the apparent promoter advantage.** Promoters look easiest on the full test set (0.93-0.94), but two numbers, GC and CpG o/e, already give 0.88-0.90, and once the test subset is matched the ranking changes: pELS is as separable as PLS (0.685-0.728 against 0.697-0.717) in all three biosamples, and only dELS stays clearly lowest (0.658-0.672).
- **The matching works as intended:** the [GC, CpG] model falls to 0.50-0.53 on the matched subsets while the k-mer model keeps 0.66-0.73, so k-mer content beyond GC and CpG distinguishes every class from background, modestly.
- **Distal enhancers are the hard class.** dELS has the lowest AUROC, the lowest F1 (0.22-0.34) and only 14-25% recall at the argmax; most dELS are called background. This agrees with the per-class result in `transfer_by_class.md`.
- **Promoter vs proximal enhancer confusion is large.** In T cell, 911 of 2,462 test pELS are predicted PLS and only 599 correct (24% recall); in K562 pELS recall is 46%, in GM12878 39%. PLS and pELS share much of their sequence signal.
- **The SHAP drivers form a CpG gradient.** The share of |SHAP| mass on CG-containing 4-mers is 0.37-0.42 for PLS, 0.30-0.34 for pELS and 0.27-0.29 for dELS, against 0.18 if CG features were used in proportion to their number (47 of 256). CGCG is the top feature for PLS and pELS (higher frequency raises the class score; Spearman 0.80-0.83) and lowers the dELS score (-0.71); CGGC, CCGG and GCCG also push toward "not dELS". The one non-CG driver shared by all biosamples for dELS is TGTG (positive).
- **The drivers are stable and partly shared.** Top-20 feature sets are stable across seeds (mean Jaccard 0.70-0.88). Across biosamples PLS drivers overlap most (12-15 of 20 shared; Jaccard 0.43-0.60), pELS less (0.29-0.48) and dELS least (0.18-0.43). Twelve PLS, seven pELS and five dELS k-mers are in the top 20 of all three biosamples, each with a consistent direction.

## What this does not show
- **Correlation, not mechanism.** SHAP values describe this model's use of 4-mer frequencies, not causal regulatory effects. A CpG-rich 4-mer raising the promoter score is consistent with promoters sitting at CpG islands, which is my background knowledge and was not tested here.
- **The matched AUROCs are not mutually comparable.** Each class is matched against background on its own sub-population (different sizes and bins); only the within-class comparison to the composition-only model is clean.
- **Untuned model and short k-mers.** XGBoost parameters are fixed and k=4 only; a tuned or longer-k model would score higher (see `xgb_tuned.md`), and SHAP on k=6 might show finer structure. No CNN was run on the four-class task.
- **Background is not validated.** Negatives are GC-matched genomic windows, not confirmed non-regulatory, and the matching is coarse, so the matched AUROCs are upper bounds on composition-independent signal.
- **"CG share" baseline is crude.** The 0.18 reference counts features, not their expected importance; it shows over-use of CG features, not by how much a CpG-free model would lose (notebook 05: 0.007-0.021 for the binary task).
- **Repeats are uncontrolled** in this notebook, and test rows used for SHAP are a stratified sample, not the full test set.
