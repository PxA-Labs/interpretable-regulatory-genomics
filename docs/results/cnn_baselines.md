# Do CNNs beat k-mer XGBoost? (notebook 06)

> **Update (2026-11-19):** partly superseded by `cnn_tuned.md`. Every run here hit the 15-epoch cap; with a longer schedule and a learning-rate sweep, ShallowCNN gains 0.007-0.012 AUROC and edges k=6 XGBoost, so the "no clear CNN advantage" conclusion below is too strong for ShallowCNN.

Run `20261119_irg-06-cnn-baselines` (Kaggle v1, GPU, 3,802 s). GC-matched v2.1 tables, canonical split (train chr1-14,X,Y; val chr15-18; test chr19-22), 5 seeds. Both CNNs use reverse-complement augmentation and strand-averaged prediction.

Models: ShallowCNN (7,937 parameters) and AttnCNN (189,185 parameters), both trained from scratch.

## Test AUROC (mean ± SD over 5 seeds for CNNs)
"Matched" is the GC+CpG-matched test subset from notebook 05 (52-68% of test rows; not comparable to the full-task number).

| biosample | XGB k=4 | XGB k=6 | ShallowCNN | AttnCNN |
|:--|--:|--:|--:|--:|
| tcell, full | 0.870 | 0.881 | 0.878 ± 0.002 | 0.883 ± 0.003 |
| tcell, matched | 0.699 | 0.726 | 0.713 ± 0.003 | 0.733 ± 0.009 |
| k562, full | 0.850 | 0.863 | 0.855 ± 0.002 | 0.861 ± 0.003 |
| k562, matched | 0.733 | 0.759 | 0.740 ± 0.003 | 0.758 ± 0.005 |
| gm12878, full | 0.823 | 0.834 | 0.835 ± 0.002 | 0.848 ± 0.002 |
| gm12878, matched | 0.718 | 0.743 | 0.743 ± 0.003 | 0.766 ± 0.005 |

XGBoost numbers come from notebook 05 (`k4_all`, `k6_all`); they are seed 0 only, so they have no SD here. Excluding homology-flagged rows changes the CNN AUROCs by at most 0.004.

## What this shows
- **No clear CNN advantage over a k=6 XGBoost.** AttnCNN is within about 0.002 of it on tcell and k562 (full test) and 0.014 above it on gm12878. ShallowCNN, with about 8k parameters, roughly matches k=4 to k=6 XGBoost.
- **On the matched subset the ranking barely changes.** AttnCNN leads on gm12878 (0.766 vs 0.743) and tcell (0.733 vs 0.726), and ties k=6 on k562. The one gap larger than seed noise is gm12878.
- **Composition control lowers every model by a similar amount (about 0.10-0.15).** The shortcut affects CNNs and k-mer models alike.

## What this does not show
- **All 30 runs used the full 15 epochs.** Early stopping (patience 3) never triggered, so the CNNs may be undertrained and these numbers are likely a lower bound for them. A longer schedule and learning-rate tuning are untested.
- **Unequal comparison.** XGBoost k=6 has a single seed and was not tuned for this comparison. No tuning was done for either model family.
- **Nothing here tests interpretability.** Filter-to-motif matching with FDR control is still to do.
- **Repeats (Alu, L1) are not controlled** in the matched subset.
