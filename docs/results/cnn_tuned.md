# Was the notebook-06 CNN result an undertraining artefact? (notebook 07)

Run `20261119_irg-07-cnn-tuned` (Kaggle v2, GPU, 11,663 s). Same GC-matched v2.1 tables, canonical split (train chr1-14,X,Y; val chr15-18; test chr19-22), same models, strand averaging and evaluation sets as notebook 06. Changes: up to 60 epochs, patience 8, ReduceLROnPlateau, and a learning-rate sweep over {3e-4, 1e-3, 3e-3} (seed 0, selected on **validation** AUROC only), then 5 seeds at the selected rate. Test chromosomes were never used for any selection.

## Test AUROC, mean ± SD over 5 seeds
"Matched" is the GC+CpG-matched test subset (a sub-population; do not compare it with the full-task number).

| biosample | set | XGB k=6 (nb05) | ShallowCNN nb06 | ShallowCNN nb07 | AttnCNN nb06 | AttnCNN nb07 |
|:--|:--|--:|--:|--:|--:|--:|
| tcell | full | 0.881 | 0.878 | 0.886 ± 0.002 | 0.883 | 0.891 ± 0.012 |
| tcell | matched | 0.726 | 0.713 | 0.737 ± 0.004 | 0.733 | 0.749 ± 0.026 |
| k562 | full | 0.863 | 0.855 | 0.868 ± 0.002 | 0.861 | 0.858 ± 0.007 |
| k562 | matched | 0.759 | 0.740 | 0.768 ± 0.004 | 0.758 | 0.749 ± 0.015 |
| gm12878 | full | 0.834 | 0.835 | 0.847 ± 0.001 | 0.848 | 0.848 ± 0.005 |
| gm12878 | matched | 0.743 | 0.743 | 0.764 ± 0.002 | 0.766 | 0.766 ± 0.009 |

Selected learning rates: 3e-3 for five of the six model/biosample pairs, 1e-3 for tcell ShallowCNN. Epochs used: ShallowCNN 39-50 on average, AttnCNN about 22. One of the 30 final runs still hit the 60-epoch cap (tcell ShallowCNN).

## What this shows
- **Undertraining affected ShallowCNN, not AttnCNN.** ShallowCNN gained 0.007-0.012 AUROC on the full test set and 0.021-0.028 on the matched subset versus notebook 06. AttnCNN is unchanged on average (-0.003 to +0.007 full).
- **A tuned ShallowCNN (about 8k parameters) now edges k=6 XGBoost on all three biosamples:** +0.005 (tcell), +0.005 (k562), +0.013 (gm12878) on the full test; +0.011, +0.009, +0.021 on the matched subset. Seed SD is 0.001-0.004, so these gaps exceed seed noise for ShallowCNN.
- **AttnCNN is mixed.** On gm12878 it is clearly ahead of k=6 XGBoost (+0.014 full, +0.023 matched; seed SD 0.005 / 0.009). On tcell it is ahead on average but its seed spread is large (SD 0.012 full, 0.026 matched), so the gap is not reliable. On k562 it is slightly behind (-0.005 full, -0.010 matched). Its extra 180k parameters over ShallowCNN buy no consistent gain: it is ahead of ShallowCNN only on tcell, where it is also the least stable.
- **The earlier "no CNN advantage" claim in `cnn_baselines.md` is therefore too strong** for ShallowCNN: with tuning, a small CNN is modestly ahead of k-mer XGBoost.

## What this does not show
- **The gains are small.** The project's own gate of at least 0.02 AUROC over XGBoost is not met on the full test set (largest gap 0.014, AttnCNN on gm12878; ShallowCNN 0.013). Only gm12878 on the matched subset exceeds it (ShallowCNN +0.021, AttnCNN +0.023).
- **The comparison favours the CNN.** The CNNs got a learning-rate sweep, early stopping and decay; the XGBoost baseline got no tuning (3 seeds in notebook 08 reproduce the single-seed numbers above). A tuned XGBoost (depth, learning rate, regularisation, k=5-6) could close part of the gap. This must be run before claiming CNNs help.
- **The learning-rate grid is truncated.** Five of six selections sit at the top of the grid (3e-3), so the optimum may be higher; the improvement over notebook 06 may partly be an LR effect that a wider sweep would extend or change.
- **Sweep used one seed** and each choice was made on one validation run, so differences between neighbouring rates (often below 0.01) are not reliable.
- **Interpretability is untested.** Nothing here shows the learned filters correspond to real motifs; that needs the FDR-controlled matching planned for notebook 09.
- **Repeats (Alu, L1) are uncontrolled** until notebook 08.

## Artefacts
30 weight files (3 biosamples x 2 models x 5 seeds), published as safetensors at `PxALabs/irg-cnn-baselines` with a manifest. Raw per-run metrics are in `experiments/runs/20261119_irg-07-cnn-tuned/results/`.
