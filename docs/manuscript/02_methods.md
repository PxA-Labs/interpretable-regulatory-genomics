# 2. Methods (draft)

> Draft status: see `README.md`. All code is in the project repository; each experiment is a Kaggle notebook (`kaggle/kernels/`) with a run
> record (`experiments/runs/`).

## 2.1 Data

**Elements.** Candidate cis-regulatory elements from the ENCODE SCREEN registry, version 4 [CITE: SCREEN / ENCODE cCRE registry], GRCh38, in the
classes promoter-like (PLS), proximal enhancer-like (pELS) and distal enhancer-like (dELS). CTCF-only and CTCF-bound classes are not included.
Three biosamples were used, identified from ENCODE file metadata: naive CD8+ alpha-beta T cell (file ENCFF726XBE), K562 (ENCFF464BRU) and
GM12878 (ENCFF590IMH). Each element is represented by the 1,000 bp window centred on it (`docs/data/DATA_CARD.md`).
An earlier project dataset that was labelled K562 was found to be built from the T-cell file and is not used.

**Negatives.** For every positive window one negative window was drawn outside the biosample's cCREs by one of three strategies: *GC-matched*
(rejection sampling to the GC distribution of the positives; the primary benchmark), *random unmatched*, and *flanking*. Three K562 GC-matched
negatives that could not be matched within the sampler's time limit were filled with random coordinates and are documented. Sequences that were exact
duplicates were removed (1-12 rows per table). The final tables hold 85,180 (T cell), 114,480 (K562) and 128,845 (GM12878) rows in the GC-matched
version, balanced between classes.

**Splits.** By chromosome: train chr1-14, X, Y; validation chr15-18; test chr19-22. Model selection (early stopping, hyperparameters,
learning rates) used the validation chromosomes only; test chromosomes were not used for any selection. To flag near-duplicates across splits, the
maximum 8-mer Jaccard similarity of each validation or test sequence to any training sequence was computed and rows with similarity at least 0.3 are
flagged (about 0.4-2.3% of evaluation rows); results are reported with and without them.

## 2.2 Controlled evaluation sets

Besides the full test set, models are evaluated on (a) the test set without homology-flagged rows; (b) a **GC+CpG-matched** subset: positives and
negatives are binned by GC content and CpG observed/expected ratio (8 x 8 quantile bins) and an equal number of each is kept per bin;
(c) a **GC+CpG+repeat-matched** subset adding the fraction of the window covered by repeats (6 x 6 x 6 bins), where repeat coverage is computed from
the UCSC RepeatMasker track for hg38 [CITE: RepeatMasker / UCSC genome browser]. Composition-only reference models (logistic regression on GC and CpG
o/e, with and without repeat fractions) quantify how much a model could achieve from these features alone. Matched subsets are sub-populations and
their AUROCs are not comparable with full-task AUROCs. For the four-class analysis each element class is compared with background on its own matched
subset.

## 2.3 Models

**Classical.** Logistic regression, random forest and gradient-boosted trees (XGBoost) [CITE: XGBoost] on k-mer frequency vectors (k = 3-6).

**Convolutional.** Two networks trained from scratch on one-hot sequence with reverse-complement augmentation and strand-averaged prediction:
*ShallowCNN* (one convolution layer of 128 filters of width 15, global max pooling, linear output; 7,937 parameters) and *AttnCNN* (two convolutions,
max pooling, 4-head self-attention with residual connection and layer normalisation, mean and max pooling; 189,185 parameters). Training used AdamW
(weight decay 0.01), batch size 256, mixed precision, early stopping on validation AUROC (patience 8) and learning-rate reduction on plateau, up
to 60 epochs. The learning rate was chosen from {3e-4, 1e-3, 3e-3} on validation AUROC with one seed; final results use five seeds.

**Baseline parity.** XGBoost on 6-mer frequencies received a validation-only random search of 24 configurations (depth, learning rate, subsample,
column sampling, minimum child weight, L2 regularisation; up to 1,500 trees, early stopping on validation AUC), then five seeds at the selected
configuration. A follow-up search with a higher tree cap (the cap was nearly binding) is described in `docs/results/` when complete.

## 2.4 Metrics and statistics

The primary metric is test AUROC, with AUPRC as secondary. We report the mean and standard deviation over seeds (XGBoost: subsampling seeds; CNNs:
initialisation and data-order seeds); the data split is fixed, so seed variance does not capture variation across train/test partitions. No
hypothesis tests are used; differences smaller than a few seed standard deviations are described as indistinguishable.

## 2.5 Interpretation

**Filter-to-motif matching.** For each first-layer filter of each trained CNN, a position-frequency matrix was built from the windows where the
filter's activation exceeded 0.7 of its maximum over the validation sequences (filters with fewer than 30 windows were dropped). Each matrix was compared
with all 879 motifs of JASPAR 2024 CORE vertebrates non-redundant [CITE: JASPAR 2024] using the mean Pearson correlation over overlapping columns (at
least min(6, motif width) columns), maximised over strands and offsets. The null distribution comes from 200 column-shuffled versions of every motif,
pooled within motif-width bins; p-values are empirical and Benjamini-Hochberg false discovery rate control [CITE: Benjamini-Hochberg] is applied over
all filter-by-motif tests of a model. This is not TOMTOM [CITE: TOMTOM]. The procedure was validated on planted motifs (9 of 10 recovered, 0 of 118 random
filters flagged) and stability across five seeds was assessed.

**Feature attributions.** For the four-class task, XGBoost's built-in TreeSHAP [CITE: TreeSHAP] was computed on a stratified sample of test rows for
4-mer frequency features; for each class we report the share of mean absolute SHAP on CG-containing 4-mers, the stability of the top-20 features across
seeds and their overlap between biosamples.

## 2.6 Reproducibility

Every experiment is a Kaggle notebook run on Kaggle GPUs or CPUs with pinned inputs; smoke tests precede each full run. For each run the repository
holds the notebook source, the execution log, the kernel version, seeds, runtime and result tables (`experiments/runs/`, indexed in
`experiments/REGISTRY.md`). The benchmark tables, the run records and the trained models (30 CNNs; tuned XGBoost models) are published as Hugging Face
and Kaggle artefacts with SHA-256 manifests [TODO: final repository names, DOIs and licences once the authors decide]. Weights are stored as
safetensors or XGBoost native binaries (no pickle), and saved models were re-scored on the real test sets to confirm that they reproduce the reported
numbers. The Kaggle command line does not expose executed notebook files, so the log and pinned kernel version are the execution record.

## 2.7 Threats to validity

- Negatives are GC-matched genomic windows, not confirmed non-regulatory sequence, and matching is by quantile bins, so residual composition signal remains
  (a six-feature composition-and-repeat model still reaches AUROC 0.53-0.55 on the fully matched subsets).
- Hyperparameter searches for both model families were truncated (learning-rate and regularisation optima at the edge of the grid; tree cap nearly binding);
  the CNN architectures were not tuned. Relative rankings of XGBoost and CNNs could change with larger searches.
- Motif matching has limited power for noisy learned filters, uses a shuffle null that is pooled across filters, and has no orthogonal validation
  (for example ChIP-seq overlap or in-silico mutagenesis). A match shows resemblance, not that the model uses that factor.
- One fixed chromosome split; three biosamples (two are cell lines); a single reference genome; and cCRE labels derived from the same consortium
  data that define the biosamples.
- SHAP values describe the fitted model, not causal regulatory effects.
