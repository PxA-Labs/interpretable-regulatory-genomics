# Experiment Registry

Central index of experiments. **Section 1 is the authoritative record.** Section 2 is legacy and must not be cited.

## 1. Reproducible runs (v2.1 tables, canonical split: train chr1-14,X,Y / val chr15-18 / test chr19-22)

Each run has a folder `experiments/runs/<run_id>/` (`run.yaml`, log, `results/`), a write-up in `docs/results/`, and is published in the
run-record datasets `PxALabs/irg-experiment-runs` (Hugging Face) and `purvanshj009/irg-experiment-runs` (Kaggle). Headlines are one-line
summaries; read the write-up for caveats before quoting a number. Executed notebook files are not stored: the Kaggle CLI cannot download
them, so the log plus the pinned `kernel_version` in `run.yaml` is the execution record.

| Run | Notebook | Question | Headline (see write-up for caveats) | Write-up | Published artefacts |
|---|---|---|---|---|---|
| `20261118_irg-01-data-curation` | 01 | Build the 9 tables (3 biosamples x 3 negative strategies), splits, validation report | The legacy "K562" tables were T-cell elements (ENCFF726XBE) | `docs/data/DATA_CARD.md` | data |
| `20261119_irg-02-package-dataset` | 02 | Verify hashes, deduplicate, package v2.1 | All input hashes verified; 1-12 duplicates removed per table | `docs/data/DATA_CARD.md` | `PxALabs/encode-regulatory-elements` (HF) |
| `20261119_irg-03-classical-baselines` | 03 | LR / RF / XGBoost, k-mer sweep, cross-negative and cross-cell-type | Random negatives inflate AUROC (0.93-0.96 vs 0.82-0.87 GC-matched); the < 0.05 GC-trap gate fails | `docs/results/phase1_unified_baselines.md` | run record |
| `20261119_irg-04-transfer-by-class` | 04 | Is cross-biosample transfer inflated by shared elements? | Class mix explains part (30-50%) of the shared-vs-specific gap; about 70% remains within class | `docs/results/transfer_by_class.md` | run record |
| `20261119_irg-05-cpg-ablation` | 05 | Is the surviving shortcut CpG? | Signal is distributed: removing CG k-mers costs 0.007-0.021; GC+CpG-matched test lowers AUROC 0.09-0.17 | `docs/results/cpg_ablation.md` | run record |
| `20261119_irg-06-cnn-baselines` | 06 | CNNs vs k-mer XGBoost (15 epochs) | Superseded by 07 (all runs hit the epoch cap) | `docs/results/cnn_baselines.md` | run record |
| `20261119_irg-07-cnn-tuned` | 07 | Was 06 undertrained? LR sweep, 60 epochs | ShallowCNN +0.007-0.012 over 06; AttnCNN mixed; advantage measured vs untuned XGBoost | `docs/results/cnn_tuned.md` | `PxALabs/irg-cnn-baselines` (30 weights) |
| `20261119_irg-08-repeat-control` | 08 | Is the signal repeat content? | Repeat matching lowers AUROC only 0.010-0.025; model ordering unchanged | `docs/results/repeat_control.md` | run record, `irg-repeat-coverage` (Kaggle) |
| `20261119_irg-09-xgb-tuned` | 09 | Is the CNN advantage real vs a tuned XGBoost? | Tuning +0.005-0.008; CNN advantage mostly gone, only GM12878 stays ahead | `docs/results/xgb_tuned.md` | `PxALabs/irg-xgboost-baselines` (15 models) |
| `20261119_irg-10-motif-fdr` | 10 | Do CNN filters match JASPAR motifs under FDR control? | 0.4% of 3,699 filters (own matcher, not Tomtom); Discussion #11 claim not supported | `docs/results/motif_fdr.md` | run record, `irg-jaspar2024` (Kaggle) |
| `20261120_irg-11-xgb-wide` | 11 | Was the tuned XGBoost still under-tuned (tree cap binding)? | Slightly: +0.001-0.003 full AUROC, +0.003-0.006 matched; XGBoost now level with or ahead of ShallowCNN except GM12878 matched sets | `docs/results/xgb_wide.md` | `PxALabs/irg-xgboost-wide` (15 models) |
| `20261119_irg-12-multiclass-shap` | 12 | What separates PLS / pELS / dELS? | Composition explains most of the promoter advantage; after GC+CpG matching pELS ~ PLS > dELS; drivers form a CpG gradient | `docs/results/multiclass_shap.md` | run record |

Not yet done: tuned CNN architecture search, language-model embeddings or fine-tuning (held back by decision), orthogonal validation of any motif.

## 2. Legacy experiments (do not cite)

The folders under `experiments/logs/` pre-date the v2.1 tables, hold values that disagree with the project's own discussions, and were
built from tables later found to be T-cell elements. See `experiments/logs/README.md`. Their IDs are kept for traceability only.

| Experiment ID | Intended purpose | Status |
|---------------|------------------|--------|
| `exp_001_logistic_kmer4`, `exp_002_rf_kmer4`, `exp_003_rf_kmer_sweep`, `exp_004_xgb_kmer4`, `exp_011_xgb_ablation`, `exp_012_xgb_neg_sensitivity` | Phase 1 classical baselines | Legacy, unreliable; superseded by runs 03 and 05 |
| `exp_005_shallow_cnn_onehot_k562`, `exp_006_deep_cnn_onehot_k562`, `exp_007_attention_cnn_onehot_k562` | Phase 2 CNNs | Never tracked; superseded by runs 06-07 |
| `exp_014_attention_cnn_multiclass_k562` | Multiclass CNN | Never tracked; XGBoost multiclass in run 12 |
| `exp_015_attention_cnn_zeroshot_gm12878` | Cross-cell-type zero-shot | Never tracked; XGBoost transfer in runs 03-04 |
| `exp_008_nt_embeddings_k562`, `exp_009_dnago_embeddings_k562` | Phase 3 foundation-model embeddings | Not reproducible from the repo; language-model work is held back |
