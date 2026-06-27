# Experiment Registry

This file serves as a central index for tracking all formal model training and evaluation experiments across Phases 1–3.

## Phase 1: Classical Interpretable Baselines

| Experiment ID | Model / Script | Purpose | Status |
|---------------|----------------|---------|--------|
| `exp_001_logistic_kmer4` | Logistic Regression | Trivial/Simple linear baseline on k=4 kmers | Pending |
| `exp_002_rf_kmer4` | Random Forest | Non-linear tree baseline on k=4 kmers | Pending |
| `exp_003_rf_kmer_sweep` | Random Forest | Hyperparameter sweep over k-mer sizes (k=3, 4, 5, 6) | Pending |
| `exp_004_xgb_kmer4` | XGBoost | Optimized gradient boosting baseline | Pending |
| `exp_011_xgb_ablation` | XGBoost | Feature ablation study (k-mer vs GC vs dinucleotide vs motifs) | Pending |
| `exp_012_xgb_neg_sensitivity` | XGBoost | Negative sampling strategy sensitivity (E7) | Pending |

## Phase 2: Deep Learning & Generalization

| Experiment ID | Model / Script | Purpose | Status |
|---------------|----------------|---------|--------|
| `exp_005_shallow_cnn_onehot_k562` | ShallowCNN | Simple CNN on one-hot DNA | Pending |
| `exp_006_deep_cnn_onehot_k562` | DeepCNN | Deep CNN on one-hot DNA | Pending |
| `exp_007_attention_cnn_onehot_k562` | AttentionCNN | CNN with attention for motif discovery | Pending |
| `exp_014_attention_cnn_multiclass_k562` | AttentionCNN | Multiclass prediction (PLS vs pELS vs dELS) | Pending |
| `exp_015_attention_cnn_zeroshot_gm12878` | AttentionCNN | Cross-cell-type generalization test on GM12878 | Pending |

## Phase 3: Foundation Models (Optional)

| Experiment ID | Model / Script | Purpose | Status |
|---------------|----------------|---------|--------|
| `exp_008_nt_embeddings_k562` | Nucleotide Transformer | Pre-trained embeddings + Logistic Regression | Pending |
| `exp_009_dnago_embeddings_k562` | DNAGo | Foundation model fine-tuning or embeddings | Pending |

---

*Note: The corresponding `config.yaml` and `metrics.json` for each experiment are saved in the `experiments/logs/` directory using these Experiment IDs as folder names.*
