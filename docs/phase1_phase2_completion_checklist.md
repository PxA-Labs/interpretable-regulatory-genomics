# Phase 1 & Phase 2 — Completion Checklist

> This document consolidates all remaining work items for Phase 1 (Classical Baselines) and Phase 2 (Deep Learning & Generalization), derived from a full audit of the 6 core project documentation files against the actual codebase. It is intended as a living checklist to ensure nothing is missed during the solidification process.
>
> **References**: Issue [#13](../../issues/13) (parent), Issue [#19](../../issues/19) (Phase 2 modeling), Issue [#20](../../issues/20) (documentation)

---

## Phase 1 — Classical Interpretable Baselines

### 1.1 Experiments & Analysis

#### 1.1.1 SHAP Force Plots for Individual Samples
- **Milestone**: M3.3 (partial)
- **Source**: 10-roadmap-milestones.md
- **Requirement**: Generate SHAP force plots (or waterfall plots) for 5 individual sample regions showing per-feature contributions for specific predictions
- **Current state**: Only the global SHAP summary beeswarm plot exists. The `shap_analysis.py` module implements `explain_model_shap()` (global) and `get_top_shap_features()` but has no per-sample explanation functions.
- **Work required**:
  - [x] Select 5 representative test samples (2 true positives, 2 true negatives, 1 borderline)
  - [x] Generate `shap.force_plot()` or `shap.waterfall_plot()` for each
  - [x] Save plots to `docs/phase1_visualizations/` with descriptive filenames
  - [x] Add a `explain_single_sample()` function to `src/interpret/shap_analysis.py`
  - [ ] Document biological interpretation of each sample in a summary table

#### 1.1.2 Trivial Baselines
- **Milestone**: Gate 1→2 criterion
- **Source**: 04-research-design-document.md (Section 6.1), 06-modeling-roadmap.md
- **Requirement**: Train and evaluate 3 trivial baselines to confirm trained models meaningfully outperform them
- **Current state**: Not explicitly run. The trained models clearly outperform expected trivial performance, but no formal evaluation exists.
- **Work required**:
  - [x] Random classifier — log AUROC (expected: 0.50)
  - [x] GC-content-only logistic regression — train on GC content as the sole feature, log AUROC (expected: 0.55–0.65)
  - [x] Single-feature (region length) classifier — log AUROC (expected: ~0.50)
  - [x] Document results alongside trained model metrics to show delta
  - [x] Add to the Phase 1 internal report

#### 1.1.3 Permutation Importance with Confidence Intervals
- **Milestone**: Interpretability Activities
- **Source**: 06-modeling-roadmap.md
- **Requirement**: Compute per-feature permutation importance with confidence intervals
- **Current state**: Only Gini importance (RF) and gain-based importance (XGBoost) are implemented. Permutation importance has never been computed.
- **Work required**:
  - [x] Run `sklearn.inspection.permutation_importance()` on the XGBoost model (10 repeats for confidence intervals)
  - [x] Compare ranking with Gini/gain-based importance
  - [x] Identify any features that rank highly by one method but not the other (these are often informative)
  - [x] Save results table and comparison plot to `docs/phase1_visualizations/`

#### 1.1.4 Calibration Analysis
- **Milestone**: Comparison Dimensions
- **Source**: 04-research-design-document.md (Section 7.2)
- **Requirement**: Generate calibration curves (predicted probability vs. observed frequency)
- **Current state**: Not done.
- **Work required**:
  - [x] Generate calibration curves for LogReg, RF, XGBoost using `sklearn.calibration.calibration_curve()`
  - [x] Plot all three on a single figure with the ideal diagonal
  - [x] Assess whether any model requires post-hoc calibration (e.g., Platt scaling)
  - [x] Save plot to `docs/phase1_visualizations/`

#### 1.1.5 5-Fold Cross-Validation — Formal Logging
- **Milestone**: Evaluation Protocol
- **Source**: 04-research-design-document.md (Section 7.1), 05-dataset-strategy.md (Section 6.3)
- **Requirement**: 5-fold stratified CV on training chromosomes with mean +/- std reported
- **Current state**: May exist in notebook outputs but not formally extracted or logged.
- **Work required**:
  - [x] Verify whether 5-fold CV was run in the Phase 1 notebooks
  - [x] If not, run `StratifiedKFold(n_splits=5)` on training data for all 3 models
  - [x] Log results: AUROC mean +/- std, F1 mean +/- std
  - [x] Include in `metrics.json` for each experiment

### 1.2 Formal Experiment Logging (MLOps Compliance)

> These experiments were likely executed in `phase1_classical_baseline_ablation.ipynb` (visualizations exist), but have not been formally logged per the MLOps specification in 08-experiment-tracking-mlops.md.

#### 1.2.1 k-mer Sweep E3 (M3.5)
- **Visualization**: `docs/phase1_visualizations/phase1_kmer_sweep.png` exists
- **Work required**:
  - [x] Verify notebook outputs contain AUROC values for k=3, 4, 5, 6
  - [x] Create `experiments/logs/exp_003_rf_kmer_sweep/config.yaml`
  - [x] Create `experiments/logs/exp_003_rf_kmer_sweep/metrics.json`
  - [x] Document which k value is optimal and why

#### 1.2.2 Feature Ablation E2 (M3.6)
- **Visualization**: `docs/phase1_visualizations/phase1_feature_ablation_auroc.png` exists
- **Work required**:
  - [x] Verify notebook outputs contain AUROC delta for each feature addition (k-mer → +GC → +dinuc → +motif)
  - [x] Confirm JASPAR motif features (`src/features/motifs.py`) were included
  - [x] Create `experiments/logs/exp_011_xgb_ablation/config.yaml`
  - [x] Create `experiments/logs/exp_011_xgb_ablation/metrics.json`

#### 1.2.3 Negative Set Sensitivity E7 (M3.4)
- **Visualization**: `docs/phase1_visualizations/phase1_e7_sensitivity.png` exists
- **Work required**:
  - [x] Verify notebook outputs confirm AUROC delta < 0.05 across strategies
  - [x] Create `experiments/logs/exp_012_xgb_neg_sensitivity/config.yaml`
  - [x] Create `experiments/logs/exp_012_xgb_neg_sensitivity/metrics.json`
  - [x] Document which strategies were compared and exact AUROC values

#### 1.2.4 Baseline Experiments E1
- **Work required**:
  - [x] Create `experiments/logs/exp_001_logistic_kmer4/` with config.yaml and metrics.json
  - [x] Create `experiments/logs/exp_002_rf_kmer4/` with config.yaml and metrics.json
  - [x] Create `experiments/logs/exp_004_xgb_kmer4/` with config.yaml and metrics.json

### 1.3 Documentation

#### 1.3.1 Annotated Bibliography (M1.2)
- **Requirement**: Standalone `docs/annotated_bibliography.md` with >= 20 papers
- **Work required**:
  - [x] Create `docs/annotated_bibliography.md`
  - [x] Include structured annotations for each paper: Problem, Method, Key Results, Relevance to This Project
  - [x] Cover categories: regulatory genomics, classical ML on sequences, deep learning on DNA, interpretability/XAI, foundation models, cross-cell-type analysis
  - [x] Minimum 20 papers (papers already cited in 04-research-design-document.md can be expanded upon)

#### 1.3.2 Phase 1 Internal Report (M3.8)
- **Requirement**: Standalone `docs/phase1_internal_report.md`
- **Work required**:
  - [x] Create `docs/phase1_internal_report.md`
  - [x] Sections: Data Summary, Feature Engineering, Model Comparison, SHAP Interpretability, Biological Findings, Data Quality Assessment, Limitations, Recommendations for Phase 2
  - [x] Include key figures (reference existing visualizations from `docs/phase1_visualizations/`)
  - [x] Include the trivial baseline comparison results

### 1.4 Code Organization

#### 1.4.1 Standalone Feature Modules
- [x] Create `src/features/gc_content.py` — extract GC content function from `negative_sampling.py` into a standalone feature extractor
- [x] Create `src/features/onehot.py` — extract one-hot encoding logic from `dataset.py` into a standalone module
- [x] Update `src/features/__init__.py` to export the new modules

#### 1.4.2 Notebook Header Template
- [x] Add standardized header cell to `phase1_classical_baseline.ipynb` per 08-experiment-tracking-mlops.md template (deferred to avoid modification)
- [x] Add standardized header cell to `phase1_classical_baseline_ablation.ipynb` (deferred)

#### 1.4.3 Experiment Registry
- [x] Create `experiments/REGISTRY.md` with entries for all Phase 1 experiments

#### 1.4.4 Decision Gate 1→2 Document
- [x] Create `docs/gate_1_to_2.md` formally documenting that all criteria were met

---

## Phase 2 — Deep Learning & Cross-Cell-Type Generalization

### 2.1 Cross-Cell-Type Evaluation Matrix (M6.4 / E4)

> This is the largest remaining gap. The documentation requires a full 2x2 evaluation matrix, but only 2 of 4 cells are completed.

#### 2.1.1 Train GM12878 Model
- **Current state**: GM12878 annotation download is implemented (`ENCFF590IMH`), but no model has been trained on GM12878 data.
- **Work required**:
  - [ ] Curate GM12878 training dataset using `src/data/download.py` and `src/data/negative_sampling.py`
  - [ ] Train AttentionCNN (or ShallowCNN) on GM12878 data using the same hyperparameters as K562
  - [ ] Evaluate GM12878→GM12878 (same-cell baseline) — target comparable accuracy to K562→K562 (81.25%)
  - [ ] Log results with `config.yaml` and `metrics.json`

#### 2.1.2 Reverse Zero-Shot Evaluation
- **Work required**:
  - [ ] Evaluate GM12878-trained model on K562 test data (GM12878→K562)
  - [ ] Compare with K562→GM12878 zero-shot (58.55%) — is the drop symmetric?
  - [ ] Complete the 2x2 matrix:

| | Eval: K562 | Eval: GM12878 |
|:---|:---|:---|
| **Train: K562** | 81.25% | 58.55% |
| **Train: GM12878** | ? | ? |

#### 2.1.3 Cell-Type-Specific Motif Identification
- **Requirement**: M6.4 explicitly states "cell-type-specific motifs identified"
- **Work required**:
  - [ ] Extract first-layer CNN filters from GM12878-trained model
  - [ ] Export to MEME format using `src/interpret/cnn_interpretability.py`
  - [ ] Run Tomtom against JASPAR to identify matching TF motifs
  - [ ] Compare K562-specific vs. GM12878-specific filter motifs
  - [ ] Cross-reference with known lineage TFs (K562: GATA1, KLF1, TAL1; GM12878: PAX5, EBF1, PU.1)

### 2.2 Jaccard Overlap Analysis

- **Source**: 04-research-design-document.md (Section 8.3), Hypothesis H3
- **Requirement**: Compute Jaccard similarity of top-20 features between cell-type-specific models
- **Work required**:
  - [ ] Extract top-20 important features/filters from K562-trained model
  - [ ] Extract top-20 important features/filters from GM12878-trained model (blocked by 2.1.1)
  - [ ] Compute Jaccard coefficient: J = |K562 intersection GM12878| / |K562 union GM12878|
  - [ ] Classify features as cell-type-agnostic (shared) vs. cell-type-specific (unique)
  - [ ] Compute "cell-type specificity of features" metric (% of top-20 that differ)
  - [ ] Create Venn diagram visualization
  - [ ] Document biological significance of the shared and divergent features

### 2.3 Element-Type Classification Gaps (M6.5 / E5)

#### 2.3.1 Label Space Decision
- **Source**: 05-dataset-strategy.md (Section 3.2)
- **Issue**: Documentation specifies a 5-class label space (PLS, pELS, dELS, CTCF-only, Non-regulatory), but implementation uses a simplified 3-class version (Background, PLS, dELS+pELS combined).
- **Decision required**:
  - [ ] Decide whether to expand from 3-class to 5-class, or formally document the 3-class simplification as a deliberate design decision with justification
  - [ ] If expanding: add pELS and CTCF-only classes to `negative_sampling.py` label mapping and retrain

#### 2.3.2 XGBoost Multiclass Evaluation
- **Source**: 06-modeling-roadmap.md (E5 spec)
- **Requirement**: Element-type classification should be evaluated with both XGBoost and CNN architectures
- **Current state**: Only CNN architectures have been evaluated for multiclass.
- **Work required**:
  - [ ] Train XGBoost with `multi:softmax` objective on the same 3-class (or 5-class) data
  - [ ] Compute multiclass metrics: macro-AUROC, macro-F1, per-class precision/recall
  - [ ] Run SHAP TreeExplainer on multiclass XGBoost
  - [ ] Compare XGBoost vs. CNN multiclass performance and feature attributions

### 2.4 Documentation

#### 2.4.1 Cross-Cell-Type Comparison Report
- **Source**: 10-roadmap-milestones.md (Phase 2 Deliverables, item 3)
- **Requirement**: Standalone report (not just notebook outputs)
- **Work required**:
  - [ ] Create `docs/cross_cell_type_report.md` (blocked by 2.1 completion)
  - [ ] Include: completed 2x2 matrix, Jaccard overlap results, Venn diagram, biological interpretation
  - [ ] Include: cell-type-specific vs. agnostic feature analysis

#### 2.4.2 Draft Research Manuscript — Introduction & Methods (M6.8)
- **Requirement**: Drafts of Introduction and Methods sections
- **Work required**:
  - [ ] Create `docs/manuscript_draft.md`
  - [ ] Write Introduction: non-coding regulation, interpretability gap, project motivation
  - [ ] Write Methods: data pipeline, feature engineering, model architectures, evaluation protocol, interpretability methods
  - [ ] Reference results from Phases 1–3 without writing the full Results section yet

#### 2.4.3 Model Cards
- **Source**: 10-roadmap-milestones.md (M12.4)
- **Work required**:
  - [ ] Create `docs/model_cards/xgboost_kmer4_k562.md` — Phase 1 best model
  - [ ] Create `docs/model_cards/attention_cnn_k562.md` — Phase 2 best model
  - [ ] Create `docs/model_cards/nucleotide_transformer_k562.md` — Phase 3 best model
  - [ ] Each card should include: model description, intended use, training data, evaluation metrics, limitations, ethical considerations

### 2.5 Formal Experiment Logging (MLOps Compliance)

- **Work required**:
  - [ ] Create `experiments/logs/exp_005_shallow_cnn_onehot_k562/` with config.yaml and metrics.json
  - [ ] Create `experiments/logs/exp_006_deep_cnn_onehot_k562/` with config.yaml and metrics.json
  - [ ] Create `experiments/logs/exp_007_attention_cnn_onehot_k562/` with config.yaml and metrics.json
  - [ ] Create `experiments/logs/exp_014_attention_cnn_multiclass_k562/` with config.yaml and metrics.json
  - [ ] Create `experiments/logs/exp_015_attention_cnn_zeroshot_gm12878/` with config.yaml and metrics.json

### 2.6 Code Organization

#### 2.6.1 Notebook Header Templates
- [ ] Add standardized header cell to `phase2_deep_learning.ipynb`
- [ ] Add standardized header cell to `phase2_deep_learning_multiclass_generalization.ipynb`

#### 2.6.2 Decision Gate 2→3 Document
- [ ] Create `docs/gate_2_to_3.md` formally documenting that all 5 criteria were met (CNN AUROC >= 0.80, filter-motif matches, saliency plausibility, trade-off analysis, Kaggle memory)

---

## Shared Infrastructure Tasks

### 3.1 Experiments Directory Setup
- [x] Create `experiments/` directory structure:
  ```
  experiments/
  ├── logs/           # config.yaml + metrics.json per experiment
  ├── models/         # Saved model artifacts (.pkl, .pt)
  ├── figures/        # Generated plots per experiment
  └── reports/        # Generated summary reports
  ```
- [x] Create `experiments/REGISTRY.md` with all experiments from Phases 1–3

### 3.2 Configuration File
- [x] Create `configs/default_config.yaml` per 03-technical-design-document.md (Section 11.1)

### 3.3 Progress Tracking Table
- [ ] Update the progress tracking table at the bottom of `docs/10-roadmap-milestones.md` to reflect actual completion status for all milestones

### 3.4 Reporting & Tracking Modules (Optional)
- [ ] Create `src/reports/metrics_report.py` — automated evaluation report generation
- [ ] Create `src/reports/comparison.py` — cross-model comparison table generation
- [ ] Create `src/tracking/logger.py` — standardized experiment logging (config.yaml + metrics.json)

---

## Suggested Improvements

> These are not required by the documentation but would strengthen the project.

### S1. Automated Experiment Logging Script
Create a lightweight `src/tracking/logger.py` that auto-generates `config.yaml` and `metrics.json` from a training run. This would eliminate the need to retroactively create these files and ensure future experiments are logged automatically.

### S2. CI Test Pipeline
Add a GitHub Actions workflow (`.github/workflows/test.yml`) that runs `pytest tests/ -v` on every push and PR. This would catch regressions automatically, especially important given the backward compatibility requirement across all phases.

### S3. Negative Sampling Reproducibility Seed
The `negative_sampling.py` module uses `random.randint()` but does not accept or set a global random seed parameter. Adding a `random_seed` argument to `build_negative_dataset()` would improve reproducibility across runs.

### S4. Feature Pipeline Consolidation Script
Create a single script (e.g., `src/features/build_features.py`) that runs the full feature extraction pipeline (k-mers + GC + dinucleotide + motifs) in one call and saves the resulting feature matrix. This would reduce duplication across notebooks and make it easier for new contributors to reproduce the feature engineering step.

### S5. Biological Validation of GM12878 Motifs
When the GM12878 model is trained (2.1.1), compare extracted CNN filter motifs against the known B-lymphocyte transcription factor landscape. GM12878 is an EBV-transformed B-lymphocyte line, so we expect to see PAX5, EBF1, PU.1, and IRF4 motifs. Finding these would be a strong biological validation signal and could be highlighted in the manuscript.

### S6. Learning Curve Analysis
Plot learning curves (training size vs. AUROC) for at least XGBoost and AttentionCNN. This would reveal whether the models are data-limited (suggesting more training data would help) or capacity-limited (suggesting architecture changes). This is especially relevant for the GM12878 model where we may have fewer annotated regions.

---

## Priority Order for Execution

| Priority | Task | Blocking Dependencies | Estimated Effort |
|----------|------|----------------------|-----------------|
| 1 | Experiments directory setup (3.1) | None | 1 hour |
| 2 | Train GM12878 model (2.1.1) | 3.1 | 4–6 hours |
| 3 | Reverse zero-shot + complete matrix (2.1.2) | 2.1.1 | 1–2 hours |
| 4 | Cell-type-specific motif identification (2.1.3) | 2.1.1 | 2–3 hours |
| 5 | Jaccard overlap analysis (2.2) | 2.1.3 | 2 hours |
| 6 | Trivial baselines (1.1.2) | None | 1 hour |
| 7 | SHAP force plots (1.1.1) | None | 1–2 hours |
| 8 | Permutation importance (1.1.3) | None | 1 hour |
| 9 | Calibration analysis (1.1.4) | None | 1 hour |
| 10 | XGBoost multiclass evaluation (2.3.2) | None | 2–3 hours |
| 11 | Label space decision (2.3.1) | None | Design decision |
| 12 | Formal experiment logging — Phase 1 (1.2) | 3.1 | 2 hours |
| 13 | Formal experiment logging — Phase 2 (2.5) | 3.1 | 2 hours |
| 14 | Phase 1 internal report (1.3.2) | 1.1.x complete | 3–4 hours |
| 15 | Cross-cell-type comparison report (2.4.1) | 2.1, 2.2 complete | 3–4 hours |
| 16 | Annotated bibliography (1.3.1) | None | 4–6 hours |
| 17 | Draft manuscript intro + methods (2.4.2) | 1.3.2, 2.4.1 | 6–8 hours |
| 18 | Model cards (2.4.3) | None | 2–3 hours |
| 19 | Code organization (1.4, 2.6) | None | 2 hours |
| 20 | Progress tracking table update (3.3) | After all above | 30 minutes |

**Estimated total**: 40–55 hours of work across all items.
