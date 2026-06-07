# 04 — Research Design Document

## Interpretable Machine Learning for Discovering Hidden Regulatory Switches in Non-Coding DNA

| Field | Value |
|-------|-------|
| Document ID | RDD-001 |
| Version | 1.0 |
| Status | Active |
| Created | 2026-06-07 |
| Related docs | [Charter](./01-project-charter.md), [Dataset Strategy](./05-dataset-strategy.md), [Modeling Roadmap](./06-modeling-roadmap.md) |

---

## 1. Scientific Background

### 1.1 The Non-Coding Genome and Gene Regulation

The human genome consists of approximately 3.2 billion base pairs, of which only ~1.5–2% encodes proteins. The remaining ~98% — the non-coding genome — was historically dismissed as "junk DNA," but decades of research have revealed that it contains critical regulatory elements that control gene expression.

Key types of non-coding regulatory elements:

| Element | Function | Typical size | Distance from gene |
|---------|----------|--------------|-------------------|
| **Promoter** | Initiates transcription; RNA polymerase binding site | 100–1000 bp | 0–2 kb upstream of TSS |
| **Enhancer** | Increases transcription of target genes; can act over long distances | 200–1500 bp | 1 kb – 1 Mb from target |
| **Silencer** | Represses transcription | 200–1000 bp | Variable |
| **Insulator** | Blocks enhancer-promoter communication | 300–2000 bp | Variable |
| **Open chromatin region** | Accessible DNA indicative of regulatory activity | Variable | Variable |

These elements function through binding of transcription factors (TFs) to specific DNA sequence motifs, recruitment of co-activators or co-repressors, and modulation of local chromatin structure.

### 1.2 The Regulatory Switch Concept

In this project, a **"regulatory switch"** is defined as:

> A sequence region or local sequence-pattern combination in non-coding DNA that appears to strongly influence regulatory activity — such as enhancer-like behavior, promoter-like behavior, chromatin accessibility, or context-specific control of gene expression.

This is a **working project definition**, not a claim of molecular proof. The term "switch" reflects the idea that specific sequence patterns can toggle regulatory states (active/inactive), analogous to a switch controlling a circuit. The project aims to discover candidate switch-like patterns computationally.

### 1.3 Computational Approaches to Regulatory Prediction

Sequence-based regulatory prediction is an established computational direction. Key approaches include:

1. **Motif-based methods**: Scan sequences for known TF binding motifs (e.g., using PWMs from JASPAR). Simple, interpretable, but limited to known motifs.
2. **k-mer-based methods**: Represent sequences as frequency vectors of all subsequences of length k. Captures composition without requiring known motifs.
3. **Convolutional neural networks (CNNs)**: Learn sequence features automatically via convolutional filters. DeepSEA, Basset, and similar models demonstrated this approach.
4. **Recurrent / attention models**: Capture long-range dependencies. DanQ combined CNN + RNN for regulatory prediction.
5. **Genomic foundation models**: Large pretrained models (Enformer, DNABERT, Nucleotide Transformer, HyenaDNA) trained on massive genomic corpora. High accuracy but require substantial computational resources for training.

### 1.4 The Interpretability Gap

Machine learning in genomics involves trade-offs between predictive performance and human interpretability:

- **High interpretability, lower capacity**: Logistic regression on k-mers or motif features. Every prediction can be traced to specific features. May miss complex interactions.
- **Medium interpretability**: Shallow CNNs where learned filters can be mapped to motifs. SHAP or saliency methods provide post-hoc explanations.
- **Low interpretability, higher capacity**: Large transformers. Strong predictions but difficult to explain which sequence elements drive the output.

This project explicitly operates in the high-to-medium interpretability zone, prioritizing biological insight over raw accuracy.

### 1.5 ENCODE as a Data Foundation

The Encyclopedia of DNA Elements (ENCODE) project aims to identify all functional elements in the human and mouse genomes. It provides:

- Chromatin accessibility maps (DNase-seq, ATAC-seq)
- Histone modification profiles (ChIP-seq for H3K27ac, H3K4me3, H3K4me1, etc.)
- Transcription factor binding sites (ChIP-seq)
- Candidate cis-regulatory elements (cCREs) — pre-classified regulatory annotations

ENCODE's candidate cis-regulatory elements (cCREs) are specifically designed to identify regulatory regions and serve as appropriate labels or candidate regions for this project.

---

## 2. Hypothesis Framing

### 2.1 Central Hypothesis

> Non-coding DNA regions annotated as regulatory by ENCODE contain sequence-level patterns (motifs, k-mer compositions, local subsequence arrangements) that are learnable by interpretable machine learning models and that correspond to known or novel regulatory mechanisms.

### 2.2 Operational Hypotheses

| ID | Hypothesis | Testable by |
|----|-----------|-------------|
| H1 | ENCODE-annotated cCREs have distinguishable k-mer composition from randomly sampled non-coding regions. | Train k-mer classifier; AUROC > 0.5 |
| H2 | Top-importance k-mers or features overlap with known TF binding motif cores. | Cross-reference top features with JASPAR/HOCOMOCO |
| H3 | Regulatory classification accuracy differs across cell types, reflecting cell-type-specific regulation. | Compare AUROC across K562, GM12878, HepG2 models |
| H4 | Different regulatory element types (enhancer vs. promoter) have distinct sequence signatures. | Train separate classifiers; compare top features |
| H5 | Shallow CNN filters learn motif-like patterns that overlap with known TF motifs. | Visualize CNN filters; compare to JASPAR PWMs |
| H6 | Combining interpretable features with pretrained embeddings improves prediction without destroying interpretability. | Stage 3 experiment |

---

## 3. Research Questions

### 3.1 Primary Research Questions

| # | Question |
|---|---------|
| RQ1 | Can a lightweight, interpretable ML model distinguish ENCODE-annotated regulatory regions from non-regulatory non-coding DNA with AUROC ≥ 0.80? |
| RQ2 | Which sequence features (k-mers, motifs, compositional statistics) are most predictive of regulatory activity, and do they align with known biology? |
| RQ3 | How do regulatory sequence signatures differ between cell types (e.g., K562 vs. GM12878)? |

### 3.2 Secondary Research Questions

| # | Question |
|---|---------|
| RQ4 | Does a shallow CNN discover motif-like convolutional filters, and how do they compare to known TF motifs? |
| RQ5 | What is the interpretability-accuracy trade-off when comparing logistic regression, tree-based models, CNNs, and pretrained-embedding classifiers? |
| RQ6 | Can the pipeline identify novel candidate regulatory patterns not currently in motif databases? |

---

## 4. Experimental Design

### 4.1 Experiment Matrix

| Experiment | Independent variable | Dependent variable | Controls |
|------------|--------------------|--------------------|----------|
| **E1**: Baseline classification | Model family (LogReg, RF, XGB) | AUROC, AUPRC, F1 | Same data split, same features, same seed |
| **E2**: Feature ablation | Feature set (k-mer only, k-mer+GC, k-mer+GC+motif) | AUROC delta | Same model (RF), same split |
| **E3**: k-mer resolution | Value of k (3, 4, 5, 6) | AUROC, top features | Same model (RF), same split |
| **E4**: Cell-type comparison | Cell type (K562, GM12878, HepG2) | AUROC, top features overlap | Same model, same pipeline |
| **E5**: Element-type classification | Element type (enhancer-like, promoter-like) | AUROC, top features | Same model, same split |
| **E6**: CNN motif discovery | — | Learned filter similarity to JASPAR | Fixed architecture |
| **E7**: Negative set sensitivity | Negative sampling strategy | AUROC variation | Same model, same positives |
| **E8**: Pretrained embeddings | Feature source (k-mer vs. pretrained) | AUROC, interpretability | Same classifier head |

### 4.2 Experiment Execution Order

```
E1 (Baseline)
 └──→ E3 (k-mer resolution) — informs feature choice for remaining experiments
       └──→ E2 (Feature ablation) — determines final feature set
             └──→ E7 (Negative set sensitivity) — validates data robustness
                   └──→ E4 (Cell-type comparison)
                   └──→ E5 (Element-type classification)
                         └──→ E6 (CNN motif discovery)
                               └──→ E8 (Pretrained embeddings) — Stage 3 only
```

---

## 5. Dataset Strategy

Detailed in [Document 05 — Dataset Strategy](./05-dataset-strategy.md). Summary:

| Dataset | Use | Phase |
|---------|-----|-------|
| ENCODE cCREs (human, hg38) | Primary labels for regulatory regions | Phase 1 |
| hg38 reference genome | Sequence extraction | Phase 1 |
| JASPAR motif database | Motif cross-referencing | Phase 1–2 |
| ENCODE DNase-seq peaks | Cell-type-specific accessibility labels | Phase 2 |
| ENCODE histone ChIP-seq peaks | Enhancer/promoter mark labels | Phase 2 |
| GTEx expression data | Expression-linked regulatory analysis | Phase 3+ |

---

## 6. Baselines

### 6.1 Trivial Baselines (Must Outperform)

| Baseline | Expected AUROC | Purpose |
|----------|---------------|---------|
| Random classifier | 0.50 | Sanity check |
| GC-content-only logistic regression | 0.55–0.65 | Tests whether composition alone is predictive |
| Single-feature (region length) | ~0.50 | Tests data leakage through length |

### 6.2 Interpretable Baselines (Project Core)

| Model | Features | Expected AUROC range |
|-------|----------|---------------------|
| Logistic regression | k-mer (k=4,5,6) | 0.70–0.80 |
| Random forest | k-mer + GC + dinucleotide | 0.75–0.85 |
| XGBoost | k-mer + GC + dinucleotide + motif | 0.78–0.88 |

### 6.3 Deep Learning Baselines (Phase 2)

| Model | Input | Expected AUROC range |
|-------|-------|---------------------|
| Shallow CNN (2–3 conv layers) | One-hot encoded sequence | 0.80–0.90 |
| CNN + SHAP/saliency | One-hot encoded sequence | Same accuracy, +interpretability |

### 6.4 Reference Points (Literature)

| Published method | Task | Reported AUROC | Note |
|-----------------|------|---------------|------|
| DeepSEA (2015) | Multi-task regulatory prediction | ~0.90+ per task | Large model, extensive training |
| Basset (2016) | Chromatin accessibility | ~0.89 mean | CNN on one-hot |
| gkmSVM | Regulatory vs. non-regulatory | ~0.85–0.90 | Gapped k-mer SVM; interpretable |

These are reference points only. The project does not aim to beat state-of-the-art large models but to achieve competitive performance with interpretable methods.

---

## 7. Model Comparison Plan

### 7.1 Standardized Evaluation Protocol

Every model evaluated using:

1. **Same data**: Identical train/val/test split (chromosome-holdout).
2. **Same metrics**: AUROC, AUPRC, accuracy, F1, precision, recall.
3. **Same seed**: Random seed = 42 for all stochastic operations.
4. **Same reporting**: Standardized metrics table + interpretability report.

### 7.2 Comparison Dimensions

| Dimension | How measured |
|-----------|-------------|
| Predictive accuracy | AUROC, AUPRC on test set |
| Calibration | Calibration plot (predicted probability vs. observed frequency) |
| Interpretability utility | Qualitative: do top features match known biology? |
| Interpretability depth | Number of samples for which explanations are available |
| Compute cost | Training time (seconds), memory usage (MB) |
| Kaggle feasibility | Does it run within Kaggle constraints? (binary) |
| Robustness | Performance variance across 5-fold CV on training data |

### 7.3 Output: Model Comparison Matrix

| Model | Interpretable? | Kaggle? |
|-------|----------------|---------|
| LogReg | Yes | Yes |
| RF | Yes | Yes |
| XGBoost | Yes (SHAP) | Yes |
| CNN | Partial | Yes |
| Pretrained | Limited | Depends |

*Columns to be filled: AUROC, F1, Train time (populated after experiments).*

---

## 8. Evaluation Metrics

### 8.1 Primary Metrics

| Metric | Why |
|--------|-----|
| **AUROC** | Standard for binary classification; threshold-independent; comparable across models |
| **AUPRC** | More informative than AUROC when classes are imbalanced |
| **F1 (macro)** | Balances precision and recall |

### 8.2 Secondary Metrics

| Metric | Why |
|--------|-----|
| Accuracy | Simple interpretability for non-specialists |
| Precision at 90% recall | Practical: how many predictions are correct when we catch most positives? |
| Recall at 90% precision | Practical: how many positives do we catch when we're mostly correct? |

### 8.3 Interpretability Metrics

| Metric | How measured |
|--------|-------------|
| Known-motif recovery rate | % of top-20 important features matching JASPAR/HOCOMOCO motifs |
| Cross-model feature agreement | Jaccard similarity of top-20 features between model families |
| Cell-type specificity of features | % of top features that differ between cell-type-specific models |

---

## 9. Validity Threats

| Threat | Type | Mitigation |
|--------|------|------------|
| **Data leakage from overlapping regions** | Internal | Chromosome-holdout split; verify no region overlap |
| **Label noise in ENCODE annotations** | External | Use high-confidence cCREs only; sensitivity analysis with stricter filters |
| **Negative set bias** | Construct | GC-matched negatives; exclude all known annotations; sensitivity analysis with multiple negative strategies |
| **Overfitting to specific cell types** | External | Cross-cell-type experiments; report per-cell-type and pooled results |
| **Confounding by GC content** | Construct | Include GC as feature; ablation experiment removing GC; GC-matched negatives |
| **Interpretability confirmation bias** | Researcher | Blind motif evaluation; compare against random features; report negative results |
| **Sequence length artifacts** | Construct | Fixed-length regions (1 kb); verify length has no predictive power |
| **Publication bias in motif databases** | External | Acknowledge that motif databases are incomplete; report novel patterns cautiously |

---

## 10. Limitations

1. **Computational**: Results are limited by what Kaggle-scale compute can explore. Larger models and longer sequences may yield better accuracy.
2. **Label quality**: ENCODE annotations are computational predictions themselves (for cCREs), not all experimentally validated. Predictions are predictions of predictions.
3. **Sequence-only**: Initial phases ignore chromatin context, 3D genome structure, epigenetic marks, and expression data. Sequence alone does not capture the full regulatory picture.
4. **Cell-type coverage**: Initial experiments cover 2–3 well-annotated cell types. Results may not generalize to under-studied cell types.
5. **Causality**: The project identifies correlative patterns, not causal mechanisms. "Important features" are statistically associated with regulatory labels, not proven to cause regulatory activity.
6. **Species specificity**: All initial work is human (hg38). Cross-species generalization is not tested.
7. **Static labels**: ENCODE annotations represent steady-state regulatory landscapes. Dynamic regulation (developmental, stimulus-response) is not captured.

---

## 11. Future Publication Potential

### 11.1 Candidate Venue Types

| Venue type | Examples | Fit |
|------------|---------|-----|
| Bioinformatics workshop | MLCB (ML in Computational Biology) @ NeurIPS, ISMB workshops | Good fit for initial results |
| Computational biology journal | Bioinformatics, BMC Bioinformatics, PLoS Computational Biology | Good fit for full study |
| ML conference (applications track) | ICML, NeurIPS (application workshops) | If novel methodology emerges |
| Preprint | bioRxiv, arXiv (q-bio) | For early dissemination |

### 11.2 Minimum Publication-Ready Package

1. Clear problem statement and biological motivation.
2. Reproducible data processing pipeline.
3. Multi-model comparison with standardized metrics.
4. Interpretability analysis with biological validation (motif overlap).
5. Cross-cell-type analysis.
6. Honest limitations section.
7. Public code repository.

### 11.3 Potential Paper Structure

1. Introduction: Non-coding regulation, interpretability gap.
2. Related work: Sequence-based regulatory prediction, interpretable ML in genomics.
3. Methods: Data, features, models, interpretability, evaluation.
4. Results: Classification performance, feature importance, motif analysis, cell-type comparison.
5. Discussion: Insights, limitations, future work.
6. Conclusion.

---

*End of Document 04 — Research Design Document*  
*Next: [05 — Dataset Strategy](./05-dataset-strategy.md)*
