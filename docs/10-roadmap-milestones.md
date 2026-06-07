# 10 — Roadmap and Milestones

## Interpretable Machine Learning for Discovering Hidden Regulatory Switches in Non-Coding DNA

| Field | Value |
|-------|-------|
| Document ID | RM-001 |
| Version | 1.0 |
| Status | Active |
| Created | 2026-06-07 |
| Related docs | [Charter](./01-project-charter.md), [Modeling Roadmap](./06-modeling-roadmap.md), [Risk Register](./09-risk-register.md) |

---

## Roadmap Overview

```
Month:   1          2          3          4          5          6          7-9        10-12
         ├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
Phase 0: ████████                                                                    
Phase 1:      ██████████████████                                                     
Phase 2:                        ████████████████████                                 
Phase 3:                                              ████████████████               
Phase 4:                                                                 ████████████
Writing:                   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

████ = Active development
░░░░ = Writing / documentation (parallel track)
```

---

## Month 1: Foundation

### Objective
Establish data pipeline, environment, and first baseline model.

### Milestones

| # | Milestone | Target date | Success criterion |
|---|-----------|-------------|-------------------|
| M1.1 | **Project setup complete** | Week 1 | Git repo initialized; directory structure created; requirements.txt defined; docs pack reviewed by team |
| M1.2 | **Literature survey complete** | Week 2 | ≥ 20 papers read and annotated; annotated bibliography committed |
| M1.3 | **Data pipeline functional** | Week 3 | ENCODE cCREs downloaded for K562; sequences extracted from hg38; negative set generated; chromosome-holdout split defined |
| M1.4 | **First baseline model trained** | Week 4 | Logistic regression on k=4 k-mers; AUROC computed on test set; results logged |

### Deliverables
- [ ] Repository with directory structure and documentation
- [ ] Annotated bibliography
- [ ] Data pipeline: download → parse → extract → split
- [ ] EDA notebook with data visualizations
- [ ] First experiment log (`exp_001_logistic_kmer4`)
- [ ] Experiment registry initialized

### Definition of Success at Month 1
> A reproducible pipeline that goes from raw ENCODE annotations to a trained classifier with logged metrics. At least one model produces AUROC > 0.60 (proving the task is learnable).

---

## Month 3: Interpretable Baseline Complete

### Objective
Complete Phase 1 (classical ML) with full interpretability analysis and robust validation.

### Milestones

| # | Milestone | Target date | Success criterion |
|---|-----------|-------------|-------------------|
| M3.1 | **Multi-model comparison** | Month 2, Week 2 | LogReg, RF, XGBoost trained and compared on same split; comparison table produced |
| M3.2 | **Feature importance analysis** | Month 2, Week 3 | Top-20 features extracted per model; cross-referenced with JASPAR; motif overlap table generated |
| M3.3 | **SHAP analysis complete** | Month 2, Week 4 | SHAP summary plot for XGBoost; SHAP force plots for 5 sample regions |
| M3.4 | **Negative set sensitivity analysis** | Month 3, Week 1 | Experiment E7 complete; AUROC stable (Δ < 0.05) across ≥ 2 negative sampling strategies |
| M3.5 | **k-mer resolution analysis** | Month 3, Week 1 | Experiment E3 complete; optimal k value identified |
| M3.6 | **Feature ablation study** | Month 3, Week 2 | Experiment E2 complete; contribution of each feature group quantified |
| M3.7 | **Decision Gate 1→2 passed** | Month 3, Week 3 | AUROC ≥ 0.75; interpretability verified; all gate criteria met |
| M3.8 | **Phase 1 report written** | Month 3, Week 4 | Internal report summarizing baseline results, interpretability findings, data quality assessment |

### Deliverables
- [ ] ≥ 5 completed experiments with full logging
- [ ] Model comparison table (accuracy + interpretability)
- [ ] Motif cross-reference report
- [ ] SHAP analysis artifacts
- [ ] Feature ablation results
- [ ] Negative set sensitivity analysis
- [ ] Internal Phase 1 report
- [ ] Kaggle notebook reproducing key results
- [ ] Updated experiment registry

### Definition of Success at Month 3
> At least one classical model achieves AUROC ≥ 0.75. Feature importance analysis reveals biologically plausible patterns (≥ 30% of top-20 features match known motifs). All results are reproducible. Decision Gate 1→2 is passed.

---

## Month 6: Deep Learning and Cross-Cell-Type Analysis

### Objective
Complete Phase 2 (CNN) and cross-cell-type experiments. Begin research writing.

### Milestones

| # | Milestone | Target date | Success criterion |
|---|-----------|-------------|-------------------|
| M6.1 | **Shallow CNN trained** | Month 4, Week 2 | CNN achieves AUROC ≥ 0.80 on test set; training completes on Kaggle GPU |
| M6.2 | **CNN filter analysis** | Month 4, Week 4 | First-layer filters visualized as sequence logos; ≥ 3 filters match JASPAR motifs |
| M6.3 | **Saliency / Integrated Gradients** | Month 5, Week 2 | Saliency maps generated for 100+ test regions; qualitative analysis documented |
| M6.4 | **Cross-cell-type experiment** | Month 5, Week 4 | Experiment E4 complete for K562 + GM12878; cell-type-specific motifs identified |
| M6.5 | **Element-type classification** | Month 5, Week 4 | Experiment E5 complete; enhancer vs. promoter classifier evaluated |
| M6.6 | **Accuracy vs. interpretability report** | Month 6, Week 1 | Trade-off analysis comparing all model families documented |
| M6.7 | **Decision Gate 2→3 passed** | Month 6, Week 2 | All gate criteria met; team decides whether to pursue pretrained embeddings |
| M6.8 | **Draft research manuscript (introduction + methods)** | Month 6, Week 4 | Introduction and methods sections drafted |

### Deliverables
- [ ] Trained CNN model with filter analysis
- [ ] Saliency map gallery (representative examples)
- [ ] Cross-cell-type comparison report
- [ ] Enhancer vs. promoter classification results
- [ ] Full model comparison matrix (classical + CNN)
- [ ] Accuracy-vs-interpretability analysis document
- [ ] Draft manuscript (intro + methods)
- [ ] Updated Kaggle notebooks

### Definition of Success at Month 6
> CNN achieves AUROC ≥ 0.80. Cross-cell-type analysis reveals cell-type-specific regulatory features. At least one novel (non-JASPAR) candidate motif pattern is identified by the CNN. Research manuscript is in progress.

---

## Month 12 Vision: Research-Ready Platform

### Objective
Complete remaining phases (as time allows), submit or post publication, and ensure the project is documented for long-term continuity.

### Milestones (Months 7–12)

| # | Milestone | Target window | Success criterion |
|---|-----------|--------------|-------------------|
| M12.1 | **Pretrained embedding experiments** (Phase 3) | Months 7–9 | Embeddings extracted from ≥ 1 pretrained model; comparison to baselines documented |
| M12.2 | **Hybrid model** (k-mer + embedding) | Month 8–9 | Hybrid classifier evaluated; SHAP on hybrid features |
| M12.3 | **Full results manuscript** | Month 9–10 | Complete draft including results, discussion, limitations |
| M12.4 | **Model card(s)** | Month 10 | Model card for best model documenting performance, limitations, intended use |
| M12.5 | **Public code release** | Month 10 | Cleaned repository with README, license, installation instructions |
| M12.6 | **Publication submission** | Month 11 | Submitted to target venue (workshop or journal) |
| M12.7 | **Demo or visualization tool** (optional) | Month 12 | Gradio app or static HTML showing interactive regulatory predictions |
| M12.8 | **Project handoff documentation** | Month 12 | Updated all docs; contributor onboarding verified with a new team member |

### Deliverables (Months 7–12)
- [ ] Pretrained embedding comparison report
- [ ] Complete research manuscript
- [ ] Model card(s)
- [ ] Public repository release
- [ ] Submitted paper or preprint
- [ ] (Optional) Interactive demo
- [ ] Full project documentation update

### Definition of Success at Month 12
> The project has produced:
> 1. A published or submitted research paper on interpretable regulatory prediction.
> 2. A reproducible codebase usable by other researchers.
> 3. Documented evidence of regulatory switch-like sequence patterns.
> 4. A platform that can be extended by future contributors.

---

## What "Success" Means at Each Stage

| Stage | Success definition | Minimum acceptable outcome |
|-------|-------------------|---------------------------|
| **Month 1** | Pipeline works; first model trained | Data loads correctly; any AUROC > 0.5 |
| **Month 3** | Interpretable baseline with motif validation | AUROC ≥ 0.75; ≥ 3 known motifs in top-20 features |
| **Month 6** | CNN + cross-cell-type analysis | AUROC ≥ 0.80; cell-type-specific patterns documented |
| **Month 12** | Publication + reproducible platform | Paper submitted; code public; handoff-ready |

### Minimum Viable Research Contribution (MVRC)

Even if the project stops at Month 3, the following constitutes a valid research contribution:

1. Reproducible pipeline for ENCODE-based regulatory classification.
2. Multi-model comparison showing accuracy-interpretability trade-offs.
3. Feature importance analysis connecting k-mer/motif features to known biology.
4. Honest documentation of what works and what doesn't.

This can be published as a workshop paper or technical report.

---

## Milestone Dependency Graph

```
M1.1 (Setup)
  │
  ├── M1.2 (Literature) ─────────────────────────────────────────┐
  │                                                               │
  └── M1.3 (Data pipeline) ──→ M1.4 (First baseline)            │
                                  │                               │
                          M3.1 (Multi-model) ──→ M3.5 (k sweep)  │
                            │                     │               │
                          M3.2 (Importance) ──→ M3.6 (Ablation)  │
                            │                                     │
                          M3.3 (SHAP) ──→ M3.4 (Neg. sensitivity)│
                                           │                      │
                                         M3.7 (Gate 1→2) ────────┘
                                           │                      │
                                         M3.8 (Phase 1 report)   │
                                           │                      │
                                 ┌─────────┘                      │
                                 │                                │
                          M6.1 (CNN) ──→ M6.2 (Filters)          │
                            │             │                       │
                          M6.3 (Saliency) M6.4 (Cell types)      │
                            │                    │                │
                          M6.5 (Enh vs Prom)     │                │
                            │                    │                │
                          M6.6 (Trade-off) ──────┘                │
                            │                                     │
                          M6.7 (Gate 2→3)                         │
                            │                                     │
                          M6.8 (Draft manuscript) ←───────────────┘
                            │
                    M12.1 (Pretrained) ──→ M12.2 (Hybrid)
                                            │
                                          M12.3 (Full manuscript)
                                            │
                                    M12.4 (Model card)
                                    M12.5 (Code release)
                                    M12.6 (Submission)
```

---

## Progress Tracking

This section should be updated as milestones are completed.

| Milestone | Status | Date completed | Notes |
|-----------|--------|---------------|-------|
| M1.1 | ☐ Not started | — | — |
| M1.2 | ☐ Not started | — | — |
| M1.3 | ☐ Not started | — | — |
| M1.4 | ☐ Not started | — | — |
| M3.1 | ☐ Not started | — | — |
| M3.2 | ☐ Not started | — | — |
| M3.3 | ☐ Not started | — | — |
| M3.4 | ☐ Not started | — | — |
| M3.5 | ☐ Not started | — | — |
| M3.6 | ☐ Not started | — | — |
| M3.7 | ☐ Not started | — | — |
| M3.8 | ☐ Not started | — | — |
| M6.1 | ☐ Not started | — | — |
| M6.2 | ☐ Not started | — | — |
| M6.3 | ☐ Not started | — | — |
| M6.4 | ☐ Not started | — | — |
| M6.5 | ☐ Not started | — | — |
| M6.6 | ☐ Not started | — | — |
| M6.7 | ☐ Not started | — | — |
| M6.8 | ☐ Not started | — | — |
| M12.1 | ☐ Not started | — | — |
| M12.2 | ☐ Not started | — | — |
| M12.3 | ☐ Not started | — | — |
| M12.4 | ☐ Not started | — | — |
| M12.5 | ☐ Not started | — | — |
| M12.6 | ☐ Not started | — | — |
| M12.7 | ☐ Not started | — | — |
| M12.8 | ☐ Not started | — | — |

---

*End of Document 10 — Roadmap and Milestones*  
*Next: [11 — Glossary & Project Memory](./11-glossary-project-memory.md)*
