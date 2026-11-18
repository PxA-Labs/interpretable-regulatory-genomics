# 09 — Risk Register

## Interpretable Machine Learning for Discovering Hidden Regulatory Switches in Non-Coding DNA

| Field | Value |
|-------|-------|
| Document ID | RR-001 |
| Version | 1.0 |
| Status | Active |
| Created | 2026-06-07 |
| Related docs | [Charter](./01-project-charter.md), [Compute Memo](./07-compute-feasibility-memo.md), [Roadmap](./10-roadmap-milestones.md) |

---

## Risk Assessment Framework

Each risk is assessed on:

| Dimension | Scale |
|-----------|-------|
| **Likelihood** | Low / Medium / High |
| **Impact** | Low / Medium / High / Critical |
| **Overall severity** | Low / Medium / High / Critical |

---

## 1. Scientific Risks

| ID | Risk | Likelihood | Impact | Severity | Mitigation | Fallback |
|----|------|-----------|--------|----------|------------|----------|
| SR-01 | **k-mer features are insufficient** to capture regulatory signal; models learn GC content or simple composition instead of biologically meaningful patterns. | Medium | High | High | Feature ablation experiment (remove GC, test remaining signal). Compare k-mer importances to random expectations. Validate top features against JASPAR. | Add motif scan features. Move to CNN (Phase 2) which learns features directly. |
| SR-02 | **ENCODE cCRE labels are noisy or circular**: cCREs are themselves computationally predicted, so the model may learn to replicate the ENCODE pipeline rather than discover new patterns. | Medium | Medium | Medium | Use multiple label sources (DNase peaks, histone marks) as independent validation. Clearly state limitation in publications. | Frame results as "consistent with ENCODE annotations" rather than "discovered new biology." |
| SR-03 | **Negative set design introduces bias**: if negatives are systematically different from positives (e.g., in sequence complexity, repeat content), the model exploits these artifacts. | Medium | High | High | GC-matched + exclude-known negatives. Multiple negative sampling strategies with sensitivity analysis (E7). Check feature importance for compositional vs. motif features. | If all negative strategies show same features, the bias is minimal. If results vary wildly, report the sensitivity analysis honestly. |
| SR-04 | **Motif databases are incomplete**: top features may not match known motifs because the motifs are novel, not because they are noise. | High | Low | Medium | Report unmatched motifs as "candidate novel patterns" with appropriate caveats. Do not claim biological significance without experimental validation. | Treat this as a positive research finding if the patterns are consistent and reproducible. |
| SR-05 | **Cell-type-specific results may not generalize**: models trained on K562 may not transfer to other cell types, limiting biological insight. | Medium | Medium | Medium | Cross-cell-type experiments (E4). Report per-cell-type and pooled results. Document which findings are cell-type-specific vs. general. | If cell-type-specific, this is actually an interesting result about regulatory diversity. |
| SR-06 | **Overfitting to training chromosomes**: despite chromosome-holdout, the model may learn chromosome-specific patterns that don't generalize. | Low | Medium | Low | Verify that test chromosomes (chr19–22) have representative regulatory element density. Add leave-one-chromosome-out analysis as sanity check. | If test-set performance is much lower than CV, investigate chromosome-specific confounders. |

---

## 2. Data Risks

| ID | Risk | Likelihood | Impact | Severity | Mitigation | Fallback |
|----|------|-----------|--------|----------|------------|----------|
| DR-01 | **ENCODE portal becomes unavailable** or changes data access policy. | Low | High | Medium | Download and cache all required files early. Store versioned copies as Kaggle datasets. Document exact file URLs and checksums. | Use cached data. If ENCODE is permanently unavailable, switch to Roadmap Epigenomics or FANTOM5 (alternative sources). |
| DR-02 | **Reference genome version mismatch**: using hg38 annotations with wrong genome build causes incorrect sequence extraction. | Low | Critical | Medium | Always verify genome assembly version. Include assembly version in data pipeline logging. Cross-check extracted sequences against UCSC Genome Browser for sample regions. | Detect early by checking sequence content at known loci. |
| DR-03 | **Data too large for Kaggle**: curated dataset exceeds Kaggle storage limits. | Low | Medium | Low | Start with K562-only subset (~100K regions). Upload per-chromosome FASTA as separate datasets. Use compressed formats (.npz). | Process locally and upload only feature matrices to Kaggle. |
| DR-04 | **Class imbalance**: far more regulatory or non-regulatory regions depending on sampling strategy. | Medium | Medium | Medium | Control positive:negative ratio (default 1:1). Apply class weights in training. Report results for multiple ratios. | Use oversampling (SMOTE for tabular, augmentation for sequences) if imbalance is severe. |
| DR-05 | **Data leakage** through overlapping regions, shared regulatory domains, or chromosomal proximity. | Medium | Critical | High | Chromosome-holdout split. Verify no region overlap between splits. Check for data leakage using a "leakage detector" that tests whether split membership is predictable from features. | If leakage detected, re-design split. In worst case, use leave-one-chromosome-out evaluation. |

---

## 3. Modeling Risks

| ID | Risk | Likelihood | Impact | Severity | Mitigation | Fallback |
|----|------|-----------|--------|----------|------------|----------|
| MR-01 | **All models achieve < 0.70 AUROC**: the task may be harder than expected, or features may be wrong. | Low-Medium | High | High | Check data pipeline for bugs first. Verify labels are correct. Try different feature sets. Ensure trivial baselines (random, GC-only) are reasonable. | If data and labels are correct and AUROC < 0.70, document as negative result. Pivot to multi-class (enhancer vs. promoter vs. background) which may have stronger signal. |
| MR-02 | **CNN does not improve over tree-based models**: diminishing returns from deep learning at this scale. | Medium | Low | Low | Document as a valid finding. It means interpretable models are sufficient — this supports the project thesis. | Stay with tree-based models. This is a defensible research outcome. |
| MR-03 | **Pretrained embeddings do not improve classification**: foundation model representations may not capture regulatory-relevant features at 1 kb scale. | Medium | Low | Low | Document as negative result. Compare embedding quality across different pretrained models. | Stay with Phase 2 results. Report that pretrained models didn't add value for this specific task and scale. |
| MR-04 | **Hyperparameter sensitivity**: results change significantly with different hyperparameters, making findings unreliable. | Medium | Medium | Medium | Report best, worst, and median across a reasonable hyperparameter search. Use cross-validation for stability. | If results are unstable, increase dataset size or use ensemble methods. |
| MR-05 | **Overfitting** (train AUROC ≫ test AUROC). | Medium | Medium | Medium | Monitor train/val gap throughout training. Use regularization (L1/L2, dropout, early stopping). | Simplify model (fewer features, shallower architecture). |

---

## 4. Interpretability Risks

| ID | Risk | Likelihood | Impact | Severity | Mitigation | Fallback |
|----|------|-----------|--------|----------|------------|----------|
| IR-01 | **Feature importance is dominated by GC content**: the model relies primarily on overall nucleotide composition rather than specific motifs or patterns. | Medium | High | High | Include GC content as an explicit feature. If GC is top-important, run ablation with GC removed. Use GC-matched negatives. | If GC dominates even with matching, report this as a finding (composition is the primary signal). Add motif-scan features to capture finer-grained patterns. |
| IR-02 | **SHAP computation is too slow**: for large feature sets (k=6, 4096 features) × many samples, SHAP may take hours. | Medium | Low | Low | Use `TreeExplainer` (fast for tree models). Subsample to 1K–5K samples. Use approximate SHAP methods. | Report feature importance from built-in model methods (Gini, permutation) instead of SHAP. |
| IR-03 | **CNN filters do not match known motifs**: learned patterns may be biologically uninterpretable. | Medium | Medium | Medium | Visualize filters as sequence logos. Compare quantitatively (e.g., TOMTOM motif comparison). Some filters may capture non-motif structural features. | Report unmatched filters as "learned patterns" with appropriate caveats. Not every learned feature needs to match a known motif. |
| IR-04 | **Saliency maps are noisy or inconsistent**: gradient-based attribution can be sensitive to input perturbations. | Medium | Medium | Medium | Use Integrated Gradients (more stable than vanilla gradients). Average over multiple reference baselines. Compare saliency patterns across samples. | Use SmoothGrad or ensemble saliency. Report limitations of gradient-based interpretability. |
| IR-05 | **Attention weights are misinterpreted**: attention weights do not necessarily reflect feature importance, a known limitation in the interpretability literature. | Medium | Medium | Medium | If using attention, clearly state that attention weights are descriptive, not causal. Use attention only as supplementary to gradient-based methods. | Do not rely on attention as the primary interpretability mechanism. |

---

## 5. Compute Risks

| ID | Risk | Likelihood | Impact | Severity | Mitigation | Fallback |
|----|------|-----------|--------|----------|------------|----------|
| CR-01 | **Kaggle reduces free-tier GPU quota**: quota changes could limit weekly GPU hours. | Low-Medium | High | Medium | Monitor Kaggle announcements. Use CPU runtime for classical ML. Minimize GPU usage. | Migrate to Google Colab free tier. Consider Colab Pro ($12/month). |
| CR-02 | **Kaggle session disconnects** during long training runs, losing unsaved progress. | Medium | Medium | Medium | Save model checkpoints every 5 epochs. Save intermediate results to `/kaggle/working/`. Use try/except to save on error. | Re-run from last checkpoint. Design experiments to complete within 4 hours. |
| CR-03 | **Out-of-memory on Kaggle** when loading pretrained models or large feature matrices. | Medium | Medium | Medium | Use float16 for pretrained model inference. Process in batches. Use sparse matrices for k-mer features. | Use smallest pretrained model variant. Pre-compute embeddings in multiple sessions. |
| CR-04 | **Library version conflicts on Kaggle**: Kaggle's pre-installed packages may conflict with required versions. | Low | Low | Low | Pin versions in `requirements.txt`. Test on Kaggle early. Use `pip install --upgrade` for specific packages. | Adapt code to available versions. |

---

## 6. Timeline Risks

| ID | Risk | Likelihood | Impact | Severity | Mitigation | Fallback |
|----|------|-----------|--------|----------|------------|----------|
| TR-01 | **Data pipeline takes longer than expected**: parsing ENCODE formats, handling edge cases, debugging sequence extraction. | Medium | Medium | Medium | Start data pipeline work immediately (Phase 0). Use well-tested libraries (pybedtools, pysam). Test with small subsets first. | Allocate extra week. Use pre-processed datasets if available. |
| TR-02 | **Team member attrition**: students leave the project mid-stream. | Medium | High | High | Comprehensive documentation (this pack). Modular code. Onboarding brief. No single points of failure. | Remaining team members can continue with documented codebase. Recruit replacements using onboarding doc. |
| TR-03 | **Scope creep**: team members want to add features beyond current scope (multi-omics, variant analysis, web interface). | High | Medium | Medium | Scope lock in Charter and Glossary. Decision gates in Modeling Roadmap. All scope changes require documented justification. | Review scope lock document. Defer new ideas to Phase 4 or a separate project. |
| TR-04 | **Semester/academic calendar disruption**: exam periods, breaks reduce available effort. | High | Medium | Medium | Build buffer into timeline. Design experiments that can be paused and resumed. Prioritize early milestones. | Accept slower progress during disruptions. Extend timeline by 1–2 months if needed. |
| TR-05 | **Delayed publication**: results are not novel enough or reviewers raise issues. | Medium | Medium | Medium | Start writing early (background section during Phase 0). Target workshop papers first (lower bar). Plan for revisions. | Publish as preprint on bioRxiv. Present at internal seminars. |

---

## 7. Risk Severity Summary

| Severity | Count | Most critical risks |
|----------|-------|-------------------|
| **Critical** | 0 (none currently critical) | — |
| **High** | 5 | SR-01, SR-03, DR-05, MR-01, IR-01 |
| **Medium** | 14 | Most modeling, data, and timeline risks |
| **Low** | 5 | Compute version conflicts, CNN not improving |

### Top 5 Risks to Monitor

1. **SR-03 / DR-05**: Data leakage or negative set bias → fundamentally undermines all results.
2. **SR-01 / IR-01**: Features capturing composition rather than regulatory patterns → undermines interpretability thesis.
3. **MR-01**: Models fail to achieve meaningful accuracy → blocks all downstream work.
4. **TR-02**: Team attrition → documentation and modularity are the primary mitigation.
5. **CR-01**: Kaggle quota reduction → have backup compute plan.

---

*End of Document 09 — Risk Register*  
*Next: [10 — Roadmap & Milestones](./10-roadmap-milestones.md)*
