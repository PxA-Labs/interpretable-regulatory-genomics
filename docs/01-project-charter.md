# 01 — Project Charter

## Interpretable Machine Learning for Discovering Hidden Regulatory Switches in Non-Coding DNA

| Field | Value |
|-------|-------|
| Document ID | CHARTER-001 |
| Version | 1.0 |
| Status | Active |
| Created | 2026-06-07 |
| Related docs | [PRD](./02-product-requirements-document.md), [Technical Design](./03-technical-design-document.md), [Glossary](./11-glossary-project-memory.md) |

---

## 1. Problem Statement

The human genome is approximately 3.2 billion base pairs long, yet only about 1.5–2% encodes proteins. The remaining ~98% — the non-coding genome — contains regulatory elements (enhancers, promoters, silencers, insulators) that control when, where, and how genes are expressed. Many disease-associated genetic variants identified by genome-wide association studies (GWAS) fall in non-coding regions, yet we lack systematic, interpretable methods to understand which non-coding sequences act as regulatory "switches" and why.

Current computational approaches face a tension:

- **High-capacity models** (large genomic transformers, foundation models) can achieve strong predictive accuracy but require substantial computational resources — often hundreds of GPU-hours and large institutional infrastructure — placing them beyond the reach of undergraduate research teams.
- **Simpler models** are computationally feasible but may miss complex regulatory patterns.
- **Both categories** often lack interpretability: they predict regulatory activity without explaining which sequence features drive the prediction.

**This project addresses the gap between predictive power and interpretability at feasible compute scale.** We aim to build a Kaggle-friendly, interpretable ML system that predicts regulatory behavior in non-coding DNA and explains which sequence patterns contribute to that prediction — enabling candidate "regulatory switch" discovery without requiring industrial-scale infrastructure.

---

## 2. Vision

> Build the most accessible, interpretable, and reproducible research pipeline for discovering regulatory switch-like patterns in non-coding DNA — starting from public data, running on student-accessible compute, and evolving into a research platform that connects sequence-level predictions to biological insight.

---

## 3. Mission

Develop and maintain an interpretable machine learning pipeline that:

1. **Predicts** whether a non-coding genomic region is likely regulatory (enhancer-like, promoter-like, accessible chromatin) or non-regulatory.
2. **Explains** which sequence motifs, k-mers, or learned features drive each prediction.
3. **Runs** entirely within Kaggle notebooks or equivalent environments during the initial build.
4. **Evolves** toward stronger models, richer interpretability, and broader datasets over time.
5. **Documents** everything for reproducibility and knowledge continuity.

---

## 4. Goals and Non-Goals

### 4.1 Goals

| ID | Goal | Measurable outcome |
|----|------|--------------------|
| G1 | Build a working regulatory vs. non-regulatory classifier for non-coding DNA | AUROC ≥ 0.80 on held-out test set using ENCODE-derived labels |
| G2 | Produce interpretable outputs for every prediction | Feature importance / motif attribution available for each test region |
| G3 | Validate that identified "important" motifs overlap known biology | ≥ 50% of top-10 motifs per cell type match known TF binding motifs in JASPAR/HOCOMOCO |
| G4 | Maintain Kaggle-executable pipeline | All core experiments runnable within Kaggle notebook constraints (16 GB RAM, single GPU, ~12 hr session) |
| G5 | Publish reproducible experiment artifacts | All experiments tracked with logged hyperparameters, metrics, and versioned data references |
| G6 | Create a research-ready codebase | Modular code with tests, documentation, and clear contribution guidelines |

### 4.2 Non-Goals (Explicit Exclusions)

| ID | Non-Goal | Rationale |
|----|----------|-----------|
| NG1 | Train a genomic foundation model from scratch | Requires compute far beyond student-accessible infrastructure |
| NG2 | Build a medical diagnostic product | No clinical validation pathway; regulatory and ethical barriers |
| NG3 | Perform wet-lab experimental validation | No lab access; computational prediction only |
| NG4 | Create a universal genome understanding system | Scope is limited to non-coding regulatory prediction |
| NG5 | Build a CRISPR editing platform | Entirely different domain and infrastructure |
| NG6 | Develop a generic "AI for biology" dashboard | Must stay anchored in the specific scientific task |
| NG7 | Process very long genomic sequences (>10 kb) in early phases | Memory and compute limitations on Kaggle |
| NG8 | Make claims of molecular causality | The project identifies candidate regulatory patterns, not proven causal mechanisms |

---

## 5. Stakeholders

| Stakeholder | Role | Needs |
|-------------|------|-------|
| Core student team (2–5 CS undergraduates) | Builders, researchers, maintainers | Clear scope, feasible compute, learning opportunities |
| Faculty advisor / PI | Scientific oversight, publication guidance | Sound methodology, reproducibility, publishable results |
| Future collaborators (students joining later) | Contributors | Onboarding documentation, clear codebase, protected scope |
| Computational biology community | Peer reviewers, users of released tools | Reproducible results, honest claims, open data |
| Open-source community | Potential downstream users | Well-documented code, permissive licensing, clean APIs |

---

## 6. Long-Term Impact

If successful, this project:

1. **Demonstrates** that meaningful regulatory pattern discovery is achievable without billion-parameter models or institutional GPU clusters.
2. **Produces** an interpretable analysis pipeline reusable by other groups studying non-coding regulation.
3. **Generates** candidate regulatory switch hypotheses testable by experimental collaborators.
4. **Establishes** a research platform extensible to multi-omics, cross-species, or disease-variant analysis.
5. **Publishes** findings in a computational biology venue (target: workshop paper → conference → journal).

---

## 7. Project Constraints

### 7.1 Compute Constraints

> **Critical constraint.** This shapes every technical decision.

| Resource | Available | Limit |
|----------|-----------|-------|
| GPU | Kaggle (1× NVIDIA T4 or P100, 16 GB VRAM) | 30 hrs/week GPU quota |
| RAM | Kaggle kernel: 16 GB (CPU), 13 GB (GPU) | Per-session limit |
| Disk | Kaggle: ~20 GB scratch + dataset storage | Per-notebook limit |
| Session duration | ~12 hours maximum | Auto-termination |
| Multi-GPU | Not available on Kaggle free tier | N/A |
| TPU | Kaggle TPU v3-8 available (limited weekly quota) | Useful for specific workloads but not default |

**Implication**: Large transformer-based genomic models demand substantial computational resources. Models such as Enformer (249M parameters, 196 kb input), DNABERT-2, Nucleotide Transformer, and HyenaDNA were trained on multi-GPU/TPU setups with significant infrastructure. Training these from scratch is out of scope. Inference or limited fine-tuning of smaller pretrained checkpoints may be explored in later phases only.

### 7.2 Data Constraints

- Public datasets only (no proprietary or restricted-access data in initial phases).
- ENCODE, GEO, and UCSC Genome Browser are primary sources.
- Data must be downloadable, parseable, and processable within Kaggle storage limits.
- Curated subsets preferred over full-database downloads.

### 7.3 Team Constraints

- 2–5 undergraduate CS students.
- Strong in software engineering and ML; variable in computational biology background.
- No dedicated bioinformatician on the team (biology knowledge will be acquired through literature).
- Asynchronous work expected; no guaranteed full-time commitment.

### 7.4 Ethical Constraints

- No human-subjects data requiring IRB approval.
- ENCODE data is publicly available under consortium data use policies.
- No personally identifiable genomic information is used.
- All model claims must be framed as computational predictions, not clinical assertions.

### 7.5 Timeline Constraints

- Initial working baseline: within 3 months.
- Research-grade results: within 6 months.
- Publication-ready work: within 12 months.
- The project is designed for incremental progress, not a single deliverable deadline.

---

## 8. Success Criteria (Summary)

| Timeframe | Definition of success |
|-----------|----------------------|
| 1 month | Data pipeline functional; at least one baseline model trained and evaluated |
| 3 months | Interpretable baseline with motif analysis operational; reproducible results |
| 6 months | Multi-model comparison complete; interpretability report generated; draft research write-up |
| 12 months | Platform extensible; potential publication submitted; documented for handoff |

Detailed milestones are in [Document 10 — Roadmap & Milestones](./10-roadmap-milestones.md).

---

## 9. Governance

- **Scope changes** require documented justification and team consensus (see [Glossary — Scope Lock](./11-glossary-project-memory.md)).
- **Model upgrades** beyond lightweight architectures must pass decision gates defined in [Modeling Roadmap](./06-modeling-roadmap.md).
- **Data additions** must be reviewed for licensing, size, and relevance before integration.
- **Claims** in any report or publication must distinguish between computational prediction and biological causality.

---

## 10. Scope Lock Statement

This section is canonical. It must be preserved and referenced in all project documents.

| Dimension | Locked scope |
|-----------|--------------|
| Domain | Non-coding regulatory genomics |
| Core objective | Discover and interpret candidate regulatory switch-like patterns in non-coding DNA |
| Early execution environment | Kaggle-friendly |
| Early model philosophy | Interpretable and lightweight first |
| Large genomic transformer training | Out of scope for initial build (compute limits) |
| Preferred starting data | Public datasets such as ENCODE |
| Long-term evolution | Stronger models, richer interpretation, broader generalization — only after a working baseline exists |

---

*End of Document 01 — Project Charter*
*Next: [02 — Product Requirements Document](./02-product-requirements-document.md)*
