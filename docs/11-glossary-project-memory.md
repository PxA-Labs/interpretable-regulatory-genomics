# 11 — Glossary and Project Memory Document

## Interpretable Machine Learning for Discovering Hidden Regulatory Switches in Non-Coding DNA

| Field | Value |
|-------|-------|
| Document ID | GPM-001 |
| Version | 1.0 |
| Status | Active — Living Document |
| Created | 2026-06-07 |
| Related docs | [Charter](./01-project-charter.md), [Onboarding](./12-contributor-onboarding.md) |

---

## Purpose

This document serves three critical functions:

1. **Glossary**: Defines all recurring terms to ensure consistent language across the team and all documents.
2. **Project Memory**: Preserves decisions already made so future assistants, collaborators, or AI tools do not re-open settled questions.
3. **Scope Guard**: Defines forbidden scope assumptions to prevent drift.

> **This is a living document.** Update it whenever a new term is introduced, a decision is made, or a scope boundary is clarified.

---

## Part 1: Glossary of Project Terms

### Biology Terms

| Term | Definition as used in this project |
|------|------------------------------------|
| **Non-coding DNA** | DNA that does not encode proteins. Constitutes ~98% of the human genome. Contains regulatory elements that control gene expression. |
| **Regulatory element** | A region of non-coding DNA that influences the expression of one or more genes. Includes enhancers, promoters, silencers, and insulators. |
| **Enhancer** | A regulatory element that increases transcription of a target gene. Can act over long genomic distances (up to ~1 Mb). Typically marked by H3K27ac and H3K4me1 histone modifications. |
| **Promoter** | A regulatory element located near the transcription start site (TSS) of a gene. Initiates transcription. Typically marked by H3K4me3. |
| **Silencer** | A regulatory element that represses transcription of target genes. |
| **Insulator** | A regulatory element that blocks enhancer-promoter communication or prevents spreading of chromatin states. Often bound by CTCF. |
| **Transcription factor (TF)** | A protein that binds to specific DNA sequence motifs and regulates transcription. |
| **Motif** | A short, conserved DNA sequence pattern (typically 6–20 bp) recognized by a transcription factor. Represented as a position weight matrix (PWM). |
| **Chromatin** | The complex of DNA and histone proteins that packages DNA in the nucleus. "Open" chromatin is accessible and often regulatory; "closed" chromatin is compact and often inactive. |
| **Histone modification** | Chemical modifications (methylation, acetylation) to histone proteins that mark regulatory states. Key marks: H3K27ac (active enhancers), H3K4me3 (active promoters), H3K4me1 (poised enhancers). |
| **DNase-seq / ATAC-seq** | Assays that measure chromatin accessibility. DNase-seq uses DNase I enzyme; ATAC-seq uses Tn5 transposase. Open regions are "peaks." |
| **ChIP-seq** | Chromatin immunoprecipitation followed by sequencing. Identifies where specific proteins (TFs, modified histones) bind to DNA. |
| **TSS** | Transcription Start Site — the position where transcription of a gene begins. |
| **cCRE** | Candidate Cis-Regulatory Element — a computationally predicted regulatory region defined by ENCODE based on chromatin accessibility and histone marks. |
| **GWAS** | Genome-Wide Association Study — identifies genetic variants (SNPs) associated with traits or diseases. Many GWAS hits fall in non-coding regions. |
| **hg38 / GRCh38** | The current human reference genome assembly (Genome Reference Consortium Human Build 38). |

### Project-Specific Terms

| Term | Definition as used in this project |
|------|------------------------------------|
| **Regulatory switch** | A sequence region or local sequence-pattern combination in non-coding DNA that appears to strongly influence regulatory activity. This is the project's interpretive framing for biologically meaningful local regulatory patterns. **This is NOT a claim of proven molecular causality.** It is a working label for candidate patterns identified computationally. |
| **Switch-like pattern** | A learned or engineered feature (k-mer, motif, CNN filter pattern) that the model identifies as highly predictive of regulatory activity. The "switch" metaphor reflects the idea that specific sequence patterns can toggle regulatory states (active/inactive). |
| **Positive region** | A genomic region labeled as regulatory in the project's classification task. Derived from ENCODE cCRE annotations or cell-type-specific peaks. |
| **Negative region** | A genomic region labeled as non-regulatory. Sampled from the genome with specific controls (GC-matching, exclusion of known annotations). |
| **Chromosome-holdout split** | Data splitting strategy where entire chromosomes are assigned to train, validation, or test sets. Prevents leakage from nearby regulatory elements. Train: chr1–14,X; Val: chr15–18; Test: chr19–22. |
| **Feature importance** | A measure of how much a specific feature (k-mer, motif score, etc.) contributes to the model's predictions. Methods: Gini importance, permutation importance, SHAP, saliency. |
| **Motif cross-reference** | The process of comparing model-identified important features against databases of known TF binding motifs (JASPAR, HOCOMOCO) to validate biological plausibility. |
| **Kaggle-friendly** | A design constraint meaning the system must run within Kaggle notebook limits: 16 GB RAM, 1 GPU (T4/P100, 16 GB VRAM), ~12 hr session, ~20 GB disk, 30 hr/week GPU quota. |
| **Decision gate** | A set of criteria that must be met before the project progresses from one modeling phase to the next. Defined in [Modeling Roadmap](./06-modeling-roadmap.md). |

### Data Terms

| Term | Definition |
|------|-----------| 
| **BED file** | Tab-delimited text format for genomic intervals: chromosome, start, end, and optional additional columns. Standard format for regulatory annotations. |
| **narrowPeak** | Extended BED format used for ChIP-seq and DNase-seq peaks. Includes signal value, p-value, q-value, and peak summit position. |
| **FASTA** | Text format for DNA sequences. Each entry has a header line (>name) followed by the sequence. |
| **One-hot encoding** | Representation of DNA as a binary matrix: each nucleotide (A, C, G, T) encoded as a 4-dimensional binary vector. A 1 kb sequence becomes a (1000, 4) matrix. |
| **k-mer** | A subsequence of length k. k-mer frequency = count of each possible k-length subsequence in a region. For k=4, there are 4^4 = 256 possible k-mers. |
| **PWM / PFM** | Position Weight Matrix / Position Frequency Matrix — represents a TF binding motif as a matrix of nucleotide frequencies at each position. |
| **GC content** | The fraction of a DNA sequence that is G or C nucleotides. Varies across the genome and between regulatory and non-regulatory regions. |

### Model Terms

| Term | Definition |
|------|-----------| 
| **AUROC** | Area Under the Receiver Operating Characteristic curve. Measures classifier performance across all thresholds. 0.5 = random; 1.0 = perfect. Primary metric for this project. |
| **AUPRC** | Area Under the Precision-Recall Curve. More informative than AUROC for imbalanced datasets. |
| **SHAP** | SHapley Additive exPlanations — a method for explaining individual predictions by computing the contribution of each feature based on cooperative game theory. |
| **Saliency map** | A per-input-element attribution map showing which parts of the input most affect the model's output. Computed via gradients for neural networks. |
| **Integrated Gradients** | An attribution method that computes feature importance by integrating gradients along a path from a reference input (e.g., all-zeros) to the actual input. More stable than vanilla saliency. |
| **Convolutional filter** | A learned pattern in a CNN. First-layer filters on one-hot DNA often learn motif-like patterns. |

---

## Part 2: Project Memory — Decisions Already Made

### Locked Decisions

These decisions have been made and should NOT be re-opened without documented justification and team consensus.

| ID | Decision | Rationale | Date decided | Document reference |
|----|----------|-----------|-------------|-------------------|
| D-01 | **Project scope is interpretable ML for non-coding regulatory DNA.** | Selected from a broader idea-selection process. Provides clear scientific focus and feasible execution path. | 2026-06-07 | Charter §10 |
| D-02 | **"Regulatory switch" is the project's interpretive label, not a molecular claim.** | Avoids overstatement; keeps language honest and defensible. | 2026-06-07 | Charter §1, Glossary |
| D-03 | **Initial execution on Kaggle.** | Team lacks institutional compute; Kaggle is free, accessible, and sufficient for Phases 0–2. | 2026-06-07 | Compute Memo §1 |
| D-04 | **Training large genomic transformers from scratch is out of scope.** | Requires multi-GPU, multi-day training costing $10K+; not feasible for a student team. | 2026-06-07 | Compute Memo §2 |
| D-05 | **ENCODE cCREs are the starting dataset.** | Pre-classified, well-curated, freely available, widely used, Kaggle-compatible. | 2026-06-07 | Dataset Strategy §2 |
| D-06 | **Chromosome-holdout is the primary split strategy.** | Prevents leakage from nearby regulatory elements sharing chromatin domains. Train: chr1–14,X; Val: chr15–18; Test: chr19–22. | 2026-06-07 | Dataset Strategy §6 |
| D-07 | **K562 is the primary cell type; GM12878 is secondary.** | Both are well-annotated Tier 1 ENCODE cell lines with extensive data. | 2026-06-07 | Dataset Strategy §2 |
| D-08 | **Interpretability is a first-class requirement, not an afterthought.** | The project's thesis is that interpretable models can discover regulatory patterns. Accuracy without interpretability does not meet the project's goals. | 2026-06-07 | Charter §4, Research Design §1.4 |
| D-09 | **Staged modeling approach (classical → CNN → pretrained).** | Builds complexity incrementally; validates feasibility before investing in expensive experiments. | 2026-06-07 | Modeling Roadmap |
| D-10 | **Fixed region length of 1 kb for initial experiments.** | Captures most enhancer/promoter-scale regulatory elements; manageable for compute; avoids length-based artifacts. | 2026-06-07 | Technical Design §2 |
| D-11 | **GC-matched negative sampling with exclusion of known annotations.** | Controls for compositional bias; reduces false negatives in negative set. | 2026-06-07 | Dataset Strategy §4 |
| D-12 | **No clinical claims.** | The project produces computational predictions, not medical diagnostics. | 2026-06-07 | Charter §4 (NG5, NG6) |

### Open Decisions (Not Yet Resolved)

| ID | Question | Options under consideration | Decision needed by |
|----|----------|---------------------------|-------------------|
| OD-01 | Which exact ENCODE experiments (accession IDs) to use for K562? | Multiple DNase-seq and histone ChIP-seq experiments available | Phase 0 data curation |
| OD-02 | Exact k-mer range for baseline features? | k=4 only, k=4+5, k=4+5+6, k=3+4+5+6 | Experiment E3 (Phase 1) |
| OD-03 | Whether to include motif scan features in Phase 1 or defer to Phase 2? | Include early (more features) vs. defer (simpler baseline) | Phase 1 planning |
| OD-04 | Which pretrained model to use for Phase 3? | DNABERT-2, Nucleotide Transformer (small), HyenaDNA (small) | Decision Gate 2→3 |
| OD-05 | Target publication venue? | MLCB workshop, Bioinformatics journal, bioRxiv preprint | Month 6 |
| OD-06 | Whether to build a Gradio demo? | Build demo (outreach value) vs. skip (focus on research) | Month 9 |
| OD-07 | License for public code release? | MIT, Apache 2.0, BSD-3 | Month 9 |

---

## Part 3: Forbidden Scope Assumptions

The following assumptions are **explicitly forbidden**. Any contributor (human or AI) must not make these assumptions when working on this project.

| # | Forbidden assumption | Why it's forbidden |
|---|---------------------|--------------------|
| FA-01 | "We have access to a multi-GPU cluster." | We don't. Initial compute is Kaggle free-tier only. |
| FA-02 | "We can train a genomic transformer from scratch." | Requires $10K+ of compute and multi-GPU hardware. Out of scope. |
| FA-03 | "The model's predictions prove molecular causality." | We identify statistical associations, not causal mechanisms. |
| FA-04 | "ENCODE labels are ground truth." | cCREs are computational predictions by the ENCODE consortium, not experimentally validated for every element. |
| FA-05 | "We can validate predictions in a wet lab." | No lab access. Computational validation only. |
| FA-06 | "This is a medical product." | No clinical validation, no regulatory pathway, no diagnostic claims. |
| FA-07 | "We need very long sequences (>10 kb) from the start." | 1 kb is sufficient for initial enhancer/promoter-scale analysis. Longer sequences require more compute. |
| FA-08 | "Accuracy is more important than interpretability." | Both matter, but interpretability is the project's differentiating thesis. Don't sacrifice it for marginal accuracy gains. |
| FA-09 | "We should download all of ENCODE." | Start with a curated subset. All of ENCODE is terabytes of data. |
| FA-10 | "Phase 3 and 4 are required deliverables." | They are aspirational extensions. The project is successful if Phase 1–2 produces strong results. |

---

## Part 4: Project Labels and Model Classes

### Classification Labels

| Label name | Code | Definition | Source |
|------------|------|-----------|--------|
| Regulatory | `1` | Region annotated as cCRE with active signal in target cell type | ENCODE SCREEN |
| Non-regulatory | `0` | Randomly sampled region NOT in any cCRE, NOT in coding regions | Generated |

### Multi-Class Labels (Phase 2)

| Label name | Code | Definition |
|------------|------|-----------|
| Promoter-Like Signature (PLS) | `pls` | cCRE with high H3K4me3 and DNase near TSS |
| Proximal Enhancer-Like (pELS) | `pels` | cCRE with high H3K27ac within 2 kb of TSS |
| Distal Enhancer-Like (dELS) | `dels` | cCRE with high H3K27ac > 2 kb from TSS |
| CTCF-only | `ctcf` | cCRE with CTCF binding, no other marks |
| Non-regulatory | `neg` | Generated negative region |

### Cell Type Identifiers

| Cell type | ENCODE tier | Primary use |
|-----------|-------------|-------------|
| K562 | Tier 1 | Primary cell type for all experiments |
| GM12878 | Tier 1 | Secondary cell type for cross-cell-type comparison |
| HepG2 | Tier 2 | Optional third cell type |

---

## Part 5: Acronyms and Abbreviations

| Abbreviation | Full form |
|-------------|-----------|
| AUROC | Area Under the Receiver Operating Characteristic |
| AUPRC | Area Under the Precision-Recall Curve |
| BED | Browser Extensible Data (genomic interval format) |
| cCRE | Candidate Cis-Regulatory Element |
| ChIP-seq | Chromatin Immunoprecipitation followed by Sequencing |
| CNN | Convolutional Neural Network |
| CTCF | CCCTC-Binding Factor (a transcription factor) |
| dELS | Distal Enhancer-Like Signature |
| DL | Deep Learning |
| ENCODE | Encyclopedia of DNA Elements |
| FASTA | Fast-All (sequence file format) |
| GC | Guanine-Cytosine (nucleotides) |
| GEO | Gene Expression Omnibus |
| GTEx | Genotype-Tissue Expression project |
| GWAS | Genome-Wide Association Study |
| hg38 | Human Genome assembly 38 |
| IG | Integrated Gradients |
| JASPAR | A database of transcription factor binding profiles |
| k-mer | A subsequence of length k |
| LogReg | Logistic Regression |
| ML | Machine Learning |
| mm10 | Mouse Genome assembly 10 |
| MLOps | Machine Learning Operations |
| pELS | Proximal Enhancer-Like Signature |
| PFM | Position Frequency Matrix |
| PLS | Promoter-Like Signature |
| PWM | Position Weight Matrix |
| RF | Random Forest |
| SHAP | SHapley Additive exPlanations |
| SNP | Single Nucleotide Polymorphism |
| TF | Transcription Factor |
| TSS | Transcription Start Site |
| XGB | XGBoost (Extreme Gradient Boosting) |

---

## Part 6: Update Log

| Date | Change | Author |
|------|--------|--------|
| 2026-06-07 | Document created with initial glossary, decisions, and scope guard. | Project initialization |

---

*End of Document 11 — Glossary and Project Memory*  
*Next: [12 — Contributor Onboarding Brief](./12-contributor-onboarding.md)*
