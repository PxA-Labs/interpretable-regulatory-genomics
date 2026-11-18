# Phase 4: Extensibility into Multi-Omics or Cell-Type-Specific Prediction

- Discussion: https://github.com/PxA-Labs/interpretable-regulatory-genomics/discussions/12
- Category: Show and tell
- Author: archittmittal
- Created: 2026-06-16T17:46:11Z
- Updated: 2026-06-18T17:16:54Z
- Labels: none

## Original post

With the completion of Phase 1 (CpG Island baselines), Phase 2 (PyTorch CNNs & biological motif validation), and Phase 3 (Foundation Model embeddings), we are now looking toward the future extensibility of the codebase. 

As outlined in the modeling roadmap, **Phase 4: Extensibility into Multi-Omics or Cell-Type-Specific Prediction** is an aspirational phase designed to scale the system beyond sequence-only, single-cell-type models. 

Below are the 5 candidate extensions proposed for Phase 4:

---

### Candidate Extensions

| Extension | Objective | Data Source | Complexity |
| :--- | :--- | :--- | :--- |
| **1. Multi-Cell-Type Regulatory Prediction** | Predict cell-type-specific regulatory activity directly from DNA sequences. | ENCODE cCREs across 5+ cell types | Moderate |
| **2. Expression-Linked Regulatory Scoring** | Link regulatory predictions of non-coding elements to nearby target gene expression. | GTEx summary data | Moderate-High |
| **3. Cross-Species Regulatory Conservation** | Compare regulatory predictions between human (hg38) and mouse (mm10) to assess evolutionary conservation. | ENCODE mouse + mm10 | High |
| **4. Variant Effect Prediction** | Perform in silico mutagenesis to predict how non-coding SNPs alter regulatory predictions and disrupt transcription factor binding. | dbSNP / ClinVar | High |
| **5. Single-Cell Accessibility** | Predict cell-type-specific chromatin accessibility at single-cell resolution. | Published scATAC-seq datasets | High |

---

### Open Discussion & Feedback

We would love to hear feedback from the team and community on which direction we should prioritize:

1. **Variant Effect Prediction (In Silico Mutagenesis):** This has high clinical utility (e.g., scoring GWAS variants or mutations in oncology). Should we implement a pipeline that mutates each base pair in identified enhancers and scores the delta in CNN predictions?
2. **Multi-Cell-Type Regulatory Prediction:** Capturing the cooperative combinatorial binding logic across multiple cell lines is crucial for publishing in high-impact genomics journals. Is this the most logical engineering step?
3. **Multi-Omics & Expression Linking:** Linking our regulatory predictions to GTEx expression data would bridge the gap between "regulatory switches" and actual gene transcription.


## Comments

### purvanshjoshi — 2026-06-18T17:16:53Z

Hi @archittmittal and team,

After performing a comprehensive audit of our current codebase against the project's official 12-month delivery roadmap (available in `10-roadmap-milestones.pdf`), we are **halting the start of Phase 4 (Multi-Omics Extensibility) for now**. 

While Phase 3 completed successfully, jumping into the complexity of external multi-omics data integration (like GTEx expression or single-cell accessibility) is premature because several critical sequence-only validation checks and baseline experiments from prior phases remain incomplete. 

To ensure our findings are scientifically rigorous and robust before adding more modalities, we need to first focus on strengthening our baselines. The core reasons we are holding on Phase 4 are:

### 1. Phase 1 (Classical Baseline) Gaps
* **GC-Bias Trap (Experiment E7)**: We must formally validate our models against multiple negative sampling strategies (GC-matched vs. random unmatched vs. flanking negatives) to prove they are learning regulatory syntax rather than exploiting overall GC-content.
* **$k$-mer sweep (Experiment E3)**: We need to evaluate performance for $k \in [3, 4, 5, 6]$ to confirm if $k=4$ is actually optimal or if higher-order $k$-mers provide significantly better features.
* **Feature Ablation (Experiment E2)**: We need to run step-by-step feature ablation (k-mers $\rightarrow$ +GC $\rightarrow$ +dinuc $\rightarrow$ +motifs) to quantify feature contributions.

### 2. Phase 2 (Deep Learning) Gaps
* **Cross-Cell-Type Verification (Experiment E4)**: The model is currently locked to the K562 cell line. To ensure the CNN is generalizable, we need to run cross-cell-type tests with the **GM12878** dataset.
* **Enhancer vs. Promoter Multiclass classification (Experiment E5)**: Currently, promoters (PLS) and enhancers (dELS) are pooled into a single class. Since they represent very different biology, we need to train a 3-class model to distinguish between `[Non-regulatory, Promoter-like, Enhancer-like]`.

### 3. Phase 3 (Pretrained Embeddings) Gaps
* **Hybrid k-mer + Embedding Model (Experiment E8c)**: We have not yet trained a hybrid model combining raw $k$-mer frequencies and Nucleotide Transformer embeddings. This is a critical baseline to check if a joint model can surpass our current $0.8830$ AUROC limit.

We have opened a tracking issue [Issue #13](https://github.com/PxA-Labs/interpretable-regulatory-genomics/issues/13) detailing these gaps. Let's focus on closing these items first to establish a solid foundation before opening the door to multi-omics.

