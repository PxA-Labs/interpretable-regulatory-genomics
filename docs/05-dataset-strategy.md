# 05 — Dataset Strategy Document

## Interpretable Machine Learning for Discovering Hidden Regulatory Switches in Non-Coding DNA

| Field | Value |
|-------|-------|
| Document ID | DSS-001 |
| Version | 1.0 |
| Status | Active |
| Created | 2026-06-07 |
| Related docs | [Technical Design](./03-technical-design-document.md), [Research Design](./04-research-design-document.md), [Compute Memo](./07-compute-feasibility-memo.md) |

---

## 1. Candidate Datasets

### 1.1 Priority Sources (Phase 1–2)

#### ENCODE Candidate Cis-Regulatory Elements (cCREs)

| Property | Detail |
|----------|--------|
| **Source** | ENCODE portal (https://www.encodeproject.org/) and ENCODE SCREEN (https://screen.encodeproject.org/) |
| **What it provides** | Pre-classified candidate regulatory regions in the human genome: promoter-like signatures (PLS), proximal enhancer-like signatures (pELS), distal enhancer-like signatures (dELS), CTCF-only, DNase-H3K4me3 |
| **Genome assembly** | hg38 (GRCh38) |
| **Format** | BED files downloadable from SCREEN or ENCODE portal |
| **Size** | ~926,535 human cCREs (as of ENCODE phase 3); varies by filter |
| **Why it matters** | Provides ready-made, expert-curated labels for regulatory regions; eliminates need for raw signal processing |
| **Licensing** | ENCODE data use policy: freely available for research; requires citation |

#### ENCODE DNase-seq / ATAC-seq Peak Files

| Property | Detail |
|----------|--------|
| **Source** | ENCODE portal; individual experiment pages |
| **What it provides** | Cell-type-specific open chromatin peaks (regions of accessible DNA) |
| **Format** | BED / narrowPeak |
| **Size** | ~50–200 MB per experiment (peaks); hundreds of experiments available |
| **Why it matters** | Enables cell-type-specific regulatory classification (a region is "regulatory in K562" if it has a DNase peak in K562) |
| **Recommended experiments** | Start with well-characterized cell types: K562 (ENCSR000EMT), GM12878 (ENCSR000EMU) |

#### ENCODE Histone ChIP-seq Peak Files

| Property | Detail |
|----------|--------|
| **Source** | ENCODE portal |
| **What it provides** | Peaks for histone marks associated with regulatory activity (H3K27ac = active enhancers; H3K4me3 = active promoters; H3K4me1 = poised enhancers) |
| **Format** | BED / narrowPeak |
| **Why it matters** | Distinguishes enhancer-like vs. promoter-like regions; enables element-type classification |
| **Recommended experiments** | H3K27ac and H3K4me3 in K562 and GM12878 |

#### Human Reference Genome (hg38)

| Property | Detail |
|----------|--------|
| **Source** | UCSC Genome Browser (https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/) or NCBI |
| **What it provides** | Full DNA sequence of the human genome |
| **Format** | FASTA (per-chromosome files recommended for memory efficiency) |
| **Size** | ~3.1 GB total; ~250 MB per large chromosome |
| **Why it matters** | Required for extracting DNA sequences at annotated coordinates |

#### JASPAR Motif Database

| Property | Detail |
|----------|--------|
| **Source** | https://jaspar.elixir.no/ |
| **What it provides** | Position frequency matrices (PFMs) for known transcription factor binding motifs |
| **Format** | MEME / PFM / JASPAR format |
| **Size** | ~10 MB |
| **Why it matters** | Cross-referencing learned features against known biology; motif enrichment analysis |

#### GENCODE Gene Annotations

| Property | Detail |
|----------|--------|
| **Source** | https://www.gencodegenes.org/ |
| **What it provides** | Gene coordinates, TSS positions, exon/intron boundaries |
| **Format** | GTF / GFF3 |
| **Size** | ~50 MB |
| **Why it matters** | Needed to exclude coding regions; identify promoter-proximal vs. distal regulatory regions |

### 1.2 Secondary Sources (Phase 3+)

| Dataset | What it provides | When to use |
|---------|-----------------|-------------|
| **GTEx** (https://gtexportal.org/) | Tissue-specific gene expression across 54 tissues | When linking regulatory predictions to expression |
| **Roadmap Epigenomics** | Epigenetic maps across 111 reference epigenomes | When expanding beyond ENCODE cell types |
| **FANTOM5** | CAGE-defined enhancers and promoters | Alternative enhancer labels for cross-validation |
| **Single-cell ATAC-seq** (e.g., from Satpathy et al.) | Cell-type-resolved chromatin accessibility | When exploring cell-type-specific regulation at finer resolution |

### 1.3 Kaggle-Available Resources

| Resource | Kaggle availability |
|----------|-------------------|
| AlphaGenome model | Available on Kaggle Models (for Stage 3 embedding extraction) |
| hg38 reference | Can be uploaded as a Kaggle dataset (per-chromosome FASTA) |
| ENCODE cCREs | Must be curated and uploaded as a Kaggle dataset |
| JASPAR PFMs | Small enough to include directly in notebook or dataset |

---

## 2. Which Dataset to Start With

### Recommended Starting Dataset: ENCODE cCREs (SCREEN V3)

**Rationale:**

1. **Pre-classified**: cCREs come with element-type annotations (PLS, pELS, dELS, CTCF-only), eliminating the need for raw signal processing.
2. **Well-curated**: Generated by the ENCODE consortium using standardized pipelines across multiple assays.
3. **Manageable size**: ~926K elements, but can be filtered to ~50K–200K per cell type for initial experiments.
4. **Directly downloadable**: Available as BED files from SCREEN.
5. **Widely cited**: Used in numerous computational biology studies; results are comparable to literature.

### Starting Subset Recommendation

| Filter | Value | Resulting size (approx.) |
|--------|-------|------------------------|
| Assembly | hg38 | Full set |
| Cell type | K562 (primary) | ~150K–200K active cCREs |
| Element type | dELS + PLS (enhancer + promoter) | ~100K–150K |
| Quality | High-confidence only (default SCREEN filters) | ~80K–120K |
| Region length | 200–2000 bp (filter extremes) | ~70K–100K |

This gives ~70K–100K positive regions — sufficient for training and evaluation.

---

## 3. Label Strategy

### 3.1 Binary Classification Labels

| Label | Definition | Source |
|-------|-----------|--------|
| **Positive (regulatory)** | Region annotated as cCRE by ENCODE SCREEN with active signal in target cell type | ENCODE cCRE BED file, filtered by cell-type activity |
| **Negative (non-regulatory)** | Random non-coding region NOT annotated as any cCRE and NOT overlapping any ENCODE annotation | Generated by negative sampling pipeline |

### 3.2 Multi-Class Labels (Phase 2)

| Label | Definition | Source |
|-------|-----------|--------|
| **PLS** (Promoter-Like Signature) | cCRE with high H3K4me3 and high DNase signal near TSS | ENCODE SCREEN classification |
| **pELS** (Proximal Enhancer-Like) | cCRE with high H3K27ac, within 2 kb of TSS | ENCODE SCREEN classification |
| **dELS** (Distal Enhancer-Like) | cCRE with high H3K27ac, > 2 kb from TSS | ENCODE SCREEN classification |
| **CTCF-only** | cCRE with CTCF binding but no other marks | ENCODE SCREEN classification |
| **Non-regulatory** | Sampled negative region | Generated |

### 3.3 Cell-Type-Specific Labels

A region may be:
- Active in K562 but inactive in GM12878 → labeled differently per cell type.
- Active in both → labeled positive in both.
- The system trains separate models per cell type and compares feature importances.

---

## 4. Positive / Negative Set Design

### 4.1 Positive Set Construction

```
ENCODE cCREs (BED)
    │
    ├── Filter by cell-type activity (e.g., K562-active)
    ├── Filter by element type (e.g., dELS + PLS)
    ├── Filter by region length (200–2000 bp)
    ├── Remove regions overlapping GENCODE exons (exclude coding)
    ├── Resize to fixed length (e.g., center ± 500 bp = 1 kb)
    │
    └── → Positive set: ~50K–100K regions
```

### 4.2 Negative Set Construction

```
hg38 genome
    │
    ├── Generate random non-overlapping 1 kb regions
    ├── Exclude: all ENCODE cCREs (any cell type)
    ├── Exclude: all GENCODE exons, UTRs, known promoters (±2 kb of TSS)
    ├── Exclude: ENCODE blacklist regions (problematic mappability)
    ├── Match: GC content distribution of positive set (±5%)
    ├── Match: chromosome distribution of positive set
    │
    └── → Negative set: 1× positive set size (default)
```

### 4.3 Positive-to-Negative Ratio

| Ratio | When to use |
|-------|-------------|
| 1:1 | Default for initial experiments; balanced training |
| 1:3 | Sensitivity analysis; tests classifier behavior under mild imbalance |
| 1:5 | Stress test; closer to genome-wide base rate |

---

## 5. Preprocessing Steps

### 5.1 Sequence Extraction Pipeline

| Step | Action | Tool | Output |
|------|--------|------|--------|
| 1 | Download cCRE BED file | `wget` / ENCODE API | BED file |
| 2 | Filter BED by criteria (cell type, element type, length) | `pandas` / `pybedtools` | Filtered BED |
| 3 | Generate negative regions | Custom script (`negative_sampling.py`) | Negative BED |
| 4 | Merge positive + negative BED | `pandas` | Combined BED with labels |
| 5 | Extract DNA sequences from hg38 FASTA | `pysam` / `pyfaidx` | FASTA or NumPy arrays |
| 6 | One-hot encode sequences | Custom (`onehot.py`) | NumPy array (N × L × 4) |
| 7 | Compute k-mer features | Custom (`kmer.py`) | NumPy array (N × 4^k) |
| 8 | Compute GC content, dinucleotide freq | Custom (`gc_content.py`) | NumPy array |
| 9 | Apply chromosome-holdout split | Custom | Train/val/test indices |
| 10 | Cache to disk | `np.savez_compressed` / `parquet` | Versioned cache files |

### 5.2 Sequence Representation

| Representation | Shape | Size per sample (1 kb) | Used by |
|----------------|-------|----------------------|---------|
| One-hot | (1000, 4) | 4 KB | CNN |
| k-mer frequency (k=4) | (256,) | 2 KB | Classical ML |
| k-mer frequency (k=5) | (1024,) | 8 KB | Classical ML |
| k-mer frequency (k=6) | (4096,) | 32 KB | Classical ML (sparse) |
| GC + dinucleotide | (17,) | <1 KB | All models (auxiliary) |

### 5.3 Quality Control Checks

| Check | Purpose | Action if failed |
|-------|---------|-----------------|
| No N-bases > 10% of sequence | Ensures sequence quality | Exclude region |
| No overlap between positive and negative sets | Prevents label contamination | Re-generate negatives |
| No overlap between train/val/test regions | Prevents leakage | Verify chromosome assignment |
| GC distribution match (KS test p > 0.05) | Ensures negatives are composition-matched | Re-sample negatives |
| Balanced class distribution (within 1:1 to 1:5) | Prevents degenerate training | Resample or apply class weights |
| Sequence length consistency | All regions are exactly fixed length | Resize or exclude |

---

## 6. Train / Validation / Test Split Policy

### 6.1 Chromosome-Holdout Split (Primary)

| Split | Chromosomes | Approx. genome fraction |
|-------|-------------|------------------------|
| **Train** | chr1–chr14, chrX | ~70% |
| **Validation** | chr15–chr18 | ~15% |
| **Test** | chr19–chr22 | ~15% |

### 6.2 Rationale

- **Why chromosome-holdout?** Regulatory elements on the same chromosome may share regulatory landscapes, chromatin domains, or be in linkage disequilibrium. Splitting by chromosome eliminates within-chromosome leakage.
- **Why chr19–22 for test?** chr19 is the most gene-dense human chromosome, providing a challenging and distinct test distribution. chr22 was the first chromosome sequenced and is well-characterized.
- **Why chrX in training?** chrX has unique regulatory biology (X-inactivation); including it in training allows the model to learn from this diversity. If X-specific analysis is needed, hold out chrX separately.

### 6.3 Cross-Validation (Within Training Set)

- 5-fold stratified cross-validation on training chromosomes for hyperparameter tuning.
- Each fold is **stratified by class label**, not by chromosome (since chromosome-holdout is already enforced at the train/test boundary).

---

## 7. Leakage Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Region overlap** | Same genomic coordinates in train and test | Chromosome-holdout eliminates this |
| **Nearby regulatory elements** | Train and test regions from adjacent genomic positions | Chromosome-holdout; minimum 100 kb distance between nearest train/test regions (if within-chromosome split is ever used) |
| **Negative set contamination** | Negative regions that are actually regulatory but unannotated | Exclude all ENCODE annotations from negatives, not just target cell type |
| **Feature leakage via GC content** | If negatives are not GC-matched, GC alone becomes predictive | GC-matched negative sampling |
| **Batch effects** | Data from different ENCODE phases processed differently | Use only ENCODE phase 3 data with consistent processing |
| **Label leakage through sequence length** | If positives and negatives have different native lengths | Fixed-length regions (resize all to 1 kb) |

---

## 8. Ethics and Licensing Considerations

### 8.1 Data Licensing

| Dataset | License / Policy | Obligation |
|---------|-----------------|------------|
| ENCODE | ENCODE Data Use Policy (https://www.encodeproject.org/help/data-use-policy/) | Free for research use; cite ENCODE consortium publications |
| hg38 reference genome | Public domain (NCBI/UCSC) | No restrictions |
| JASPAR | CC BY 4.0 | Attribute JASPAR in publications |
| GENCODE | Freely available | Cite GENCODE |
| GTEx (future) | dbGaP controlled access for individual-level data; summary data publicly available | Use only summary/aggregate data unless dbGaP access is obtained |

### 8.2 Ethical Considerations

| Consideration | Status |
|---------------|--------|
| Human subjects | Not applicable — no individual-level data; reference genome and population-level annotations only |
| IRB approval | Not required for this data |
| Personally identifiable information | None present in reference genome or ENCODE peak files |
| Dual-use risk | Low — regulatory prediction is basic research, not weaponizable |
| Bias | The reference genome (hg38) is derived primarily from a few individuals; regulatory annotations are biased toward well-studied cell types. This limits generalizability to under-represented populations and tissues. Acknowledge in publications. |
| Clinical misuse | Explicitly state that results are computational predictions, not clinical diagnostics. No claims about disease risk, treatment, or individual genome interpretation. |

### 8.3 Responsible Reporting

- Frame all results as "candidate regulatory patterns" or "computational predictions."
- Do not claim that models have "discovered" regulatory switches — they have identified statistical associations.
- Report limitations prominently.
- Do not overstate generalizability beyond tested cell types and conditions.

---

## 9. Data Versioning

### 9.1 Versioning Policy

| Item | Version strategy |
|------|-----------------|
| ENCODE cCRE download | Tag by download date and ENCODE release version (e.g., `cCRE_v3_2026-06-07`) |
| hg38 reference | Use UCSC hg38 with specific patch level (e.g., `GRCh38.p14`) |
| JASPAR | Tag by JASPAR release (e.g., `JASPAR2024`) |
| Processed datasets | Semantic version (e.g., `v0.1.0`) with changelog |
| Feature caches | Keyed by dataset version + feature config hash |

### 9.2 Reproducibility Contract

For any experiment result, the following must be recoverable:
1. Exact dataset version used.
2. Exact feature configuration.
3. Exact data split (chromosome assignment).
4. Random seed.
5. Model hyperparameters.
6. Software versions (requirements.txt / conda environment).

---

## 10. Data Budget (Kaggle Constraints)

| Data item | Size on disk | Fits in Kaggle? |
|-----------|-------------|-----------------|
| Curated cCRE BED (K562 + GM12878) | ~50 MB | ✓ |
| hg38 per-chromosome FASTA (chr1–22, X) | ~3.1 GB | ✓ (as Kaggle dataset) |
| Processed sequences (100K regions × 1 kb, one-hot) | ~400 MB | ✓ |
| k-mer features (100K × 4096, sparse) | ~200 MB | ✓ |
| JASPAR PFMs | ~10 MB | ✓ |
| GENCODE GTF | ~50 MB | ✓ |
| **Total estimated** | **~4–5 GB** | **✓ (within 20 GB Kaggle limit)** |

---

*End of Document 05 — Dataset Strategy*  
*Next: [06 — Modeling Roadmap](./06-modeling-roadmap.md)*
