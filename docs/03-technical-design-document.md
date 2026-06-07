# 03 — Technical Design Document / Architecture Doc

## Interpretable Machine Learning for Discovering Hidden Regulatory Switches in Non-Coding DNA

| Field | Value |
|-------|-------|
| Document ID | TDD-001 |
| Version | 1.0 |
| Status | Active |
| Created | 2026-06-07 |
| Related docs | [PRD](./02-product-requirements-document.md), [Dataset Strategy](./05-dataset-strategy.md), [MLOps](./08-experiment-tracking-mlops.md) |

---

## 1. System Architecture

### 1.1 High-Level Architecture (Text Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REGULATORY SWITCH PIPELINE                         │
├─────────────┬───────────────┬──────────────┬──────────────┬─────────────────┤
│  DATA LAYER │  FEATURE LAYER│  MODEL LAYER │ INTERPRET.   │ REPORTING LAYER │
│             │               │              │ LAYER        │                 │
│ ┌─────────┐ │ ┌───────────┐ │ ┌──────────┐ │ ┌──────────┐ │ ┌─────────────┐ │
│ │ ENCODE  │ │ │ k-mer     │ │ │ LogReg   │ │ │ Feature  │ │ │ Metrics     │ │
│ │ Download│→│ │ Features  │→│ │ RF / XGB │→│ │ Import.  │→│ │ Report      │ │
│ │ & Parse │ │ │ Motif Scan│ │ │ CNN      │ │ │ SHAP     │ │ │ Motif Report│ │
│ │         │ │ │ One-Hot   │ │ │ (Later:  │ │ │ Saliency │ │ │ Figures     │ │
│ │ Ref     │ │ │ GC / Dinuc│ │ │ Pretrain)│ │ │ Motif    │ │ │ Comparison  │ │
│ │ Genome  │ │ │           │ │ │          │ │ │ Enrich.  │ │ │ Tables      │ │
│ └─────────┘ │ └───────────┘ │ └──────────┘ │ └──────────┘ │ └─────────────┘ │
├─────────────┴───────────────┴──────────────┴──────────────┴─────────────────┤
│                         EXPERIMENT TRACKING LAYER                           │
│                  (Logging, Versioning, Artifact Management)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Layer Responsibilities

| Layer | Responsibility | Key modules |
|-------|---------------|-------------|
| **Data Layer** | Download, parse, cache ENCODE annotations; extract sequences from reference genome; generate negative sets | `data/download.py`, `data/parse_encode.py`, `data/sequence_extractor.py`, `data/negative_sampling.py` |
| **Feature Layer** | Transform raw sequences into model-ready features (k-mers, one-hot, motif scores, GC content) | `features/kmer.py`, `features/onehot.py`, `features/motif_scanner.py`, `features/gc_content.py` |
| **Model Layer** | Train, evaluate, and store classifiers across multiple model families | `models/logistic.py`, `models/tree_ensemble.py`, `models/cnn.py`, `models/registry.py` |
| **Interpretability Layer** | Extract and visualize feature importance, SHAP values, saliency maps, motif enrichment | `interpret/importance.py`, `interpret/shap_analysis.py`, `interpret/saliency.py`, `interpret/motif_enrichment.py` |
| **Reporting Layer** | Generate evaluation reports, comparison tables, publication-ready figures | `reports/metrics_report.py`, `reports/motif_report.py`, `reports/comparison.py` |
| **Experiment Tracking Layer** | Log hyperparameters, metrics, data versions; manage model artifacts | `tracking/logger.py`, `tracking/artifact_store.py` |

---

## 2. Data Pipeline

### 2.1 Pipeline Flow

```
ENCODE Portal ──→ BED/narrowPeak files
                        │
                        ▼
              ┌───────────────────┐
              │  Parse annotations │  ← Filter by cell type, assay type,
              │  (parse_encode.py) │     quality tier, region length
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │  Extract sequences │  ← hg38 FASTA (per-chromosome)
              │  (sequence_ext.)   │     pysam / pyfaidx
              └────────┬──────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Positive │ │ Negative │ │ Ambiguous│
    │ regions  │ │ regions  │ │ (exclude)│
    └────┬─────┘ └────┬─────┘ └──────────┘
         │            │
         ▼            ▼
    ┌────────────────────────┐
    │  Split: train/val/test │  ← Chromosome-holdout scheme
    │  (by chromosome)       │     (see §2.3)
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  Feature extraction    │  ← k-mer, one-hot, GC, motif
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  Cache to disk         │  ← .npz / .parquet / .h5
    │  (versioned snapshot)  │
    └────────────────────────┘
```

### 2.2 Negative Region Sampling Strategy

Negative regions require careful design to avoid trivially separable datasets.

**Strategy options (implemented as configurable parameter):**

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Random genomic** | Sample random non-overlapping regions from the same chromosomes, matched by length | Simple, fast | May sample other regulatory regions (false negatives) |
| **GC-matched** | Match GC content distribution of positive regions | Controls for compositional bias | More complex sampling |
| **Flanking** | Sample from flanking regions around positives (±2–10 kb offset) | Controls for chromosomal context | May capture nearby regulatory activity |
| **Exclude-known** | Random sampling but exclude all known ENCODE-annotated regions | Reduces false negatives | Requires comprehensive annotation set |

**Default**: GC-matched + exclude-known, with a 1:1 positive-to-negative ratio (configurable up to 1:5).

### 2.3 Train / Validation / Test Split

**Chromosome-holdout scheme** to prevent spatial leakage:

| Split | Chromosomes | Rationale |
|-------|-------------|-----------|
| Train | chr1–chr14, chrX | ~70% of genome |
| Validation | chr15–chr18 | ~15% of genome |
| Test | chr19–chr22 | ~15% of genome; chr19 is gene-dense (harder test) |

This ensures no region in the test set shares a chromosome with any training region, eliminating leakage from nearby regulatory elements.

---

## 3. Training Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Load cached  │────→│ Select model │────→│ Train model  │
│ features     │     │ from registry│     │ (fit)        │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌──────────────┐     ┌───────▼──────┐
                     │ Log to       │←────│ Evaluate on  │
                     │ experiment   │     │ val set      │
                     │ tracker      │     └──────────────┘
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │ Save model   │
                     │ artifact     │
                     └──────────────┘
```

### 3.1 Model Registry Pattern

```python
# models/registry.py — conceptual interface
class ModelRegistry:
    """Central registry for all model families."""

    @staticmethod
    def get_model(name: str, params: dict) -> BaseModel:
        """Return an instantiated model by name."""
        registry = {
            "logistic_regression": LogisticRegulatoryModel,
            "random_forest": RandomForestRegulatoryModel,
            "xgboost": XGBoostRegulatoryModel,
            "cnn": CNNRegulatoryModel,
        }
        return registry[name](**params)

class BaseModel(ABC):
    """Abstract base for all regulatory classifiers."""
    def fit(self, X_train, y_train): ...
    def predict(self, X): ...
    def predict_proba(self, X): ...
    def get_feature_importance(self) -> pd.DataFrame: ...
    def save(self, path: str): ...
    def load(cls, path: str): ...
```

---

## 4. Validation Workflow

| Step | Action | Output |
|------|--------|--------|
| 1 | Load test-set features (chromosome-holdout) | Feature matrix |
| 2 | Load trained model artifact | Model object |
| 3 | Run `predict_proba` on test set | Probability vector |
| 4 | Compute AUROC, AUPRC, F1, accuracy, precision, recall | Metrics dict |
| 5 | Generate confusion matrix | Plot |
| 6 | Run cross-validation on training chromosomes (sanity check) | CV metrics |
| 7 | Compare metrics across model families | Comparison table |
| 8 | Log all results to experiment tracker | JSON / CSV log |

### 4.1 Stratification

- Within-split class balance is maintained via stratified sampling when doing cross-validation on training data.
- Chromosome-holdout split does not guarantee balanced classes per split; class weights or oversampling used if imbalance exceeds 1:3.

---

## 5. Inference Workflow

For batch inference on new/unseen genomic regions:

```
Input: BED file of query regions
         │
         ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ Extract seqs │────→│ Compute      │────→│ Load trained │
  │ from hg38    │     │ features     │     │ model        │
  └──────────────┘     └──────────────┘     └──────┬───────┘
                                                    │
                       ┌──────────────┐     ┌───────▼──────┐
                       │ Output:      │←────│ predict_proba│
                       │ predictions  │     └──────────────┘
                       │ + confidence │
                       └──────────────┘
```

**Output format**: CSV/Parquet with columns: `chrom`, `start`, `end`, `prediction`, `probability`, `model_name`, `model_version`.

---

## 6. Interpretability Workflow

```
Trained Model + Test Data
         │
         ├──→ Feature Importance (built-in)
         │         │
         │         ▼
         │    Top-k features ranked
         │
         ├──→ SHAP Analysis
         │         │
         │         ▼
         │    SHAP summary plot
         │    SHAP force plot (per-sample)
         │    SHAP dependence plots
         │
         ├──→ Saliency Maps (CNN only)
         │         │
         │         ▼
         │    Per-nucleotide attribution
         │    Aggregated motif-like patterns
         │
         └──→ Motif Enrichment
                   │
                   ▼
              Top features → map to known TF motifs
              (JASPAR/HOCOMOCO PWM scan)
              Cross-reference with literature
```

### 6.1 Interpretability Output Artifacts

| Artifact | Format | Model families |
|----------|--------|----------------|
| Feature importance table | CSV | All |
| Feature importance bar chart | PNG/SVG | All |
| SHAP summary plot | PNG/SVG | Tree-based, Linear |
| SHAP waterfall (per-sample) | PNG/SVG | Tree-based, Linear |
| Saliency heatmap | PNG/SVG | CNN |
| Integrated gradients map | PNG/SVG | CNN |
| Motif enrichment table | CSV | All (post-processing) |
| Motif logo plots | PNG/SVG | All (post-processing) |

---

## 7. Storage and Artifact Strategy

### 7.1 Directory Structure

```
project-root/
├── data/
│   ├── raw/                    # Downloaded ENCODE files (BED, narrowPeak)
│   ├── reference/              # Reference genome FASTA (or symlinks)
│   ├── processed/              # Extracted sequences, labeled datasets
│   ├── features/               # Cached feature matrices (.npz, .parquet)
│   └── splits/                 # Train/val/test split definitions (BED files)
├── src/
│   ├── data/                   # Data pipeline modules
│   ├── features/               # Feature engineering modules
│   ├── models/                 # Model definitions and registry
│   ├── interpret/              # Interpretability modules
│   ├── reports/                # Report generation
│   ├── tracking/               # Experiment logging
│   └── utils/                  # Shared utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_training.ipynb
│   ├── 04_interpretability.ipynb
│   ├── 05_model_comparison.ipynb
│   └── kaggle/                 # Kaggle-specific self-contained notebooks
│       ├── kaggle_baseline.ipynb
│       └── kaggle_full_pipeline.ipynb
├── experiments/
│   ├── logs/                   # Experiment logs (JSON/CSV)
│   ├── models/                 # Saved model artifacts
│   ├── figures/                # Generated plots and figures
│   └── reports/                # Generated reports
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_interpret.py
├── docs/                       # This documentation pack
├── configs/
│   ├── default_config.yaml     # Default hyperparameters and paths
│   └── experiment_configs/     # Per-experiment configuration files
├── requirements.txt
├── setup.py (or pyproject.toml)
└── README.md
```

### 7.2 Artifact Naming Convention

```
experiments/models/{experiment_name}/{model_family}_{timestamp}_{dataset_version}.pkl
experiments/logs/{experiment_name}/metrics_{timestamp}.json
experiments/figures/{experiment_name}/{plot_type}_{model_family}_{timestamp}.png
```

### 7.3 Kaggle Dataset Integration

On Kaggle, datasets are mounted at `/kaggle/input/`. The project should:

1. Package curated ENCODE subsets as Kaggle datasets (private or public).
2. Reference genome chromosomes as a separate Kaggle dataset.
3. Use `/kaggle/working/` for outputs.
4. Kaggle notebooks should import `src/` modules via a utility dataset containing the codebase, or use `%%writefile` magic to define modules inline.

---

## 8. Notebook vs. Modular Code Structure

### 8.1 Dual-Mode Strategy

| Context | Approach |
|---------|----------|
| **Kaggle execution** | Self-contained notebooks that inline critical functions or import from a bundled `src/` package uploaded as a Kaggle dataset |
| **Local development** | Standard Python package (`src/`) with notebooks for exploration only; scripts for production runs |
| **Testing** | `pytest`-based tests in `tests/`; run locally, not on Kaggle |

### 8.2 Notebook Hygiene Rules

1. Every notebook starts with a header cell: experiment name, date, purpose, data version.
2. Notebooks must not define critical logic inline — use `src/` modules.
3. Exception: Kaggle-specific notebooks may inline code when necessary, but must clearly mark inlined sections.
4. Notebooks must be runnable top-to-bottom without manual intervention.
5. Output cells should be cleared before committing to version control.

---

## 9. Recommended Stack

### 9.1 Core Dependencies

| Category | Package | Version constraint | Purpose |
|----------|---------|-------------------|---------|
| **Data** | `pandas` | ≥ 1.5 | Tabular data handling |
| **Data** | `numpy` | ≥ 1.24 | Numerical computation |
| **Data** | `pysam` | ≥ 0.21 | FASTA/BAM file access |
| **Data** | `pybedtools` | ≥ 0.9 | BED file operations |
| **Data** | `biopython` | ≥ 1.81 | Sequence utilities |
| **ML** | `scikit-learn` | ≥ 1.3 | Classical ML models, metrics, preprocessing |
| **ML** | `xgboost` | ≥ 1.7 | Gradient boosting |
| **ML** | `lightgbm` | ≥ 4.0 | Alternative gradient boosting |
| **DL** | `torch` | ≥ 2.0 | CNN and attention models |
| **Interpret** | `shap` | ≥ 0.42 | SHAP values |
| **Interpret** | `captum` | ≥ 0.6 | PyTorch interpretability (saliency, IG) |
| **Viz** | `matplotlib` | ≥ 3.7 | Plotting |
| **Viz** | `seaborn` | ≥ 0.12 | Statistical visualization |
| **Tracking** | `pyyaml` | ≥ 6.0 | Configuration files |
| **Motif** | `logomaker` | ≥ 0.8 | Sequence logo visualization |

### 9.2 Optional / Later-Stage Dependencies

| Package | Purpose | When to add |
|---------|---------|-------------|
| `transformers` (HuggingFace) | Pretrained genomic model loading | Stage 3 |
| `wandb` | Experiment tracking UI | When local tracking becomes insufficient |
| `mlflow` | Alternative experiment tracking | When local tracking becomes insufficient |
| `gradio` | Simple demo UI | When a public demo is desired |
| `onnx` / `onnxruntime` | Model export for deployment | If deployment is ever needed |

---

## 10. Kaggle-Oriented Execution Pattern

### 10.1 Recommended Kaggle Setup

```
Kaggle Notebook
├── Inputs (mounted datasets):
│   ├── encode-regulatory-annotations-v1/    # Curated BED files
│   ├── hg38-reference-chromosomes/          # Per-chromosome FASTA
│   ├── regulatory-switch-src/               # Project source code
│   └── jaspar-motif-database/               # JASPAR PWMs
├── Working directory (/kaggle/working/):
│   ├── features/                            # Computed feature matrices
│   ├── models/                              # Trained model artifacts
│   ├── logs/                                # Experiment logs
│   └── figures/                             # Generated plots
└── Output (saved after session):
    └── [auto-saved from /kaggle/working/]
```

### 10.2 Kaggle Session Budget

| Task | Estimated time | GPU needed? |
|------|---------------|-------------|
| Data parsing + sequence extraction | 20–40 min | No |
| k-mer feature computation (100k regions, k≤6) | 10–30 min | No (CPU) |
| One-hot encoding (100k regions × 1 kb) | 5–10 min | No |
| Logistic regression training | 1–5 min | No |
| Random forest / XGBoost training | 5–20 min | No |
| Shallow CNN training (50 epochs) | 30–90 min | Yes |
| SHAP computation (tree-based) | 10–30 min | No |
| Saliency maps (CNN, 1000 samples) | 5–15 min | Yes |
| Full pipeline | 2–4 hours | Partially |

### 10.3 Memory Management on Kaggle

- Load reference genome chromosomes individually, not the full FASTA.
- Use `float32` for features, not `float64`.
- Compute features in batches and write to disk, then load batch-by-batch.
- Use sparse matrices for high-dimensional k-mer features (k ≥ 5).
- Delete intermediate variables and call `gc.collect()` after major operations.
- For CNN: use `DataLoader` with reasonable `batch_size` (64–256) to manage GPU memory.

---

## 11. Configuration Management

### 11.1 Default Configuration File

```yaml
# configs/default_config.yaml
project:
  name: "regulatory-switch-discovery"
  version: "0.1.0"

data:
  genome_assembly: "hg38"
  encode_experiment_types:
    - "DNase-seq"      # Open chromatin
    - "H3K27ac"        # Active enhancer mark
    - "H3K4me3"        # Active promoter mark
  cell_types:
    - "K562"           # Primary cell type (well-annotated)
    - "GM12878"        # Secondary cell type
  region_length: 1000  # bp; fixed-length regions
  negative_strategy: "gc_matched_exclude_known"
  positive_negative_ratio: 1.0

features:
  kmer_k_values: [3, 4, 5, 6]
  include_gc_content: true
  include_dinucleotide_freq: true
  include_motif_scores: false  # Enable in Stage 2

splits:
  train_chromosomes: ["chr1","chr2","chr3","chr4","chr5","chr6","chr7",
                       "chr8","chr9","chr10","chr11","chr12","chr13","chr14","chrX"]
  val_chromosomes: ["chr15","chr16","chr17","chr18"]
  test_chromosomes: ["chr19","chr20","chr21","chr22"]

models:
  default_family: "random_forest"
  random_seed: 42

training:
  cv_folds: 5
  early_stopping_patience: 10  # For neural models

interpretability:
  shap_max_samples: 1000
  saliency_method: "integrated_gradients"
  top_k_features: 20

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  features: "data/features"
  models: "experiments/models"
  logs: "experiments/logs"
  figures: "experiments/figures"
```

---

## 12. Security and Access Considerations

- All data is public. No authentication or access control is required for data.
- Kaggle API keys should be stored in Kaggle Secrets, not hardcoded.
- No PHI (Protected Health Information) is involved.
- Model artifacts do not contain sensitive information (they contain learned weights, not raw sequences).

---

## 13. Future Architecture Extensions (Later Phases Only)

> **Note**: The following are documented for planning purposes only. They are not part of the initial build.

| Extension | Architecture impact | Phase |
|-----------|-------------------|-------|
| Pretrained model embeddings | New feature extraction pathway using HuggingFace transformers | Stage 3 |
| Multi-omics integration | Additional data loaders (expression, methylation) | Stage 4 |
| Web-based demo (Gradio) | New serving layer; model export to ONNX | Post-Stage 3 |
| Distributed training | Migration from Kaggle to cloud compute | Only if funded |
| Database-backed experiment tracking | Replace file-based logging with MLflow/W&B server | When team size > 5 |

---

*End of Document 03 — Technical Design Document*  
*Next: [04 — Research Design Document](./04-research-design-document.md)*
