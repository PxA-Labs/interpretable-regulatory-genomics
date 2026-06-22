# Interpretable Regulatory Genomics

**Interpretable Machine Learning on ENCODE Regulatory Data to Discover Hidden Switches in Non-Coding DNA**

This project applies interpretable machine learning to ENCODE-annotated candidate cis-regulatory elements (cCREs) to identify sequence-level patterns — "regulatory switches" — that distinguish active regulatory regions from non-regulatory non-coding DNA. The work progresses through four phases: classical baselines, convolutional neural networks, pre-trained foundation model embeddings, and multi-omics extensibility.

Optimized to run on accessible Kaggle-level compute infrastructure (T4 GPU).

---

## Current Status

> [!NOTE]
> **Phases 1–3 are complete. Phase 2 solidification (multi-class classification and cross-cell-type generalization) has been merged via PR [#17](../../pull/17).**

| Phase | Description | Best Model | Test AUROC | Status |
|-------|-------------|------------|------------|--------|
| **Phase 1** | Classical Interpretable Baselines | XGBoost (k=4 k-mers) | 0.8830 | Complete |
| **Phase 2** | Deep Learning (CNNs) | AttentionCNN (one-hot) | 0.8604 | Complete |
| **Phase 2+** | Multi-class & Cross-Cell-Type | AttentionCNN (3-class) | 81.25% acc | Complete |
| **Phase 3** | Pre-trained Foundation Models | Nucleotide Transformer (500M) | 0.9176 | Complete |

Remaining work is tracked in [Issue #13](../../issues/13) with sub-issues [#19](../../issues/19) (cross-cell-type matrix) and [#20](../../issues/20) (documentation deliverables).

---

## Discussions

| # | Topic | Phase |
|---|-------|-------|
| [#1](../../discussions/1) | Classical Interpretable Baselines & SHAP Explainability | Phase 1 |
| [#16](../../discussions/16) | Phase 1 Solidification & Baseline Verification | Phase 1 |
| [#4](../../discussions/4) | Deep Learning Baselines, Convolutional Filters & Attribution Maps | Phase 2 |
| [#18](../../discussions/18) | Multi-class Deep Learning & Cross-Cell-Type Generalization (K562 → GM12878) | Phase 2 |
| [#5](../../discussions/5) | Pre-trained Foundation Models & Embeddings | Phase 3 |
| [#11](../../discussions/11) | Interpretable CNN Discovers Major Regulatory Switches in Non-Coding DNA | Cross-Phase |
| [#12](../../discussions/12) | Extensibility into Multi-Omics or Cell-Type-Specific Prediction | Phase 4 |

---

## Quickstart

```bash
git clone https://github.com/PxA-Labs/interpretable-regulatory-genomics.git
cd interpretable-regulatory-genomics
pip install -r requirements.txt
pytest tests/ -v
```

Notebooks are located in `notebook/` and are designed to run on [Kaggle](https://www.kaggle.com/) with T4 GPU acceleration. They can also be run locally with the appropriate data files (see `src/data/download.py`).

---

## Documentation

All specifications are compiled into professional PDF documents inside [docs/](docs/):

- [01-Project Charter](docs/01-project-charter.pdf) — Problem statement, vision, goals, non-goals, and constraints.
- [02-Product Requirements Document](docs/02-product-requirements-document.pdf) — User personas, use cases, and acceptance criteria.
- [03-Technical Design Document](docs/03-technical-design-document.pdf) — System architecture, data flow, and pipeline layers.
- [04-Research Design Document](docs/04-research-design-document.pdf) — Scientific hypotheses, evaluation matrix, and experiments.
- [05-Dataset Strategy](docs/05-dataset-strategy.pdf) — Data acquisition, labelling, leakage prevention, and QC checks.
- [06-Modeling Roadmap](docs/06-modeling-roadmap.pdf) — Phase-wise modeling plan and baseline configurations.
- [07-Compute Feasibility Memo](docs/07-compute-feasibility-memo.pdf) — Compute budget and resource constraints.
- [08-Experiment Tracking & MLOps](docs/08-experiment-tracking-mlops.pdf) — Notebook hygiene, config structures, and reproducibility.
- [09-Risk Register](docs/09-risk-register.pdf) — Scientific, modeling, and timeline risks with mitigations.
- [10-Roadmap & Milestones](docs/10-roadmap-milestones.pdf) — 12-month delivery roadmap and check-off criteria.
- [11-Glossary & Project Memory](docs/11-glossary-project-memory.pdf) — Definitions, locked decisions, and scope limitations.
- [12-Contributor Onboarding Brief](docs/12-contributor-onboarding.pdf) — Setup checklist and coding guidelines.

---

## License

MIT License — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
