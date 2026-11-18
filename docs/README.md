# Project-Definition Pack — Index

## Interpretable Machine Learning for Discovering Hidden Regulatory Switches in Non-Coding DNA

> **Project Identity**: A Kaggle-friendly, interpretable ML research pipeline for identifying candidate regulatory switch-like patterns in non-coding DNA, built on public datasets such as ENCODE.

This directory contains the complete project-definition pack. Each document is self-contained but cross-referential. Together they form the operating manual for the project.

---

## Document Index

| # | Document | Purpose | File |
|---|----------|---------|------|
| 01 | [Project Charter](./01-project-charter.md) | Problem, vision, mission, goals, constraints | `01-project-charter.md` |
| 02 | [Product Requirements Document](./02-product-requirements-document.md) | Features, use cases, requirements, acceptance criteria | `02-product-requirements-document.md` |
| 03 | [Technical Design Document](./03-technical-design-document.md) | Architecture, pipelines, stack, execution patterns | `03-technical-design-document.md` |
| 04 | [Research Design Document](./04-research-design-document.md) | Hypotheses, experiments, baselines, evaluation | `04-research-design-document.md` |
| 05 | [Dataset Strategy](./05-dataset-strategy.md) | Data sources, labels, splits, preprocessing, ethics | `05-dataset-strategy.md` |
| 06 | [Modeling Roadmap](./06-modeling-roadmap.md) | Phased model plan, decision gates | `06-modeling-roadmap.md` |
| 07 | [Compute & Feasibility Memo](./07-compute-feasibility-memo.md) | Why feasible, what Kaggle supports, bottlenecks | `07-compute-feasibility-memo.md` |
| 08 | [Experiment Tracking & MLOps Lite](./08-experiment-tracking-mlops.md) | Conventions, reproducibility, versioning | `08-experiment-tracking-mlops.md` |
| 09 | [Risk Register](./09-risk-register.md) | Risks, mitigations, fallback plans | `09-risk-register.md` |
| 10 | [Roadmap & Milestones](./10-roadmap-milestones.md) | 1/3/6/12-month plans, success criteria | `10-roadmap-milestones.md` |
| 11 | [Glossary & Project Memory](./11-glossary-project-memory.md) | Definitions, locked decisions, forbidden scope | `11-glossary-project-memory.md` |
| 12 | [Contributor Onboarding Brief](./12-contributor-onboarding.md) | How to join, what's decided, expectations | `12-contributor-onboarding.md` |

---

## How to Use This Pack

1. **New contributors** — start with Document 12 (Onboarding Brief), then read Document 01 (Charter) and Document 11 (Glossary).
2. **Technical contributors** — additionally read Documents 03, 05, 06, and 08.
3. **Research contributors** — additionally read Documents 04, 05, and 06.
4. **Project leads / advisors** — read Documents 01, 02, 09, and 10 for governance and milestones.

## Scope Lock (Reproduced for Visibility)

- **Domain**: Non-coding regulatory genomics.
- **Core Objective**: Discover and interpret candidate regulatory switch-like patterns in non-coding DNA.
- **Early Execution Environment**: Kaggle-friendly.
- **Early Model Philosophy**: Interpretable and lightweight first.
- **Large Genomic Transformer Training**: Out of scope for initial build because of compute limits.
- **Preferred Data Sources**: Public datasets such as ENCODE.
- **Long-Term Evolution**: Stronger models, richer interpretation, broader generalization — but only after a working baseline exists.

---

*Last updated: 2026-06-07*
