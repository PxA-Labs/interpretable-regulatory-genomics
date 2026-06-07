# Contributing to Interpretable Regulatory Genomics

First off, thank you for considering contributing to this project! It is a research-grade computational genomics initiative aimed at discovering regulatory switches in non-coding DNA.

Following these guidelines helps ensure the codebase remains robust, reproducible, and easy to onboarding new contributors.

---

## 🚀 How to Get Started

1. **Review the Specifications**: Before writing any code, review the project documentation in the [docs/](docs/) directory (particularly [12-contributor-onboarding.pdf](docs/12-contributor-onboarding.pdf) and [11-glossary-project-memory.pdf](docs/11-glossary-project-memory.pdf)).
2. **Search Issues**: Check the GitHub Issues tab to see if your bug or feature request has already been discussed.
3. **Fork and Clone**: Fork this repository to your account and clone it locally.

---

## 🛠️ Development & Coding Standards

To ensure reproducibility (crucial for genomic research), we follow strict rules:

### 1. Python Stack & Dependencies
* **Core Language**: Python 3.10+
* **Dependencies**: Any new libraries must be approved (no large GPU clusters or scratch-trained transformers; keep the code runnable on Kaggle-level compute).
* Keep dependencies defined in `requirements.txt` or environment yaml files.

### 2. Code Quality & Format
* **Style Guide**: Conform to PEP 8.
* **Code Documentation**: Maintain thorough comments and docstrings explaining biological assumptions and logic (e.g., negative sampling ratios, chromosome-holdout choices).
* **Notebook Hygiene**: Jupyter notebooks must be cleaned of temporary testing cells and run top-to-bottom without error before commit.

### 3. Chromosome Holdout Split
* **Never** use random train/test splits. Always hold out specific chromosomes (e.g., Chr 8, 9) for test sets to prevent genomic data leakage.

---

## 📬 Git Workflow & Pull Requests

1. **Branch Naming**: Use a prefix indicating the change type:
   * `feat/` for new features or modeling approaches.
   * `fix/` for bug fixes.
   * `docs/` for documentation updates.
   * `refactor/` for code cleanup.
2. **Commit Messages**: Write clear, descriptive commit messages (e.g., `feat: Add CNN baseline with one-hot sequence encoding`).
3. **Pull Request Checklist**:
   * Ensure your branch is up-to-date with `main`.
   * Include a description of the changes and the problem they solve.
   * Verify all code runs successfully under standard Kaggle/local environments.
   * A project maintainer will review your code before merging.

Thank you for helping us advance interpretable regulatory genomics!
