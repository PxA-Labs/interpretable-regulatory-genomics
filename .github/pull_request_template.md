## Description
Provide a brief summary of the changes introduced by this PR and the problem it solves.

## Related Issue / Spec Document
- Connects to issue # or references docs file (e.g. `docs/03-technical-design-document.pdf`)

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Refactoring (code cleanup, formatting, or optimization)
- [ ] Documentation update (PDF/markdown adjustments)

## Verification & Testing
Describe how you verified these changes. Include:
1. Steps to run or test the code.
2. The runtime environment (e.g. Kaggle GPU T4x2, local CPU).
3. Metric outputs (e.g. AUROC, training time) if applicable.

## Checklist
- [ ] My code follows the PEP 8 style guide for Python.
- [ ] I have documented my code, particularly complex biological modeling details or assumptions.
- [ ] I have verified that no random split is used, and the chromosome-holdout partition (e.g. Chr 8, 9) is strictly preserved to prevent leakage.
- [ ] Jupyter Notebooks have been run sequentially and cleared of temporary execution cells.
- [ ] I have updated the documentation or run-scripts if necessary.
