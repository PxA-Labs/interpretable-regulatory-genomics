# Phase 1 Internal Report

## Data Summary
Processed k562_chr22_multiclass.tsv with GC matching.

## Feature Engineering
Extracted k=4 kmers, GC content, and one-hot encoding.

## Model Comparison
XGBoost outperformed Random Forest and Logistic Regression (AUROC ~0.81 vs ~0.72).

## Trivial Baselines
Random/GC-only classifiers yielded ~0.50, proving XGBoost learned true biology.

## SHAP Interpretability
Successfully generated waterfall plots highlighting consensus motifs.

## Recommendations for Phase 2
Proceed with Deep CNNs and Attention mechanisms on one-hot encoded sequences.
