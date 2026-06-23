"""
=============================================================================
Phase 1 Analysis: Single-Sample SHAP Explanations
=============================================================================

Context / Where it was used:
This script was created to fulfill Checklist Item 1.1.1. It generates localized
SHAP waterfall plots for 5 representative test samples without modifying the 
legacy Phase 1 Jupyter Notebooks.

How to use this script:
1. Ensure your environment has `shap` and `xgboost` installed.
2. Ensure you have run the data extraction pipeline so that 
   `data/processed/k562_chr22_multiclass.tsv` exists.
3. Run the script from the repository root:
   `python scripts/phase1_analysis/generate_shap_plots.py`
4. The generated plots will be saved to `docs/phase1_visualizations/`.
=============================================================================
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.features.kmer
import src.models.tree_ensemble
from src.interpret.shap_analysis import explain_single_sample

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("data/processed/k562_chr22_multiclass.tsv", sep="\t")

# Map labels to binary for the classical baseline: background (0) vs active (1)
df["label"] = df["label"].apply(lambda x: 1 if x > 0 else 0)

# 2. Extract k-mers
print("Extracting k-mers...")
X, kmer_names = src.features.kmer.extract_kmer_features(df, k=4)
y = df["label"].values

# 3. Train XGBoost model
print("Training XGBoost model...")
model = src.models.tree_ensemble.XGBoostRegulatoryModel(n_estimators=100, n_jobs=-1, eval_metric="logloss", random_state=42)
model.fit(X, y)

# 4. Predict
y_prob = model.predict_proba(X)
y_pred = model.predict(X)

# 5. Select 5 samples
# 2 True Positives
tp_indices = np.where((y == 1) & (y_pred == 1))[0]
# 2 True Negatives
tn_indices = np.where((y == 0) & (y_pred == 0))[0]
# 1 Borderline (prob near 0.5)
borderline_indices = np.argsort(np.abs(y_prob - 0.5))

selected_indices = [
    tp_indices[0], tp_indices[1], 
    tn_indices[0], tn_indices[1], 
    borderline_indices[0]
]

labels = ["TruePositive_1", "TruePositive_2", "TrueNegative_1", "TrueNegative_2", "Borderline"]

out_dir = "docs/phase1_visualizations"
os.makedirs(out_dir, exist_ok=True)

# 6. Generate plots
for idx, label in zip(selected_indices, labels):
    print(f"Generating SHAP plot for {label} (Index: {idx}, Prob: {y_prob[idx]:.3f})...")
    X_sample = X[idx]
    plot_path = os.path.join(out_dir, f"shap_waterfall_{label.lower()}.png")
    explain_single_sample(
        model=model,
        X_sample=X_sample,
        feature_names=kmer_names,
        output_plot_path=plot_path,
        show_plot=False
    )

print("All plots generated successfully!")
