"""
=============================================================================
Phase 1 Analysis: Permutation Importance
=============================================================================

Context / Where it was used:
This script was created to fulfill Checklist Item 1.1.3. It computes the 
permutation importance of the k-mer features for the XGBoost model using 10 
repeats to generate confidence intervals, and compares it against standard 
gain-based importance.

How to use this script:
1. Ensure `scikit-learn`, `xgboost`, and `matplotlib` are installed.
2. Ensure `data/processed/k562_chr22_multiclass.tsv` exists.
3. Run the script from the repository root:
   `python scripts/phase1_analysis/evaluate_permutation_importance.py`
4. The script trains an XGBoost model, runs permutation shuffling (which may 
   take a few moments), and saves both a plot (`permutation_importance.png`) 
   and a CSV table (`permutation_importance_results.csv`) into 
   `docs/phase1_visualizations/`.
=============================================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, roc_auc_score

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tree_ensemble import XGBoostRegulatoryModel

def get_kmer_features(sequences, k=4):
    """Simple feature extraction purely for this analysis script."""
    from itertools import product
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(kmers)}
    
    X = np.zeros((len(sequences), len(kmers)))
    for i, seq in enumerate(sequences):
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            if kmer in kmer_to_idx:
                X[i, kmer_to_idx[kmer]] += 1
    # Normalize by sequence length
    X = X / (1000 - k + 1)
    return X, kmers

def main():
    print("Loading dataset...")
    dataset_path = "data/processed/k562_chr22_multiclass.tsv"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path, sep="\t")
    # Subsample for speed during permutation importance (which is very slow)
    # Using 500 samples to keep it fast while retaining signal
    df = df.sample(n=min(500, len(df)), random_state=42)
    
    df["binary_label"] = df["label"].apply(lambda x: 1 if x > 0 else 0)
    y = df["binary_label"].values
    
    print("Extracting k-mer features (k=4)...")
    X, kmer_names = get_kmer_features(df["sequence"].values, k=4)
    
    print("Training XGBoost Model...")
    model = XGBoostRegulatoryModel(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    # Get built-in Gain importance
    print("Extracting Gain-based Importance...")
    gain_df = model.get_feature_importance(feature_names=kmer_names)
    gain_importance = gain_df["importance"].values
    
    print("Computing Permutation Importance (10 repeats)...")
    # Custom scorer for predict_proba since AUROC needs probabilities
    def auroc_scorer(estimator, X_test, y_test):
        # We know estimator has predict_proba
        probs = estimator.predict(X_test)
        return roc_auc_score(y_test, probs)
    
    # We must pass the underlying model because permutation_importance expects a standard scikit-learn estimator
    # XGBoostRegulatoryModel wraps it in self.model
    result = permutation_importance(
        model.model, X, y, scoring=auroc_scorer, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    perm_importances = result.importances_mean
    perm_std = result.importances_std
    
    # Compare rankings
    print("Analyzing Results...")
    
    # Create comparison table for top 20 features by Gain
    results_list = []
    for i in range(len(kmer_names)):
        results_list.append({
            "kmer": kmer_names[i],
            "gain_importance": gain_importance[i],
            "perm_importance_mean": perm_importances[i],
            "perm_importance_std": perm_std[i]
        })
    
    results_df = pd.DataFrame(results_list)
    results_df["gain_rank"] = results_df["gain_importance"].rank(ascending=False)
    results_df["perm_rank"] = results_df["perm_importance_mean"].rank(ascending=False)
    results_df["rank_diff"] = np.abs(results_df["gain_rank"] - results_df["perm_rank"])
    
    # Sort by Permutation Importance
    results_df = results_df.sort_values(by="perm_importance_mean", ascending=False)
    
    # Plotting
    print("Generating Plot...")
    top_n = 15
    top_results = results_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_results["kmer"][::-1], top_results["perm_importance_mean"][::-1], xerr=top_results["perm_importance_std"][::-1], capsize=4, color='skyblue', edgecolor='black')
    ax.set_xlabel("Permutation Importance (Mean AUROC drop +/- STD)")
    ax.set_title(f"Top {top_n} Features by Permutation Importance (XGBoost)")
    plt.tight_layout()
    
    out_dir = "docs/phase1_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "permutation_importance.png")
    plt.savefig(plot_path, dpi=300)
    
    csv_path = os.path.join(out_dir, "permutation_importance_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"Results saved to {out_dir}/")
    print(f"Top 5 by Permutation:")
    print(results_df.head(5)[["kmer", "perm_importance_mean", "perm_rank", "gain_rank"]])

if __name__ == "__main__":
    main()
