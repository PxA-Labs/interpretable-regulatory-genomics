"""
=============================================================================
Phase 1 Analysis: Calibration Curves
=============================================================================

Context / Where it was used:
This script was created to fulfill Checklist Item 1.1.4. It generates 
calibration curves (reliability diagrams) for Logistic Regression, Random 
Forest, and XGBoost models to evaluate if the predicted probabilities map 
accurately to observed frequencies.

How to use this script:
1. Ensure `scikit-learn`, `xgboost`, and `matplotlib` are installed.
2. Ensure `data/processed/k562_chr22_multiclass.tsv` exists.
3. Run the script from the repository root:
   `python scripts/phase1_analysis/evaluate_calibration.py`
4. A single plot comparing the calibration of all three models against the 
   ideal diagonal will be saved to `docs/phase1_visualizations/calibration_curves.png`.
=============================================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    X = X / (1000 - k + 1)
    return X, kmers

def main():
    print("Loading dataset for Calibration Analysis...")
    dataset_path = "data/processed/k562_chr22_multiclass.tsv"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path, sep="\t")
    # Using 1000 samples to keep training fast but ensure enough bins for calibration
    df = df.sample(n=min(1000, len(df)), random_state=42)
    
    df["binary_label"] = df["label"].apply(lambda x: 1 if x > 0 else 0)
    y = df["binary_label"].values
    
    print("Extracting features...")
    X, _ = get_kmer_features(df["sequence"].values, k=4)
    
    # We will do a simple train/test split to evaluate calibration on unseen data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
        "XGBoost": XGBoostRegulatoryModel(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    plt.figure(figsize=(10, 8))
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    for name, model in models.items():
        print(f"Training and Evaluating {name}...")
        if name == "XGBoost":
            model.fit(X_train, y_train)
            probs = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name}")
        
    plt.ylabel("Fraction of Positives")
    plt.xlabel("Mean Predicted Probability")
    plt.title("Calibration Curves (Reliability Diagram)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    out_dir = "docs/phase1_visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "calibration_curves.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nCalibration curves saved to {out_path}")

if __name__ == "__main__":
    main()
