"""
=============================================================================
Phase 1 Analysis: 5-Fold Stratified Cross-Validation
=============================================================================

Context / Where it was used:
This script was created to fulfill Checklist Items 1.1.5 and 1.2.4. It 
performs a formal 5-Fold Stratified Cross-Validation for the classical 
baselines (LogReg, RF, XGBoost) and automatically logs the resulting 
metrics (Mean/Std for AUROC and F1) into the `experiments/logs/` directory.

How to use this script:
1. Ensure `scikit-learn`, `xgboost`, and `pandas` are installed.
2. Ensure `data/processed/k562_chr22_multiclass.tsv` exists.
3. Run the script from the repository root:
   `python scripts/phase1_analysis/evaluate_cv.py`
4. The script will output metrics to the console and generate formal MLOps 
   `metrics.json` and `config.yaml` files inside `experiments/logs/`.
=============================================================================
"""
import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
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
    print("Loading dataset for 5-Fold CV...")
    dataset_path = "data/processed/k562_chr22_multiclass.tsv"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path, sep="\t")
    # Using 1000 samples for reasonable runtime in CV
    df = df.sample(n=min(1000, len(df)), random_state=42)
    
    df["binary_label"] = df["label"].apply(lambda x: 1 if x > 0 else 0)
    y = df["binary_label"].values
    
    print("Extracting features...")
    X, _ = get_kmer_features(df["sequence"].values, k=4)
    
    models = {
        "exp_001_logistic_kmer4": LogisticRegression(max_iter=1000, random_state=42),
        "exp_002_rf_kmer4": RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
        "exp_004_xgb_kmer4": XGBoostRegulatoryModel(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for exp_id, model in models.items():
        print(f"\nRunning 5-Fold CV for {exp_id}...")
        auroc_scores = []
        f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            if exp_id == "exp_004_xgb_kmer4":
                # Need a fresh instance per fold
                fold_model = XGBoostRegulatoryModel(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42)
                fold_model.fit(X_train, y_train)
                probs = fold_model.predict(X_val)
                preds = (probs > 0.5).astype(int)
            else:
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
                probs = fold_model.predict_proba(X_val)[:, 1]
                preds = fold_model.predict(X_val)
                
            auroc = roc_auc_score(y_val, probs)
            f1 = f1_score(y_val, preds)
            
            auroc_scores.append(auroc)
            f1_scores.append(f1)
            
        mean_auroc = np.mean(auroc_scores)
        std_auroc = np.std(auroc_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        print(f"Results: AUROC {mean_auroc:.4f} +/- {std_auroc:.4f} | F1 {mean_f1:.4f} +/- {std_f1:.4f}")
        
        # Save to experiments/logs
        log_dir = f"experiments/logs/{exp_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        metrics = {
            "cv_splits": 5,
            "metrics": {
                "auroc": {"mean": float(mean_auroc), "std": float(std_auroc)},
                "f1_score": {"mean": float(mean_f1), "std": float(std_f1)}
            }
        }
        
        metrics_path = os.path.join(log_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Write dummy config
        config_path = os.path.join(log_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(f"experiment_id: {exp_id}\n")
            f.write("feature_set: kmer_k4\n")
            f.write("model_type: " + exp_id.split("_")[1] + "\n")
            
    print("\nAll CV metrics saved to experiments/logs/")

if __name__ == "__main__":
    main()
