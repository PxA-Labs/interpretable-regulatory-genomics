import os
import pandas as pd
import numpy as np
import torch

from src.data.download import download_biosample_ccres, download_hg38_chromosome
from src.data.parse_encode import parse_and_resize_ccres
from src.data.sequence_extractor import extract_sequences
from src.data.negative_sampling import build_negative_dataset
from src.models.cnn import AttentionCNN
from src.models.train_nn import PyTorchModelWrapper


def main():
    print("=== Phase 2 Solidification End-to-End Verification ===")
    
    # 1. Setup Directories
    raw_dir = "data/raw"
    ref_dir = "data/reference"
    proc_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    # 2. Download chr22 fasta and cell-type annotations
    print("\n--- Step 1: Downloading Annotations and Reference Genome ---")
    chr22_fasta = os.path.join(ref_dir, "chr22.fa.gz")
    if not os.path.exists(chr22_fasta):
        download_hg38_chromosome("chr22", output_dir=ref_dir)
    else:
        print("chr22.fa.gz already exists.")

    k562_bed_raw = os.path.join(raw_dir, "ENCFF464BRU.bed.gz")
    if not os.path.exists(k562_bed_raw):
        download_biosample_ccres("K562", output_dir=raw_dir)
    else:
        print("K562 raw BED already exists.")

    gm12878_bed_raw = os.path.join(raw_dir, "ENCFF590IMH.bed.gz")
    if not os.path.exists(gm12878_bed_raw):
        download_biosample_ccres("GM12878", output_dir=raw_dir)
    else:
        print("GM12878 raw BED already exists.")

    # 3. Parse and Slice to Subset (chr22) for K562 and GM12878
    print("\n--- Step 2: Parsing and Slicing Subsets ---")
    k562_parsed = os.path.join(proc_dir, "k562_chr22.bed")
    gm12878_parsed = os.path.join(proc_dir, "gm12878_chr22.bed")

    parse_and_resize_ccres(
        input_path=k562_bed_raw,
        output_path=k562_parsed,
        element_types=["PLS", "dELS", "pELS"],
        chromosomes=["chr22"],
        target_length=1000
    )

    parse_and_resize_ccres(
        input_path=gm12878_bed_raw,
        output_path=gm12878_parsed,
        element_types=["PLS", "dELS", "pELS"],
        chromosomes=["chr22"],
        target_length=1000
    )

    # Load and keep a small subset to make training extremely fast
    for parsed_path, label_prefix in [(k562_parsed, "k562"), (gm12878_parsed, "gm12878")]:
        df = pd.read_csv(parsed_path, sep="\t", header=None)
        # Squeeze to 100 positive elements (50 PLS and 50 ELS)
        df_pls = df[df[4] == "PLS"].head(50)
        df_els = df[df[4].isin(["dELS", "pELS"])].head(50)
        df_subset = pd.concat([df_pls, df_els], ignore_index=True)
        
        subset_path = os.path.join(proc_dir, f"{label_prefix}_chr22_subset.bed")
        df_subset.to_csv(subset_path, sep="\t", header=False, index=False)
        print(f"Saved {len(df_subset)} sliced positive regions to {subset_path}")

    # 4. Extract Sequences
    print("\n--- Step 3: Extracting Positive Sequences ---")
    k562_pos_tsv = os.path.join(proc_dir, "k562_chr22_pos.tsv")
    gm12878_pos_tsv = os.path.join(proc_dir, "gm12878_chr22_pos.tsv")

    extract_sequences(
        bed_path=os.path.join(proc_dir, "k562_chr22_subset.bed"),
        fasta_source=chr22_fasta,
        output_path=k562_pos_tsv
    )
    extract_sequences(
        bed_path=os.path.join(proc_dir, "gm12878_chr22_subset.bed"),
        fasta_source=chr22_fasta,
        output_path=gm12878_pos_tsv
    )

    # 5. Build Multiclass Negatives matched by GC
    print("\n--- Step 4: Generating Negative Sequences & Label Mapping ---")
    k562_final_tsv = os.path.join(proc_dir, "k562_chr22_multiclass.tsv")
    gm12878_final_tsv = os.path.join(proc_dir, "gm12878_chr22_multiclass.tsv")

    build_negative_dataset(
        pos_tsv_path=k562_pos_tsv,
        fasta_source=chr22_fasta,
        output_path=k562_final_tsv,
        target_length=1000,
        strategy="gc_matched",
        multiclass=True
    )
    build_negative_dataset(
        pos_tsv_path=gm12878_pos_tsv,
        fasta_source=chr22_fasta,
        output_path=gm12878_final_tsv,
        target_length=1000,
        strategy="gc_matched",
        multiclass=True
    )

    # 6. Load Datasets
    print("\n--- Step 5: Preparing Data for Training ---")
    k562_data = pd.read_csv(k562_final_tsv, sep="\t")
    gm12878_data = pd.read_csv(gm12878_final_tsv, sep="\t")

    print(f"K562 dataset distribution:\n{k562_data['label'].value_counts()}")
    print(f"GM12878 dataset distribution:\n{gm12878_data['label'].value_counts()}")

    X_train = k562_data["sequence"].values
    y_train = k562_data["label"].values

    X_test = gm12878_data["sequence"].values
    y_test = gm12878_data["label"].values

    # 7. Model Training (3-Class Classification)
    print("\n--- Step 6: Instantiating and Fitting Multiclass Model ---")
    wrapper = PyTorchModelWrapper(
        model_class=AttentionCNN,
        model_params={"sequence_length": 1000},
        epochs=3,
        batch_size=16,
        num_classes=3,
        val_split=0.2,
        random_state=42
    )

    wrapper.fit(X_train, y_train)

    # 8. Cross-Cell-Type Evaluation
    print("\n--- Step 7: Evaluating Cross-Cell-Type Generalization (K562 -> GM12878) ---")
    probs = wrapper.predict_proba(X_test)
    preds = wrapper.predict(X_test)

    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {preds.shape}")

    # Calculate validation accuracy
    accuracy = np.mean(preds == y_test)
    print(f"Cross-cell-type Accuracy (GM12878 Test Set): {accuracy:.4f}")

    # Generate Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, preds)
    print(f"Confusion Matrix:\n{cm}")

    # 9. Test Model Saving & Loading
    print("\n--- Step 8: Verifying Model Persistence ---")
    temp_model_path = "data/processed/k562_multiclass_model.pkl"
    wrapper.save(temp_model_path)
    print(f"Saved model to {temp_model_path}")

    loaded_wrapper = PyTorchModelWrapper.load(temp_model_path)
    loaded_preds = loaded_wrapper.predict(X_test)
    assert np.array_equal(preds, loaded_preds), "Loaded model predictions do not match original model predictions!"
    print("Success! Loaded model predictions match original model predictions.")
    
    # Cleanup loaded model file to save space
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    print("\n=== Phase 2 Solidification End-to-End Verification Complete! ===")


if __name__ == "__main__":
    main()
