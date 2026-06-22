import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch

from src.data.download import download_biosample_ccres
from src.data.negative_sampling import build_negative_dataset
from src.models.cnn import ShallowCNN, DeepCNN, AttentionCNN
from src.models.train_nn import PyTorchModelWrapper


def test_download_biosample_ccres_validation():
    # Verify validation works for unsupported cell lines
    with pytest.raises(ValueError, match="Unsupported biosample"):
        download_biosample_ccres("hepg2")

    # Mock download_file to prevent actual network calls
    with patch("src.data.download.download_file") as mock_download:
        dest_k562 = download_biosample_ccres("K562", output_dir="temp_raw")
        assert dest_k562 == os.path.join("temp_raw", "ENCFF464BRU.bed.gz")
        mock_download.assert_called_once_with(
            "https://www.encodeproject.org/files/ENCFF464BRU/@@download/ENCFF464BRU.bed.gz",
            os.path.join("temp_raw", "ENCFF464BRU.bed.gz"),
        )

    with patch("src.data.download.download_file") as mock_download:
        dest_gm12878 = download_biosample_ccres("gm12878", output_dir="temp_raw")
        assert dest_gm12878 == os.path.join("temp_raw", "ENCFF590IMH.bed.gz")
        mock_download.assert_called_once_with(
            "https://www.encodeproject.org/files/ENCFF590IMH/@@download/ENCFF590IMH.bed.gz",
            os.path.join("temp_raw", "ENCFF590IMH.bed.gz"),
        )


def test_multiclass_negative_sampling_label_mapping():
    # Setup temporary directory and files
    with tempfile.TemporaryDirectory() as tmp_dir:
        pos_tsv_path = os.path.join(tmp_dir, "positives.tsv")
        output_tsv_path = os.path.join(tmp_dir, "dataset.tsv")
        fasta_path = os.path.join(tmp_dir, "chr22.fa")

        # Write mock FASTA (needs to be longer to allow sampling)
        with open(fasta_path, "w") as f:
            f.write(">chr22\n")
            f.write("ATCG" * 500 + "\n")  # 2000 bp

        # Create positive regions DataFrame with PLS, dELS, and pELS
        pos_df = pd.DataFrame(
            {
                "chrom": ["chr22", "chr22", "chr22"],
                "start": [100, 500, 900],
                "end": [200, 600, 1000],
                "accession": ["POS1", "POS2", "POS3"],
                "classification": ["PLS", "dELS", "pELS"],
                "sequence": ["A" * 100, "C" * 100, "G" * 100],
            }
        )
        pos_df.to_csv(pos_tsv_path, sep="\t", index=False)

        # 1. Test in multiclass mode (multiclass=True)
        build_negative_dataset(
            pos_tsv_path=pos_tsv_path,
            fasta_source=fasta_path,
            output_path=output_tsv_path,
            target_length=100,
            strategy="random_unmatched",
            multiclass=True,
        )

        df_multi = pd.read_csv(output_tsv_path, sep="\t")
        # Output should contain positives (3) and sampled negatives (3)
        assert len(df_multi) == 6

        # Check label assignments:
        # PLS -> 1
        # dELS / pELS -> 2
        # negatives (non-regulatory) -> 0
        pls_row = df_multi[df_multi["classification"] == "PLS"].iloc[0]
        dels_row = df_multi[df_multi["classification"] == "dELS"].iloc[0]
        pels_row = df_multi[df_multi["classification"] == "pELS"].iloc[0]
        neg_rows = df_multi[df_multi["classification"] == "non-regulatory"]

        assert pls_row["label"] == 1
        assert dels_row["label"] == 2
        assert pels_row["label"] == 2
        assert all(neg_rows["label"] == 0)

        # 2. Test in binary mode (multiclass=False) to ensure backward compatibility
        build_negative_dataset(
            pos_tsv_path=pos_tsv_path,
            fasta_source=fasta_path,
            output_path=output_tsv_path,
            target_length=100,
            strategy="random_unmatched",
            multiclass=False,
        )

        df_binary = pd.read_csv(output_tsv_path, sep="\t")
        assert len(df_binary) == 6

        # In binary mode: positives -> 1, negatives -> 0
        binary_pos = df_binary[
            df_binary["classification"].isin(["PLS", "dELS", "pELS"])
        ]
        binary_neg = df_binary[df_binary["classification"] == "non-regulatory"]
        assert all(binary_pos["label"] == 1)
        assert all(binary_neg["label"] == 0)


def test_cnn_output_shapes():
    # Setup dummy input: batch of 4 sequences of length 1000
    x = torch.randn(4, 4, 1000)

    # 1. ShallowCNN
    shallow_bin = ShallowCNN(num_classes=1)
    shallow_multi = ShallowCNN(num_classes=3)
    assert shallow_bin(x).shape == (4,)
    assert shallow_multi(x).shape == (4, 3)

    # 2. DeepCNN
    deep_bin = DeepCNN(num_classes=1)
    deep_multi = DeepCNN(num_classes=3)
    assert deep_bin(x).shape == (4,)
    assert deep_multi(x).shape == (4, 3)

    # 3. AttentionCNN
    attn_bin = AttentionCNN(num_classes=1)
    attn_multi = AttentionCNN(num_classes=3)
    assert attn_bin(x).shape == (4,)
    assert attn_multi(x).shape == (4, 3)


def test_pytorch_model_wrapper_multiclass_training():
    # Create synthetic dataset (20 samples, 4 channels, length 1000)
    X = np.random.randn(20, 4, 1000).astype(np.float32)

    # 3-class target labels (0, 1, or 2)
    y = np.random.choice([0, 1, 2], size=20)

    wrapper = PyTorchModelWrapper(
        model_class=ShallowCNN,
        model_params={"sequence_length": 1000},
        epochs=2,
        batch_size=4,
        num_classes=3,
        val_split=0.2,
        random_state=42,
    )

    # Fit the wrapper
    wrapper.fit(X, y)

    # Check predictions probabilities shape
    probs = wrapper.predict_proba(X)
    assert probs.shape == (20, 3)
    # Check that rows sum to 1 (softmax behavior)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Check predictions shape
    preds = wrapper.predict(X)
    assert preds.shape == (20,)
    assert np.all(np.isin(preds, [0, 1, 2]))


def test_pytorch_model_wrapper_save_load_multiclass():
    X = np.random.randn(10, 4, 1000).astype(np.float32)
    y = np.random.choice([0, 1, 2], size=10)

    wrapper = PyTorchModelWrapper(
        model_class=ShallowCNN,
        epochs=1,
        batch_size=4,
        num_classes=3,
        val_split=0.0,
        random_state=42,
    )
    wrapper.fit(X, y)
    preds_before = wrapper.predict(X)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_model_path = f.name

    try:
        wrapper.save(temp_model_path)
        loaded_wrapper = PyTorchModelWrapper.load(temp_model_path)

        assert loaded_wrapper.num_classes == 3
        preds_after = loaded_wrapper.predict(X)
        assert np.array_equal(preds_before, preds_after)
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)


def test_backwards_compatibility():
    # Test that binary mode works out-of-the-box by default (num_classes=1)
    X = np.random.randn(10, 4, 1000).astype(np.float32)
    y = np.random.choice([0, 1], size=10)

    # Initialize without specifying num_classes explicitly
    wrapper = PyTorchModelWrapper(
        model_class=ShallowCNN, epochs=1, batch_size=4, val_split=0.0, random_state=42
    )
    assert wrapper.num_classes == 1

    wrapper.fit(X, y)

    # Probabilities should be single probability (sigmoid)
    probs = wrapper.predict_proba(X)
    assert probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1)

    preds = wrapper.predict(X)
    assert np.all(np.isin(preds, [0, 1]))
