import itertools
import numpy as np
import pandas as pd


def generate_dinucleotides() -> list:
    """
    Generate all possible DNA dinucleotides of length 2 in alphabetical order.
    E.g. -> ['AA', 'AC', 'AG', 'AT', 'CA', ...]
    """
    bases = ["A", "C", "G", "T"]
    dinucleotides = ["".join(p) for p in itertools.product(bases, repeat=2)]
    return dinucleotides


def sequence_to_dinucleotide_frequencies(seq: str, dinuc_to_idx: dict) -> np.ndarray:
    """
    Compute normalized dinucleotide frequencies for a single DNA sequence.
    """
    counts = np.zeros(len(dinuc_to_idx), dtype=np.float32)
    num_windows = len(seq) - 1

    if num_windows <= 0:
        return counts

    seq_upper = seq.upper()
    for i in range(num_windows):
        dinuc = seq_upper[i : i + 2]
        if dinuc in dinuc_to_idx:
            counts[dinuc_to_idx[dinuc]] += 1

    # Normalize to get frequencies
    frequencies = counts / num_windows
    return frequencies


def extract_dinucleotide_frequencies(
    df: pd.DataFrame, sequence_col: str = "sequence"
) -> tuple:
    """
    Extract dinucleotide frequency features for a DataFrame of genomic sequences.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing a column of sequence strings.
    sequence_col : str, default 'sequence'
        Name of the column containing the sequence strings.

    Returns:
    --------
    X : np.ndarray
        Feature matrix of shape (N, 16) containing normalized frequencies.
    dinuc_names : list
        List of all 16 dinucleotide names corresponding to columns of X.
    """
    print("Extracting dinucleotide features...")
    dinuc_names = generate_dinucleotides()
    dinuc_to_idx = {dinuc: idx for idx, dinuc in enumerate(dinuc_names)}

    num_samples = len(df)
    X = np.zeros((num_samples, len(dinuc_names)), dtype=np.float32)

    for idx, seq in enumerate(df[sequence_col]):
        X[idx] = sequence_to_dinucleotide_frequencies(seq, dinuc_to_idx)

    print("Dinucleotide feature extraction complete.")
    return X, dinuc_names
