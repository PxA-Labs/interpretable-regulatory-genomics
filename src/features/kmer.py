import itertools
import numpy as np
import pandas as pd

def generate_kmers(k: int) -> list:
    """
    Generate all possible DNA k-mers of length k in alphabetical order.
    E.g. k=2 -> ['AA', 'AC', 'AG', 'AT', 'CA', ...]
    """
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    return kmers

def sequence_to_kmer_frequencies(seq: str, k: int, kmer_to_idx: dict) -> np.ndarray:
    """
    Compute normalized k-mer frequencies for a single DNA sequence.
    """
    counts = np.zeros(len(kmer_to_idx), dtype=np.float32)
    num_windows = len(seq) - k + 1
    
    if num_windows <= 0:
        return counts
        
    for i in range(num_windows):
        kmer = seq[i:i+k]
        if kmer in kmer_to_idx:
            counts[kmer_to_idx[kmer]] += 1
            
    # Normalize to get frequencies (densities)
    frequencies = counts / num_windows
    return frequencies

def extract_kmer_features(
    df: pd.DataFrame,
    k: int,
    sequence_col: str = 'sequence'
) -> tuple:
    """
    Extract k-mer features for a DataFrame of genomic sequences.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing a column of sequence strings.
    k : int
        Length of the k-mers to count.
    sequence_col : str, default 'sequence'
        Name of the column containing the sequence strings.
        
    Returns:
    --------
    X : np.ndarray
        Feature matrix of shape (N, 4^k) containing normalized frequencies.
    kmer_names : list
        List of all 4^k k-mer names corresponding to columns of X.
    """
    print(f"Extracting {k}-mer features (total features: {4**k:,})...")
    
    # Generate vocabulary
    kmer_names = generate_kmers(k)
    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(kmer_names)}
    
    # Extract frequencies
    num_samples = len(df)
    X = np.zeros((num_samples, len(kmer_names)), dtype=np.float32)
    
    for idx, seq in enumerate(df[sequence_col]):
        X[idx] = sequence_to_kmer_frequencies(seq, k, kmer_to_idx)
        if (idx + 1) % 1000 == 0 or (idx + 1) == num_samples:
            print(f"  Processed {idx + 1} / {num_samples} sequences.")
            
    print("k-mer feature extraction complete.")
    return X, kmer_names
