import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

def sequence_to_onehot(seq: str) -> np.ndarray:
    """
    Convert a DNA sequence string to a one-hot encoded numpy array.
    
    Mapping:
        A/a -> [1, 0, 0, 0]
        C/c -> [0, 1, 0, 0]
        G/g -> [0, 0, 1, 0]
        T/t -> [0, 0, 0, 1]
        Any other character (e.g. N) -> [0.25, 0.25, 0.25, 0.25]
        
    Parameters:
    -----------
    seq : str
        DNA sequence string.
        
    Returns:
    --------
    np.ndarray
        One-hot encoded array of shape (4, len(seq)).
    """
    seq = seq.upper()
    seq_len = len(seq)
    
    # Initialize one-hot matrix of shape (4, seq_len)
    onehot = np.zeros((4, seq_len), dtype=np.float32)
    
    # Mapping bases to channel indices
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for i, base in enumerate(seq):
        if base in mapping:
            onehot[mapping[base], i] = 1.0
        else:
            # Handle unknown bases (N) by assigning uniform probability
            onehot[:, i] = 0.25
            
    return onehot

class GenomicDataset(Dataset):
    """
    PyTorch Dataset for regulatory sequences.
    Converts genomic sequence strings into one-hot encoded tensors.
    """
    def __init__(self, df: pd.DataFrame, sequence_col: str = 'sequence', label_col: str = 'label'):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing sequence strings and labels.
        sequence_col : str, default 'sequence'
            Column name containing DNA sequences.
        label_col : str, default 'label'
            Column name containing binary labels.
        """
        self.sequences = df[sequence_col].values
        self.labels = df[label_col].values if label_col in df.columns else None
        
    def __len__(self) -> int:
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> tuple:
        seq_str = self.sequences[idx]
        onehot_seq = sequence_to_onehot(seq_str)
        
        # Convert to PyTorch float tensor
        x = torch.tensor(onehot_seq, dtype=torch.float32)
        
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        return x
