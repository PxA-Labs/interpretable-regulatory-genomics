import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


from src.features.onehot import sequence_to_onehot


class GenomicDataset(Dataset):
    """
    PyTorch Dataset for regulatory sequences.
    Converts genomic sequence strings into one-hot encoded tensors.
    """

    def __init__(
        self, df: pd.DataFrame, sequence_col: str = "sequence", label_col: str = "label"
    ):
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
