import numpy as np


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
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}

    for i, base in enumerate(seq):
        if base in mapping:
            onehot[mapping[base], i] = 1.0
        else:
            # Handle unknown bases (N) by assigning uniform probability
            onehot[:, i] = 0.25

    return onehot
