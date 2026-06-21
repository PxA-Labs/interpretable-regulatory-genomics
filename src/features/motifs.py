import re
import numpy as np
import pandas as pd


def parse_jaspar_pfm(file_path: str) -> dict:
    """
    Parse a single or multi-motif JASPAR PFM file.
    Returns a dict mapping motif ID/name (e.g. 'MA0004.1_Arnt') to a 4xW numpy array.
    Rows correspond to A, C, G, T in order.
    """
    motifs = {}
    current_id = None
    current_name = None
    matrix_rows = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id and len(matrix_rows) == 4:
                    motifs[f"{current_id}_{current_name}"] = np.array(
                        matrix_rows, dtype=np.float32
                    )
                parts = line[1:].split()
                current_id = parts[0]
                current_name = parts[1] if len(parts) > 1 else parts[0]
                matrix_rows = []
            else:
                nums = re.findall(r"\d+", line)
                if nums:
                    matrix_rows.append([float(x) for x in nums])

        # Add the last motif
        if current_id and len(matrix_rows) == 4:
            motifs[f"{current_id}_{current_name}"] = np.array(
                matrix_rows, dtype=np.float32
            )

    return motifs


def pfm_to_pwm(
    pfm: np.ndarray, pseudocount: float = 0.1, bg: dict = None
) -> np.ndarray:
    """
    Convert a 4xW PFM to a 4xW Position Weight Matrix (log-odds score).
    bg is a dict containing background probabilities for A, C, G, T (defaults to 0.25 each).
    """
    if bg is None:
        bg = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

    bg_arr = np.array(
        [bg["A"], bg["C"], bg["G"], bg["T"]], dtype=np.float32
    ).reshape(4, 1)

    # Column sums (total counts at each position)
    col_sums = pfm.sum(axis=0, keepdims=True)
    col_sums = np.maximum(col_sums, 1e-9)

    # Calculate position probabilities with pseudocounts
    # p = (counts + pseudocount * background) / (total_counts + pseudocount)
    p = (pfm + pseudocount * bg_arr) / (col_sums + pseudocount)

    # Log-odds score: log2(p / bg)
    pwm = np.log2(p / bg_arr)
    return pwm


def extract_motif_scores(
    df: pd.DataFrame,
    pfm_file: str,
    sequence_col: str = "sequence",
    pseudocount: float = 0.1,
    bg: dict = None,
) -> tuple:
    """
    For each sequence in df, compute the maximum log-odds score for each motif in pfm_file.
    Utilizes numpy vectorization via sliding_window_view for extreme speed.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing a column of sequence strings.
    pfm_file : str
        Path to a JASPAR format PFM file.
    sequence_col : str, default 'sequence'
        Name of the column containing the sequence strings.
    pseudocount : float, default 0.1
        Pseudocount value for probability correction.
    bg : dict, optional
        Background nucleotide distribution (A, C, G, T).

    Returns:
    --------
    X : np.ndarray
        Feature matrix of shape (N, M) containing maximum log-odds scores.
    motif_names : list
        List of motif names corresponding to columns of X.
    """
    print(f"Loading PFMs from {pfm_file}...")
    motifs = parse_jaspar_pfm(pfm_file)
    motif_names = sorted(list(motifs.keys()))

    if not motif_names:
        print("Warning: No motifs found in PFM file.")
        return np.zeros((len(df), 0), dtype=np.float32), []

    print(f"Loaded {len(motif_names)} motifs. Converting to PWMs...")
    pwms = {}
    for name in motif_names:
        pwm = pfm_to_pwm(motifs[name], pseudocount, bg)
        # Pad with a 5th row of zeros representing invalid bases or Ns (index 4)
        pwm_padded = np.vstack([pwm, np.zeros((1, pwm.shape[1]), dtype=np.float32)])
        pwms[name] = pwm_padded

    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    num_samples = len(df)
    num_motifs = len(motif_names)

    seqs = df[sequence_col].tolist()
    L = len(seqs[0]) if num_samples > 0 else 0

    # Build sequence matrix where A=0, C=1, G=2, T=3, invalid/N=4
    seq_matrix = np.full((num_samples, L), 4, dtype=np.int32)
    for idx, seq in enumerate(seqs):
        seq_upper = seq.upper()
        for pos, char in enumerate(seq_upper):
            if pos < L:
                seq_matrix[idx, pos] = base_to_idx.get(char, 4)

    X = np.zeros((num_samples, num_motifs), dtype=np.float32)

    print(f"Scanning {num_samples} sequences with {num_motifs} motifs...")
    for m_idx, name in enumerate(motif_names):
        pwm = pwms[name]
        W = pwm.shape[1]
        if L >= W:
            # sliding_window_view returns view of shape (N, L - W + 1, W)
            windows = np.lib.stride_tricks.sliding_window_view(seq_matrix, W, axis=1)
            scores = np.zeros((num_samples, L - W + 1), dtype=np.float32)
            for p in range(W):
                scores += pwm[windows[:, :, p], p]
            X[:, m_idx] = scores.max(axis=1)
        else:
            X[:, m_idx] = -999.0

    print("Motif scanning complete.")
    return X, motif_names
