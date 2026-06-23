def calculate_gc_content(seq: str) -> float:
    """
    Calculate the GC content fraction of a DNA sequence.
    """
    if not seq:
        return 0.0
    # Count both upper and lower case G/C
    seq_upper = seq.upper()
    gc_count = seq_upper.count("G") + seq_upper.count("C")
    return gc_count / len(seq)
