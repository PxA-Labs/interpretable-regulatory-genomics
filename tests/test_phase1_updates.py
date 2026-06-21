import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.negative_sampling import calculate_gc_content, sample_negatives_for_chromosome
from src.features.dinucleotide import extract_dinucleotide_frequencies, generate_dinucleotides
from src.features.motifs import parse_jaspar_pfm, pfm_to_pwm, extract_motif_scores


def test_calculate_gc_content():
    assert calculate_gc_content("ATCG") == 0.5
    assert calculate_gc_content("AAAA") == 0.0
    assert calculate_gc_content("GGCC") == 1.0
    assert calculate_gc_content("AtCg") == 0.5
    assert calculate_gc_content("") == 0.0


def test_dinucleotide_frequencies():
    seqs = pd.DataFrame({"sequence": ["ATCG", "AAAA", "GGCCGG"]})
    X, names = extract_dinucleotide_frequencies(seqs)
    
    assert X.shape == (3, 16)
    assert len(names) == 16
    assert "AA" in names
    
    # "AAAA" has length 4, overlapping dinucs: AA, AA, AA. So freq of AA = 1.0, others = 0.0
    aa_idx = names.index("AA")
    assert X[1, aa_idx] == 1.0
    assert X[1].sum() == 1.0
    
    # "ATCG" has overlapping dinucs: AT, TC, CG. Freq of each is 1/3.
    assert pytest.approx(X[0].sum()) == 1.0
    assert pytest.approx(X[0, names.index("AT")]) == 1.0 / 3.0


def test_motif_scanning():
    # Write a temporary JASPAR PFM file
    pfm_content = """
>MA0004.1 Arnt
A [  4  19   0   0 ]
C [ 16   0  20  20 ]
G [  0   1   0   0 ]
T [  0   0   0   0 ]
"""
    with tempfile.NamedTemporaryFile(suffix=".jaspar", delete=False, mode="w") as f:
        f.write(pfm_content.strip())
        temp_pfm_path = f.name

    try:
        motifs = parse_jaspar_pfm(temp_pfm_path)
        assert "MA0004.1_Arnt" in motifs
        assert motifs["MA0004.1_Arnt"].shape == (4, 4)
        
        # In position 2 (0-indexed 1), A count is 19/20, representing highly conserved A
        # Let's run extract_motif_scores on sample sequences
        seqs = pd.DataFrame({"sequence": ["ACGG", "AAAA", "CGAT"]})
        X, names = extract_motif_scores(seqs, temp_pfm_path)
        
        assert X.shape == (3, 1)
        assert names == ["MA0004.1_Arnt"]
        # Log odds should be real values
        assert not np.isnan(X).any()
    finally:
        if os.path.exists(temp_pfm_path):
            os.remove(temp_pfm_path)


def test_negative_sampling_strategies():
    # Setup mock positive regions and chromosome sequence
    chrom_seq = "ATCGATCG" * 100 # length 800
    pos_regions = pd.DataFrame({
        'chrom': ['chr1', 'chr1'],
        'start': [10, 100],
        'end': [20, 110],
        'accession': ['POS1', 'POS2'],
        'classification': ['PLS', 'dELS'],
        'sequence': ['ATCGATCGAT', 'ATCGATCGAT']
    })
    
    # Strategy 1: gc_matched (default, backward-compatible)
    negs_gc = sample_negatives_for_chromosome(
        chrom='chr1',
        chrom_seq=chrom_seq,
        pos_regions_chrom=pos_regions,
        target_length=10,
        strategy='gc_matched'
    )
    assert len(negs_gc) == 2
    assert all(n['chrom'] == 'chr1' for n in negs_gc)
    assert all(n['classification'] == 'non-regulatory' for n in negs_gc)
    
    # Strategy 2: random_unmatched
    negs_random = sample_negatives_for_chromosome(
        chrom='chr1',
        chrom_seq=chrom_seq,
        pos_regions_chrom=pos_regions,
        target_length=10,
        strategy='random_unmatched'
    )
    assert len(negs_random) == 2
    
    # Strategy 3: flanking
    negs_flanking = sample_negatives_for_chromosome(
        chrom='chr1',
        chrom_seq=chrom_seq,
        pos_regions_chrom=pos_regions,
        target_length=10,
        strategy='flanking'
    )
    assert len(negs_flanking) == 2
    # Check that flanking generates regions shifted from original start coordinates
    # Original starts: 10 and 100.
    # Flanking should shift by multipliers of 5000, or fall back to random if bounds exceeded.
    # Since chrom_seq length is 800, flanking shift of 5000 is out of bounds, so it will fall back to random unmatched sampling.
    # Let's test bounds success with a long chromosome sequence
    chrom_seq_long = "ATCGATCG" * 10000 # length 80000
    negs_flanking_long = sample_negatives_for_chromosome(
        chrom='chr1',
        chrom_seq=chrom_seq_long,
        pos_regions_chrom=pos_regions,
        target_length=10,
        strategy='flanking'
    )
    assert len(negs_flanking_long) == 2
    # Ensure no overlaps with pos_intervals: 10-20 and 100-110
    for neg in negs_flanking_long:
        assert not (neg['start'] < 20 and neg['end'] > 10)
        assert not (neg['start'] < 110 and neg['end'] > 100)
