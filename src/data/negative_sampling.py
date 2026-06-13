import os
import random
import pandas as pd
import numpy as np

def calculate_gc_content(seq: str) -> float:
    """
    Calculate the GC content fraction of a DNA sequence.
    """
    if not seq:
        return 0.0
    # Count both upper and lower case G/C
    seq_upper = seq.upper()
    gc_count = seq_upper.count('G') + seq_upper.count('C')
    return gc_count / len(seq)

def is_overlapping(start: int, end: int, positive_intervals: list) -> bool:
    """
    Check if the interval [start, end] overlaps with any positive interval.
    Assumes positive_intervals is a sorted list of (start, end) tuples.
    Uses binary search for speed.
    """
    # Binary search to find the insertion point
    low = 0
    high = len(positive_intervals) - 1
    
    while low <= high:
        mid = (low + high) // 2
        p_start, p_end = positive_intervals[mid]
        
        # Check for overlap
        if start < p_end and end > p_start:
            return True
        elif start >= p_end:
            low = mid + 1
        else:
            high = mid - 1
            
    return False

def sample_negatives_for_chromosome(
    chrom: str,
    chrom_seq: str,
    pos_regions_chrom: pd.DataFrame,
    target_length: int = 1000,
    max_attempts_per_sample: int = 10000,
    gc_bins: np.ndarray = None
) -> list:
    """
    Sample GC-matched negative regions for a single chromosome.
    """
    print(f"Sampling negatives for chromosome {chrom}...")
    
    # Sort positive intervals for fast overlap checking
    pos_intervals = sorted(list(zip(pos_regions_chrom['start'], pos_regions_chrom['end'])))
    
    # Calculate GC content for all positives in this chromosome
    pos_regions_chrom = pos_regions_chrom.copy()
    if 'gc' not in pos_regions_chrom.columns:
        # If sequence is not present, we can't calculate GC. We assume sequence is present.
        pos_regions_chrom['gc'] = pos_regions_chrom['sequence'].apply(calculate_gc_content)
        
    pos_gcs = pos_regions_chrom['gc'].values
    
    # Define GC bins if not provided (default 20 bins from 0.0 to 1.0)
    if gc_bins is None:
        gc_bins = np.linspace(0.0, 1.0, 21)
        
    # Assign each positive to a GC bin
    pos_bin_indices = np.digitize(pos_gcs, gc_bins) - 1
    
    # Count how many negatives we need per bin
    bin_counts = pd.Series(pos_bin_indices).value_counts().to_dict()
    print(f"Target negative bin counts for {chrom}: {bin_counts}")
    
    negatives = []
    chrom_len = len(chrom_seq)
    
    # Track how many we have collected per bin
    collected_bins = {b: 0 for b in bin_counts.keys()}
    
    total_needed = len(pos_regions_chrom)
    attempts = 0
    max_total_attempts = total_needed * max_attempts_per_sample
    
    while len(negatives) < total_needed and attempts < max_total_attempts:
        attempts += 1
        
        # 1. Sample a random start coordinate
        start = random.randint(0, chrom_len - target_length)
        end = start + target_length
        
        # 2. Overlap check with positives
        if is_overlapping(start, end, pos_intervals):
            continue
            
        # 3. Extract candidate sequence
        seq = chrom_seq[start:end]
        
        # 4. Check for gaps / Ns (exclude if any Ns are found)
        if 'N' in seq:
            continue
            
        # 5. Compute GC content and find its bin
        gc = calculate_gc_content(seq)
        bin_idx = int(np.digitize(gc, gc_bins) - 1)
        
        # 6. Check if we still need a negative in this GC bin
        if bin_idx in bin_counts and collected_bins[bin_idx] < bin_counts[bin_idx]:
            negatives.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'accession': f"NEG_{chrom}_{len(negatives):06d}",
                'classification': 'non-regulatory',
                'sequence': seq,
                'gc': gc
            })
            collected_bins[bin_idx] += 1
            
    print(f"Sampled {len(negatives)} / {total_needed} negatives for {chrom} (attempts: {attempts}).")
    
    # If we failed to get a perfect match in some bins due to random sampling limits,
    # fill the remaining slots with any valid non-overlapping sequences
    remaining = total_needed - len(negatives)
    if remaining > 0:
        print(f"Notice: Filled {remaining} remaining negative slots with random coordinates due to sampling timeout.")
        fill_attempts = 0
        while len(negatives) < total_needed and fill_attempts < remaining * 100:
            fill_attempts += 1
            start = random.randint(0, chrom_len - target_length)
            end = start + target_length
            if is_overlapping(start, end, pos_intervals):
                continue
            seq = chrom_seq[start:end]
            if 'N' in seq:
                continue
            gc = calculate_gc_content(seq)
            negatives.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'accession': f"NEG_{chrom}_{len(negatives):06d}",
                'classification': 'non-regulatory',
                'sequence': seq,
                'gc': gc
            })
            
    return negatives

def build_negative_dataset(
    pos_tsv_path: str,
    fasta_source: str,
    output_path: str,
    target_length: int = 1000
):
    """
    Load positive regions, sample matching GC/chromosome negative regions, 
    and save the combined dataset.
    """
    print(f"Loading positive regions with sequences from {pos_tsv_path}...")
    pos_df = pd.read_csv(pos_tsv_path, sep='\t')
    
    # Ensure GC content is calculated for positives
    if 'gc' not in pos_df.columns:
        pos_df['gc'] = pos_df['sequence'].apply(calculate_gc_content)
        
    all_negatives = []
    
    # Load fasta source chromosomes as needed
    from .sequence_extractor import load_fasta_chromosome
    
    for chrom, group in pos_df.groupby('chrom'):
        # Locate chromosome FASTA file
        fasta_path = None
        if os.path.isdir(fasta_source):
            for ext in ['.fa', '.fa.gz', '.fasta', '.fasta.gz']:
                candidate = os.path.join(fasta_source, f"{chrom}{ext}")
                if os.path.exists(candidate):
                    fasta_path = candidate
                    break
        elif os.path.isfile(fasta_source):
            fasta_path = fasta_source
            
        if not fasta_path or not os.path.exists(fasta_path):
            print(f"Warning: Reference sequence for {chrom} not found. Skipping negative sampling for this chromosome.")
            continue
            
        try:
            chrom_seq = load_fasta_chromosome(fasta_path)
        except Exception as e:
            print(f"Error loading FASTA for {chrom}: {e}. Skipping negative sampling.")
            continue
            
        # Sample negatives
        chrom_negs = sample_negatives_for_chromosome(
            chrom=chrom,
            chrom_seq=chrom_seq,
            pos_regions_chrom=group,
            target_length=target_length
        )
        all_negatives.extend(chrom_negs)
        
    neg_df = pd.DataFrame(all_negatives)
    
    # Combine positives and negatives
    # Add a label column (1 for positive, 0 for negative)
    pos_df['label'] = 1
    neg_df['label'] = 0
    
    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
    
    # Save combined dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved combined positive & negative dataset to {output_path} (Total regions: {len(combined_df):,}).")
    return output_path
