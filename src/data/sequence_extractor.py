import os
import gzip
import pandas as pd

def load_fasta_chromosome(fasta_path: str) -> str:
    """
    Load a single chromosome sequence from a FASTA file (handles both plain text and gzipped files).
    """
    print(f"Loading chromosome sequence from {fasta_path}...")
    open_fn = gzip.open if fasta_path.endswith('.gz') else open
    seq_parts = []
    with open_fn(fasta_path, 'rt') as f:
        # Skip header line
        header = f.readline()
        if not header.startswith('>'):
            raise ValueError(f"Invalid FASTA file format in {fasta_path}. Missing '>' header.")
        
        # Read sequence lines
        for line in f:
            if line.startswith('>'):
                break  # Stop if another chromosome starts (should be single-chrom files)
            seq_parts.append(line.strip())
            
    # Combine lines and convert to uppercase
    chrom_seq = "".join(seq_parts).upper()
    print(f"Loaded sequence of length {len(chrom_seq):,} bp.")
    return chrom_seq

def extract_sequences(
    bed_path: str,
    fasta_source: str,
    output_path: str,
    max_n_fraction: float = 0.10
):
    """
    Extract DNA sequences for coordinates in a BED file from a reference genome fasta source.
    Filters out regions containing too many 'N' (unknown) bases.
    
    Parameters:
    -----------
    bed_path : str
        Path to the parsed BED file containing coordinates (chrom, start, end, accession, classification).
    fasta_source : str
        Path to a single FASTA file (e.g. hg38.fa) OR path to a directory containing 
        per-chromosome FASTA files (e.g., chr1.fa.gz, chr2.fa.gz).
    output_path : str
        Path to save the output tab-separated file containing coordinates and extracted sequences.
    max_n_fraction : float, default 0.10
        Maximum allowed fraction of 'N' (unknown) bases in a sequence. Regions exceeding this are excluded.
    """
    print(f"Reading BED coordinates from {bed_path}...")
    # Read processed BED file
    bed_df = pd.read_csv(
        bed_path, 
        sep='\t', 
        header=None, 
        names=['chrom', 'start', 'end', 'accession', 'classification']
    )
    print(f"Total input regions: {len(bed_df):,}")

    extracted_records = []
    skipped_n_count = 0
    skipped_boundary_count = 0

    # Group by chromosome to load each chromosome sequence into memory only once
    for chrom, group in bed_df.groupby('chrom'):
        # 1. Locate the FASTA file for this chromosome
        fasta_path = None
        if os.path.isdir(fasta_source):
            # Look for common extensions: .fa, .fa.gz, .fasta, .fasta.gz
            for ext in ['.fa', '.fa.gz', '.fasta', '.fasta.gz']:
                candidate = os.path.join(fasta_source, f"{chrom}{ext}")
                if os.path.exists(candidate):
                    fasta_path = candidate
                    break
        elif os.path.isfile(fasta_source):
            # If a single file was passed, we fall back to it
            fasta_path = fasta_source

        if not fasta_path or not os.path.exists(fasta_path):
            print(f"Warning: Reference sequence for {chrom} not found at source '{fasta_source}'. Skipping {len(group)} regions.")
            continue

        # 2. Load chromosome sequence
        try:
            chrom_seq = load_fasta_chromosome(fasta_path)
        except Exception as e:
            print(f"Error loading FASTA for {chrom}: {e}. Skipping this chromosome.")
            continue

        # 3. Extract sequences for all coordinates in this chromosome
        for _, row in group.iterrows():
            start, end = int(row['start']), int(row['end'])
            
            # Boundary check
            if start < 0 or end > len(chrom_seq):
                skipped_boundary_count += 1
                continue
                
            # Slice sequence
            seq = chrom_seq[start:end]
            
            # Quality control: check fraction of N-bases
            n_count = seq.count('N')
            if len(seq) > 0 and (n_count / len(seq)) > max_n_fraction:
                skipped_n_count += 1
                continue
                
            extracted_records.append({
                'chrom': row['chrom'],
                'start': start,
                'end': end,
                'accession': row['accession'],
                'classification': row['classification'],
                'sequence': seq
            })

    # Convert to DataFrame
    out_df = pd.DataFrame(extracted_records)
    print(f"Sequence extraction complete.")
    print(f"  Extracted regions: {len(out_df):,}")
    print(f"  Skipped (out of bounds): {skipped_boundary_count:,}")
    print(f"  Skipped (high N-base fraction): {skipped_n_count:,}")

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved extracted sequences to {output_path}.")
    return output_path
