import os
import pandas as pd


def parse_and_resize_ccres(
    input_path: str,
    output_path: str,
    element_types: list = None,
    target_length: int = 1000,
    chromosomes: list = None,
):
    """
    Parse a cCRE BED file, filter by element classification/chromosomes,
    and resize each region to a fixed target length centered around the original region's midpoint.

    Parameters:
    -----------
    input_path : str
        Path to the raw BED file (can be gzipped .bed.gz).
        Expected ENCODE format: first 3 columns are chrom, start, end,
        and column 10 (or 4, depending on the file) contains the classification.
    output_path : str
        Path where the processed and resized BED file will be saved.
    element_types : list, optional
        List of classifications to keep (e.g., ['PLS', 'dELS', 'pELS']).
        If None, all classifications are kept.
    target_length : int, default 1000
        The fixed length (in base pairs) to resize the regions to.
    chromosomes : list, optional
        List of chromosomes to keep (e.g., ['chr1', 'chr2']).
        If None, all chromosomes are kept.
    """
    print(f"Parsing cCRE annotations from {input_path}...")

    # Read BED file. We only require the first 4 columns, plus column 10 for classification if present.
    # We read without a header.
    # ENCODE cCRE BED files are tab-separated.
    try:
        # We read up to column 10 (0-indexed 9) for classification
        df = pd.read_csv(
            input_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2, 3, 9],
            names=["chrom", "start", "end", "accession", "classification"],
            dtype={
                "chrom": str,
                "start": int,
                "end": int,
                "accession": str,
                "classification": str,
            },
        )
    except Exception as e:
        print(
            f"Warning: Failed to load column 10, trying standard 4-column BED format: {e}"
        )
        df = pd.read_csv(
            input_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2, 3],
            names=["chrom", "start", "end", "accession"],
            dtype={"chrom": str, "start": int, "end": int, "accession": str},
        )
        df["classification"] = "unknown"

    initial_count = len(df)
    print(f"Loaded {initial_count} initial regions.")

    # 1. Filter by Chromosomes
    if chromosomes is not None:
        df = df[df["chrom"].isin(chromosomes)]
        print(f"Filtered by chromosomes. Remaining: {len(df)}")

    # 2. Filter by regulatory element classification type
    if element_types is not None and "classification" in df.columns:
        # Standardize matching (some files contain e.g., "promoter-like" or "PLS")
        # We perform a case-insensitive sub-string or exact match
        df = df[df["classification"].isin(element_types)]
        print(f"Filtered by element types {element_types}. Remaining: {len(df)}")

    # 3. Center and Resize to fixed target length
    print(f"Resizing regions to fixed length of {target_length} bp...")
    midpoints = df["start"] + (df["end"] - df["start"]) // 2
    df["start"] = midpoints - target_length // 2
    df["end"] = df["start"] + target_length

    # Ensure coordinates are non-negative
    df = df[df["start"] >= 0]
    print(f"Filtered out regions with invalid coordinates. Remaining: {len(df)}")

    # 4. Save to tab-separated BED format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, sep="\t", header=False, index=False)
    print(f"Processed BED file saved to {output_path} (Total regions: {len(df)}).")
    return output_path
