import os
import urllib.request
import shutil


def download_file(url: str, dest_path: str):
    """
    Download a file from a URL to a local destination.
    """
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Custom User-Agent to avoid getting blocked by standard server policies
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    print("Download complete.")


def download_encode_ccres(output_dir: str = "data/raw"):
    """
    Download the ENCODE registry of human candidate cis-Regulatory Elements (cCREs) on hg38.
    Default file is the ENCODE SCREEN V4 registry: ENCFF726XBE.bed.gz
    """
    url = (
        "https://www.encodeproject.org/files/ENCFF726XBE/@@download/ENCFF726XBE.bed.gz"
    )
    dest_path = os.path.join(output_dir, "ENCFF726XBE.bed.gz")
    download_file(url, dest_path)
    return dest_path


def download_hg38_chromosome(chrom: str, output_dir: str = "data/reference"):
    """
    Download a single chromosome FASTA file from UCSC hg38 goldenPath.
    E.g. chrom="chr1" -> downloads chr1.fa.gz
    """
    url = f"https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{chrom}.fa.gz"
    dest_path = os.path.join(output_dir, f"{chrom}.fa.gz")
    download_file(url, dest_path)
    return dest_path
