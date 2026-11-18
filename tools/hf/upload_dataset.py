"""Upload the packaged dataset (output of Kaggle notebook irg-02-package-dataset) to a Hugging Face dataset repo.

Memory-light: fetches ONE file at a time from the Kaggle kernel output, uploads it, deletes it.
Credentials: uses the locally cached logins (`kaggle auth login`, `hf auth login`). No tokens are read or printed here.

    python tools/hf/upload_dataset.py --repo PxALabs/encode-regulatory-elements --workdir D:/_hf_staging
The repo is created PRIVATE; making it public is a deliberate manual step on huggingface.co.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

KERNEL = "purvanshj009/irg-02-package-dataset"
SMALL_FIRST = ["README.md", "splits.json", "validation_report.json", "MANIFEST.sha256"]
BIOS = ["tcell", "k562", "gm12878"]
STRAT = ["gc_matched", "random_unmatched", "flanking"]


def kaggle_fetch(name: str, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    pattern = r"(^|/)" + re.escape(name) + "$"  # matched against the relative path, e.g. irg_curated/README.md
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", KERNEL, "-p", str(dest), "-q", "--file-pattern", pattern], check=True)
    hits = [p for p in dest.rglob("*") if p.is_file() and p.name == Path(name).name]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one file for {name}, found {hits}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--only", default=None, help="regex; upload only matching paths (resume)")
    a = ap.parse_args()

    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    print("repo ready (private):", a.repo)
    existing = set(api.list_repo_files(a.repo, repo_type="dataset"))

    targets = list(SMALL_FIRST)
    targets += [f"{b}_{s}.parquet" for b in BIOS for s in STRAT]
    targets += [f"slm_ready/{b}_gc_matched.jsonl.gz" for b in BIOS]
    work = Path(a.workdir)
    for t in targets:
        if a.only and not re.search(a.only, t):
            continue
        if t in existing:
            print("skip (already uploaded):", t)
            continue
        f = kaggle_fetch(t, work / "dl")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=t, repo_id=a.repo, repo_type="dataset", commit_message=f"Add {t}")
        print(f"uploaded {t} ({f.stat().st_size/1e6:.1f} MB)")
        shutil.rmtree(work / "dl")
    print("done")


if __name__ == "__main__":
    main()
