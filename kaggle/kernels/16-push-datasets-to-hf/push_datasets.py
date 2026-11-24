# %% [markdown]
# # 16 - publish the two Kaggle-only datasets to Hugging Face from inside Kaggle (no local download)
# Needs a Kaggle Secret HF_TOKEN (write access) attached to this notebook; never printed.
# * PxALabs/irg-repeat-coverage : per-window RepeatMasker fractions (public).
# * PxALabs/irg-jaspar2024      : unmodified JASPAR 2024 CORE vertebrates non-redundant MEME copy (created PRIVATE until the licence/redistribution question is settled by the owner).

# %%
import glob, hashlib, shutil
from pathlib import Path
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
CODE = "github.com/PxA-Labs/interpretable-regulatory-genomics"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def push(repo, private, src_glob, card):
    files = glob.glob(src_glob, recursive=True)
    assert files, src_glob
    st = Path("/kaggle/working") / repo.split("/")[1]
    st.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, st / Path(f).name)
    (st / "README.md").write_text(card)
    (st / "MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.name}\n" for p in sorted(st.iterdir()) if p.name not in ("README.md", "MANIFEST.sha256")))
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(st), repo_id=repo, repo_type="dataset", commit_message="initial upload")
    print("uploaded", repo, "private" if private else "public", [p.name for p in st.iterdir()])


push("PxALabs/irg-repeat-coverage", False, "/kaggle/input/**/repeat_fractions.parquet", f"""---
license: other
tags: [genomics, regulatory-elements, ENCODE, repeatmasker, annotation]
---
# Repeat coverage for ENCODE regulatory elements v2.1 (GC-matched tables)

Per-row fraction of the 1,000 bp window covered by UCSC RepeatMasker (hg38 `rmsk`) annotations: all repeats (`rep_frac`), Alu (`alu_frac`), L1 (`l1_frac`),
and simple/low-complexity (`simple_frac`). Join on `id` (+ `biosample`, `negative_strategy`) with the `*_gc_matched.parquet` tables of `PxALabs/encode-regulatory-elements` v2.1.

- Source: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz (Last-Modified 2022-10-19, 155,633,856 bytes). Check UCSC's terms of use for the annotation.
- Code: `tools/data/repeat_annotation.py` in {CODE}
- Covers only the three `gc_matched` tables (328,505 rows). Coordinates are 0-based half-open, as in the tables.
- parquet sha256: 4c8ee141bca908beed14d4d4ee91ca6d243c7512822b47b6cf2abef842da580e (also in `MANIFEST.sha256`).
- Used by notebooks 08, 09, 11, 13, 14, 15 for the repeat-matched test set.
""")

push("PxALabs/irg-jaspar2024", True, "/kaggle/input/**/jaspar2024_core_vert_nr.meme", f"""---
license: cc-by-4.0
tags: [genomics, transcription-factors, motifs, JASPAR]
---
# JASPAR 2024 CORE, vertebrates, non-redundant (MEME format)

Unmodified copy of `JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt` (879 motifs) downloaded 2026-11-19 from https://jaspar.elixir.no/download/data/2024/CORE/.
SHA-256: dd494278d356a4e170908c74d6d8eb746a2c7504d6210cd51017141f21a65b18.

Mirror for offline use in Kaggle notebooks of {CODE} (motif-FDR analysis, notebook 10). JASPAR states it is licensed under CC BY 4.0; cite
Rauluseviciute I, et al. JASPAR 2024: 20th anniversary of the open-access database of transcription factor binding profiles. Nucleic Acids Res. 2024;52(D1):D174-D182. doi:10.1093/nar/gkad1059.
All credit to the JASPAR database. The JASPAR pages do not explicitly address mirroring; this repository is private until that is confirmed.
""")
