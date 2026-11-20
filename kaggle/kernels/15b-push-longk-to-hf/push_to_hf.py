# %% [markdown]
# # 15b - publish notebook-15 models to a private Hugging Face repo from inside Kaggle (no local download)
# Needs a Kaggle Secret named HF_TOKEN (write access) attached to this notebook. The token is read from the secret and never printed.
# Uploads: the .ubj models of kernel irg-15-longk-xgb, a SHA-256 manifest, and the small result tables. Repo is created PRIVATE.

# %%
import glob, hashlib, json, os, shutil
from pathlib import Path
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

REPO = "PxALabs/irg-xgboost-longk"
tok = UserSecretsClient().get_secret("HF_TOKEN")
api = HfApi(token=tok)
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
models = sorted(glob.glob("/kaggle/input/**/models/*.ubj", recursive=True))
assert models, "no .ubj found; attach kernel irg-15-longk-xgb"
stage = Path("/kaggle/working/stage")
(stage / "models").mkdir(parents=True, exist_ok=True)
man = {}
for p in models:
    shutil.copy(p, stage / "models" / Path(p).name)
    man[Path(p).name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
for p in glob.glob("/kaggle/input/**/results/*.csv", recursive=True) + glob.glob("/kaggle/input/**/results/protocol.json", recursive=True):
    (stage / "results").mkdir(exist_ok=True)
    shutil.copy(p, stage / "results" / Path(p).name)
(stage / "MANIFEST.sha256").write_text("".join(f"{h}  models/{n}\n" for n, h in sorted(man.items())))
api.upload_folder(folder_path=str(stage), repo_id=REPO, repo_type="model", commit_message="notebook 15 models, results tables, manifest")
print("uploaded", len(man), "models to", REPO, "(private)")
