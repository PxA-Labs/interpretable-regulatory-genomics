# kaggle/

Reproducible compute lives on Kaggle (free T4/P100 GPU). This folder holds the *metadata* only; data and outputs are Kaggle Datasets.

```
kaggle/
├── datasets/<slug>/        dataset-metadata.json + README.md (what is uploaded and from where)
└── kernels/<slug>/         kernel-metadata.json for each pushed notebook
```

## Run protocol (every notebook)

1. Smoke-test first: push a tiny version (small subsample, 1 epoch, 1 seed), check it, then push the full version.
2. Push: `kaggle kernels push -p kaggle/kernels/<slug>` (note the version number in `run.yaml`).
3. Pull only the small outputs and the log, never the whole working folder (it can be hundreds of MB):
   `kaggle kernels output <user>/<slug> -p <dir> --file-pattern '\.(csv|json)$'`.
   Large artefacts (weights, tables) are fetched by exact filename only when needed.
4. Create `experiments/runs/<run_id>/` with `run.yaml`, the log, and `results/`; write the analysis to `docs/results/`.
   The Kaggle CLI cannot download the rendered executed notebook (`__notebook__.ipynb`); the log plus the pinned
   `kernel_version` is the execution record.
5. Publish: add the run to the private run-record datasets `PxALabs/irg-experiment-runs` (Hugging Face, `hf upload`) and
   `<user>/irg-experiment-runs` (Kaggle, `kaggle datasets version --dir-mode zip`), regenerating `MANIFEST.sha256`.
   Datasets stay private until a maintainer reviews them and chooses the licence.
6. Open a PR with the notebook, run record and results doc. Reference the relevant issue.

Never commit Kaggle credentials. Authenticate locally with `kaggle auth login`.
