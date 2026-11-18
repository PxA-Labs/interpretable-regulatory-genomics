# kaggle/

Reproducible compute lives on Kaggle (free T4/P100 GPU). This folder holds the *metadata* only; data and outputs are Kaggle Datasets.

```
kaggle/
├── datasets/<slug>/        dataset-metadata.json + README.md (what is uploaded and from where)
└── kernels/<slug>/         kernel-metadata.json for each pushed notebook
```

## Run protocol (every notebook)

1. Push the notebook: `kaggle kernels push -p kaggle/kernels/<slug>`.
2. Pull executed output: `kaggle kernels output <user>/<slug> -p experiments/runs/<run_id>/output`.
3. Save the *executed* notebook copy to `experiments/runs/<run_id>/<name>.executed.ipynb`.
4. Publish `experiments/runs/<run_id>/output` as a new Kaggle Dataset version for reuse.
5. Record `config.yaml` + `metrics.json` + git SHA + dataset versions in `experiments/REGISTRY.md`.

Never commit Kaggle credentials. Authenticate locally with `kaggle auth login`.
