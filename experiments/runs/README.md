# experiments/runs/

One folder per executed Kaggle run: `<YYYYMMDD>_<kernel-slug>/` containing
`<name>.executed.ipynb`, `output/` (metrics, figures, predictions), and `run.yaml`
(git SHA, dataset versions, seeds, runtime). Large artifacts are published as Kaggle Datasets; only small text/JSON/figures are committed.
