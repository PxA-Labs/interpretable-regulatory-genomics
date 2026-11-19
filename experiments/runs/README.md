# experiments/runs/

One folder per executed Kaggle run: `<YYYYMMDD>_<kernel-slug>/` containing `run.yaml` (kernel, pinned kernel version, inputs, seeds,
accelerator, runtime, notes), the execution log (`<slug>.log`), and `results/` (small CSV/JSON outputs and `protocol.json`).
Large artefacts are not committed: model weights go to a Hugging Face model repository (one repository per experiment), and the
run records are also published as the datasets `PxALabs/irg-experiment-runs` (Hugging Face) and `purvanshj009/irg-experiment-runs`
(Kaggle). The executed notebook file itself cannot be downloaded with the Kaggle CLI, so the log plus the pinned kernel version is
the execution record. See `experiments/REGISTRY.md` for the index and `kaggle/README.md` for the run protocol.
