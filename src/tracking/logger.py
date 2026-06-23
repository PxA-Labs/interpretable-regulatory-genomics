"""
Lightweight experiment tracking module for file-based MLOps.

Generates config.yaml and metrics.json files conforming to the specification
in docs/08-experiment-tracking-mlops.md. Designed to be called from notebooks
at the end of an experiment to persist all metadata and results.

Usage:
    from src.tracking import ExperimentLogger

    logger = ExperimentLogger(
        experiment_name="exp_001_logistic_kmer4",
        base_dir="experiments"
    )

    logger.log_config(
        model_family="logistic_regression",
        hyperparameters={"C": 1.0, "max_iter": 1000},
        data_config={"cell_type": "K562", "dataset_version": "cCRE_v3_2026-06-07"},
        feature_config={"type": "kmer", "k_values": [4]},
        split_version="chrom_holdout_v0.1.0",
        random_seed=42,
    )

    logger.log_metrics(
        test_metrics={"auroc": 0.78, "f1_macro": 0.75},
        val_metrics={"auroc": 0.80},
        training_time_seconds=120,
    )
"""

import os
import json
import yaml
from datetime import datetime, timezone


class ExperimentLogger:
    """
    File-based experiment logger that generates config.yaml and metrics.json
    under experiments/logs/{experiment_name}/.

    All parameters are optional with sensible defaults to support both
    full experiment logging and lightweight retroactive logging.
    """

    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        """
        Initialize the logger for a given experiment.

        Parameters
        ----------
        experiment_name : str
            Experiment identifier following the naming convention:
            exp_{NNN}_{model}_{features}[_{variant}]
        base_dir : str, default "experiments"
            Root directory for experiment outputs.
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.log_dir = os.path.join(base_dir, "logs", experiment_name)
        self.model_dir = os.path.join(base_dir, "models", experiment_name)
        self.figure_dir = os.path.join(base_dir, "figures", experiment_name)

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)

    def log_config(
        self,
        model_family: str,
        hyperparameters: dict = None,
        data_config: dict = None,
        feature_config: dict = None,
        split_version: str = "chrom_holdout_v0.1.0",
        random_seed: int = 42,
    ):
        """
        Save experiment configuration to config.yaml.

        Parameters
        ----------
        model_family : str
            Model type (e.g., "logistic_regression", "random_forest", "xgboost",
            "shallow_cnn", "attention_cnn", "nucleotide_transformer").
        hyperparameters : dict, optional
            Model hyperparameters (e.g., {"n_estimators": 500, "max_depth": 20}).
        data_config : dict, optional
            Data configuration (e.g., {"cell_type": "K562",
            "dataset_version": "cCRE_v3_2026-06-07",
            "element_types": ["dELS", "PLS"],
            "negative_strategy": "gc_matched"}).
        feature_config : dict, optional
            Feature configuration (e.g., {"type": "kmer", "k_values": [4],
            "include_gc": True}).
        split_version : str, default "chrom_holdout_v0.1.0"
            Split definition version identifier.
        random_seed : int, default 42
            Random seed used for reproducibility.
        """
        config = {
            "experiment_name": self.experiment_name,
            "model": {"family": model_family},
            "random_seed": random_seed,
        }

        if hyperparameters:
            config["model"].update(hyperparameters)

        if data_config:
            config["data"] = data_config

        if feature_config:
            config["features"] = feature_config

        config["split"] = {
            "version": split_version,
            "train_chroms": [
                "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7",
                "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14",
                "chrX",
            ],
            "val_chroms": ["chr15", "chr16", "chr17", "chr18"],
            "test_chroms": ["chr19", "chr20", "chr21", "chr22"],
        }

        config_path = os.path.join(self.log_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Config saved to {config_path}")
        return config_path

    def log_metrics(
        self,
        test_metrics: dict = None,
        val_metrics: dict = None,
        cv_metrics: dict = None,
        training_time_seconds: float = None,
        peak_memory_mb: float = None,
        environment: dict = None,
        notes: str = None,
    ):
        """
        Save evaluation metrics to metrics.json.

        Parameters
        ----------
        test_metrics : dict, optional
            Test set metrics (e.g., {"auroc": 0.88, "auprc": 0.85,
            "accuracy": 0.82, "f1_macro": 0.81, "precision": 0.83,
            "recall": 0.79}).
        val_metrics : dict, optional
            Validation set metrics (same keys as test_metrics).
        cv_metrics : dict, optional
            Cross-validation metrics (e.g., {"auroc_mean": 0.85,
            "auroc_std": 0.02}).
        training_time_seconds : float, optional
            Total training time in seconds.
        peak_memory_mb : float, optional
            Peak memory usage in megabytes.
        environment : dict, optional
            Environment info (e.g., {"platform": "kaggle", "gpu": "T4",
            "python_version": "3.10.12"}).
        notes : str, optional
            Free-text notes about the experiment.
        """
        metrics_data = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": {},
        }

        if test_metrics:
            metrics_data["metrics"]["test"] = test_metrics

        if val_metrics:
            metrics_data["metrics"]["validation"] = val_metrics

        if cv_metrics:
            metrics_data["metrics"]["cv_5fold"] = cv_metrics

        if training_time_seconds is not None:
            metrics_data["training_time_seconds"] = round(training_time_seconds, 1)

        if peak_memory_mb is not None:
            metrics_data["peak_memory_mb"] = round(peak_memory_mb, 1)

        if environment:
            metrics_data["environment"] = environment

        if notes:
            metrics_data["notes"] = notes

        metrics_path = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Metrics saved to {metrics_path}")
        return metrics_path

    def log_notes(self, content: str):
        """
        Save free-text experiment notes to notes.md.

        Parameters
        ----------
        content : str
            Markdown-formatted notes about the experiment.
        """
        notes_path = os.path.join(self.log_dir, "notes.md")
        with open(notes_path, "w") as f:
            f.write(f"# {self.experiment_name}\n\n")
            f.write(content)

        print(f"Notes saved to {notes_path}")
        return notes_path

    def get_figure_path(self, filename: str) -> str:
        """
        Return the full path for saving a figure artifact.

        Parameters
        ----------
        filename : str
            Figure filename (e.g., "confusion_matrix.png", "roc_curve.png").

        Returns
        -------
        str
            Full path under experiments/figures/{experiment_name}/{filename}.
        """
        return os.path.join(self.figure_dir, filename)

    def get_model_path(self, filename: str = "model.pkl") -> str:
        """
        Return the full path for saving a model artifact.

        Parameters
        ----------
        filename : str, default "model.pkl"
            Model filename (e.g., "model.pkl", "model.pt").

        Returns
        -------
        str
            Full path under experiments/models/{experiment_name}/{filename}.
        """
        return os.path.join(self.model_dir, filename)

    def summary(self) -> str:
        """
        Print a summary of what has been logged for this experiment.
        """
        files = []
        for subdir in [self.log_dir, self.model_dir, self.figure_dir]:
            if os.path.exists(subdir):
                for f in os.listdir(subdir):
                    files.append(os.path.join(subdir, f))

        summary_text = f"Experiment: {self.experiment_name}\n"
        summary_text += f"Log directory: {self.log_dir}\n"
        summary_text += f"Artifacts logged: {len(files)}\n"
        for f in sorted(files):
            summary_text += f"  - {f}\n"

        print(summary_text)
        return summary_text
