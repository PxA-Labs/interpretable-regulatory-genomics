import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Try importing shap, handle gracefully if not installed
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def explain_model_shap(
    model,
    X_explain: np.ndarray,
    feature_names: list,
    output_plot_path: str = None,
    max_display: int = 20,
    show_plot: bool = False,
) -> np.ndarray:
    """
    Compute SHAP values for a trained model and generate a global summary plot.

    Parameters:
    -----------
    model : BaseModel
        Trained model instance (Random Forest or XGBoost).
    X_explain : np.ndarray
        Feature matrix (N, num_features) to compute SHAP values for.
    feature_names : list
        List of k-mer strings or feature names.
    output_plot_path : str, optional
        Path where the SHAP summary plot will be saved.
    max_display : int, default 20
        Number of top features to show in the summary plot.
    show_plot : bool, default False
        If True, displays the plot in the active environment (e.g. Jupyter notebook).

    Returns:
    --------
    shap_values : np.ndarray
        Calculated SHAP values matrix of shape X_explain.shape.
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError(
            "shap package is not installed. Please run 'pip install shap'."
        )

    print("Initializing SHAP TreeExplainer...")

    # Check if the underlying model is a tree-based classifier
    # We pass the raw sklearn/xgboost object to the TreeExplainer
    raw_model = model.model

    explainer = shap.TreeExplainer(raw_model)
    print("Computing SHAP values (this may take a minute depending on model depth)...")

    # Calculate SHAP values
    # For binary classification, TreeExplainer returns a list of arrays [class_0_shap, class_1_shap]
    # for sklearn Random Forest, or a single array for xgboost
    raw_shap = explainer.shap_values(X_explain)

    if isinstance(raw_shap, list):
        # In multi-class or sklearn binary RF, we take class 1 (regulatory class)
        shap_values = raw_shap[1]
    else:
        # For xgboost binary, it is a single array of shape (N, num_features)
        shap_values = raw_shap

    print("SHAP values calculated successfully.")

    # Plot summary figure if path is provided
    if output_plot_path:
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

        plt.figure(figsize=(10, 6.5))
        # Plot beeswarm summary plot
        # Using shap.summary_plot directly
        shap.summary_plot(
            shap_values,
            X_explain,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )

        # Save figure
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close()
        print(f"SHAP summary plot saved to {output_plot_path}.")

    return shap_values


def get_top_shap_features(
    shap_values: np.ndarray, feature_names: list, top_n: int = 20
) -> pd.DataFrame:
    """
    Summarize the global impact of features by taking the mean absolute SHAP value.
    """
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    return (
        df.sort_values(by="mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def explain_single_sample(
    model,
    X_sample: np.ndarray,
    feature_names: list,
    output_plot_path: str = None,
    show_plot: bool = False,
):
    """
    Generate and save a SHAP explanation plot (waterfall/force plot) for a single sample.

    Parameters:
    -----------
    model : BaseModel
        Trained model instance (Random Forest or XGBoost).
    X_sample : np.ndarray
        Feature vector for a single sample, shape (1, num_features) or (num_features,).
    feature_names : list
        List of k-mer strings or feature names.
    output_plot_path : str, optional
        Path where the SHAP plot will be saved.
    show_plot : bool, default False
        If True, displays the plot in the active environment.
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError(
            "shap package is not installed. Please run 'pip install shap'."
        )

    print("Initializing SHAP TreeExplainer...")
    raw_model = model.model
    explainer = shap.TreeExplainer(raw_model)

    # Ensure shape is (1, num_features)
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)

    print("Computing SHAP explanation for the single sample...")

    try:
        # Explainer __call__ returns an Explanation object (preferred for waterfall)
        explanation = explainer(X_sample)

        # If binary classification returns list (like sklearn RF)
        if isinstance(explanation.base_values, list) or (explanation.values.ndim == 3):
            # We select class 1 (regulatory class)
            val = explanation.values[0, :, 1]
            base_val = explanation.base_values[0, 1]
            data = explanation.data[0]
        else:
            # For xgboost, it's (num_samples, num_features)
            val = explanation.values[0]
            base_val = explanation.base_values[0]
            data = explanation.data[0]

        exp_obj = shap.Explanation(
            values=val, base_values=base_val, data=data, feature_names=feature_names
        )

        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(exp_obj, show=False)

    except Exception as e:
        print(
            f"Warning: explanation object plotting failed: {e}. Falling back to standard force/beeswarm."
        )
        # Fallback to manual force plot or summary plotting
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 5))
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]

        shap.plots.force(
            expected_value,
            shap_values[0],
            X_sample[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )

    if output_plot_path:
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        print(f"SHAP explanation plot saved to {output_plot_path}.")

    if show_plot:
        plt.show()
    else:
        plt.close()
