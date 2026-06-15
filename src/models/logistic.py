import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel


class LogisticRegulatoryModel(BaseModel):
    """
    Logistic Regression classifier for regulatory switch identification.
    Uses L1/L2 regularization for feature selection and coefficients for interpretability.
    """

    def __init__(self, **kwargs):
        # Set default random state if not provided
        if "random_state" not in kwargs:
            kwargs["random_state"] = 42
        # Set max_iter higher to ensure convergence on high-dimensional k-mers
        if "max_iter" not in kwargs:
            kwargs["max_iter"] = 1000

        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Return probability of active class (index 1)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Return coefficients as feature importances.
        Features are sorted by absolute coefficient magnitude (retaining direction).
        """
        coefs = self.model.coef_[0]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefs))]

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": coefs,
                "abs_importance": np.abs(coefs),
            }
        )

        # Sort by absolute coefficient size (impact strength)
        df = df.sort_values(by="abs_importance", ascending=False)
        return df.drop(columns=["abs_importance"]).reset_index(drop=True)
