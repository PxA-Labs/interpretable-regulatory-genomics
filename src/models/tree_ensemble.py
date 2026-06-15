import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

# Try importing xgboost, handle gracefully if not installed
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class RandomForestRegulatoryModel(BaseModel):
    """
    Random Forest classifier for regulatory sequence classification.
    Captures non-linear interactions between k-mer sequence features.
    """

    def __init__(self, **kwargs):
        if "random_state" not in kwargs:
            kwargs["random_state"] = 42
        if "n_estimators" not in kwargs:
            kwargs["n_estimators"] = 500
        if "n_jobs" not in kwargs:
            kwargs["n_jobs"] = -1  # Parallelize across all CPU cores

        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Return Gini importances.
        """
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


class XGBoostRegulatoryModel(BaseModel):
    """
    XGBoost gradient boosting classifier for regulatory sequence classification.
    Highly optimized tree ensemble capturing complex k-mer combinations.
    """

    def __init__(self, **kwargs):
        if not XGB_AVAILABLE:
            raise RuntimeError(
                "xgboost package is not installed. Please run 'pip install xgboost'."
            )

        if "random_state" not in kwargs:
            kwargs["random_state"] = 42
        if "n_estimators" not in kwargs:
            kwargs["n_estimators"] = 500
        if "n_jobs" not in kwargs:
            kwargs["n_jobs"] = -1
        if "eval_metric" not in kwargs:
            kwargs["eval_metric"] = "logloss"

        self.model = xgb.XGBClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Return gain-based feature importances.
        """
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)
