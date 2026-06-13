from abc import ABC, abstractmethod
import pickle
import pandas as pd

class BaseModel(ABC):
    """
    Abstract base class for all regulatory sequence classifiers.
    Provides standard ML operations and enforces interpretability interfaces.
    """
    @abstractmethod
    def fit(self, X, y):
        """
        Train the model on features X and labels y.
        """
        pass
        
    @abstractmethod
    def predict(self, X):
        """
        Predict binary labels (0 or 1) for features X.
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict probability of active regulatory class (1) for features X.
        """
        pass
        
    @abstractmethod
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Extract model-derived feature importances.
        
        Returns:
        --------
        pd.DataFrame
            Sorted DataFrame with columns ['feature', 'importance']
        """
        pass
        
    def save(self, path: str):
        """
        Serialize model instance to disk.
        """
        print(f"Saving model to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path: str):
        """
        Deserialize model instance from disk.
        """
        print(f"Loading model from {path}...")
        with open(path, 'rb') as f:
            return pickle.load(f)
