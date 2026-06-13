from .base_model import BaseModel
from .logistic import LogisticRegulatoryModel
from .tree_ensemble import RandomForestRegulatoryModel, XGBoostRegulatoryModel

class ModelRegistry:
    """
    Factory registry class to instantiate and fetch model families by name.
    """
    @staticmethod
    def get_model(name: str, params: dict = None) -> BaseModel:
        """
        Retrieve and instantiate a model class by registration name.
        """
        if params is None:
            params = {}
            
        registry = {
            "logistic_regression": LogisticRegulatoryModel,
            "random_forest": RandomForestRegulatoryModel,
            "xgboost": XGBoostRegulatoryModel,
        }
        
        if name not in registry:
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(registry.keys())}"
            )
            
        return registry[name](**params)
