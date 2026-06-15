from .base_model import BaseModel
from .logistic import LogisticRegulatoryModel
from .tree_ensemble import RandomForestRegulatoryModel, XGBoostRegulatoryModel
from .cnn import ShallowCNN, DeepCNN, AttentionCNN
from .train_nn import PyTorchModelWrapper


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

        def create_pytorch_model(model_class, **kwargs):
            # Separate wrapper parameters from model architecture parameters
            wrapper_keys = {
                "epochs",
                "batch_size",
                "lr",
                "weight_decay",
                "val_split",
                "patience",
                "device",
                "random_state",
            }
            wrapper_params = {}
            model_params = {}
            for k, v in kwargs.items():
                if k in wrapper_keys:
                    wrapper_params[k] = v
                else:
                    model_params[k] = v
            return PyTorchModelWrapper(
                model_class=model_class, model_params=model_params, **wrapper_params
            )

        registry = {
            "logistic_regression": LogisticRegulatoryModel,
            "random_forest": RandomForestRegulatoryModel,
            "xgboost": XGBoostRegulatoryModel,
            "shallow_cnn": lambda **kwargs: create_pytorch_model(ShallowCNN, **kwargs),
            "deep_cnn": lambda **kwargs: create_pytorch_model(DeepCNN, **kwargs),
            "attention_cnn": lambda **kwargs: create_pytorch_model(
                AttentionCNN, **kwargs
            ),
        }

        if name not in registry:
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(registry.keys())}"
            )

        return registry[name](**params)
