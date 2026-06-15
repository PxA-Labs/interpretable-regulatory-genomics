import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import pandas as pd
import pickle

from src.models.base_model import BaseModel
from src.data.dataset import GenomicDataset

class PyTorchModelWrapper(BaseModel):
    """
    Unified wrapper that brings PyTorch Neural Network models under the standard
    BaseModel interface (fit, predict, predict_proba, save, load).
    """
    def __init__(self, model_class, model_params=None, epochs=25, batch_size=64, 
                 lr=0.001, weight_decay=1e-5, val_split=0.15, patience=5,
                 device=None, random_state=42):
        """
        Parameters:
        -----------
        model_class : class
            Uninstantiated PyTorch nn.Module class.
        model_params : dict, optional
            Arguments to pass to PyTorch model constructor.
        epochs : int, default 25
            Maximum training epochs.
        batch_size : int, default 64
            Mini-batch size.
        lr : float, default 0.001
            Learning rate.
        weight_decay : float, default 1e-5
            L2 regularization weight decay.
        val_split : float, default 0.15
            Fraction of training data used for early stopping validation.
        patience : int, default 5
            Early stopping patience.
        device : str, optional
            'cuda' or 'cpu'. If None, auto-detected.
        random_state : int, default 42
            Seed for reproducibility.
        """
        self.model_class = model_class
        self.model_params = model_params if model_params is not None else {}
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_split = val_split
        self.patience = patience
        self.random_state = random_state
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Instantiate model
        self.model = self.model_class(**self.model_params).to(self.device)
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the PyTorch neural network.
        
        Parameters:
        -----------
        X : list, pd.Series, np.ndarray, or torch.Tensor
            Sequence strings of shape (N,) or pre-computed one-hot tensors of shape (N, 4, 1000).
        y : np.ndarray or torch.Tensor
            Binary labels of shape (N,).
        X_val : list, pd.Series, np.ndarray, or torch.Tensor, optional
            Validation sequences or one-hot tensors.
        y_val : np.ndarray or torch.Tensor, optional
            Validation labels.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # 1. Setup Train and Validation Datasets
        train_dataset = self._create_dataset(X, y)
        
        if X_val is not None and y_val is not None:
            val_dataset = self._create_dataset(X_val, y_val)
        elif self.val_split > 0:
            val_len = int(len(train_dataset) * self.val_split)
            train_len = len(train_dataset) - val_len
            train_dataset, val_dataset = random_split(
                train_dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(self.random_state)
            )
        else:
            val_dataset = None
            
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset else None
        
        # 2. Setup Optimization
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        # 3. Training Loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation Step
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_x.size(0)
                        
                        preds = (torch.sigmoid(outputs) >= 0.5).float()
                        correct += (preds == batch_y).sum().item()
                        total += batch_y.size(0)
                        
                val_loss /= len(val_loader.dataset)
                val_acc = correct / total
                
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch:02d}/{self.epochs:02d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")
                
                # Check for Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch}.")
                        break
            else:
                print(f"Epoch {epoch:02d}/{self.epochs:02d} | Train Loss: {train_loss:.4f}")
                
        # Load best weights
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            print(f"Loaded best validation model (Loss: {best_val_loss:.4f}).")
            
        return self
        
    def predict_proba(self, X):
        """
        Predict probability of active regulatory class (1) for input sequences.
        """
        self.model.eval()
        dataset = self._create_dataset(X, y=None)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs_batch = torch.sigmoid(outputs).cpu().numpy()
                probs.extend(probs_batch)
                
        return np.array(probs, dtype=np.float32)
        
    def predict(self, X):
        """
        Predict binary labels (0 or 1) for input sequences.
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.int32)
        
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Feature importance is not directly defined on input variables for CNNs.
        Returns a placeholder DataFrame. Saliency maps should be used instead.
        """
        print("Warning: Feature importance coefficients are not supported directly for CNN models. "
              "Use saliency maps or integrated gradients for sequence-level attributions.")
        return pd.DataFrame(columns=['feature', 'importance'])
        
    def _create_dataset(self, X, y=None):
        """
        Utility to create GenomicDataset or standard TensorDataset based on input types.
        """
        # If X is a DataFrame, Series, list, or ndarray of string type
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            df = pd.DataFrame(X)
            # Find the sequence column or name it
            seq_col = df.columns[0]
            if y is not None:
                df['label'] = y
                return GenomicDataset(df, sequence_col=seq_col, label_col='label')
            else:
                return GenomicDataset(df, sequence_col=seq_col, label_col=None)
        elif isinstance(X, (list, np.ndarray)) and (isinstance(X[0], str) or isinstance(X[0], np.str_)):
            df = pd.DataFrame({'sequence': X})
            if y is not None:
                df['label'] = y
                return GenomicDataset(df, sequence_col='sequence', label_col='label')
            else:
                return GenomicDataset(df, sequence_col='sequence', label_col=None)
        # If X is already encoded numeric array or tensor
        else:
            if isinstance(X, np.ndarray):
                x_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                x_tensor = X.clone().detach().float()
                
            if y is not None:
                if isinstance(y, np.ndarray):
                    y_tensor = torch.tensor(y, dtype=torch.float32)
                else:
                    y_tensor = y.clone().detach().float()
                return TensorDataset(x_tensor, y_tensor)
            else:
                return TensorDataset(x_tensor)
                
    def save(self, path: str):
        """
        Custom serialization for PyTorch wrapper saving weights and hyper-parameters.
        """
        print(f"Saving PyTorch wrapper model to {path}...")
        save_dict = {
            'model_class': self.model_class,
            'model_params': self.model_params,
            'model_state_dict': self.model.state_dict(),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'val_split': self.val_split,
            'patience': self.patience,
            'random_state': self.random_state
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)  # nosec B301 – internal checkpoint, trusted path
            
    @classmethod
    def load(cls, path: str):
        """
        Custom deserialization loading weights and reconstructing wrapper.

        Security note: pickle.load() is used here intentionally. Checkpoint
        files are produced only by our own save() method within a controlled
        research environment — never from user-supplied or network-sourced paths.
        """
        print(f"Loading PyTorch wrapper model from {path}...")
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)  # nosec B301 – internal checkpoint, trusted path
            
        # Initialize wrapper instance
        wrapper = cls(
            model_class=save_dict['model_class'],
            model_params=save_dict['model_params'],
            epochs=save_dict['epochs'],
            batch_size=save_dict['batch_size'],
            lr=save_dict['lr'],
            weight_decay=save_dict['weight_decay'],
            val_split=save_dict['val_split'],
            patience=save_dict['patience'],
            random_state=save_dict['random_state']
        )
        # Load weights
        wrapper.model.load_state_dict(save_dict['model_state_dict'])
        return wrapper
