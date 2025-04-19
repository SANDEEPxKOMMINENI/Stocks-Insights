import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Any, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from pathlib import Path

class StockModelTrainer:
    """
    Trainer class for stock prediction models
    """
    def __init__(self, model: nn.Module, 
                device: Optional[torch.device] = None,
                experiment_name: str = None):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            device: The device to use for training
            experiment_name: Name of the experiment for saving results
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set experiment name with timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"stock_model_{timestamp}"
        self.experiment_name = experiment_name
        
        # Create directories for experiment
        self.experiment_dir = os.path.join("experiments", experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "plots"), exist_ok=True)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_metrics': {},
            'hyperparameters': {},
            'epochs_trained': 0
        }
        
    def create_dataloaders(self, 
                          data_dict: Dict[str, np.ndarray], 
                          batch_size: int = 64,
                          shuffle_train: bool = True) -> Dict[str, DataLoader]:
        """
        Create DataLoaders from numpy arrays
        
        Args:
            data_dict: Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test
            batch_size: Batch size for DataLoaders
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Dictionary with train, val, test DataLoaders
        """
        dataloaders = {}
        
        # Training data
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).to(self.device),
            torch.tensor(y_train, dtype=torch.float32).to(self.device)
        )
        dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        
        # Validation data
        if 'X_val' in data_dict and 'y_val' in data_dict:
            X_val, y_val = data_dict['X_val'], data_dict['y_val']
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32).to(self.device),
                torch.tensor(y_val, dtype=torch.float32).to(self.device)
            )
            dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Test data
        if 'X_test' in data_dict and 'y_test' in data_dict:
            X_test, y_test = data_dict['X_test'], data_dict['y_test']
            test_dataset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32).to(self.device),
                torch.tensor(y_test, dtype=torch.float32).to(self.device)
            )
            dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
        return dataloaders
        
    def train(self, dataloaders: Dict[str, DataLoader], 
             epochs: int = 100, 
             lr: float = 0.001,
             criterion: nn.Module = None,
             optimizer: optim.Optimizer = None,
             scheduler: optim.lr_scheduler._LRScheduler = None,
             early_stopping_patience: int = 10,
             save_best_model: bool = True,
             verbose: bool = True) -> Dict:
        """
        Train the model
        
        Args:
            dataloaders: Dictionary with train, val, test DataLoaders
            epochs: Number of training epochs
            lr: Learning rate
            criterion: Loss function (default: MSELoss)
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler
            early_stopping_patience: Patience for early stopping
            save_best_model: Whether to save the best model
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        # Default loss criterion
        if criterion is None:
            criterion = nn.MSELoss()
            
        # Default optimizer
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
        # Save hyperparameters
        self.history['hyperparameters'] = {
            'epochs': epochs,
            'learning_rate': lr,
            'batch_size': dataloaders['train'].batch_size,
            'optimizer': optimizer.__class__.__name__,
            'criterion': criterion.__class__.__name__
        }
        
        # Initialize training variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in dataloaders['train']:
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(dataloaders['train'])
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            if 'val' in dataloaders:
                self.model.eval()
                with torch.no_grad():
                    for inputs, targets in dataloaders['val']:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        
                val_loss /= len(dataloaders['val'])
                self.history['val_loss'].append(val_loss)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if save_best_model:
                        self.save_model('best_model')
                else:
                    patience_counter += 1
                    
                # Print progress
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                # Check early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
                    
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                
        # Update epochs trained
        self.history['epochs_trained'] += epoch + 1
        
        # Save final model
        self.save_model('final_model')
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def evaluate(self, dataloader: DataLoader, criterion: nn.Module = None) -> Dict[str, float]:
        """
        Evaluate the model on a dataset
        
        Args:
            dataloader: DataLoader with evaluation data
            criterion: Loss function (default: MSELoss)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if criterion is None:
            criterion = nn.MSELoss()
            
        self.model.eval()
        
        all_targets = []
        all_predictions = []
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Save predictions and targets for further metrics
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        # Calculate metrics
        y_true = np.vstack(all_targets)
        y_pred = np.vstack(all_predictions)
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate directional accuracy (for stock price movement prediction)
        # For each window, compare the direction of the last step vs. target
        y_true_dir = np.sign(y_true[:, -1, 0] - y_true[:, 0, 0])
        y_pred_dir = np.sign(y_pred[:, -1, 0] - y_pred[:, 0, 0])
        direction_accuracy = np.mean(y_true_dir == y_pred_dir)
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'mse': mse,
            'mae': mae,
            'direction_accuracy': float(direction_accuracy)
        }
        
        # Save test metrics
        self.history['test_metrics'] = metrics
        self.save_history()
        
        return metrics
    
    def save_model(self, name: str = 'model') -> None:
        """
        Save model weights and architecture
        
        Args:
            name: Name of the saved model
        """
        model_path = os.path.join(self.experiment_dir, "models", name)
        torch.save(self.model.state_dict(), f"{model_path}.pt")
        
        # Save model architecture/config if available
        if hasattr(self.model, 'config'):
            with open(f"{model_path}_config.json", 'w') as f:
                json.dump(self.model.config, f)
    
    def load_model(self, name: str = 'best_model') -> None:
        """
        Load saved model
        
        Args:
            name: Name of the model to load
        """
        model_path = os.path.join(self.experiment_dir, "models", name)
        self.model.load_state_dict(torch.load(f"{model_path}.pt"))
        self.model.to(self.device)
        
    def save_history(self) -> None:
        """
        Save training history to file
        """
        history_path = os.path.join(self.experiment_dir, "logs", "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
            
    def load_history(self) -> Dict:
        """
        Load training history from file
        
        Returns:
            Training history dictionary
        """
        history_path = os.path.join(self.experiment_dir, "logs", "history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        return self.history
        
    def plot_training_history(self, save: bool = True) -> None:
        """
        Plot training and validation loss
        
        Args:
            save: Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        
        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
            
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(os.path.join(self.experiment_dir, "plots", "training_history.png"))
        
        plt.show()
        
    def plot_predictions(self, 
                        inputs: np.ndarray, 
                        targets: np.ndarray,
                        steps_ahead: int = 1,
                        save: bool = True,
                        filename: str = "predictions.png") -> None:
        """
        Plot model predictions vs. actual values
        
        Args:
            inputs: Input sequences
            targets: Target values
            steps_ahead: Steps ahead to predict
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        # Convert to tensors
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(inputs_tensor).cpu().numpy()
            
        # Plot the results
        plt.figure(figsize=(12, 6))
        
        # For simplicity, just plot first sample
        plt.plot(targets[0, :, 0], 'b-', label='Actual')
        plt.plot(predictions[0, :, 0], 'r-', label='Predicted')
        
        plt.title(f'Stock Price Prediction (Steps Ahead: {steps_ahead})')
        plt.xlabel('Time Steps')
        plt.ylabel('Price (normalized)')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(os.path.join(self.experiment_dir, "plots", filename))
            
        plt.show()
        
    def save_for_huggingface(self, model_name: str, model_description: str) -> str:
        """
        Prepare model for Hugging Face upload
        
        Args:
            model_name: Name for the Hugging Face model
            model_description: Description of the model
            
        Returns:
            Path to the prepared model directory
        """
        # Create directory for Hugging Face model
        hf_dir = os.path.join(self.experiment_dir, "huggingface")
        os.makedirs(hf_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(hf_dir, "pytorch_model.bin"))
        
        # Create model config
        if hasattr(self.model, 'config'):
            model_config = self.model.config
        else:
            model_config = {}
            
        # Add training metrics
        model_config.update({
            'metrics': self.history.get('test_metrics', {}),
            'hyperparameters': self.history.get('hyperparameters', {}),
            'model_name': model_name
        })
        
        # Save config
        with open(os.path.join(hf_dir, "config.json"), 'w') as f:
            json.dump(model_config, f)
            
        # Create README for model card
        metrics = self.history.get('test_metrics', {})
        hyperparams = self.history.get('hyperparameters', {})
        
        readme = f"""# {model_name}

{model_description}

## Model Description

This is a stock price prediction model based on the Informer architecture.

### Technical Details

- **Architecture:** Informer (Transformer-based model for time series forecasting)
- **Training Epochs:** {self.history.get('epochs_trained', 'N/A')}
- **Optimizer:** {hyperparams.get('optimizer', 'N/A')}
- **Learning Rate:** {hyperparams.get('learning_rate', 'N/A')}
- **Batch Size:** {hyperparams.get('batch_size', 'N/A')}

## Performance Metrics

- **MSE:** {metrics.get('mse', 'N/A')}
- **MAE:** {metrics.get('mae', 'N/A')}
- **Direction Accuracy:** {metrics.get('direction_accuracy', 'N/A')}

## Usage

```python
import torch
from transformers import AutoModelForSequenceRegression

# Load model
model = AutoModelForSequenceRegression.from_pretrained("{model_name}")

# Prepare input sequence (example)
# Shape: [batch_size, sequence_length, features]
inputs = torch.randn(1, 60, 10)  # Adjust dimensions based on your model

# Generate predictions
with torch.no_grad():
    predictions = model(inputs)
    
# Process predictions
print(predictions.shape)  # [batch_size, forecast_horizon, 1]
```

## Citation

If you use this model, please cite the original Informer paper:

```
@inproceedings{zhou2021informer,
  title={Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={Proceedings of AAAI},
  year={2021}
}
```
"""
        
        with open(os.path.join(hf_dir, "README.md"), 'w') as f:
            f.write(readme)
            
        return hf_dir