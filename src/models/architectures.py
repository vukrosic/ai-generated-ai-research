"""Neural network architectures for curve fitting experiments."""

import torch
import torch.nn as nn
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional


class BaseModel(nn.Module, ABC):
    """Abstract base class for all neural network models."""
    
    def __init__(self, input_dim: int):
        """Initialize base model.
        
        Args:
            input_dim: Dimension of input features
        """
        super().__init__()
        self.input_dim = input_dim
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters and metadata.
        
        Returns:
            Dictionary containing parameter information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'model_type': self.__class__.__name__,
            'parameter_details': {
                name: {
                    'shape': list(param.shape),
                    'requires_grad': param.requires_grad,
                    'num_elements': param.numel()
                }
                for name, param in self.named_parameters()
            }
        }
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def model_summary(self) -> str:
        """Generate a summary string of the model architecture.
        
        Returns:
            Formatted string with model information
        """
        params = self.get_parameters()
        summary = f"Model: {params['model_type']}\n"
        summary += f"Input dimension: {params['input_dim']}\n"
        summary += f"Total parameters: {params['total_parameters']:,}\n"
        summary += f"Trainable parameters: {params['trainable_parameters']:,}\n"
        summary += "\nLayer details:\n"
        
        for name, details in params['parameter_details'].items():
            summary += f"  {name}: {details['shape']} ({details['num_elements']:,} params)\n"
            
        return summary
    
    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model state and architecture information.
        
        Args:
            filepath: Path to save the model (without extension)
            metadata: Additional metadata to save with the model
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Prepare save data
        save_data = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'model_parameters': self.get_parameters(),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        # Add constructor arguments for model recreation
        if hasattr(self, '_constructor_args'):
            save_data['constructor_args'] = self._constructor_args
        else:
            # Try to infer constructor arguments from model attributes
            save_data['constructor_args'] = self._infer_constructor_args()
        
        # Add metadata if provided
        if metadata:
            save_data['metadata'] = metadata
        
        # Save model state
        torch.save(save_data, f"{filepath}.pth")
        
        # Save architecture info as JSON for human readability
        arch_info = {
            'model_class': save_data['model_class'],
            'constructor_args': save_data['constructor_args'],
            'model_parameters': save_data['model_parameters'],
            'timestamp': save_data['timestamp']
        }
        
        with open(f"{filepath}_info.json", 'w') as f:
            json.dump(arch_info, f, indent=2)
    
    def _infer_constructor_args(self) -> Dict[str, Any]:
        """Infer constructor arguments from model attributes."""
        args = {'input_dim': self.input_dim}
        
        # Add class-specific arguments
        if hasattr(self, 'hidden_dims'):
            args['hidden_dims'] = self.hidden_dims
        if hasattr(self, 'activation_name'):
            args['activation'] = self.activation_name
        if hasattr(self, 'dropout_rate'):
            args['dropout_rate'] = self.dropout_rate
        if hasattr(self, 'batch_norm'):
            args['batch_norm'] = self.batch_norm
            
        return args
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[torch.device] = None) -> 'BaseModel':
        """Load a saved model.
        
        Args:
            filepath: Path to the saved model (without extension)
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load saved data
        save_data = torch.load(f"{filepath}.pth", map_location=device, weights_only=False)
        
        # Get model class and constructor arguments
        model_class_name = save_data['model_class']
        constructor_args = save_data['constructor_args']
        
        # Create model instance
        if model_class_name == 'LinearModel':
            model = LinearModel(**constructor_args)
        elif model_class_name == 'ShallowNetwork':
            model = ShallowNetwork(**constructor_args)
        elif model_class_name == 'DeepNetwork':
            model = DeepNetwork(**constructor_args)
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")
        
        # Load state dict
        model.load_state_dict(save_data['model_state_dict'])
        model.to(device)
        
        return model
    
    def validate_architecture(self, reference_model: 'BaseModel') -> bool:
        """Validate that this model has the same architecture as reference model.
        
        Args:
            reference_model: Model to compare architecture against
            
        Returns:
            True if architectures match, False otherwise
        """
        # Compare model classes
        if self.__class__ != reference_model.__class__:
            return False
        
        # Compare parameter counts
        if self.count_parameters() != reference_model.count_parameters():
            return False
        
        # Compare parameter shapes
        self_params = dict(self.named_parameters())
        ref_params = dict(reference_model.named_parameters())
        
        if set(self_params.keys()) != set(ref_params.keys()):
            return False
        
        for name in self_params:
            if self_params[name].shape != ref_params[name].shape:
                return False
        
        return True


class LinearModel(BaseModel):
    """Simple linear regression model for baseline comparison."""
    
    def __init__(self, input_dim: int = 1):
        """Initialize linear model.
        
        Args:
            input_dim: Dimension of input features (default: 1 for curve fitting)
        """
        super().__init__(input_dim)
        self._constructor_args = {'input_dim': input_dim}
        
        self.linear = nn.Linear(input_dim, 1)
        
        # Initialize weights with small random values
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.linear(x)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get the linear model coefficients.
        
        Returns:
            Dictionary with weight and bias values
        """
        return {
            'weight': self.linear.weight.item() if self.input_dim == 1 else self.linear.weight.detach().numpy(),
            'bias': self.linear.bias.item()
        }


class ShallowNetwork(BaseModel):
    """Shallow neural network with 1-2 hidden layers."""
    
    def __init__(self, input_dim: int = 1, hidden_dims: List[int] = [64], 
                 activation: str = 'relu', dropout_rate: float = 0.0):
        """Initialize shallow network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions (1-2 layers)
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(input_dim)
        
        if len(hidden_dims) > 2:
            raise ValueError("ShallowNetwork supports maximum 2 hidden layers")
        
        self._constructor_args = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'activation': activation,
            'dropout_rate': dropout_rate
        }
        
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.01)
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class DeepNetwork(BaseModel):
    """Deep neural network with 3+ hidden layers."""
    
    def __init__(self, input_dim: int = 1, hidden_dims: List[int] = [128, 64, 32], 
                 activation: str = 'relu', dropout_rate: float = 0.1, 
                 batch_norm: bool = False):
        """Initialize deep network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions (3+ layers)
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
        """
        super().__init__(input_dim)
        
        if len(hidden_dims) < 3:
            raise ValueError("DeepNetwork requires minimum 3 hidden layers")
        
        self._constructor_args = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'activation': activation,
            'dropout_rate': dropout_rate,
            'batch_norm': batch_norm
        }
        
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                if self.activation_name in ['relu', 'leaky_relu']:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ModelFactory:
    """Factory class for creating model instances from configuration."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> BaseModel:
        """Create a model instance based on type and configuration.
        
        Args:
            model_type: Type of model ('linear', 'shallow', 'deep')
            config: Configuration dictionary with model parameters
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        input_dim = config.get('input_dim', 1)
        
        if model_type.lower() == 'linear':
            return LinearModel(input_dim=input_dim)
        
        elif model_type.lower() == 'shallow':
            return ShallowNetwork(
                input_dim=input_dim,
                hidden_dims=config.get('hidden_dims', [64]),
                activation=config.get('activation', 'relu'),
                dropout_rate=config.get('dropout_rate', 0.0)
            )
        
        elif model_type.lower() == 'deep':
            return DeepNetwork(
                input_dim=input_dim,
                hidden_dims=config.get('hidden_dims', [128, 64, 32]),
                activation=config.get('activation', 'relu'),
                dropout_rate=config.get('dropout_rate', 0.1),
                batch_norm=config.get('batch_norm', False)
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_supported_models() -> List[str]:
        """Get list of supported model types."""
        return ['linear', 'shallow', 'deep']
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict[str, Any]:
        """Get default configuration for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Default configuration dictionary
        """
        defaults = {
            'linear': {
                'input_dim': 1
            },
            'shallow': {
                'input_dim': 1,
                'hidden_dims': [64],
                'activation': 'relu',
                'dropout_rate': 0.0
            },
            'deep': {
                'input_dim': 1,
                'hidden_dims': [128, 64, 32],
                'activation': 'relu',
                'dropout_rate': 0.1,
                'batch_norm': False
            }
        }
        
        if model_type.lower() not in defaults:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return defaults[model_type.lower()]


class ModelCheckpoint:
    """Utility class for model checkpointing during training."""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints', save_best_only: bool = True,
                 monitor: str = 'val_loss', mode: str = 'min'):
        """Initialize model checkpoint utility.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
            monitor: Metric to monitor for best model selection
            mode: 'min' for metrics that should be minimized, 'max' for maximized
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model: BaseModel, epoch: int, metrics: Dict[str, float],
                       optimizer_state: Optional[Dict] = None, 
                       scheduler_state: Optional[Dict] = None) -> bool:
        """Save model checkpoint.
        
        Args:
            model: Model to checkpoint
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            optimizer_state: Optimizer state dict
            scheduler_state: Learning rate scheduler state dict
            
        Returns:
            True if checkpoint was saved, False otherwise
        """
        current_value = metrics.get(self.monitor)
        if current_value is None:
            print(f"Warning: Monitor metric '{self.monitor}' not found in metrics")
            return False
        
        is_best = self._is_better(current_value, self.best_value)
        
        if not self.save_best_only or is_best:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'constructor_args': model._constructor_args,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            if optimizer_state:
                checkpoint_data['optimizer_state_dict'] = optimizer_state
            if scheduler_state:
                checkpoint_data['scheduler_state_dict'] = scheduler_state
            
            # Save checkpoint
            if is_best:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                self.best_value = current_value
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # Also save model info
            info_path = checkpoint_path.replace('.pth', '_info.json')
            with open(info_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'metrics': metrics,
                    'model_class': checkpoint_data['model_class'],
                    'constructor_args': checkpoint_data['constructor_args'],
                    'timestamp': checkpoint_data['timestamp']
                }, f, indent=2)
            
            return True
        
        return False
    
    def load_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load checkpoint on
            
        Returns:
            Dictionary containing checkpoint data
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return checkpoint
    
    def load_best_model(self, device: Optional[torch.device] = None) -> BaseModel:
        """Load the best saved model.
        
        Args:
            device: Device to load model on
            
        Returns:
            Best model instance
        """
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best model found at {best_path}")
        
        checkpoint = self.load_checkpoint(best_path, device)
        
        # Recreate model
        model_class_name = checkpoint['model_class']
        constructor_args = checkpoint['constructor_args']
        
        if model_class_name == 'LinearModel':
            model = LinearModel(**constructor_args)
        elif model_class_name == 'ShallowNetwork':
            model = ShallowNetwork(**constructor_args)
        elif model_class_name == 'DeepNetwork':
            model = DeepNetwork(**constructor_args)
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best value."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best