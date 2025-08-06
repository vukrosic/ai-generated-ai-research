"""
Experiment configuration management with validation and serialization.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration dataclass for curve fitting experiments."""
    
    # Data generation parameters
    polynomial_degree: int
    noise_level: float
    train_val_split: float
    
    # Model architecture parameters
    model_architecture: str
    hidden_dims: List[int]
    
    # Training parameters
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int
    random_seed: int
    
    # Optional parameters with defaults
    num_data_points: int = 1000
    x_range: tuple = (-5.0, 5.0)
    activation_function: str = "relu"
    early_stopping_patience: int = 10
    save_checkpoints: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate all configuration parameters."""
        # Validate polynomial degree
        if not isinstance(self.polynomial_degree, int) or not (1 <= self.polynomial_degree <= 6):
            raise ValueError("polynomial_degree must be an integer between 1 and 6")
        
        # Validate noise level
        if not isinstance(self.noise_level, (int, float)) or self.noise_level < 0:
            raise ValueError("noise_level must be a non-negative number")
        
        # Validate train/validation split
        if not isinstance(self.train_val_split, (int, float)) or not (0.1 <= self.train_val_split <= 0.9):
            raise ValueError("train_val_split must be a number between 0.1 and 0.9")
        
        # Validate model architecture
        valid_architectures = ["linear", "shallow", "deep"]
        if self.model_architecture not in valid_architectures:
            raise ValueError(f"model_architecture must be one of {valid_architectures}")
        
        # Validate hidden dimensions
        if not isinstance(self.hidden_dims, list) or not all(isinstance(dim, int) and dim > 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must be a list of positive integers")
        
        # Validate optimizer
        valid_optimizers = ["sgd", "adam", "rmsprop", "adagrad"]
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")
        
        # Validate learning rate
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        
        # Validate batch size
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        
        # Validate epochs
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        
        # Validate random seed
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        
        # Validate optional parameters
        if not isinstance(self.num_data_points, int) or self.num_data_points <= 0:
            raise ValueError("num_data_points must be a positive integer")
        
        if not isinstance(self.x_range, (tuple, list)) or len(self.x_range) != 2:
            raise ValueError("x_range must be a tuple or list of two numbers")
        
        if self.x_range[0] >= self.x_range[1]:
            raise ValueError("x_range must have first element less than second element")
        
        valid_activations = ["relu", "tanh", "sigmoid", "leaky_relu"]
        if self.activation_function not in valid_activations:
            raise ValueError(f"activation_function must be one of {valid_activations}")
        
        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be a positive integer")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Convert configuration to JSON string or save to file.
        
        Args:
            filepath: Optional path to save JSON file
            
        Returns:
            JSON string representation
        """
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, sort_keys=True)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            ExperimentConfig instance
        """
        # Convert x_range from list to tuple if needed (JSON serialization converts tuples to lists)
        if 'x_range' in config_dict and isinstance(config_dict['x_range'], list):
            config_dict = config_dict.copy()  # Don't modify original dict
            config_dict['x_range'] = tuple(config_dict['x_range'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str_or_path: str) -> 'ExperimentConfig':
        """
        Create configuration from JSON string or file.
        
        Args:
            json_str_or_path: JSON string or path to JSON file
            
        Returns:
            ExperimentConfig instance
        """
        # Check if it's a file path by looking for file extension and checking if it exists
        try:
            path = Path(json_str_or_path)
            if path.suffix == '.json' and path.exists():
                # It's a file path
                with open(json_str_or_path, 'r') as f:
                    config_dict = json.load(f)
            else:
                # It's a JSON string
                config_dict = json.loads(json_str_or_path)
        except (OSError, json.JSONDecodeError):
            # If path checking fails, try as JSON string
            try:
                config_dict = json.loads(json_str_or_path)
            except json.JSONDecodeError:
                # If JSON parsing fails, try as file path
                with open(json_str_or_path, 'r') as f:
                    config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def get_default_config(cls) -> 'ExperimentConfig':
        """
        Get a default configuration for testing and examples.
        
        Returns:
            ExperimentConfig with sensible defaults
        """
        return cls(
            polynomial_degree=3,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture="shallow",
            hidden_dims=[64, 32],
            optimizer="adam",
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            random_seed=42,
            num_data_points=1000,
            x_range=(-5.0, 5.0),
            activation_function="relu",
            early_stopping_patience=10,
            save_checkpoints=True
        )
    
    def copy(self, **kwargs) -> 'ExperimentConfig':
        """
        Create a copy of the configuration with optional parameter updates.
        
        Args:
            **kwargs: Parameters to update in the copy
            
        Returns:
            New ExperimentConfig instance
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"ExperimentConfig(degree={self.polynomial_degree}, arch={self.model_architecture}, opt={self.optimizer})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.to_json()


class ConfigValidator:
    """Utility class for advanced configuration validation."""
    
    @staticmethod
    def validate_config_compatibility(config: ExperimentConfig) -> List[str]:
        """
        Validate configuration compatibility and return warnings.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check if hidden dimensions make sense for architecture
        if config.model_architecture == "linear" and config.hidden_dims:
            warnings.append("Linear model specified but hidden_dims provided - hidden_dims will be ignored")
        
        if config.model_architecture == "shallow" and len(config.hidden_dims) > 2:
            warnings.append("Shallow network typically uses 1-2 hidden layers, but more were specified")
        
        if config.model_architecture == "deep" and len(config.hidden_dims) < 3:
            warnings.append("Deep network typically uses 3+ hidden layers, but fewer were specified")
        
        # Check learning rate compatibility with optimizer
        if config.optimizer.lower() == "sgd" and config.learning_rate > 0.1:
            warnings.append("High learning rate for SGD optimizer may cause instability")
        
        if config.optimizer.lower() in ["adam", "rmsprop"] and config.learning_rate > 0.01:
            warnings.append(f"High learning rate for {config.optimizer} optimizer may cause instability")
        
        # Check batch size vs data points
        if config.batch_size > config.num_data_points // 2:
            warnings.append("Batch size is large relative to dataset size")
        
        # Check polynomial degree vs noise level
        if config.polynomial_degree >= 5 and config.noise_level < 0.05:
            warnings.append("High-degree polynomials with low noise may lead to overfitting")
        
        return warnings
    
    @staticmethod
    def suggest_improvements(config: ExperimentConfig) -> List[str]:
        """
        Suggest configuration improvements.
        
        Args:
            config: Configuration to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Suggest better train/val split for small datasets
        if config.num_data_points < 500 and config.train_val_split < 0.7:
            suggestions.append("Consider increasing train_val_split for small datasets")
        
        # Suggest early stopping patience based on epochs
        if config.early_stopping_patience > config.epochs // 5:
            suggestions.append("Early stopping patience might be too high relative to total epochs")
        
        # Suggest batch size optimization
        if config.num_data_points > 10000 and config.batch_size < 64:
            suggestions.append("Consider increasing batch size for large datasets")
        
        return suggestions