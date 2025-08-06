"""
Test suite for experiment configuration management.
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.experiments.config import ExperimentConfig, ConfigValidator


class TestExperimentConfig:
    """Test cases for ExperimentConfig class."""
    
    def test_default_config_creation(self):
        """Test creating a default configuration."""
        config = ExperimentConfig.get_default_config()
        
        assert config.polynomial_degree == 3
        assert config.noise_level == 0.1
        assert config.train_val_split == 0.8
        assert config.model_architecture == "shallow"
        assert config.hidden_dims == [64, 32]
        assert config.optimizer == "adam"
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.random_seed == 42
    
    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config = ExperimentConfig(
            polynomial_degree=2,
            noise_level=0.05,
            train_val_split=0.7,
            model_architecture="deep",
            hidden_dims=[128, 64, 32],
            optimizer="sgd",
            learning_rate=0.01,
            batch_size=16,
            epochs=50,
            random_seed=123
        )
        
        assert config.polynomial_degree == 2
        assert config.optimizer == "sgd"
        assert len(config.hidden_dims) == 3
    
    def test_invalid_polynomial_degree(self):
        """Test validation of polynomial degree."""
        with pytest.raises(ValueError, match="polynomial_degree must be an integer between 1 and 6"):
            ExperimentConfig(
                polynomial_degree=0,  # Invalid
                noise_level=0.1,
                train_val_split=0.8,
                model_architecture="shallow",
                hidden_dims=[64],
                optimizer="adam",
                learning_rate=0.001,
                batch_size=32,
                epochs=100,
                random_seed=42
            )
    
    def test_invalid_noise_level(self):
        """Test validation of noise level."""
        with pytest.raises(ValueError, match="noise_level must be a non-negative number"):
            ExperimentConfig(
                polynomial_degree=3,
                noise_level=-0.1,  # Invalid
                train_val_split=0.8,
                model_architecture="shallow",
                hidden_dims=[64],
                optimizer="adam",
                learning_rate=0.001,
                batch_size=32,
                epochs=100,
                random_seed=42
            )
    
    def test_invalid_train_val_split(self):
        """Test validation of train/validation split."""
        with pytest.raises(ValueError, match="train_val_split must be a number between 0.1 and 0.9"):
            ExperimentConfig(
                polynomial_degree=3,
                noise_level=0.1,
                train_val_split=1.5,  # Invalid
                model_architecture="shallow",
                hidden_dims=[64],
                optimizer="adam",
                learning_rate=0.001,
                batch_size=32,
                epochs=100,
                random_seed=42
            )
    
    def test_invalid_model_architecture(self):
        """Test validation of model architecture."""
        with pytest.raises(ValueError, match="model_architecture must be one of"):
            ExperimentConfig(
                polynomial_degree=3,
                noise_level=0.1,
                train_val_split=0.8,
                model_architecture="invalid",  # Invalid
                hidden_dims=[64],
                optimizer="adam",
                learning_rate=0.001,
                batch_size=32,
                epochs=100,
                random_seed=42
            )
    
    def test_invalid_optimizer(self):
        """Test validation of optimizer."""
        with pytest.raises(ValueError, match="optimizer must be one of"):
            ExperimentConfig(
                polynomial_degree=3,
                noise_level=0.1,
                train_val_split=0.8,
                model_architecture="shallow",
                hidden_dims=[64],
                optimizer="invalid",  # Invalid
                learning_rate=0.001,
                batch_size=32,
                epochs=100,
                random_seed=42
            )
    
    def test_invalid_learning_rate(self):
        """Test validation of learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be a positive number"):
            ExperimentConfig(
                polynomial_degree=3,
                noise_level=0.1,
                train_val_split=0.8,
                model_architecture="shallow",
                hidden_dims=[64],
                optimizer="adam",
                learning_rate=0,  # Invalid
                batch_size=32,
                epochs=100,
                random_seed=42
            )
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original_config = ExperimentConfig.get_default_config()
        
        # Test to_json
        json_str = original_config.to_json()
        assert isinstance(json_str, str)
        
        # Test from_json with string
        restored_config = ExperimentConfig.from_json(json_str)
        assert restored_config.polynomial_degree == original_config.polynomial_degree
        assert restored_config.optimizer == original_config.optimizer
        assert restored_config.hidden_dims == original_config.hidden_dims
    
    def test_json_file_operations(self):
        """Test JSON file save and load operations."""
        original_config = ExperimentConfig.get_default_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to file
            original_config.to_json(temp_path)
            assert Path(temp_path).exists()
            
            # Load from file
            restored_config = ExperimentConfig.from_json(temp_path)
            assert restored_config.polynomial_degree == original_config.polynomial_degree
            assert restored_config.optimizer == original_config.optimizer
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_copy(self):
        """Test configuration copying with updates."""
        original_config = ExperimentConfig.get_default_config()
        
        # Create copy with updates
        new_config = original_config.copy(
            polynomial_degree=5,
            optimizer="sgd",
            learning_rate=0.01
        )
        
        # Check original is unchanged
        assert original_config.polynomial_degree == 3
        assert original_config.optimizer == "adam"
        
        # Check new config has updates
        assert new_config.polynomial_degree == 5
        assert new_config.optimizer == "sgd"
        assert new_config.learning_rate == 0.01
        
        # Check other values are preserved
        assert new_config.batch_size == original_config.batch_size
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        config = ExperimentConfig.get_default_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['polynomial_degree'] == 3
        assert config_dict['optimizer'] == 'adam'
        assert config_dict['hidden_dims'] == [64, 32]
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'polynomial_degree': 4,
            'noise_level': 0.2,
            'train_val_split': 0.75,
            'model_architecture': 'deep',
            'hidden_dims': [128, 64],
            'optimizer': 'rmsprop',
            'learning_rate': 0.005,
            'batch_size': 64,
            'epochs': 200,
            'random_seed': 999
        }
        
        config = ExperimentConfig.from_dict(config_dict)
        assert config.polynomial_degree == 4
        assert config.optimizer == 'rmsprop'
        assert config.hidden_dims == [128, 64]


class TestConfigValidator:
    """Test cases for ConfigValidator class."""
    
    def test_compatibility_warnings(self):
        """Test configuration compatibility warnings."""
        # Linear model with hidden dims
        config = ExperimentConfig(
            polynomial_degree=3,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture="linear",
            hidden_dims=[64, 32],  # Should warn
            optimizer="adam",
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            random_seed=42
        )
        
        warnings = ConfigValidator.validate_config_compatibility(config)
        assert len(warnings) > 0
        assert any("Linear model" in warning for warning in warnings)
    
    def test_learning_rate_warnings(self):
        """Test learning rate compatibility warnings."""
        config = ExperimentConfig(
            polynomial_degree=3,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture="shallow",
            hidden_dims=[64],
            optimizer="sgd",
            learning_rate=0.5,  # High for SGD
            batch_size=32,
            epochs=100,
            random_seed=42
        )
        
        warnings = ConfigValidator.validate_config_compatibility(config)
        assert any("High learning rate" in warning for warning in warnings)
    
    def test_suggestions(self):
        """Test configuration improvement suggestions."""
        config = ExperimentConfig(
            polynomial_degree=3,
            noise_level=0.1,
            train_val_split=0.6,  # Low for small dataset
            model_architecture="shallow",
            hidden_dims=[64],
            optimizer="adam",
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            random_seed=42,
            num_data_points=300  # Small dataset
        )
        
        suggestions = ConfigValidator.suggest_improvements(config)
        assert len(suggestions) > 0


if __name__ == "__main__":
    # Run basic tests
    test_config = TestExperimentConfig()
    test_config.test_default_config_creation()
    test_config.test_json_serialization()
    test_config.test_config_copy()
    
    test_validator = TestConfigValidator()
    test_validator.test_compatibility_warnings()
    
    print("All basic tests passed!")