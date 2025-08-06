"""
Simple test for experiment configuration without pytest.
"""

from src.experiments.config import ExperimentConfig, ConfigValidator
import json
import tempfile
from pathlib import Path


def test_basic_functionality():
    """Test basic configuration functionality."""
    print("Testing basic configuration creation...")
    
    # Test default config
    config = ExperimentConfig.get_default_config()
    print(f"Default config created: {config}")
    
    # Test custom config
    custom_config = ExperimentConfig(
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
    print(f"Custom config created: {custom_config}")
    
    # Test JSON serialization
    json_str = config.to_json()
    print("JSON serialization successful")
    
    # Test JSON deserialization
    restored_config = ExperimentConfig.from_json(json_str)
    print("JSON deserialization successful")
    
    # Test file operations
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        config.to_json(temp_path)
        file_config = ExperimentConfig.from_json(temp_path)
        print("File operations successful")
    finally:
        Path(temp_path).unlink(missing_ok=True)
    
    # Test copy functionality
    new_config = config.copy(polynomial_degree=5, optimizer="rmsprop")
    print(f"Config copy successful: degree={new_config.polynomial_degree}, opt={new_config.optimizer}")
    
    # Test validation warnings
    warnings = ConfigValidator.validate_config_compatibility(config)
    print(f"Validation warnings: {len(warnings)} found")
    
    suggestions = ConfigValidator.suggest_improvements(config)
    print(f"Improvement suggestions: {len(suggestions)} found")
    
    print("All tests passed!")


def test_validation():
    """Test configuration validation."""
    print("\nTesting validation...")
    
    # Test invalid polynomial degree
    try:
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
        print("ERROR: Should have raised ValueError for invalid polynomial_degree")
    except ValueError as e:
        print(f"Correctly caught invalid polynomial_degree: {e}")
    
    # Test invalid optimizer
    try:
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
        print("ERROR: Should have raised ValueError for invalid optimizer")
    except ValueError as e:
        print(f"Correctly caught invalid optimizer: {e}")
    
    print("Validation tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
    test_validation()