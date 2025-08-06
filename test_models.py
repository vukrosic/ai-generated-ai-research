"""
Comprehensive unit tests for neural network model architectures.
Tests model correctness, parameter counting, serialization, and factory patterns.
"""

import torch
import torch.nn as nn
import sys
import os
import unittest
import tempfile
import shutil
import json
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.architectures import LinearModel, ShallowNetwork, DeepNetwork, ModelFactory, ModelCheckpoint


class TestLinearModel(unittest.TestCase):
    """Test cases for LinearModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.model = LinearModel(input_dim=1)
    
    def test_model_creation(self):
        """Test model creation and basic properties."""
        self.assertEqual(self.model.input_dim, 1)
        self.assertIsInstance(self.model.linear, nn.Linear)
        self.assertEqual(self.model.linear.in_features, 1)
        self.assertEqual(self.model.linear.out_features, 1)
    
    def test_forward_pass_shapes(self):
        """Test forward pass with different input shapes."""
        test_cases = [
            (1, 1),    # Single sample
            (10, 1),   # Batch of 10
            (100, 1),  # Larger batch
        ]
        
        for batch_size, input_dim in test_cases:
            with self.subTest(batch_size=batch_size, input_dim=input_dim):
                x = torch.randn(batch_size, input_dim)
                output = self.model(x)
                expected_shape = (batch_size, 1)
                self.assertEqual(output.shape, expected_shape)
    
    def test_forward_pass_computation(self):
        """Test that forward pass computes correct linear transformation."""
        # Set known weights for predictable output
        with torch.no_grad():
            self.model.linear.weight.fill_(2.0)
            self.model.linear.bias.fill_(1.0)
        
        x = torch.tensor([[1.0], [2.0], [3.0]])
        expected_output = torch.tensor([[3.0], [5.0], [7.0]])  # 2*x + 1
        
        output = self.model(x)
        torch.testing.assert_close(output, expected_output)
    
    def test_parameter_counting(self):
        """Test parameter counting accuracy."""
        param_count = self.model.count_parameters()
        self.assertEqual(param_count, 2)  # weight + bias
        
        # Test with different input dimensions
        model_2d = LinearModel(input_dim=2)
        param_count_2d = model_2d.count_parameters()
        self.assertEqual(param_count_2d, 3)  # 2 weights + 1 bias
    
    def test_get_parameters_method(self):
        """Test get_parameters method returns correct information."""
        params = self.model.get_parameters()
        
        # Check required keys
        required_keys = ['total_parameters', 'trainable_parameters', 'input_dim', 'model_type', 'parameter_details']
        for key in required_keys:
            self.assertIn(key, params)
        
        # Check values
        self.assertEqual(params['total_parameters'], 2)
        self.assertEqual(params['trainable_parameters'], 2)
        self.assertEqual(params['input_dim'], 1)
        self.assertEqual(params['model_type'], 'LinearModel')
        
        # Check parameter details
        self.assertIn('linear.weight', params['parameter_details'])
        self.assertIn('linear.bias', params['parameter_details'])
    
    def test_model_summary(self):
        """Test model summary generation."""
        summary = self.model.model_summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn('LinearModel', summary)
        self.assertIn('Input dimension: 1', summary)
        self.assertIn('Total parameters: 2', summary)
        self.assertIn('Trainable parameters: 2', summary)
    
    def test_get_coefficients(self):
        """Test coefficient extraction."""
        coeffs = self.model.get_coefficients()
        
        self.assertIn('weight', coeffs)
        self.assertIn('bias', coeffs)
        self.assertIsInstance(coeffs['weight'], float)
        self.assertIsInstance(coeffs['bias'], float)
    
    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        # Check that weights are not zero (indicating initialization occurred)
        weight = self.model.linear.weight.item()
        bias = self.model.linear.bias.item()
        
        self.assertNotEqual(weight, 0.0)
        self.assertEqual(bias, 0.0)  # Bias should be initialized to zero


class TestShallowNetwork(unittest.TestCase):
    """Test cases for ShallowNetwork class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_single_hidden_layer(self):
        """Test shallow network with single hidden layer."""
        model = ShallowNetwork(input_dim=1, hidden_dims=[32], activation='relu')
        
        self.assertEqual(model.input_dim, 1)
        self.assertEqual(model.hidden_dims, [32])
        self.assertEqual(model.activation_name, 'relu')
        
        # Test forward pass
        x = torch.randn(10, 1)
        output = model(x)
        self.assertEqual(output.shape, (10, 1))
    
    def test_two_hidden_layers(self):
        """Test shallow network with two hidden layers."""
        model = ShallowNetwork(input_dim=1, hidden_dims=[64, 32], activation='tanh', dropout_rate=0.1)
        
        self.assertEqual(model.hidden_dims, [64, 32])
        self.assertEqual(model.activation_name, 'tanh')
        self.assertEqual(model.dropout_rate, 0.1)
        
        # Test forward pass
        x = torch.randn(10, 1)
        output = model(x)
        self.assertEqual(output.shape, (10, 1))
    
    def test_invalid_layer_count(self):
        """Test that more than 2 hidden layers raises ValueError."""
        with self.assertRaises(ValueError):
            ShallowNetwork(input_dim=1, hidden_dims=[64, 32, 16])
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
        
        for activation in activations:
            with self.subTest(activation=activation):
                model = ShallowNetwork(input_dim=1, hidden_dims=[32], activation=activation)
                x = torch.randn(5, 1)
                output = model(x)
                self.assertEqual(output.shape, (5, 1))
    
    def test_unsupported_activation(self):
        """Test that unsupported activation raises ValueError."""
        with self.assertRaises(ValueError):
            ShallowNetwork(input_dim=1, hidden_dims=[32], activation='unsupported')
    
    def test_parameter_counting(self):
        """Test parameter counting for different architectures."""
        # Single layer: (1*32 + 32) + (32*1 + 1) = 64 + 33 = 97
        model1 = ShallowNetwork(input_dim=1, hidden_dims=[32])
        expected_params1 = (1 * 32 + 32) + (32 * 1 + 1)
        self.assertEqual(model1.count_parameters(), expected_params1)
        
        # Two layers: (1*64 + 64) + (64*32 + 32) + (32*1 + 1) = 128 + 2080 + 33 = 2241
        model2 = ShallowNetwork(input_dim=1, hidden_dims=[64, 32])
        expected_params2 = (1 * 64 + 64) + (64 * 32 + 32) + (32 * 1 + 1)
        self.assertEqual(model2.count_parameters(), expected_params2)
    
    def test_dropout_behavior(self):
        """Test dropout behavior in training vs evaluation mode."""
        model = ShallowNetwork(input_dim=1, hidden_dims=[32], dropout_rate=0.5)
        x = torch.randn(100, 1)
        
        # In training mode, dropout should affect output
        model.train()
        output_train1 = model(x)
        output_train2 = model(x)
        
        # Outputs should be different due to dropout randomness
        self.assertFalse(torch.allclose(output_train1, output_train2))
        
        # In evaluation mode, dropout should be disabled
        model.eval()
        output_eval1 = model(x)
        output_eval2 = model(x)
        
        # Outputs should be identical in eval mode
        torch.testing.assert_close(output_eval1, output_eval2, rtol=1e-5, atol=1e-6)
    
    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        model = ShallowNetwork(input_dim=1, hidden_dims=[32])
        
        # Check that weights are not all zeros
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.assertFalse(torch.all(param == 0))
            elif 'bias' in name:
                torch.testing.assert_close(param, torch.zeros_like(param), rtol=1e-5, atol=1e-6)


class TestDeepNetwork(unittest.TestCase):
    """Test cases for DeepNetwork class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_minimum_layer_requirement(self):
        """Test that minimum 3 hidden layers are required."""
        # Valid: 3 layers
        model = DeepNetwork(input_dim=1, hidden_dims=[128, 64, 32])
        self.assertEqual(len(model.hidden_dims), 3)
        
        # Invalid: less than 3 layers
        with self.assertRaises(ValueError):
            DeepNetwork(input_dim=1, hidden_dims=[64, 32])
    
    def test_basic_deep_network(self):
        """Test basic deep network functionality."""
        model = DeepNetwork(input_dim=1, hidden_dims=[128, 64, 32], activation='relu')
        
        self.assertEqual(model.hidden_dims, [128, 64, 32])
        self.assertEqual(model.activation_name, 'relu')
        self.assertFalse(model.batch_norm)
        
        # Test forward pass
        x = torch.randn(10, 1)
        output = model(x)
        self.assertEqual(output.shape, (10, 1))
    
    def test_batch_normalization(self):
        """Test deep network with batch normalization."""
        model = DeepNetwork(input_dim=1, hidden_dims=[64, 32, 16], 
                          activation='relu', batch_norm=True, dropout_rate=0.2)
        
        self.assertTrue(model.batch_norm)
        
        # Check that BatchNorm1d layers are present
        has_batch_norm = any(isinstance(module, nn.BatchNorm1d) for module in model.network)
        self.assertTrue(has_batch_norm)
        
        # Test forward pass
        x = torch.randn(10, 1)
        output = model(x)
        self.assertEqual(output.shape, (10, 1))
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu']
        
        for activation in activations:
            with self.subTest(activation=activation):
                model = DeepNetwork(input_dim=1, hidden_dims=[32, 16, 8], activation=activation)
                x = torch.randn(5, 1)
                output = model(x)
                self.assertEqual(output.shape, (5, 1))
    
    def test_unsupported_activation(self):
        """Test that unsupported activation raises ValueError."""
        with self.assertRaises(ValueError):
            DeepNetwork(input_dim=1, hidden_dims=[32, 16, 8], activation='unsupported')
    
    def test_parameter_counting(self):
        """Test parameter counting for deep networks."""
        model = DeepNetwork(input_dim=1, hidden_dims=[64, 32, 16])
        
        # Calculate expected parameters
        # Layer 1: (1*64 + 64) = 128
        # Layer 2: (64*32 + 32) = 2080
        # Layer 3: (32*16 + 16) = 528
        # Output: (16*1 + 1) = 17
        # Total: 128 + 2080 + 528 + 17 = 2753
        expected_params = (1*64 + 64) + (64*32 + 32) + (32*16 + 16) + (16*1 + 1)
        self.assertEqual(model.count_parameters(), expected_params)
    
    def test_weight_initialization_relu(self):
        """Test weight initialization for ReLU networks (He initialization)."""
        model = DeepNetwork(input_dim=1, hidden_dims=[64, 32, 16], activation='relu')
        
        # Check that weights are not all zeros
        for name, param in model.named_parameters():
            if 'weight' in name and 'BatchNorm' not in name:
                self.assertFalse(torch.all(param == 0))
                # For ReLU, weights should have reasonable variance (not too large)
                self.assertLess(param.std().item(), 2.0)  # Relaxed threshold
            elif 'bias' in name and 'BatchNorm' not in name:
                torch.testing.assert_close(param, torch.zeros_like(param), rtol=1e-5, atol=1e-6)
    
    def test_weight_initialization_tanh(self):
        """Test weight initialization for tanh networks (Xavier initialization)."""
        model = DeepNetwork(input_dim=1, hidden_dims=[64, 32, 16], activation='tanh')
        
        # Check that weights are not all zeros
        for name, param in model.named_parameters():
            if 'weight' in name and 'BatchNorm' not in name:
                self.assertFalse(torch.all(param == 0))
            elif 'bias' in name and 'BatchNorm' not in name:
                torch.testing.assert_close(param, torch.zeros_like(param), rtol=1e-5, atol=1e-6)
    
    def test_dropout_and_batch_norm_combination(self):
        """Test combination of dropout and batch normalization."""
        model = DeepNetwork(input_dim=1, hidden_dims=[32, 16, 8], 
                          batch_norm=True, dropout_rate=0.3)
        
        # Check that both BatchNorm and Dropout are present
        has_batch_norm = any(isinstance(module, nn.BatchNorm1d) for module in model.network)
        has_dropout = any(isinstance(module, nn.Dropout) for module in model.network)
        
        self.assertTrue(has_batch_norm)
        self.assertTrue(has_dropout)
        
        # Test forward pass
        x = torch.randn(10, 1)
        output = model(x)
        self.assertEqual(output.shape, (10, 1))


class TestModelFactory(unittest.TestCase):
    """Test cases for ModelFactory class."""
    
    def test_create_linear_model(self):
        """Test linear model creation through factory."""
        config = {'input_dim': 1}
        model = ModelFactory.create_model('linear', config)
        
        self.assertIsInstance(model, LinearModel)
        self.assertEqual(model.input_dim, 1)
    
    def test_create_shallow_model(self):
        """Test shallow model creation through factory."""
        config = {
            'input_dim': 1,
            'hidden_dims': [64, 32],
            'activation': 'relu',
            'dropout_rate': 0.1
        }
        model = ModelFactory.create_model('shallow', config)
        
        self.assertIsInstance(model, ShallowNetwork)
        self.assertEqual(model.input_dim, 1)
        self.assertEqual(model.hidden_dims, [64, 32])
        self.assertEqual(model.activation_name, 'relu')
        self.assertEqual(model.dropout_rate, 0.1)
    
    def test_create_deep_model(self):
        """Test deep model creation through factory."""
        config = {
            'input_dim': 1,
            'hidden_dims': [128, 64, 32],
            'activation': 'tanh',
            'dropout_rate': 0.2,
            'batch_norm': True
        }
        model = ModelFactory.create_model('deep', config)
        
        self.assertIsInstance(model, DeepNetwork)
        self.assertEqual(model.input_dim, 1)
        self.assertEqual(model.hidden_dims, [128, 64, 32])
        self.assertEqual(model.activation_name, 'tanh')
        self.assertEqual(model.dropout_rate, 0.2)
        self.assertTrue(model.batch_norm)
    
    def test_case_insensitive_model_types(self):
        """Test that model type strings are case insensitive."""
        config = {'input_dim': 1}
        
        model_upper = ModelFactory.create_model('LINEAR', config)
        model_mixed = ModelFactory.create_model('Linear', config)
        model_lower = ModelFactory.create_model('linear', config)
        
        self.assertIsInstance(model_upper, LinearModel)
        self.assertIsInstance(model_mixed, LinearModel)
        self.assertIsInstance(model_lower, LinearModel)
    
    def test_unsupported_model_type(self):
        """Test that unsupported model types raise ValueError."""
        config = {'input_dim': 1}
        
        with self.assertRaises(ValueError):
            ModelFactory.create_model('unsupported', config)
    
    def test_get_supported_models(self):
        """Test getting list of supported models."""
        supported = ModelFactory.get_supported_models()
        
        self.assertIsInstance(supported, list)
        self.assertIn('linear', supported)
        self.assertIn('shallow', supported)
        self.assertIn('deep', supported)
        self.assertEqual(len(supported), 3)
    
    def test_get_default_configs(self):
        """Test getting default configurations for each model type."""
        # Test linear default config
        linear_config = ModelFactory.get_default_config('linear')
        self.assertIn('input_dim', linear_config)
        self.assertEqual(linear_config['input_dim'], 1)
        
        # Test shallow default config
        shallow_config = ModelFactory.get_default_config('shallow')
        required_keys = ['input_dim', 'hidden_dims', 'activation', 'dropout_rate']
        for key in required_keys:
            self.assertIn(key, shallow_config)
        
        # Test deep default config
        deep_config = ModelFactory.get_default_config('deep')
        required_keys = ['input_dim', 'hidden_dims', 'activation', 'dropout_rate', 'batch_norm']
        for key in required_keys:
            self.assertIn(key, deep_config)
    
    def test_default_config_unsupported_type(self):
        """Test that unsupported model types raise ValueError for default config."""
        with self.assertRaises(ValueError):
            ModelFactory.get_default_config('unsupported')
    
    def test_model_creation_with_defaults(self):
        """Test model creation using default configurations."""
        for model_type in ModelFactory.get_supported_models():
            with self.subTest(model_type=model_type):
                default_config = ModelFactory.get_default_config(model_type)
                model = ModelFactory.create_model(model_type, default_config)
                
                # Check that model was created successfully
                self.assertIsNotNone(model)
                self.assertEqual(model.input_dim, default_config['input_dim'])


class TestModelSerialization(unittest.TestCase):
    """Test cases for model serialization and loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
        self.test_model = ShallowNetwork(input_dim=1, hidden_dims=[32, 16], activation='relu')
        self.test_input = torch.randn(5, 1)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_model(self):
        """Test basic model save and load functionality."""
        save_path = os.path.join(self.temp_dir, "test_model")
        metadata = {"test": "serialization", "version": "1.0"}
        
        # Get original output
        original_output = self.test_model(self.test_input)
        
        # Save model
        self.test_model.save_model(save_path, metadata)
        
        # Check that files were created
        self.assertTrue(os.path.exists(f"{save_path}.pth"))
        self.assertTrue(os.path.exists(f"{save_path}_info.json"))
        
        # Load model
        loaded_model = ShallowNetwork.load_model(save_path)
        loaded_output = loaded_model(self.test_input)
        
        # Check outputs match
        torch.testing.assert_close(original_output, loaded_output, rtol=1e-5, atol=1e-6)
    
    def test_save_model_creates_correct_files(self):
        """Test that save_model creates the expected files with correct content."""
        save_path = os.path.join(self.temp_dir, "test_model")
        metadata = {"experiment": "test", "date": "2024-01-01"}
        
        self.test_model.save_model(save_path, metadata)
        
        # Check .pth file exists and contains expected keys
        checkpoint = torch.load(f"{save_path}.pth", weights_only=False)
        expected_keys = ['model_state_dict', 'model_class', 'model_parameters', 'timestamp', 'pytorch_version']
        for key in expected_keys:
            self.assertIn(key, checkpoint)
        
        # Check metadata was saved
        self.assertEqual(checkpoint['metadata'], metadata)
        
        # Check JSON info file
        with open(f"{save_path}_info.json", 'r') as f:
            info = json.load(f)
        
        self.assertEqual(info['model_class'], 'ShallowNetwork')
        self.assertIn('constructor_args', info)
        self.assertIn('model_parameters', info)
    
    def test_load_different_model_types(self):
        """Test loading different model types."""
        models_to_test = [
            LinearModel(input_dim=1),
            ShallowNetwork(input_dim=1, hidden_dims=[32]),
            DeepNetwork(input_dim=1, hidden_dims=[32, 16, 8])
        ]
        
        for i, model in enumerate(models_to_test):
            with self.subTest(model_type=type(model).__name__):
                save_path = os.path.join(self.temp_dir, f"model_{i}")
                
                # Save and load
                model.save_model(save_path)
                loaded_model = type(model).load_model(save_path)
                
                # Check type and architecture
                self.assertEqual(type(loaded_model), type(model))
                self.assertTrue(model.validate_architecture(loaded_model))
    
    def test_architecture_validation(self):
        """Test model architecture validation."""
        # Create models with same and different architectures
        model1 = ShallowNetwork(input_dim=1, hidden_dims=[32, 16], activation='relu')
        model2 = ShallowNetwork(input_dim=1, hidden_dims=[32, 16], activation='relu')  # Same
        model3 = ShallowNetwork(input_dim=1, hidden_dims=[64], activation='tanh')      # Different
        model4 = LinearModel(input_dim=1)                                              # Different type
        
        # Same architecture should validate
        self.assertTrue(model1.validate_architecture(model2))
        
        # Different architectures should not validate
        self.assertFalse(model1.validate_architecture(model3))
        self.assertFalse(model1.validate_architecture(model4))
    
    def test_save_without_metadata(self):
        """Test saving model without metadata."""
        save_path = os.path.join(self.temp_dir, "no_metadata_model")
        
        self.test_model.save_model(save_path)
        
        # Should still create files successfully
        self.assertTrue(os.path.exists(f"{save_path}.pth"))
        self.assertTrue(os.path.exists(f"{save_path}_info.json"))
        
        # Should be able to load
        loaded_model = ShallowNetwork.load_model(save_path)
        self.assertTrue(self.test_model.validate_architecture(loaded_model))
    
    def test_load_nonexistent_model(self):
        """Test that loading nonexistent model raises appropriate error."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent_model")
        
        with self.assertRaises(FileNotFoundError):
            LinearModel.load_model(nonexistent_path)


class TestModelCheckpoint(unittest.TestCase):
    """Test cases for ModelCheckpoint class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint = ModelCheckpoint(checkpoint_dir=self.temp_dir, save_best_only=False)
        self.model = LinearModel(input_dim=1)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint_all_epochs(self):
        """Test saving checkpoints for all epochs when save_best_only=False."""
        metrics_history = [
            {'val_loss': 1.0, 'train_loss': 1.2},
            {'val_loss': 0.8, 'train_loss': 0.9},
            {'val_loss': 0.6, 'train_loss': 0.7}
        ]
        
        saved_epochs = []
        for epoch, metrics in enumerate(metrics_history):
            was_saved = self.checkpoint.save_checkpoint(self.model, epoch, metrics)
            if was_saved:
                saved_epochs.append(epoch)
        
        # All epochs should be saved when save_best_only=False
        self.assertEqual(len(saved_epochs), len(metrics_history))
        
        # Check that best model file exists
        best_model_path = os.path.join(self.temp_dir, 'best_model.pth')
        self.assertTrue(os.path.exists(best_model_path))
    
    def test_save_checkpoint_best_only(self):
        """Test saving only best checkpoints when save_best_only=True."""
        checkpoint_best = ModelCheckpoint(checkpoint_dir=self.temp_dir, save_best_only=True)
        
        metrics_history = [
            {'val_loss': 1.0, 'train_loss': 1.2},
            {'val_loss': 1.2, 'train_loss': 1.1},  # Worse
            {'val_loss': 0.8, 'train_loss': 0.9},  # Better
            {'val_loss': 0.9, 'train_loss': 0.8}   # Worse
        ]
        
        saved_epochs = []
        for epoch, metrics in enumerate(metrics_history):
            was_saved = checkpoint_best.save_checkpoint(self.model, epoch, metrics)
            if was_saved:
                saved_epochs.append(epoch)
        
        # Only epochs 0 and 2 should be saved (initial and improvement)
        expected_saved = [0, 2]
        self.assertEqual(saved_epochs, expected_saved)
    
    def test_load_best_model(self):
        """Test loading the best saved model."""
        # Save some checkpoints
        metrics_history = [
            {'val_loss': 1.0, 'train_loss': 1.2},
            {'val_loss': 0.6, 'train_loss': 0.7},  # Best
            {'val_loss': 0.8, 'train_loss': 0.9}
        ]
        
        for epoch, metrics in enumerate(metrics_history):
            self.checkpoint.save_checkpoint(self.model, epoch, metrics)
        
        # Load best model
        best_model = self.checkpoint.load_best_model()
        
        # Check that it's the same architecture
        self.assertTrue(self.model.validate_architecture(best_model))
        self.assertIsInstance(best_model, LinearModel)
    
    def test_load_best_model_nonexistent(self):
        """Test that loading best model raises error when no model exists."""
        empty_checkpoint = ModelCheckpoint(checkpoint_dir=tempfile.mkdtemp())
        
        with self.assertRaises(FileNotFoundError):
            empty_checkpoint.load_best_model()
    
    def test_checkpoint_with_optimizer_state(self):
        """Test saving checkpoint with optimizer and scheduler state."""
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        metrics = {'val_loss': 0.5, 'train_loss': 0.6}
        
        was_saved = self.checkpoint.save_checkpoint(
            self.model, 0, metrics,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict()
        )
        
        self.assertTrue(was_saved)
        
        # Load and check that optimizer/scheduler states are saved
        checkpoint_path = os.path.join(self.temp_dir, 'best_model.pth')
        checkpoint_data = self.checkpoint.load_checkpoint(checkpoint_path)
        
        self.assertIn('optimizer_state_dict', checkpoint_data)
        self.assertIn('scheduler_state_dict', checkpoint_data)
    
    def test_monitor_metric_missing(self):
        """Test behavior when monitored metric is missing."""
        metrics_without_val_loss = {'train_loss': 0.5}
        
        was_saved = self.checkpoint.save_checkpoint(self.model, 0, metrics_without_val_loss)
        
        # Should return False when monitor metric is missing
        self.assertFalse(was_saved)
    
    def test_different_monitor_modes(self):
        """Test different monitoring modes (min vs max)."""
        # Test max mode (for metrics that should be maximized)
        checkpoint_max = ModelCheckpoint(
            checkpoint_dir=self.temp_dir, 
            monitor='accuracy', 
            mode='max'
        )
        
        metrics_history = [
            {'accuracy': 0.6, 'train_loss': 1.2},
            {'accuracy': 0.8, 'train_loss': 0.9},  # Better (higher)
            {'accuracy': 0.7, 'train_loss': 0.8}   # Worse (lower)
        ]
        
        saved_epochs = []
        for epoch, metrics in enumerate(metrics_history):
            was_saved = checkpoint_max.save_checkpoint(self.model, epoch, metrics)
            if was_saved:
                saved_epochs.append(epoch)
        
        # Epochs 0 and 1 should be saved (initial and improvement)
        expected_saved = [0, 1]
        self.assertEqual(saved_epochs, expected_saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)