"""
Comprehensive unit tests for training system implementation.
Tests training loop functionality, optimizer integration, and training utilities.
"""

import torch
import torch.nn as nn
import numpy as np
import unittest
import tempfile
import shutil
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.architectures import LinearModel, ShallowNetwork
from src.models.trainers import (
    Trainer, OptimizerFactory, OptimizerConfig, 
    EarlyStopping, LossTracker, GradientClipper,
    TrainingResults, LearningRateSchedulerFactory
)
from src.data.generators import DatasetBuilder

class TestOptimizerConfig(unittest.TestCase):
    """Test cases for OptimizerConfig dataclass."""
    
    def test_default_config(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        
        self.assertEqual(config.optimizer_type, 'adam')
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.weight_decay, 0.0)
        self.assertEqual(config.momentum, 0.9)
    
    def test_custom_config(self):
        """Test custom optimizer configuration."""
        config = OptimizerConfig(
            optimizer_type='sgd',
            learning_rate=0.01,
            momentum=0.95,
            weight_decay=1e-4
        )
        
        self.assertEqual(config.optimizer_type, 'sgd')
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.momentum, 0.95)
        self.assertEqual(config.weight_decay, 1e-4)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = OptimizerConfig(optimizer_type='adam', learning_rate=0.001)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['optimizer_type'], 'adam')
        self.assertEqual(config_dict['learning_rate'], 0.001)


class TestOptimizerFactory(unittest.TestCase):
    """Test cases for OptimizerFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.model = LinearModel(input_dim=1)
    
    def test_create_sgd_optimizer(self):
        """Test SGD optimizer creation."""
        config = OptimizerConfig(
            optimizer_type='sgd',
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        optimizer = OptimizerFactory.create_optimizer(self.model, config)
        
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(optimizer.param_groups[0]['momentum'], 0.9)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-4)
    
    def test_create_adam_optimizer(self):
        """Test Adam optimizer creation."""
        config = OptimizerConfig(
            optimizer_type='adam',
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=1e-4
        )
        
        optimizer = OptimizerFactory.create_optimizer(self.model, config)
        
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.001)
        self.assertEqual(optimizer.param_groups[0]['betas'], (0.9, 0.999))
        self.assertEqual(optimizer.param_groups[0]['eps'], 1e-8)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-4)
    
    def test_create_rmsprop_optimizer(self):
        """Test RMSprop optimizer creation."""
        config = OptimizerConfig(
            optimizer_type='rmsprop',
            learning_rate=0.01,
            alpha=0.99,
            eps=1e-8,
            weight_decay=1e-4,
            momentum=0.1,
            centered=True
        )
        
        optimizer = OptimizerFactory.create_optimizer(self.model, config)
        
        self.assertIsInstance(optimizer, torch.optim.RMSprop)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(optimizer.param_groups[0]['alpha'], 0.99)
        self.assertEqual(optimizer.param_groups[0]['eps'], 1e-8)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-4)
        self.assertEqual(optimizer.param_groups[0]['momentum'], 0.1)
        self.assertTrue(optimizer.param_groups[0]['centered'])
    
    def test_create_adagrad_optimizer(self):
        """Test AdaGrad optimizer creation."""
        config = OptimizerConfig(
            optimizer_type='adagrad',
            learning_rate=0.01,
            lr_decay=1e-4,
            weight_decay=1e-4,
            initial_accumulator_value=0.1,
            eps=1e-10
        )
        
        optimizer = OptimizerFactory.create_optimizer(self.model, config)
        
        self.assertIsInstance(optimizer, torch.optim.Adagrad)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(optimizer.param_groups[0]['lr_decay'], 1e-4)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-4)
        self.assertEqual(optimizer.param_groups[0]['initial_accumulator_value'], 0.1)
        self.assertEqual(optimizer.param_groups[0]['eps'], 1e-10)
    
    def test_unsupported_optimizer_type(self):
        """Test that unsupported optimizer types raise ValueError."""
        config = OptimizerConfig(optimizer_type='unsupported')
        
        with self.assertRaises(ValueError):
            OptimizerFactory.create_optimizer(self.model, config)
    
    def test_get_supported_optimizers(self):
        """Test getting list of supported optimizers."""
        supported = OptimizerFactory.get_supported_optimizers()
        
        expected_optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad']
        self.assertEqual(set(supported), set(expected_optimizers))
    
    def test_get_default_configs(self):
        """Test getting default configurations for each optimizer."""
        for optimizer_type in OptimizerFactory.get_supported_optimizers():
            with self.subTest(optimizer_type=optimizer_type):
                config = OptimizerFactory.get_default_config(optimizer_type)
                
                self.assertIsInstance(config, OptimizerConfig)
                self.assertEqual(config.optimizer_type, optimizer_type)
                
                # Should be able to create optimizer with default config
                optimizer = OptimizerFactory.create_optimizer(self.model, config)
                self.assertIsNotNone(optimizer)


class TestLossTracker(unittest.TestCase):
    """Test cases for LossTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, 'losses.json')
        self.tracker = LossTracker(save_path=self.save_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_loss_tracking(self):
        """Test basic loss tracking functionality."""
        # Track some losses
        losses = [(1.0, 1.2), (0.8, 0.9), (0.6, 0.7)]
        
        for train_loss, val_loss in losses:
            self.tracker.start_epoch()
            self.tracker.end_epoch(train_loss, val_loss)
        
        self.assertEqual(len(self.tracker.train_losses), 3)
        self.assertEqual(len(self.tracker.val_losses), 3)
        self.assertEqual(self.tracker.train_losses, [1.0, 0.8, 0.6])
        self.assertEqual(self.tracker.val_losses, [1.2, 0.9, 0.7])
    
    def test_get_current_epoch(self):
        """Test getting current epoch number."""
        self.assertEqual(self.tracker.get_current_epoch(), 0)
        
        self.tracker.end_epoch(1.0, 1.2)
        self.assertEqual(self.tracker.get_current_epoch(), 1)
        
        self.tracker.end_epoch(0.8, 0.9)
        self.assertEqual(self.tracker.get_current_epoch(), 2)
    
    def test_get_best_epoch(self):
        """Test getting best epoch based on validation loss."""
        losses = [(1.0, 1.2), (0.8, 0.7), (0.9, 0.8)]  # Best is epoch 1
        
        for train_loss, val_loss in losses:
            self.tracker.end_epoch(train_loss, val_loss)
        
        best_epoch, best_val_loss = self.tracker.get_best_epoch()
        self.assertEqual(best_epoch, 1)
        self.assertEqual(best_val_loss, 0.7)
    
    def test_get_loss_statistics(self):
        """Test getting comprehensive loss statistics."""
        losses = [(1.0, 1.2), (0.8, 0.9), (0.6, 0.7)]
        
        for train_loss, val_loss in losses:
            self.tracker.start_epoch()
            self.tracker.end_epoch(train_loss, val_loss)
        
        stats = self.tracker.get_loss_statistics()
        
        expected_keys = [
            'final_train_loss', 'final_val_loss', 'best_val_loss', 'best_epoch',
            'min_train_loss', 'max_train_loss', 'min_val_loss', 'max_val_loss',
            'avg_epoch_time', 'total_epochs'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['final_train_loss'], 0.6)
        self.assertEqual(stats['final_val_loss'], 0.7)
        self.assertEqual(stats['best_val_loss'], 0.7)
        self.assertEqual(stats['total_epochs'], 3)
    
    def test_save_and_load_losses(self):
        """Test saving and loading loss history."""
        losses = [(1.0, 1.2), (0.8, 0.9)]
        
        for train_loss, val_loss in losses:
            self.tracker.end_epoch(train_loss, val_loss)
        
        # Save should happen automatically
        self.assertTrue(os.path.exists(self.save_path))
        
        # Load into new tracker
        new_tracker = LossTracker()
        new_tracker.load_losses(self.save_path)
        
        self.assertEqual(new_tracker.train_losses, [1.0, 0.8])
        self.assertEqual(new_tracker.val_losses, [1.2, 0.9])


class TestGradientClipper(unittest.TestCase):
    """Test cases for GradientClipper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.model = LinearModel(input_dim=1)
        
        # Create some gradients
        x = torch.randn(10, 1)
        y = torch.randn(10, 1)
        loss = nn.MSELoss()(self.model(x), y)
        loss.backward()
    
    def test_gradient_norm_clipping(self):
        """Test gradient clipping by norm."""
        clipper = GradientClipper(clip_type='norm', clip_value=1.0)
        
        # Get original gradient norm
        original_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        
        # Reset gradients
        self.model.zero_grad()
        x = torch.randn(10, 1)
        y = torch.randn(10, 1)
        loss = nn.MSELoss()(self.model(x), y)
        loss.backward()
        
        # Apply clipping
        clipped_norm = clipper.clip_gradients(self.model)
        
        # Check that norm was clipped if it was originally > 1.0
        if original_norm > 1.0:
            self.assertLessEqual(clipped_norm, 1.0 + 1e-6)  # Small tolerance for numerical precision
    
    def test_gradient_value_clipping(self):
        """Test gradient clipping by value."""
        clipper = GradientClipper(clip_type='value', clip_value=0.5)
        
        # Apply clipping
        clipper.clip_gradients(self.model)
        
        # Check that all gradient values are within [-0.5, 0.5]
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.all(param.grad >= -0.5))
                self.assertTrue(torch.all(param.grad <= 0.5))
    
    def test_unsupported_clip_type(self):
        """Test that unsupported clip types raise ValueError."""
        with self.assertRaises(ValueError):
            GradientClipper(clip_type='unsupported', clip_value=1.0)


class TestEarlyStopping(unittest.TestCase):
    """Test cases for EarlyStopping class."""
    
    def test_early_stopping_patience(self):
        """Test early stopping with patience."""
        early_stopping = EarlyStopping(patience=3, verbose=False)
        
        # Simulate improving then worsening validation loss
        val_losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        should_stop_epochs = []
        for epoch, val_loss in enumerate(val_losses):
            early_stopping.step(val_loss)
            if early_stopping.should_stop():
                should_stop_epochs.append(epoch)
                break
        
        # Should stop after patience is exceeded (epoch 6: 3 epochs after best at epoch 2)
        self.assertEqual(should_stop_epochs, [5])  # 0-indexed, so epoch 6 is index 5
    
    def test_early_stopping_min_delta(self):
        """Test early stopping with minimum delta requirement."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1, verbose=False)
        
        # Small improvements that don't meet min_delta
        val_losses = [1.0, 0.95, 0.92, 0.91]
        
        for val_loss in val_losses:
            early_stopping.step(val_loss)
        
        # Should trigger early stopping because improvements are < min_delta
        self.assertTrue(early_stopping.should_stop())
    
    def test_early_stopping_mode_max(self):
        """Test early stopping in max mode (for metrics that should increase)."""
        early_stopping = EarlyStopping(patience=2, mode='max', verbose=False)
        
        # Simulate accuracy that increases then decreases
        accuracies = [0.6, 0.8, 0.9, 0.85, 0.8, 0.75]
        
        should_stop_epochs = []
        for epoch, accuracy in enumerate(accuracies):
            early_stopping.step(accuracy)
            if early_stopping.should_stop():
                should_stop_epochs.append(epoch)
                break
        
        # Should stop after patience is exceeded
        self.assertTrue(len(should_stop_epochs) > 0)


class TestTrainer(unittest.TestCase):
    """Test cases for Trainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create simple dataset
        self.dataset_builder = DatasetBuilder(random_seed=42)
        self.train_loader, self.val_loader, _ = self.dataset_builder.create_full_pipeline(
            degree=2, batch_size=16, num_points=100, noise_level=0.1
        )
        
        self.model = LinearModel(input_dim=1)
        self.trainer = Trainer(verbose=False)
    
    def test_basic_training(self):
        """Test basic training functionality."""
        optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
        optimizer = OptimizerFactory.create_optimizer(self.model, optimizer_config)
        
        results = self.trainer.train(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            epochs=5
        )
        
        self.assertIsInstance(results, TrainingResults)
        self.assertEqual(len(results.train_losses), 5)
        self.assertEqual(len(results.val_losses), 5)
        self.assertGreater(results.training_time, 0)
    
    def test_training_with_early_stopping(self):
        """Test training with early stopping."""
        optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
        optimizer = OptimizerFactory.create_optimizer(self.model, optimizer_config)
        early_stopping = EarlyStopping(patience=3, verbose=False)
        
        results = self.trainer.train(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            epochs=50,  # High number, but should stop early
            early_stopping=early_stopping
        )
        
        # Should stop before 50 epochs (early stopping should trigger)
        self.assertLess(len(results.train_losses), 50)
    
    def test_training_with_scheduler(self):
        """Test training with learning rate scheduler."""
        optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.1)
        optimizer = OptimizerFactory.create_optimizer(self.model, optimizer_config)
        scheduler = LearningRateSchedulerFactory.create_scheduler(
            optimizer, 'step', step_size=2, gamma=0.5
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        results = self.trainer.train(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            epochs=5,
            scheduler=scheduler
        )
        
        # Learning rate should have decreased
        final_lr = optimizer.param_groups[0]['lr']
        self.assertLess(final_lr, initial_lr)
    
    def test_model_evaluation(self):
        """Test model evaluation functionality."""
        # Train model briefly first
        optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
        optimizer = OptimizerFactory.create_optimizer(self.model, optimizer_config)
        
        self.trainer.train(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=optimizer,
            epochs=3
        )
        
        # Evaluate model
        metrics = self.trainer.evaluate(self.model, self.val_loader)
        
        expected_metrics = ['test_loss', 'num_samples', 'mse', 'rmse', 'mae', 'r2_score']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        self.assertGreater(metrics['num_samples'], 0)
        self.assertGreaterEqual(metrics['r2_score'], -1)  # R² can be negative for bad fits
    
    def test_predictions(self):
        """Test prediction functionality."""
        predictions, targets = self.trainer.predict(self.model, self.val_loader)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(targets, np.ndarray)
        self.assertEqual(predictions.shape, targets.shape)
        self.assertGreater(len(predictions), 0)
    
    def test_predictions_with_confidence(self):
        """Test predictions with confidence intervals."""
        predictions, targets, confidence_intervals = self.trainer.predict_with_confidence(
            self.model, self.val_loader, confidence_level=0.95
        )
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(targets, np.ndarray)
        self.assertIsInstance(confidence_intervals, np.ndarray)
        self.assertEqual(predictions.shape[0], confidence_intervals.shape[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)