#!/usr/bin/env python3
"""
Integration tests for complete AI curve fitting workflows.
Tests end-to-end experiment execution, configuration validation, and reproducibility.
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.generators import DatasetBuilder
from src.models.architectures import LinearModel, ShallowNetwork, DeepNetwork, ModelFactory
from src.models.trainers import Trainer, OptimizerFactory, OptimizerConfig, EarlyStopping, LossTracker
from src.experiments.config import ExperimentConfig
from src.experiments.runner import ExperimentRunner
from src.experiments.storage import ExperimentStorage
from src.visualization.plots import CurvePlotter, LossPlotter, ComparisonPlotter


class TestEndToEndExperimentExecution(unittest.TestCase):
    """Test complete end-to-end experiment execution workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.temp_dir = tempfile.mkdtemp()
        
        # Create basic experiment configuration
        self.config = ExperimentConfig(
            polynomial_degree=2,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture='shallow',
            hidden_dims=[32, 16],
            optimizer='adam',
            learning_rate=0.01,
            batch_size=16,
            epochs=10,
            random_seed=42,
            num_data_points=100
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_linear_model_workflow(self):
        """Test complete workflow with linear model."""
        # Create dataset
        dataset_builder = DatasetBuilder(random_seed=42)
        train_loader, val_loader, dataset_info = dataset_builder.create_full_pipeline(
            degree=2, batch_size=16, num_points=100, noise_level=0.1
        )
        
        # Create and train model
        model = LinearModel(input_dim=1)
        trainer = Trainer(verbose=False)
        
        optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
        optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
        
        # Train model
        results = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=5
        )
        
        # Verify training completed successfully
        self.assertEqual(len(results.train_losses), 5)
        self.assertEqual(len(results.val_losses), 5)
        self.assertGreater(results.training_time, 0)
        
        # Test evaluation
        eval_metrics = trainer.evaluate(model, val_loader)
        self.assertIn('test_loss', eval_metrics)
        self.assertIn('r2_score', eval_metrics)
        self.assertGreater(eval_metrics['num_samples'], 0)
        
        # Test predictions
        predictions, targets = trainer.predict(model, val_loader)
        self.assertEqual(predictions.shape, targets.shape)
        self.assertGreater(len(predictions), 0)
    
    def test_complete_shallow_network_workflow(self):
        """Test complete workflow with shallow network."""
        # Create dataset
        dataset_builder = DatasetBuilder(random_seed=42)
        train_loader, val_loader, dataset_info = dataset_builder.create_full_pipeline(
            degree=3, batch_size=16, num_points=100, noise_level=0.1
        )
        
        # Create and train model
        model = ShallowNetwork(input_dim=1, hidden_dims=[64, 32], activation='relu')
        trainer = Trainer(verbose=False)
        
        optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
        optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
        
        # Train with early stopping
        early_stopping = EarlyStopping(patience=5, verbose=False)
        loss_tracker = LossTracker()
        
        results = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=20,
            early_stopping=early_stopping,
            loss_tracker=loss_tracker
        )
        
        # Verify training completed
        self.assertGreater(len(results.train_losses), 0)
        self.assertLessEqual(len(results.train_losses), 20)  # May stop early
        
        # Test model serialization
        model_path = os.path.join(self.temp_dir, 'test_model')
        model.save_model(model_path)
        
        loaded_model = ShallowNetwork.load_model(model_path)
        self.assertTrue(model.validate_architecture(loaded_model))
        
        # Test that loaded model produces same outputs
        test_input = torch.randn(5, 1)
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        torch.testing.assert_close(original_output, loaded_output, rtol=1e-5, atol=1e-6)
    
    def test_complete_deep_network_workflow(self):
        """Test complete workflow with deep network."""
        # Create dataset
        dataset_builder = DatasetBuilder(random_seed=42)
        train_loader, val_loader, dataset_info = dataset_builder.create_full_pipeline(
            degree=4, batch_size=16, num_points=150, noise_level=0.15
        )
        
        # Create and train model with batch norm and dropout
        model = DeepNetwork(
            input_dim=1, 
            hidden_dims=[128, 64, 32], 
            activation='relu',
            batch_norm=True,
            dropout_rate=0.1
        )
        trainer = Trainer(verbose=False)
        
        # Use full pipeline training
        results = trainer.train_with_full_pipeline(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_config=OptimizerConfig(optimizer_type='adam', learning_rate=0.001),
            epochs=10,
            scheduler_type='step',
            scheduler_kwargs={'step_size': 5, 'gamma': 0.5},
            early_stopping_patience=8,
            gradient_clip_value=1.0
        )
        
        # Verify training completed
        self.assertGreater(len(results.train_losses), 0)
        self.assertGreater(results.training_time, 0)
        
        # Test evaluation with confidence intervals
        eval_metrics = trainer.evaluate(model, val_loader, calculate_confidence_intervals=True)
        self.assertIn('confidence_intervals', eval_metrics)
        self.assertIsInstance(eval_metrics['confidence_intervals'], np.ndarray)
    
    def test_model_factory_integration(self):
        """Test integration with ModelFactory for different model types."""
        dataset_builder = DatasetBuilder(random_seed=42)
        train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
            degree=2, batch_size=16, num_points=100
        )
        
        model_configs = [
            ('linear', {'input_dim': 1}),
            ('shallow', {'input_dim': 1, 'hidden_dims': [32], 'activation': 'relu'}),
            ('deep', {'input_dim': 1, 'hidden_dims': [64, 32, 16], 'activation': 'tanh'})
        ]
        
        trainer = Trainer(verbose=False)
        
        for model_type, config in model_configs:
            with self.subTest(model_type=model_type):
                # Create model using factory
                model = ModelFactory.create_model(model_type, config)
                
                # Train briefly
                optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
                optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
                
                results = trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=3
                )
                
                # Verify training worked
                self.assertEqual(len(results.train_losses), 3)
                self.assertGreater(results.training_time, 0)
    
    def test_different_optimizers_integration(self):
        """Test integration with different optimizers."""
        dataset_builder = DatasetBuilder(random_seed=42)
        train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
            degree=2, batch_size=16, num_points=100
        )
        
        optimizers_to_test = ['sgd', 'adam', 'rmsprop', 'adagrad']
        trainer = Trainer(verbose=False)
        
        for optimizer_type in optimizers_to_test:
            with self.subTest(optimizer_type=optimizer_type):
                # Create fresh model for each optimizer
                model = LinearModel(input_dim=1)
                
                # Get default config and create optimizer
                optimizer_config = OptimizerFactory.get_default_config(optimizer_type)
                optimizer_config.learning_rate = 0.01  # Standardize learning rate
                optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
                
                # Train model
                results = trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=5
                )
                
                # Verify training completed
                self.assertEqual(len(results.train_losses), 5)
                self.assertGreater(results.training_time, 0)
                
                # Verify loss decreased (learning occurred)
                self.assertLess(results.final_train_loss, results.train_losses[0])


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_experiment_config_validation(self):
        """Test experiment configuration validation."""
        # Test valid configuration
        valid_config = ExperimentConfig(
            polynomial_degree=3,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture='shallow',
            hidden_dims=[32, 16],
            optimizer='adam',
            learning_rate=0.01,
            batch_size=16,
            epochs=10,
            random_seed=42
        )
        
        # Should not raise any errors
        self.assertEqual(valid_config.polynomial_degree, 3)
        self.assertEqual(valid_config.model_architecture, 'shallow')
        
        # Test configuration serialization
        config_dict = valid_config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['polynomial_degree'], 3)
        
        # Test configuration from dictionary
        new_config = ExperimentConfig.from_dict(config_dict)
        self.assertEqual(new_config.polynomial_degree, valid_config.polynomial_degree)
        self.assertEqual(new_config.model_architecture, valid_config.model_architecture)
    
    def test_invalid_model_architecture_handling(self):
        """Test handling of invalid model architectures."""
        dataset_builder = DatasetBuilder(random_seed=42)
        train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
            degree=2, batch_size=16, num_points=50
        )
        
        # Test unsupported model type
        with self.assertRaises(ValueError):
            ModelFactory.create_model('unsupported_model', {'input_dim': 1})
    
    def test_invalid_optimizer_handling(self):
        """Test handling of invalid optimizer configurations."""
        model = LinearModel(input_dim=1)
        
        # Test unsupported optimizer type
        invalid_config = OptimizerConfig(optimizer_type='unsupported_optimizer')
        
        with self.assertRaises(ValueError):
            OptimizerFactory.create_optimizer(model, invalid_config)
    
    def test_data_generation_error_handling(self):
        """Test error handling in data generation."""
        generator = DatasetBuilder(random_seed=42)
        
        # Test invalid polynomial degree
        with self.assertRaises(ValueError):
            generator.build_dataset(degree=0, num_points=100)
        
        with self.assertRaises(ValueError):
            generator.build_dataset(degree=7, num_points=100)
        
        # Test invalid number of points
        with self.assertRaises(ValueError):
            generator.build_dataset(degree=2, num_points=1)
        
        # Test invalid train ratio
        with self.assertRaises(ValueError):
            generator.build_dataset(degree=2, num_points=100, train_ratio=0.0)
        
        with self.assertRaises(ValueError):
            generator.build_dataset(degree=2, num_points=100, train_ratio=1.0)
    
    def test_model_architecture_validation(self):
        """Test model architecture validation."""
        # Test shallow network with too many layers
        with self.assertRaises(ValueError):
            ShallowNetwork(input_dim=1, hidden_dims=[64, 32, 16])
        
        # Test deep network with too few layers
        with self.assertRaises(ValueError):
            DeepNetwork(input_dim=1, hidden_dims=[64, 32])
        
        # Test unsupported activation functions
        with self.assertRaises(ValueError):
            ShallowNetwork(input_dim=1, hidden_dims=[32], activation='unsupported')


class TestReproducibilityWithFixedSeeds(unittest.TestCase):
    """Test reproducibility of experiments with fixed random seeds."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.seed = 12345
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_data_generation_reproducibility(self):
        """Test that data generation is reproducible with fixed seeds."""
        # Generate data with same seed twice
        builder1 = DatasetBuilder(random_seed=self.seed)
        dataset1 = builder1.build_dataset(degree=3, num_points=100, noise_level=0.1)
        
        builder2 = DatasetBuilder(random_seed=self.seed)
        dataset2 = builder2.build_dataset(degree=3, num_points=100, noise_level=0.1)
        
        # Check that generated data is identical
        np.testing.assert_array_equal(
            dataset1['raw_data']['x_data'], 
            dataset2['raw_data']['x_data']
        )
        np.testing.assert_array_equal(
            dataset1['raw_data']['y_clean'], 
            dataset2['raw_data']['y_clean']
        )
        np.testing.assert_array_equal(
            dataset1['raw_data']['y_noisy'], 
            dataset2['raw_data']['y_noisy']
        )
        np.testing.assert_array_equal(
            dataset1['raw_data']['coefficients'], 
            dataset2['raw_data']['coefficients']
        )
    
    def test_model_training_reproducibility(self):
        """Test that model training is reproducible with fixed seeds."""
        # Set up identical training scenarios
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Create dataset
        dataset_builder = DatasetBuilder(random_seed=self.seed)
        train_loader1, val_loader1, _ = dataset_builder.create_full_pipeline(
            degree=2, batch_size=16, num_points=100, noise_level=0.1
        )
        
        # Train first model
        torch.manual_seed(self.seed)
        model1 = LinearModel(input_dim=1)
        trainer1 = Trainer(verbose=False)
        optimizer_config1 = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
        optimizer1 = OptimizerFactory.create_optimizer(model1, optimizer_config1)
        
        results1 = trainer1.train(
            model=model1,
            train_loader=train_loader1,
            val_loader=val_loader1,
            optimizer=optimizer1,
            epochs=5
        )
        
        # Reset seeds and train second model
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        dataset_builder2 = DatasetBuilder(random_seed=self.seed)
        train_loader2, val_loader2, _ = dataset_builder2.create_full_pipeline(
            degree=2, batch_size=16, num_points=100, noise_level=0.1
        )
        
        torch.manual_seed(self.seed)
        model2 = LinearModel(input_dim=1)
        trainer2 = Trainer(verbose=False)
        optimizer_config2 = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
        optimizer2 = OptimizerFactory.create_optimizer(model2, optimizer_config2)
        
        results2 = trainer2.train(
            model=model2,
            train_loader=train_loader2,
            val_loader=val_loader2,
            optimizer=optimizer2,
            epochs=5
        )
        
        # Check that training results are identical
        np.testing.assert_array_almost_equal(
            results1.train_losses, results2.train_losses, decimal=6
        )
        np.testing.assert_array_almost_equal(
            results1.val_losses, results2.val_losses, decimal=6
        )
        
        # Check that final model parameters are identical
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            self.assertEqual(name1, name2)
            torch.testing.assert_close(param1, param2, rtol=1e-6, atol=1e-8)
    
    def test_experiment_config_reproducibility(self):
        """Test that experiment configurations produce reproducible results."""
        config = ExperimentConfig(
            polynomial_degree=2,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture='linear',
            hidden_dims=[],  # Empty for linear model
            optimizer='adam',
            learning_rate=0.01,
            batch_size=16,
            epochs=5,
            random_seed=self.seed,
            num_data_points=100
        )
        
        # Run experiment twice with same configuration
        results1 = self._run_experiment_with_config(config)
        results2 = self._run_experiment_with_config(config)
        
        # Check that results are identical
        np.testing.assert_array_almost_equal(
            results1['train_losses'], results2['train_losses'], decimal=6
        )
        np.testing.assert_array_almost_equal(
            results1['val_losses'], results2['val_losses'], decimal=6
        )
    
    def _run_experiment_with_config(self, config):
        """Helper method to run experiment with given configuration."""
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Create dataset
        dataset_builder = DatasetBuilder(random_seed=config.random_seed)
        train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
            degree=config.polynomial_degree,
            batch_size=config.batch_size,
            num_points=config.num_data_points,
            noise_level=config.noise_level,
            train_ratio=config.train_val_split
        )
        
        # Create model
        torch.manual_seed(config.random_seed)
        if config.model_architecture == 'linear':
            model = LinearModel(input_dim=1)
        elif config.model_architecture == 'shallow':
            model = ShallowNetwork(input_dim=1, hidden_dims=config.hidden_dims)
        elif config.model_architecture == 'deep':
            model = DeepNetwork(input_dim=1, hidden_dims=config.hidden_dims)
        else:
            raise ValueError(f"Unsupported model architecture: {config.model_architecture}")
        
        # Train model
        trainer = Trainer(verbose=False)
        optimizer_config = OptimizerConfig(
            optimizer_type=config.optimizer,
            learning_rate=config.learning_rate
        )
        optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
        
        results = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=config.epochs
        )
        
        return {
            'train_losses': results.train_losses,
            'val_losses': results.val_losses,
            'training_time': results.training_time
        }


class TestExperimentStorageIntegration(unittest.TestCase):
    """Test integration with experiment storage and retrieval systems."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ExperimentStorage(storage_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_experiment_storage_and_retrieval(self):
        """Test storing and retrieving experiment results."""
        # Create and run a simple experiment
        torch.manual_seed(42)
        np.random.seed(42)
        
        dataset_builder = DatasetBuilder(random_seed=42)
        train_loader, val_loader, dataset_info = dataset_builder.create_full_pipeline(
            degree=2, batch_size=16, num_points=100
        )
        
        model = LinearModel(input_dim=1)
        trainer = Trainer(verbose=False)
        optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
        optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
        
        results = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=5
        )
        
        # Create experiment configuration
        config = ExperimentConfig(
            polynomial_degree=2,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture='linear',
            hidden_dims=[],  # Empty for linear model
            optimizer='adam',
            learning_rate=0.01,
            batch_size=16,
            epochs=5,
            random_seed=42
        )
        
        # Create ExperimentResults object
        from src.experiments.runner import ExperimentResults
        from datetime import datetime
        import uuid
        
        experiment_results = ExperimentResults(
            config=config,
            experiment_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            duration_seconds=results.training_time,
            training_results=results,
            final_train_loss=results.final_train_loss,
            final_val_loss=results.final_val_loss,
            best_val_loss=results.best_val_loss,
            training_time=results.training_time,
            convergence_epoch=results.best_epoch,
            model_parameters=results.model_parameters,
            model_size=model.count_parameters(),
            status="completed"
        )
        
        # Store experiment results
        success = self.storage.store_experiment(experiment_results)
        self.assertTrue(success)
        
        experiment_id = experiment_results.experiment_id
        
        self.assertIsNotNone(experiment_id)
        
        # Retrieve experiment results
        retrieved_experiment = self.storage.load_experiment(experiment_id)
        
        self.assertIsNotNone(retrieved_experiment)
        self.assertEqual(retrieved_experiment.config.polynomial_degree, 2)
        self.assertEqual(retrieved_experiment.config.model_architecture, 'linear')
        
        # Check that basic metrics were stored correctly
        self.assertEqual(retrieved_experiment.final_train_loss, results.final_train_loss)
        self.assertEqual(retrieved_experiment.final_val_loss, results.final_val_loss)
        self.assertEqual(retrieved_experiment.training_time, results.training_time)
    
    def test_multiple_experiments_storage(self):
        """Test storing and managing multiple experiments."""
        experiments = []
        
        # Create multiple experiments with different configurations
        configs = [
            {'degree': 2, 'architecture': 'linear', 'lr': 0.01},
            {'degree': 3, 'architecture': 'linear', 'lr': 0.001},
            {'degree': 2, 'architecture': 'shallow', 'lr': 0.01}
        ]
        
        for i, config_params in enumerate(configs):
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            
            # Create dataset
            dataset_builder = DatasetBuilder(random_seed=42 + i)
            train_loader, val_loader, dataset_info = dataset_builder.create_full_pipeline(
                degree=config_params['degree'], batch_size=16, num_points=50
            )
            
            # Create model
            if config_params['architecture'] == 'linear':
                model = LinearModel(input_dim=1)
            else:
                model = ShallowNetwork(input_dim=1, hidden_dims=[32])
            
            # Train model
            trainer = Trainer(verbose=False)
            optimizer_config = OptimizerConfig(
                optimizer_type='adam', 
                learning_rate=config_params['lr']
            )
            optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
            
            results = trainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                epochs=3
            )
            
            # Create and store experiment
            hidden_dims = [] if config_params['architecture'] == 'linear' else [32]
            config = ExperimentConfig(
                polynomial_degree=config_params['degree'],
                noise_level=0.1,
                train_val_split=0.8,
                model_architecture=config_params['architecture'],
                hidden_dims=hidden_dims,
                optimizer='adam',
                learning_rate=config_params['lr'],
                batch_size=16,
                epochs=3,
                random_seed=42 + i
            )
            
            # Create ExperimentResults object
            from src.experiments.runner import ExperimentResults
            from datetime import datetime
            import uuid
            
            experiment_results = ExperimentResults(
                config=config,
                experiment_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                duration_seconds=results.training_time,
                training_results=results,
                final_train_loss=results.final_train_loss,
                final_val_loss=results.final_val_loss,
                best_val_loss=results.best_val_loss,
                training_time=results.training_time,
                convergence_epoch=results.best_epoch,
                model_parameters=results.model_parameters,
                model_size=model.count_parameters(),
                status="completed"
            )
            
            success = self.storage.store_experiment(experiment_results)
            self.assertTrue(success)
            
            experiment_id = experiment_results.experiment_id
            
            experiments.append(experiment_id)
        
        # Verify all experiments were stored
        self.assertEqual(len(experiments), 3)
        
        # Test experiment querying
        all_experiments = self.storage.query_experiments()
        self.assertGreaterEqual(len(all_experiments), 3)
        
        # Test filtering by model architecture (simplified test)
        linear_count = sum(1 for exp in all_experiments 
                          if exp.config.model_architecture == 'linear')
        shallow_count = sum(1 for exp in all_experiments 
                           if exp.config.model_architecture == 'shallow')
        
        self.assertEqual(linear_count, 2)
        self.assertEqual(shallow_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)