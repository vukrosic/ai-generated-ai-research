#!/usr/bin/env python3
"""
Performance and scalability tests for AI curve fitting system.
Tests training time, memory usage, dataset scaling, and cross-platform compatibility.
"""

import sys
import os
import unittest
import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create a mock psutil for basic functionality
    class MockProcess:
        def memory_info(self):
            return type('MemInfo', (), {'rss': 100 * 1024 * 1024})()  # 100MB mock
    
    class MockPsutil:
        def Process(self):
            return MockProcess()
    
    psutil = MockPsutil()
import platform
import torch
import numpy as np
import tempfile
import shutil
from typing import Dict, List, Tuple, Any
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.generators import DatasetBuilder
from src.models.architectures import LinearModel, ShallowNetwork, DeepNetwork, ModelFactory
from src.models.trainers import Trainer, OptimizerFactory, OptimizerConfig
from src.experiments.config import ExperimentConfig


class PerformanceMonitor:
    """Utility class for monitoring performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        self.monitoring = False
        self.memory_samples = []
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.monitoring = True
        self.memory_samples = []
        
        # Start memory monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_memory)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return performance metrics."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'duration_seconds': end_time - self.start_time,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': end_memory - self.start_memory,
            'avg_memory_mb': np.mean(self.memory_samples) if self.memory_samples else end_memory
        }
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread."""
        while self.monitoring:
            try:
                current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                self.peak_memory = max(self.peak_memory, current_memory)
                self.memory_samples.append(current_memory)
                time.sleep(0.1)  # Sample every 100ms
            except:
                break


class TestTrainingPerformance(unittest.TestCase):
    """Test training time and memory usage benchmarks."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.monitor = PerformanceMonitor()
        
        # Performance thresholds (adjust based on expected performance)
        self.max_training_time_per_epoch = {
            'linear': 2.0,      # seconds per epoch
            'shallow': 5.0,     # seconds per epoch  
            'deep': 10.0        # seconds per epoch
        }
        
        self.max_memory_increase = {
            'linear': 50,       # MB
            'shallow': 100,     # MB
            'deep': 200         # MB
        }
    
    def test_linear_model_training_performance(self):
        """Test linear model training performance."""
        self._test_model_performance('linear', LinearModel(input_dim=1))
    
    def test_shallow_network_training_performance(self):
        """Test shallow network training performance."""
        model = ShallowNetwork(input_dim=1, hidden_dims=[64, 32], activation='relu')
        self._test_model_performance('shallow', model)
    
    def test_deep_network_training_performance(self):
        """Test deep network training performance."""
        model = DeepNetwork(input_dim=1, hidden_dims=[128, 64, 32], activation='relu')
        self._test_model_performance('deep', model)
    
    def _test_model_performance(self, model_type: str, model):
        """Helper method to test model performance."""
        # Create dataset
        dataset_builder = DatasetBuilder(random_seed=42)
        train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
            degree=3, batch_size=32, num_points=1000, noise_level=0.1
        )
        
        # Set up training
        trainer = Trainer(verbose=False)
        optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.001)
        optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
        
        # Monitor performance
        self.monitor.start_monitoring()
        
        # Train model
        results = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=10
        )
        
        # Get performance metrics
        perf_metrics = self.monitor.stop_monitoring()
        
        # Verify training completed successfully
        self.assertEqual(len(results.train_losses), 10)
        self.assertGreater(results.training_time, 0)
        
        # Check performance thresholds
        time_per_epoch = perf_metrics['duration_seconds'] / 10
        self.assertLess(
            time_per_epoch, 
            self.max_training_time_per_epoch[model_type],
            f"{model_type} model training too slow: {time_per_epoch:.2f}s per epoch"
        )
        
        self.assertLess(
            perf_metrics['memory_increase_mb'],
            self.max_memory_increase[model_type],
            f"{model_type} model uses too much memory: {perf_metrics['memory_increase_mb']:.1f}MB increase"
        )
        
        print(f"{model_type} model performance:")
        print(f"  Time per epoch: {time_per_epoch:.2f}s")
        print(f"  Memory increase: {perf_metrics['memory_increase_mb']:.1f}MB")
        print(f"  Peak memory: {perf_metrics['peak_memory_mb']:.1f}MB")
    
    def test_optimizer_performance_comparison(self):
        """Test performance differences between optimizers."""
        optimizers = ['sgd', 'adam', 'rmsprop', 'adagrad']
        performance_results = {}
        
        for optimizer_type in optimizers:
            with self.subTest(optimizer=optimizer_type):
                # Create fresh model and data for each optimizer
                torch.manual_seed(42)
                model = ShallowNetwork(input_dim=1, hidden_dims=[32], activation='relu')
                
                dataset_builder = DatasetBuilder(random_seed=42)
                train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
                    degree=2, batch_size=16, num_points=500
                )
                
                # Set up training
                trainer = Trainer(verbose=False)
                optimizer_config = OptimizerFactory.get_default_config(optimizer_type)
                optimizer_config.learning_rate = 0.01  # Standardize learning rate
                optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
                
                # Monitor performance
                self.monitor.start_monitoring()
                
                # Train model
                results = trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=5
                )
                
                # Get performance metrics
                perf_metrics = self.monitor.stop_monitoring()
                
                performance_results[optimizer_type] = {
                    'time_per_epoch': perf_metrics['duration_seconds'] / 5,
                    'memory_increase': perf_metrics['memory_increase_mb'],
                    'final_loss': results.final_val_loss
                }
                
                # Verify training worked
                self.assertEqual(len(results.train_losses), 5)
                self.assertGreater(results.training_time, 0)
        
        # Print comparison results
        print("\nOptimizer Performance Comparison:")
        for opt, metrics in performance_results.items():
            print(f"  {opt.upper()}: {metrics['time_per_epoch']:.3f}s/epoch, "
                  f"{metrics['memory_increase']:.1f}MB, loss={metrics['final_loss']:.6f}")
    
    def test_batch_size_performance_scaling(self):
        """Test how performance scales with batch size."""
        batch_sizes = [8, 16, 32, 64, 128]
        performance_results = {}
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Create model and data
                torch.manual_seed(42)
                model = LinearModel(input_dim=1)
                
                dataset_builder = DatasetBuilder(random_seed=42)
                train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
                    degree=2, batch_size=batch_size, num_points=1000
                )
                
                # Set up training
                trainer = Trainer(verbose=False)
                optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
                optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
                
                # Monitor performance
                self.monitor.start_monitoring()
                
                # Train model
                results = trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=3
                )
                
                # Get performance metrics
                perf_metrics = self.monitor.stop_monitoring()
                
                performance_results[batch_size] = {
                    'time_per_epoch': perf_metrics['duration_seconds'] / 3,
                    'memory_increase': perf_metrics['memory_increase_mb']
                }
                
                # Verify training worked
                self.assertEqual(len(results.train_losses), 3)
        
        # Print scaling results
        print("\nBatch Size Performance Scaling:")
        for batch_size, metrics in performance_results.items():
            print(f"  Batch {batch_size}: {metrics['time_per_epoch']:.3f}s/epoch, "
                  f"{metrics['memory_increase']:.1f}MB")
        
        # Verify that larger batch sizes don't dramatically increase memory
        max_memory = max(metrics['memory_increase'] for metrics in performance_results.values())
        min_memory = min(metrics['memory_increase'] for metrics in performance_results.values())
        memory_ratio = max_memory / max(min_memory, 1)  # Avoid division by zero
        
        self.assertLess(memory_ratio, 5.0, 
                       f"Memory usage scales too dramatically with batch size: {memory_ratio:.1f}x")


class TestDatasetScaling(unittest.TestCase):
    """Test performance with different dataset sizes and model complexities."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.monitor = PerformanceMonitor()
    
    def test_dataset_size_scaling(self):
        """Test how performance scales with dataset size."""
        dataset_sizes = [100, 500, 1000, 2000, 5000]
        performance_results = {}
        
        for size in dataset_sizes:
            with self.subTest(dataset_size=size):
                # Create model and data
                torch.manual_seed(42)
                model = ShallowNetwork(input_dim=1, hidden_dims=[32], activation='relu')
                
                dataset_builder = DatasetBuilder(random_seed=42)
                train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
                    degree=2, batch_size=32, num_points=size, noise_level=0.1
                )
                
                # Set up training
                trainer = Trainer(verbose=False)
                optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
                optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
                
                # Monitor performance
                self.monitor.start_monitoring()
                
                # Train model
                results = trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=5
                )
                
                # Get performance metrics
                perf_metrics = self.monitor.stop_monitoring()
                
                performance_results[size] = {
                    'time_per_epoch': perf_metrics['duration_seconds'] / 5,
                    'memory_increase': perf_metrics['memory_increase_mb'],
                    'final_loss': results.final_val_loss
                }
                
                # Verify training worked
                self.assertEqual(len(results.train_losses), 5)
        
        # Print scaling results
        print("\nDataset Size Performance Scaling:")
        for size, metrics in performance_results.items():
            print(f"  {size} points: {metrics['time_per_epoch']:.3f}s/epoch, "
                  f"{metrics['memory_increase']:.1f}MB, loss={metrics['final_loss']:.6f}")
        
        # Verify reasonable scaling (should be roughly linear or sub-linear)
        largest_size = max(dataset_sizes)
        smallest_size = min(dataset_sizes)
        
        largest_time = performance_results[largest_size]['time_per_epoch']
        smallest_time = performance_results[smallest_size]['time_per_epoch']
        
        size_ratio = largest_size / smallest_size
        time_ratio = largest_time / max(smallest_time, 0.001)  # Avoid division by zero
        
        # Time should scale no worse than quadratically with dataset size
        self.assertLess(time_ratio, size_ratio ** 2, 
                       f"Training time scales too poorly with dataset size: {time_ratio:.1f}x for {size_ratio:.1f}x data")
    
    def test_model_complexity_scaling(self):
        """Test how performance scales with model complexity."""
        model_configs = [
            ('linear', LinearModel(input_dim=1)),
            ('shallow_small', ShallowNetwork(input_dim=1, hidden_dims=[16], activation='relu')),
            ('shallow_medium', ShallowNetwork(input_dim=1, hidden_dims=[64, 32], activation='relu')),
            ('deep_small', DeepNetwork(input_dim=1, hidden_dims=[32, 16, 8], activation='relu')),
            ('deep_large', DeepNetwork(input_dim=1, hidden_dims=[128, 64, 32], activation='relu'))
        ]
        
        performance_results = {}
        
        for model_name, model in model_configs:
            with self.subTest(model=model_name):
                # Create dataset
                dataset_builder = DatasetBuilder(random_seed=42)
                train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
                    degree=3, batch_size=32, num_points=1000, noise_level=0.1
                )
                
                # Set up training
                trainer = Trainer(verbose=False)
                optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.001)
                optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
                
                # Monitor performance
                self.monitor.start_monitoring()
                
                # Train model
                results = trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=5
                )
                
                # Get performance metrics
                perf_metrics = self.monitor.stop_monitoring()
                
                performance_results[model_name] = {
                    'parameters': model.count_parameters(),
                    'time_per_epoch': perf_metrics['duration_seconds'] / 5,
                    'memory_increase': perf_metrics['memory_increase_mb'],
                    'final_loss': results.final_val_loss
                }
                
                # Verify training worked
                self.assertEqual(len(results.train_losses), 5)
        
        # Print complexity scaling results
        print("\nModel Complexity Performance Scaling:")
        for model_name, metrics in performance_results.items():
            print(f"  {model_name}: {metrics['parameters']} params, "
                  f"{metrics['time_per_epoch']:.3f}s/epoch, "
                  f"{metrics['memory_increase']:.1f}MB, loss={metrics['final_loss']:.6f}")
    
    def test_polynomial_degree_scaling(self):
        """Test performance with different polynomial degrees."""
        degrees = [1, 2, 3, 4, 5, 6]
        performance_results = {}
        
        for degree in degrees:
            with self.subTest(degree=degree):
                # Create model and data
                torch.manual_seed(42)
                model = ShallowNetwork(input_dim=1, hidden_dims=[64, 32], activation='relu')
                
                dataset_builder = DatasetBuilder(random_seed=42)
                train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
                    degree=degree, batch_size=32, num_points=1000, noise_level=0.1
                )
                
                # Set up training
                trainer = Trainer(verbose=False)
                optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.001)
                optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
                
                # Monitor performance
                self.monitor.start_monitoring()
                
                # Train model
                results = trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=5
                )
                
                # Get performance metrics
                perf_metrics = self.monitor.stop_monitoring()
                
                performance_results[degree] = {
                    'time_per_epoch': perf_metrics['duration_seconds'] / 5,
                    'memory_increase': perf_metrics['memory_increase_mb'],
                    'final_loss': results.final_val_loss
                }
                
                # Verify training worked
                self.assertEqual(len(results.train_losses), 5)
        
        # Print degree scaling results
        print("\nPolynomial Degree Performance Scaling:")
        for degree, metrics in performance_results.items():
            print(f"  Degree {degree}: {metrics['time_per_epoch']:.3f}s/epoch, "
                  f"{metrics['memory_increase']:.1f}MB, loss={metrics['final_loss']:.6f}")


class TestCrossPlatformCompatibility(unittest.TestCase):
    """Test cross-platform compatibility and system-specific performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.system_info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'numpy_version': np.__version__
        }
    
    def test_system_information_collection(self):
        """Test that system information can be collected properly."""
        print(f"\nSystem Information:")
        for key, value in self.system_info.items():
            print(f"  {key}: {value}")
            self.assertIsNotNone(value, f"Could not collect {key}")
    
    def test_torch_device_compatibility(self):
        """Test PyTorch device compatibility (CPU/GPU)."""
        # Test CPU device
        cpu_device = torch.device('cpu')
        model_cpu = LinearModel(input_dim=1).to(cpu_device)
        
        x_cpu = torch.randn(10, 1, device=cpu_device)
        output_cpu = model_cpu(x_cpu)
        
        self.assertEqual(output_cpu.device, cpu_device)
        self.assertEqual(output_cpu.shape, (10, 1))
        
        # Test GPU device if available
        if torch.cuda.is_available():
            print("CUDA available - testing GPU compatibility")
            gpu_device = torch.device('cuda')
            model_gpu = LinearModel(input_dim=1).to(gpu_device)
            
            x_gpu = torch.randn(10, 1, device=gpu_device)
            output_gpu = model_gpu(x_gpu)
            
            self.assertEqual(output_gpu.device, gpu_device)
            self.assertEqual(output_gpu.shape, (10, 1))
        else:
            print("CUDA not available - skipping GPU tests")
    
    def test_memory_management(self):
        """Test memory management and cleanup."""
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available - skipping memory management test")
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create and train multiple models to test memory cleanup
        for i in range(3):
            # Create model and data
            model = DeepNetwork(input_dim=1, hidden_dims=[128, 64, 32], activation='relu')
            
            dataset_builder = DatasetBuilder(random_seed=42 + i)
            train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
                degree=3, batch_size=32, num_points=500
            )
            
            # Train briefly
            trainer = Trainer(verbose=False)
            optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.001)
            optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
            
            results = trainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                epochs=2
            )
            
            # Verify training worked
            self.assertEqual(len(results.train_losses), 2)
            
            # Clean up
            del model, train_loader, val_loader, trainer, optimizer, results
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Check final memory usage
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
              f"(+{memory_increase:.1f}MB)")
        
        # Memory increase should be reasonable (less than 500MB for this test)
        self.assertLess(memory_increase, 500, 
                       f"Excessive memory usage: {memory_increase:.1f}MB increase")
    
    def test_numerical_stability(self):
        """Test numerical stability across different platforms."""
        # Test with different data ranges and noise levels
        test_cases = [
            {'x_range': (-1, 1), 'noise': 0.01, 'name': 'small_range_low_noise'},
            {'x_range': (-10, 10), 'noise': 0.1, 'name': 'large_range_high_noise'},
            {'x_range': (-0.1, 0.1), 'noise': 0.001, 'name': 'tiny_range_tiny_noise'}
        ]
        
        for i, case in enumerate(test_cases):
            with self.subTest(case=case['name']):
                # Create model
                model = LinearModel(input_dim=1)
                
                # Create dataset
                dataset_builder = DatasetBuilder(random_seed=42)
                train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
                    degree=2, batch_size=16, num_points=100, 
                    noise_level=case['noise'], x_range=case['x_range']
                )
                
                # Train model
                trainer = Trainer(verbose=False)
                optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
                optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
                
                results = trainer.train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=3
                )
                
                # Verify numerical stability
                self.assertEqual(len(results.train_losses), 3)
                self.assertFalse(np.isnan(results.final_train_loss), 
                               f"NaN loss with case {case['name']}")
                self.assertFalse(np.isinf(results.final_train_loss), 
                               f"Inf loss with case {case['name']}")
                self.assertGreater(results.training_time, 0)
        
        # Test dtype compatibility separately
        model_float32 = LinearModel(input_dim=1)
        model_float64 = LinearModel(input_dim=1).double()
        
        # Test float32
        x_float32 = torch.randn(5, 1, dtype=torch.float32)
        output_float32 = model_float32(x_float32)
        self.assertEqual(output_float32.dtype, torch.float32)
        
        # Test float64
        x_float64 = torch.randn(5, 1, dtype=torch.float64)
        output_float64 = model_float64(x_float64)
        self.assertEqual(output_float64.dtype, torch.float64)
    
    def test_concurrent_training(self):
        """Test concurrent training scenarios."""
        def train_model(model_id):
            """Train a single model."""
            torch.manual_seed(42 + model_id)
            
            # Create model and data
            model = ShallowNetwork(input_dim=1, hidden_dims=[32], activation='relu')
            
            dataset_builder = DatasetBuilder(random_seed=42 + model_id)
            train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
                degree=2, batch_size=16, num_points=200
            )
            
            # Train model
            trainer = Trainer(verbose=False)
            optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
            optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
            
            results = trainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                epochs=3
            )
            
            return {
                'model_id': model_id,
                'final_loss': results.final_val_loss,
                'training_time': results.training_time
            }
        
        # Run multiple training sessions concurrently
        num_concurrent = 3
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(train_model, i) for i in range(num_concurrent)]
            results = [future.result() for future in futures]
        
        # Verify all training sessions completed successfully
        self.assertEqual(len(results), num_concurrent)
        
        for result in results:
            self.assertIsNotNone(result['final_loss'])
            self.assertFalse(np.isnan(result['final_loss']))
            self.assertGreater(result['training_time'], 0)
        
        print(f"\nConcurrent training results:")
        for result in results:
            print(f"  Model {result['model_id']}: loss={result['final_loss']:.6f}, "
                  f"time={result['training_time']:.2f}s")


class TestMemoryLeakDetection(unittest.TestCase):
    """Test for memory leaks in long-running scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)
    
    def test_repeated_training_memory_leak(self):
        """Test for memory leaks in repeated training scenarios."""
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available - skipping memory leak test")
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_samples = []
        
        # Run multiple training iterations
        num_iterations = 10
        for i in range(num_iterations):
            # Create fresh model and data each iteration
            model = ShallowNetwork(input_dim=1, hidden_dims=[32], activation='relu')
            
            dataset_builder = DatasetBuilder(random_seed=42 + i)
            train_loader, val_loader, _ = dataset_builder.create_full_pipeline(
                degree=2, batch_size=16, num_points=200
            )
            
            # Train model
            trainer = Trainer(verbose=False)
            optimizer_config = OptimizerConfig(optimizer_type='adam', learning_rate=0.01)
            optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
            
            results = trainer.train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                epochs=2
            )
            
            # Verify training worked
            self.assertEqual(len(results.train_losses), 2)
            
            # Clean up explicitly
            del model, train_loader, val_loader, trainer, optimizer, results
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Sample memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)
        
        final_memory = memory_samples[-1]
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory leak test:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Total increase: {memory_increase:.1f}MB")
        print(f"  Average per iteration: {memory_increase/num_iterations:.1f}MB")
        
        # Check for excessive memory growth
        # Allow some growth but not more than 10MB per iteration on average
        avg_increase_per_iteration = memory_increase / num_iterations
        self.assertLess(avg_increase_per_iteration, 10.0,
                       f"Potential memory leak: {avg_increase_per_iteration:.1f}MB per iteration")
        
        # Check that memory growth is not accelerating
        if len(memory_samples) >= 5:
            early_avg = np.mean(memory_samples[:5])
            late_avg = np.mean(memory_samples[-5:])
            growth_acceleration = (late_avg - early_avg) / (len(memory_samples) - 5)
            
            self.assertLess(growth_acceleration, 5.0,
                           f"Memory growth accelerating: {growth_acceleration:.1f}MB per iteration")


if __name__ == "__main__":
    # Run with higher verbosity to see performance metrics
    unittest.main(verbosity=2)