#!/usr/bin/env python3
"""
Comprehensive unit tests for polynomial data generation system.
Tests polynomial generation accuracy, edge cases, noise injection, and dataset splitting.
"""

import sys
import os
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.generators import PolynomialGenerator, NoiseInjector, DatasetSplitter, DatasetBuilder, PolynomialDataset

class TestPolynomialGenerator(unittest.TestCase):
    """Test cases for PolynomialGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = PolynomialGenerator(random_seed=42)
    
    def test_coefficient_generation_valid_degrees(self):
        """Test coefficient generation for valid polynomial degrees 1-6."""
        for degree in range(1, 7):
            with self.subTest(degree=degree):
                coeffs = self.generator.generate_coefficients(degree=degree)
                self.assertEqual(len(coeffs), degree + 1, 
                               f"Expected {degree + 1} coefficients for degree {degree}")
                # Check that leading coefficient is not too small
                self.assertGreaterEqual(abs(coeffs[0]), 0.1, 
                                      "Leading coefficient should not be too small")
    
    def test_coefficient_generation_invalid_degrees(self):
        """Test coefficient generation with invalid degrees raises ValueError."""
        invalid_degrees = [0, 7, 8, -1, 10]
        for degree in invalid_degrees:
            with self.subTest(degree=degree):
                with self.assertRaises(ValueError):
                    self.generator.generate_coefficients(degree=degree)
    
    def test_coefficient_generation_custom_range(self):
        """Test coefficient generation with custom coefficient ranges."""
        coeff_range = (-2.0, 2.0)
        coeffs = self.generator.generate_coefficients(degree=3, coeff_range=coeff_range)
        
        # Check all coefficients are within range (allowing for leading coefficient adjustment)
        for i, coeff in enumerate(coeffs):
            if i == 0:  # Leading coefficient might be adjusted
                self.assertGreaterEqual(abs(coeff), 0.1)
            else:
                self.assertGreaterEqual(coeff, coeff_range[0])
                self.assertLessEqual(coeff, coeff_range[1])
    
    def test_polynomial_evaluation(self):
        """Test polynomial evaluation accuracy."""
        # Test with known polynomial: 2x^2 + 3x + 1
        coeffs = np.array([2.0, 3.0, 1.0])
        x = np.array([0.0, 1.0, 2.0, -1.0])
        expected_y = np.array([1.0, 6.0, 15.0, 0.0])  # Manual calculation
        
        y = self.generator.evaluate_polynomial(x, coeffs)
        np.testing.assert_array_almost_equal(y, expected_y, decimal=10)
    
    def test_polynomial_data_generation_valid_params(self):
        """Test polynomial data generation with valid parameters."""
        degree = 3
        x_range = (-1.0, 1.0)
        num_points = 50
        
        x, y, coeffs = self.generator.generate_polynomial_data(
            degree=degree, x_range=x_range, num_points=num_points
        )
        
        # Check output shapes
        self.assertEqual(len(x), num_points)
        self.assertEqual(len(y), num_points)
        self.assertEqual(len(coeffs), degree + 1)
        
        # Check x range
        self.assertGreaterEqual(x.min(), x_range[0])
        self.assertLessEqual(x.max(), x_range[1])
        
        # Verify polynomial evaluation
        y_expected = self.generator.evaluate_polynomial(x, coeffs)
        np.testing.assert_array_almost_equal(y, y_expected, decimal=10)
    
    def test_polynomial_data_generation_edge_cases(self):
        """Test polynomial data generation edge cases."""
        # Test minimum number of points
        x, y, coeffs = self.generator.generate_polynomial_data(degree=2, num_points=2)
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)
        
        # Test invalid number of points
        with self.assertRaises(ValueError):
            self.generator.generate_polynomial_data(degree=2, num_points=1)
        
        with self.assertRaises(ValueError):
            self.generator.generate_polynomial_data(degree=2, num_points=0)
    
    def test_specific_degree_functions(self):
        """Test specific degree generation functions."""
        degree_functions = [
            (1, self.generator.generate_polynomial_degree_1),
            (2, self.generator.generate_polynomial_degree_2),
            (3, self.generator.generate_polynomial_degree_3),
            (4, self.generator.generate_polynomial_degree_4),
            (5, self.generator.generate_polynomial_degree_5),
            (6, self.generator.generate_polynomial_degree_6)
        ]
        
        for expected_degree, func in degree_functions:
            with self.subTest(degree=expected_degree):
                x, y, coeffs = func()
                self.assertEqual(len(coeffs), expected_degree + 1)
                self.assertEqual(len(x), 100)  # Default num_points
                self.assertEqual(len(y), 100)
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same random seed."""
        # Test that same seed produces same results
        x1, y1, c1 = self.generator.generate_polynomial_data(degree=3, num_points=50)
        
        # Create new generator with same seed
        gen2 = PolynomialGenerator(random_seed=42)  # Same seed as setUp
        x2, y2, c2 = gen2.generate_polynomial_data(degree=3, num_points=50)
        
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(c1, c2)

class TestNoiseInjector(unittest.TestCase):
    """Test cases for NoiseInjector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.injector = NoiseInjector(random_seed=42)
        self.y_clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_gaussian_noise_properties(self):
        """Test Gaussian noise injection properties."""
        noise_level = 0.1
        y_noisy = self.injector.add_gaussian_noise(self.y_clean, noise_level)
        
        # Check output shape
        self.assertEqual(y_noisy.shape, self.y_clean.shape)
        
        # Check that noise was actually added (should be different)
        self.assertFalse(np.array_equal(y_noisy, self.y_clean))
        
        # Check noise level is approximately correct (statistical test)
        noise = y_noisy - self.y_clean
        data_std = np.std(self.y_clean)
        expected_noise_std = noise_level * data_std
        actual_noise_std = np.std(noise)
        
        # Allow some tolerance due to random sampling
        self.assertAlmostEqual(actual_noise_std, expected_noise_std, delta=expected_noise_std * 0.5)
    
    def test_uniform_noise_properties(self):
        """Test uniform noise injection properties."""
        noise_level = 0.1
        y_noisy = self.injector.add_uniform_noise(self.y_clean, noise_level)
        
        # Check output shape
        self.assertEqual(y_noisy.shape, self.y_clean.shape)
        
        # Check that noise was actually added
        self.assertFalse(np.array_equal(y_noisy, self.y_clean))
        
        # Check noise range
        noise = y_noisy - self.y_clean
        data_range = np.max(self.y_clean) - np.min(self.y_clean)
        expected_noise_range = noise_level * data_range
        
        # Noise should be within expected range
        self.assertLessEqual(np.max(np.abs(noise)), expected_noise_range / 2 * 1.1)  # Small tolerance
    
    def test_negative_noise_level_raises_error(self):
        """Test that negative noise levels raise ValueError."""
        with self.assertRaises(ValueError):
            self.injector.add_gaussian_noise(self.y_clean, noise_level=-0.1)
        
        with self.assertRaises(ValueError):
            self.injector.add_uniform_noise(self.y_clean, noise_level=-0.1)
    
    def test_zero_noise_level(self):
        """Test that zero noise level returns original data."""
        y_gaussian = self.injector.add_gaussian_noise(self.y_clean, noise_level=0.0)
        y_uniform = self.injector.add_uniform_noise(self.y_clean, noise_level=0.0)
        
        np.testing.assert_array_almost_equal(y_gaussian, self.y_clean, decimal=10)
        np.testing.assert_array_almost_equal(y_uniform, self.y_clean, decimal=10)
    
    def test_general_add_noise_method(self):
        """Test the general add_noise method with different noise types."""
        # Test Gaussian noise
        y_gaussian = self.injector.add_noise(self.y_clean, "gaussian", 0.1)
        self.assertEqual(y_gaussian.shape, self.y_clean.shape)
        self.assertFalse(np.array_equal(y_gaussian, self.y_clean))
        
        # Test uniform noise
        y_uniform = self.injector.add_noise(self.y_clean, "uniform", 0.1)
        self.assertEqual(y_uniform.shape, self.y_clean.shape)
        self.assertFalse(np.array_equal(y_uniform, self.y_clean))
        
        # Test case insensitivity
        y_gaussian_upper = self.injector.add_noise(self.y_clean, "GAUSSIAN", 0.1)
        self.assertEqual(y_gaussian_upper.shape, self.y_clean.shape)
    
    def test_unsupported_noise_type_raises_error(self):
        """Test that unsupported noise types raise ValueError."""
        with self.assertRaises(ValueError):
            self.injector.add_noise(self.y_clean, "unsupported", 0.1)
        
        with self.assertRaises(ValueError):
            self.injector.add_noise(self.y_clean, "laplacian", 0.1)
    
    def test_reproducibility_with_seed(self):
        """Test that noise injection is reproducible with same seed."""
        # Test with first injector
        y1 = self.injector.add_gaussian_noise(self.y_clean, 0.1)
        
        # Create new injector with same seed
        injector2 = NoiseInjector(random_seed=42)  # Same seed as setUp
        y2 = injector2.add_gaussian_noise(self.y_clean, 0.1)
        
        np.testing.assert_array_equal(y1, y2)

class TestDatasetSplitter(unittest.TestCase):
    """Test cases for DatasetSplitter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.splitter = DatasetSplitter(random_seed=42)
        np.random.seed(42)
        self.X = np.linspace(-1, 1, 100)
        self.y = self.X**2 + 0.1 * np.random.randn(100)
    
    def test_train_val_split_ratios(self):
        """Test train/validation split with different ratios."""
        test_ratios = [0.6, 0.7, 0.8, 0.9]
        
        for train_ratio in test_ratios:
            with self.subTest(train_ratio=train_ratio):
                X_train, X_val, y_train, y_val = self.splitter.train_val_split(
                    self.X, self.y, train_ratio=train_ratio
                )
                
                expected_train_size = int(len(self.X) * train_ratio)
                expected_val_size = len(self.X) - expected_train_size
                
                self.assertEqual(len(X_train), expected_train_size)
                self.assertEqual(len(X_val), expected_val_size)
                self.assertEqual(len(y_train), expected_train_size)
                self.assertEqual(len(y_val), expected_val_size)
                
                # Check no data overlap
                combined_size = len(X_train) + len(X_val)
                self.assertEqual(combined_size, len(self.X))
    
    def test_train_val_split_invalid_ratios(self):
        """Test that invalid train ratios raise ValueError."""
        invalid_ratios = [0.0, 1.0, -0.1, 1.1, 2.0]
        
        for ratio in invalid_ratios:
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    self.splitter.train_val_split(self.X, self.y, train_ratio=ratio)
    
    def test_train_val_split_shuffle(self):
        """Test shuffling behavior in train/val split."""
        # Test with shuffle=True (default)
        X_train1, X_val1, y_train1, y_val1 = self.splitter.train_val_split(
            self.X, self.y, train_ratio=0.8, shuffle=True
        )
        
        # Test with shuffle=False
        X_train2, X_val2, y_train2, y_val2 = self.splitter.train_val_split(
            self.X, self.y, train_ratio=0.8, shuffle=False
        )
        
        # With shuffle=False, should get first 80% for training
        expected_train_size = int(len(self.X) * 0.8)
        np.testing.assert_array_equal(X_train2, self.X[:expected_train_size])
        np.testing.assert_array_equal(X_val2, self.X[expected_train_size:])
    
    def test_standard_normalization(self):
        """Test standard normalization (zero mean, unit variance)."""
        X_train, X_val, y_train, y_val = self.splitter.train_val_split(self.X, self.y, train_ratio=0.8)
        
        X_train_norm, X_val_norm, y_train_norm, y_val_norm, norm_params = self.splitter.normalize_data(
            X_train, X_val, y_train, y_val, method="standard"
        )
        
        # Check normalization parameters exist
        self.assertIn('X_mean', norm_params)
        self.assertIn('X_std', norm_params)
        self.assertIn('y_mean', norm_params)
        self.assertIn('y_std', norm_params)
        
        # Check X normalization
        self.assertAlmostEqual(np.mean(X_train_norm), 0.0, places=10)
        self.assertAlmostEqual(np.std(X_train_norm), 1.0, places=10)
        
        # Check y normalization
        self.assertAlmostEqual(np.mean(y_train_norm), 0.0, places=10)
        self.assertAlmostEqual(np.std(y_train_norm), 1.0, places=10)
        
        # Check validation data uses training statistics
        expected_X_val_norm = (X_val - norm_params['X_mean']) / norm_params['X_std']
        expected_y_val_norm = (y_val - norm_params['y_mean']) / norm_params['y_std']
        
        np.testing.assert_array_almost_equal(X_val_norm, expected_X_val_norm)
        np.testing.assert_array_almost_equal(y_val_norm, expected_y_val_norm)
    
    def test_minmax_normalization(self):
        """Test min-max normalization to [0, 1] range."""
        X_train, X_val, y_train, y_val = self.splitter.train_val_split(self.X, self.y, train_ratio=0.8)
        
        X_train_norm, X_val_norm, y_train_norm, y_val_norm, norm_params = self.splitter.normalize_data(
            X_train, X_val, y_train, y_val, method="minmax"
        )
        
        # Check normalization parameters exist
        self.assertIn('X_min', norm_params)
        self.assertIn('X_max', norm_params)
        self.assertIn('y_min', norm_params)
        self.assertIn('y_max', norm_params)
        
        # Check X normalization range
        self.assertAlmostEqual(np.min(X_train_norm), 0.0, places=10)
        self.assertAlmostEqual(np.max(X_train_norm), 1.0, places=10)
        
        # Check y normalization range
        self.assertAlmostEqual(np.min(y_train_norm), 0.0, places=10)
        self.assertAlmostEqual(np.max(y_train_norm), 1.0, places=10)
    
    def test_no_normalization(self):
        """Test that 'none' normalization returns original data."""
        X_train, X_val, y_train, y_val = self.splitter.train_val_split(self.X, self.y, train_ratio=0.8)
        
        X_train_norm, X_val_norm, y_train_norm, y_val_norm, norm_params = self.splitter.normalize_data(
            X_train, X_val, y_train, y_val, method="none"
        )
        
        np.testing.assert_array_equal(X_train_norm, X_train)
        np.testing.assert_array_equal(X_val_norm, X_val)
        np.testing.assert_array_equal(y_train_norm, y_train)
        np.testing.assert_array_equal(y_val_norm, y_val)
        self.assertEqual(norm_params, {})
    
    def test_unsupported_normalization_method(self):
        """Test that unsupported normalization methods raise ValueError."""
        X_train, X_val, y_train, y_val = self.splitter.train_val_split(self.X, self.y, train_ratio=0.8)
        
        with self.assertRaises(ValueError):
            self.splitter.normalize_data(X_train, X_val, y_train, y_val, method="unsupported")
    
    def test_denormalize_predictions(self):
        """Test denormalization of predictions."""
        X_train, X_val, y_train, y_val = self.splitter.train_val_split(self.X, self.y, train_ratio=0.8)
        
        # Test standard denormalization
        _, _, y_train_norm, _, norm_params = self.splitter.normalize_data(
            X_train, X_val, y_train, y_val, method="standard"
        )
        
        y_denorm = self.splitter.denormalize_predictions(y_train_norm, norm_params, "standard")
        np.testing.assert_array_almost_equal(y_denorm, y_train, decimal=10)
        
        # Test minmax denormalization
        _, _, y_train_norm, _, norm_params = self.splitter.normalize_data(
            X_train, X_val, y_train, y_val, method="minmax"
        )
        
        y_denorm = self.splitter.denormalize_predictions(y_train_norm, norm_params, "minmax")
        np.testing.assert_array_almost_equal(y_denorm, y_train, decimal=10)

class TestDatasetBuilder(unittest.TestCase):
    """Test cases for DatasetBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = DatasetBuilder(random_seed=42)
    
    def test_build_dataset_structure(self):
        """Test that build_dataset creates correct data structure."""
        dataset = self.builder.build_dataset(
            degree=3, num_points=100, noise_level=0.1, train_ratio=0.8
        )
        
        # Check top-level keys
        expected_keys = ['raw_data', 'train', 'validation', 'metadata']
        self.assertEqual(set(dataset.keys()), set(expected_keys))
        
        # Check raw_data structure
        raw_data_keys = ['x_data', 'y_clean', 'y_noisy', 'coefficients']
        self.assertEqual(set(dataset['raw_data'].keys()), set(raw_data_keys))
        
        # Check train/validation structure
        for split in ['train', 'validation']:
            split_keys = ['X', 'y', 'X_original', 'y_original']
            self.assertEqual(set(dataset[split].keys()), set(split_keys))
        
        # Check metadata
        self.assertIn('degree', dataset['metadata'])
        self.assertIn('coefficients', dataset['metadata'])
        self.assertIn('random_seed', dataset['metadata'])
    
    def test_build_dataset_sizes(self):
        """Test that dataset sizes are correct."""
        num_points = 100
        train_ratio = 0.8
        
        dataset = self.builder.build_dataset(
            degree=2, num_points=num_points, train_ratio=train_ratio
        )
        
        expected_train_size = int(num_points * train_ratio)
        expected_val_size = num_points - expected_train_size
        
        # Check raw data size
        self.assertEqual(len(dataset['raw_data']['x_data']), num_points)
        self.assertEqual(len(dataset['raw_data']['y_clean']), num_points)
        self.assertEqual(len(dataset['raw_data']['y_noisy']), num_points)
        
        # Check split sizes
        self.assertEqual(len(dataset['train']['X']), expected_train_size)
        self.assertEqual(len(dataset['validation']['X']), expected_val_size)
    
    def test_build_dataset_noise_injection(self):
        """Test that noise is properly injected."""
        dataset = self.builder.build_dataset(
            degree=2, num_points=100, noise_level=0.2
        )
        
        y_clean = dataset['raw_data']['y_clean']
        y_noisy = dataset['raw_data']['y_noisy']
        
        # Noise should make data different
        self.assertFalse(np.array_equal(y_clean, y_noisy))
        
        # But should be correlated
        correlation = np.corrcoef(y_clean, y_noisy)[0, 1]
        self.assertGreater(correlation, 0.8)  # Should be highly correlated
    
    def test_build_dataset_normalization(self):
        """Test dataset normalization options."""
        # Test standard normalization
        dataset_std = self.builder.build_dataset(
            degree=2, num_points=100, normalize="standard"
        )
        
        X_train = dataset_std['train']['X']
        y_train = dataset_std['train']['y']
        
        self.assertAlmostEqual(np.mean(X_train), 0.0, places=10)
        self.assertAlmostEqual(np.std(X_train), 1.0, places=10)
        self.assertAlmostEqual(np.mean(y_train), 0.0, places=10)
        self.assertAlmostEqual(np.std(y_train), 1.0, places=10)
        
        # Test no normalization
        dataset_none = self.builder.build_dataset(
            degree=2, num_points=100, normalize="none"
        )
        
        # Should have original and normalized data the same
        np.testing.assert_array_equal(
            dataset_none['train']['X'], 
            dataset_none['train']['X_original']
        )
    
    def test_create_dataloaders(self):
        """Test PyTorch DataLoader creation."""
        dataset = self.builder.build_dataset(degree=2, num_points=100)
        
        train_loader, val_loader = self.builder.create_dataloaders(
            dataset, batch_size=16, shuffle_train=True, shuffle_val=False
        )
        
        # Check DataLoader properties
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertEqual(train_loader.batch_size, 16)
        self.assertEqual(val_loader.batch_size, 16)
        
        # Check data shapes
        for batch_x, batch_y in train_loader:
            self.assertEqual(batch_x.shape[1], 1)  # Input dimension
            self.assertEqual(batch_y.shape[1], 1)  # Output dimension
            break  # Just check first batch
    
    def test_create_full_pipeline(self):
        """Test complete pipeline from generation to DataLoaders."""
        train_loader, val_loader, dataset_info = self.builder.create_full_pipeline(
            degree=3, batch_size=32, num_points=100
        )
        
        # Check DataLoaders
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        
        # Check dataset info
        self.assertEqual(dataset_info['metadata']['degree'], 3)
        self.assertEqual(dataset_info['metadata']['num_points'], 100)
        
        # Check data can be loaded
        for batch_x, batch_y in train_loader:
            self.assertIsInstance(batch_x, torch.Tensor)
            self.assertIsInstance(batch_y, torch.Tensor)
            break


class TestPolynomialDataset(unittest.TestCase):
    """Test cases for PolynomialDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])
    
    def test_dataset_creation(self):
        """Test dataset creation and basic properties."""
        dataset = PolynomialDataset(self.X.flatten(), self.y.flatten())
        
        self.assertEqual(len(dataset), 5)
        
        # Check first sample
        x_sample, y_sample = dataset[0]
        self.assertIsInstance(x_sample, torch.Tensor)
        self.assertIsInstance(y_sample, torch.Tensor)
        self.assertEqual(x_sample.shape, (1,))
        self.assertEqual(y_sample.shape, (1,))
    
    def test_dataset_indexing(self):
        """Test dataset indexing and data retrieval."""
        dataset = PolynomialDataset(self.X.flatten(), self.y.flatten())
        
        for i in range(len(dataset)):
            x_sample, y_sample = dataset[i]
            self.assertAlmostEqual(x_sample.item(), self.X[i, 0])
            self.assertAlmostEqual(y_sample.item(), self.y[i, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)