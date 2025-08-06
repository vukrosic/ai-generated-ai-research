"""
Polynomial data generation utilities for curve fitting research.

This module provides classes for generating synthetic polynomial datasets
with configurable parameters for AI curve fitting experiments.
"""

import numpy as np
from typing import Tuple, List, Optional
import random


class PolynomialGenerator:
    """
    Generates polynomial curves with configurable degrees and coefficients.
    
    Supports polynomial degrees 1-6 with customizable coefficient ranges
    and data point generation over specified x-ranges.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the polynomial generator.
        
        Args:
            random_seed: Optional seed for reproducible random number generation
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def generate_coefficients(self, degree: int, coeff_range: Tuple[float, float] = (-5.0, 5.0)) -> np.ndarray:
        """
        Generate random coefficients for a polynomial of given degree.
        
        Args:
            degree: Polynomial degree (1-6)
            coeff_range: Tuple of (min, max) values for coefficient generation
            
        Returns:
            Array of coefficients [a_n, a_{n-1}, ..., a_1, a_0] for polynomial
            a_n*x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0
            
        Raises:
            ValueError: If degree is not between 1 and 6
        """
        if not (1 <= degree <= 6):
            raise ValueError(f"Polynomial degree must be between 1 and 6, got {degree}")
        
        min_coeff, max_coeff = coeff_range
        # Generate degree+1 coefficients (including constant term)
        coefficients = np.random.uniform(min_coeff, max_coeff, degree + 1)
        
        # Ensure the leading coefficient is not too small to avoid numerical issues
        if abs(coefficients[0]) < 0.1:
            coefficients[0] = np.sign(coefficients[0]) * 0.5 if coefficients[0] != 0 else 0.5
            
        return coefficients
    
    def evaluate_polynomial(self, x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial at given x values using given coefficients.
        
        Args:
            x: Array of x values
            coefficients: Array of coefficients [a_n, a_{n-1}, ..., a_1, a_0]
            
        Returns:
            Array of y values computed as polynomial evaluation
        """
        return np.polyval(coefficients, x)
    
    def generate_polynomial_data(self, 
                                degree: int,
                                x_range: Tuple[float, float] = (-2.0, 2.0),
                                num_points: int = 100,
                                coeff_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate polynomial data points over specified x-range.
        
        Args:
            degree: Polynomial degree (1-6)
            x_range: Tuple of (min, max) x values
            num_points: Number of data points to generate
            coeff_range: Tuple of (min, max) values for coefficient generation
            
        Returns:
            Tuple of (x_values, y_values, coefficients)
            
        Raises:
            ValueError: If degree is not between 1 and 6 or num_points < 2
        """
        if not (1 <= degree <= 6):
            raise ValueError(f"Polynomial degree must be between 1 and 6, got {degree}")
        
        if num_points < 2:
            raise ValueError(f"Number of points must be at least 2, got {num_points}")
        
        # Generate coefficients
        coefficients = self.generate_coefficients(degree, coeff_range)
        
        # Generate x values
        x_min, x_max = x_range
        x_values = np.linspace(x_min, x_max, num_points)
        
        # Evaluate polynomial
        y_values = self.evaluate_polynomial(x_values, coefficients)
        
        return x_values, y_values, coefficients
    
    def generate_polynomial_degree_1(self, x_range: Tuple[float, float] = (-2.0, 2.0),
                                   num_points: int = 100,
                                   coeff_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate degree 1 polynomial (linear): a*x + b"""
        return self.generate_polynomial_data(1, x_range, num_points, coeff_range)
    
    def generate_polynomial_degree_2(self, x_range: Tuple[float, float] = (-2.0, 2.0),
                                   num_points: int = 100,
                                   coeff_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate degree 2 polynomial (quadratic): a*x^2 + b*x + c"""
        return self.generate_polynomial_data(2, x_range, num_points, coeff_range)
    
    def generate_polynomial_degree_3(self, x_range: Tuple[float, float] = (-2.0, 2.0),
                                   num_points: int = 100,
                                   coeff_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate degree 3 polynomial (cubic): a*x^3 + b*x^2 + c*x + d"""
        return self.generate_polynomial_data(3, x_range, num_points, coeff_range)
    
    def generate_polynomial_degree_4(self, x_range: Tuple[float, float] = (-2.0, 2.0),
                                   num_points: int = 100,
                                   coeff_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate degree 4 polynomial (quartic): a*x^4 + b*x^3 + c*x^2 + d*x + e"""
        return self.generate_polynomial_data(4, x_range, num_points, coeff_range)
    
    def generate_polynomial_degree_5(self, x_range: Tuple[float, float] = (-2.0, 2.0),
                                   num_points: int = 100,
                                   coeff_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate degree 5 polynomial (quintic): a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f"""
        return self.generate_polynomial_data(5, x_range, num_points, coeff_range)
    
    def generate_polynomial_degree_6(self, x_range: Tuple[float, float] = (-2.0, 2.0),
                                   num_points: int = 100,
                                   coeff_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate degree 6 polynomial (sextic): a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + g"""
        return self.generate_polynomial_data(6, x_range, num_points, coeff_range)


class NoiseInjector:
    """
    Adds various types of noise to clean data for realistic curve fitting scenarios.
    
    Supports Gaussian and uniform noise injection with configurable parameters.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the noise injector.
        
        Args:
            random_seed: Optional seed for reproducible random number generation
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def add_gaussian_noise(self, y_data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Add Gaussian (normal) noise to y data.
        
        Args:
            y_data: Original y values
            noise_level: Standard deviation of Gaussian noise as fraction of y_data std
            
        Returns:
            Y values with added Gaussian noise
        """
        if noise_level < 0:
            raise ValueError(f"Noise level must be non-negative, got {noise_level}")
        
        # Calculate noise standard deviation based on data scale
        data_std = np.std(y_data)
        noise_std = noise_level * data_std
        
        # Generate and add Gaussian noise
        noise = np.random.normal(0, noise_std, y_data.shape)
        return y_data + noise
    
    def add_uniform_noise(self, y_data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Add uniform noise to y data.
        
        Args:
            y_data: Original y values
            noise_level: Range of uniform noise as fraction of y_data range
            
        Returns:
            Y values with added uniform noise
        """
        if noise_level < 0:
            raise ValueError(f"Noise level must be non-negative, got {noise_level}")
        
        # Calculate noise range based on data scale
        data_range = np.max(y_data) - np.min(y_data)
        noise_range = noise_level * data_range
        
        # Generate and add uniform noise
        noise = np.random.uniform(-noise_range/2, noise_range/2, y_data.shape)
        return y_data + noise
    
    def add_noise(self, y_data: np.ndarray, noise_type: str = "gaussian", 
                  noise_level: float = 0.1) -> np.ndarray:
        """
        Add noise to y data with specified type and level.
        
        Args:
            y_data: Original y values
            noise_type: Type of noise ("gaussian" or "uniform")
            noise_level: Intensity of noise
            
        Returns:
            Y values with added noise
            
        Raises:
            ValueError: If noise_type is not supported
        """
        if noise_type.lower() == "gaussian":
            return self.add_gaussian_noise(y_data, noise_level)
        elif noise_type.lower() == "uniform":
            return self.add_uniform_noise(y_data, noise_level)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}. Use 'gaussian' or 'uniform'")


class DatasetSplitter:
    """
    Handles train/validation dataset splitting with configurable ratios.
    
    Provides functionality for splitting datasets and data normalization.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the dataset splitter.
        
        Args:
            random_seed: Optional seed for reproducible random splits
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def train_val_split(self, X: np.ndarray, y: np.ndarray, 
                       train_ratio: float = 0.8, 
                       shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            X: Input features
            y: Target values
            train_ratio: Fraction of data to use for training (0 < train_ratio < 1)
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
            
        Raises:
            ValueError: If train_ratio is not between 0 and 1
        """
        if not (0 < train_ratio < 1):
            raise ValueError(f"Train ratio must be between 0 and 1, got {train_ratio}")
        
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        
        # Create indices
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Split data
        X_train = X[train_indices]
        X_val = X[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        return X_train, X_val, y_train, y_val
    
    def normalize_data(self, X_train: np.ndarray, X_val: np.ndarray = None, 
                      y_train: np.ndarray = None, y_val: np.ndarray = None,
                      method: str = "standard") -> Tuple[np.ndarray, ...]:
        """
        Normalize data using specified method.
        
        Args:
            X_train: Training input features
            X_val: Validation input features (optional)
            y_train: Training target values (optional)
            y_val: Validation target values (optional)
            method: Normalization method ("standard", "minmax", or "none")
            
        Returns:
            Tuple of normalized arrays and normalization parameters
            
        Raises:
            ValueError: If normalization method is not supported
        """
        results = []
        norm_params = {}
        
        if method.lower() == "standard":
            # Standardize X (zero mean, unit variance)
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0)
            X_std = np.where(X_std == 0, 1, X_std)  # Avoid division by zero
            
            X_train_norm = (X_train - X_mean) / X_std
            results.append(X_train_norm)
            norm_params['X_mean'] = X_mean
            norm_params['X_std'] = X_std
            
            if X_val is not None:
                X_val_norm = (X_val - X_mean) / X_std
                results.append(X_val_norm)
            
            # Standardize y if provided
            if y_train is not None:
                y_mean = np.mean(y_train)
                y_std = np.std(y_train)
                y_std = y_std if y_std != 0 else 1
                
                y_train_norm = (y_train - y_mean) / y_std
                results.append(y_train_norm)
                norm_params['y_mean'] = y_mean
                norm_params['y_std'] = y_std
                
                if y_val is not None:
                    y_val_norm = (y_val - y_mean) / y_std
                    results.append(y_val_norm)
        
        elif method.lower() == "minmax":
            # Min-max scaling to [0, 1]
            X_min = np.min(X_train, axis=0)
            X_max = np.max(X_train, axis=0)
            X_range = X_max - X_min
            X_range = np.where(X_range == 0, 1, X_range)  # Avoid division by zero
            
            X_train_norm = (X_train - X_min) / X_range
            results.append(X_train_norm)
            norm_params['X_min'] = X_min
            norm_params['X_max'] = X_max
            
            if X_val is not None:
                X_val_norm = (X_val - X_min) / X_range
                results.append(X_val_norm)
            
            # Min-max scale y if provided
            if y_train is not None:
                y_min = np.min(y_train)
                y_max = np.max(y_train)
                y_range = y_max - y_min
                y_range = y_range if y_range != 0 else 1
                
                y_train_norm = (y_train - y_min) / y_range
                results.append(y_train_norm)
                norm_params['y_min'] = y_min
                norm_params['y_max'] = y_max
                
                if y_val is not None:
                    y_val_norm = (y_val - y_min) / y_range
                    results.append(y_val_norm)
        
        elif method.lower() == "none":
            # No normalization
            results.append(X_train)
            if X_val is not None:
                results.append(X_val)
            if y_train is not None:
                results.append(y_train)
            if y_val is not None:
                results.append(y_val)
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}. Use 'standard', 'minmax', or 'none'")
        
        results.append(norm_params)
        return tuple(results)
    
    def denormalize_predictions(self, predictions: np.ndarray, norm_params: dict, 
                              method: str = "standard") -> np.ndarray:
        """
        Denormalize predictions back to original scale.
        
        Args:
            predictions: Normalized predictions
            norm_params: Normalization parameters from normalize_data
            method: Normalization method used
            
        Returns:
            Denormalized predictions
        """
        if method.lower() == "standard":
            if 'y_mean' in norm_params and 'y_std' in norm_params:
                return predictions * norm_params['y_std'] + norm_params['y_mean']
        elif method.lower() == "minmax":
            if 'y_min' in norm_params and 'y_max' in norm_params:
                y_range = norm_params['y_max'] - norm_params['y_min']
                return predictions * y_range + norm_params['y_min']
        
        return predictions
try:

    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PolynomialDataset(Dataset):
    """
    PyTorch Dataset wrapper for polynomial data.
    
    Provides PyTorch-compatible dataset interface for polynomial curve data.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            X: Input features
            y: Target values
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PolynomialDataset. Install with: pip install torch")
        
        self.X = torch.FloatTensor(X.reshape(-1, 1) if X.ndim == 1 else X)
        self.y = torch.FloatTensor(y.reshape(-1, 1) if y.ndim == 1 else y)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input, target) tensors
        """
        return self.X[idx], self.y[idx]


class DatasetBuilder:
    """
    Combines polynomial generation, noise injection, and dataset splitting.
    
    Provides a unified interface for creating complete datasets with PyTorch
    DataLoader integration for curve fitting experiments.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the dataset builder.
        
        Args:
            random_seed: Optional seed for reproducible dataset generation
        """
        self.random_seed = random_seed
        self.poly_generator = PolynomialGenerator(random_seed)
        self.noise_injector = NoiseInjector(random_seed)
        self.splitter = DatasetSplitter(random_seed)
    
    def build_dataset(self,
                     degree: int,
                     x_range: Tuple[float, float] = (-2.0, 2.0),
                     num_points: int = 100,
                     coeff_range: Tuple[float, float] = (-5.0, 5.0),
                     noise_type: str = "gaussian",
                     noise_level: float = 0.1,
                     train_ratio: float = 0.8,
                     normalize: str = "standard",
                     shuffle: bool = True) -> dict:
        """
        Build complete dataset with polynomial generation, noise, and splitting.
        
        Args:
            degree: Polynomial degree (1-6)
            x_range: Range of x values
            num_points: Number of data points to generate
            coeff_range: Range for coefficient generation
            noise_type: Type of noise to add ("gaussian" or "uniform")
            noise_level: Level of noise to add
            train_ratio: Fraction of data for training
            normalize: Normalization method ("standard", "minmax", or "none")
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Dictionary containing all dataset components and metadata
        """
        # Generate clean polynomial data
        x_data, y_clean, coefficients = self.poly_generator.generate_polynomial_data(
            degree, x_range, num_points, coeff_range
        )
        
        # Add noise
        y_noisy = self.noise_injector.add_noise(y_clean, noise_type, noise_level)
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = self.splitter.train_val_split(
            x_data, y_noisy, train_ratio, shuffle
        )
        
        # Normalize data
        if normalize != "none":
            X_train_norm, X_val_norm, y_train_norm, y_val_norm, norm_params = self.splitter.normalize_data(
                X_train, X_val, y_train, y_val, normalize
            )
        else:
            X_train_norm, X_val_norm = X_train, X_val
            y_train_norm, y_val_norm = y_train, y_val
            norm_params = {}
        
        # Create dataset dictionary
        dataset = {
            'raw_data': {
                'x_data': x_data,
                'y_clean': y_clean,
                'y_noisy': y_noisy,
                'coefficients': coefficients
            },
            'train': {
                'X': X_train_norm,
                'y': y_train_norm,
                'X_original': X_train,
                'y_original': y_train
            },
            'validation': {
                'X': X_val_norm,
                'y': y_val_norm,
                'X_original': X_val,
                'y_original': y_val
            },
            'metadata': {
                'degree': degree,
                'x_range': x_range,
                'num_points': num_points,
                'coeff_range': coeff_range,
                'noise_type': noise_type,
                'noise_level': noise_level,
                'train_ratio': train_ratio,
                'normalization': normalize,
                'norm_params': norm_params,
                'coefficients': coefficients,
                'random_seed': self.random_seed
            }
        }
        
        return dataset
    
    def create_dataloaders(self,
                          dataset: dict,
                          batch_size: int = 32,
                          shuffle_train: bool = True,
                          shuffle_val: bool = False,
                          num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders from dataset.
        
        Args:
            dataset: Dataset dictionary from build_dataset
            batch_size: Batch size for DataLoaders
            shuffle_train: Whether to shuffle training data
            shuffle_val: Whether to shuffle validation data
            num_workers: Number of worker processes for data loading
            
        Returns:
            Tuple of (train_loader, val_loader)
            
        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DataLoaders. Install with: pip install torch")
        
        # Create PyTorch datasets
        train_dataset = PolynomialDataset(dataset['train']['X'], dataset['train']['y'])
        val_dataset = PolynomialDataset(dataset['validation']['X'], dataset['validation']['y'])
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle_val,
            num_workers=num_workers
        )
        
        return train_loader, val_loader
    
    def create_full_pipeline(self,
                           degree: int,
                           batch_size: int = 32,
                           x_range: Tuple[float, float] = (-2.0, 2.0),
                           num_points: int = 100,
                           coeff_range: Tuple[float, float] = (-5.0, 5.0),
                           noise_type: str = "gaussian",
                           noise_level: float = 0.1,
                           train_ratio: float = 0.8,
                           normalize: str = "standard",
                           shuffle: bool = True,
                           shuffle_train: bool = True,
                           shuffle_val: bool = False,
                           num_workers: int = 0) -> Tuple[DataLoader, DataLoader, dict]:
        """
        Create complete pipeline from polynomial generation to PyTorch DataLoaders.
        
        Args:
            degree: Polynomial degree (1-6)
            batch_size: Batch size for DataLoaders
            x_range: Range of x values
            num_points: Number of data points to generate
            coeff_range: Range for coefficient generation
            noise_type: Type of noise to add
            noise_level: Level of noise to add
            train_ratio: Fraction of data for training
            normalize: Normalization method
            shuffle: Whether to shuffle data before splitting
            shuffle_train: Whether to shuffle training batches
            shuffle_val: Whether to shuffle validation batches
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, dataset_info)
        """
        # Build dataset
        dataset = self.build_dataset(
            degree=degree,
            x_range=x_range,
            num_points=num_points,
            coeff_range=coeff_range,
            noise_type=noise_type,
            noise_level=noise_level,
            train_ratio=train_ratio,
            normalize=normalize,
            shuffle=shuffle
        )
        
        # Create DataLoaders
        train_loader, val_loader = self.create_dataloaders(
            dataset=dataset,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            shuffle_val=shuffle_val,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, dataset