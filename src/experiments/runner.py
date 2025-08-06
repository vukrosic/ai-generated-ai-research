"""
Experiment runner and orchestration for AI curve fitting research.

This module provides the ExperimentRunner class that manages complete experiment
workflows including data generation, model training, visualization, and result storage.
"""

import torch
import numpy as np
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from threading import Lock
import logging

# Set matplotlib backend to non-GUI for thread safety
import matplotlib
matplotlib.use('Agg')

try:
    from .config import ExperimentConfig
    from ..data.generators import PolynomialGenerator, NoiseInjector, DatasetBuilder
    from ..models.architectures import ModelFactory
    from ..models.trainers import Trainer, OptimizerConfig, TrainingResults, OptimizerFactory, EarlyStopping
    from ..visualization.plots import CurvePlotter, LossPlotter, ComparisonPlotter
except ImportError:
    # Fallback for direct execution
    from config import ExperimentConfig
    try:
        from data.generators import PolynomialGenerator, NoiseInjector, DatasetBuilder
        from models.architectures import ModelFactory
        from models.trainers import Trainer, OptimizerConfig, TrainingResults, OptimizerFactory, EarlyStopping
        from visualization.plots import CurvePlotter, LossPlotter, ComparisonPlotter
    except ImportError:
        # Create minimal stubs for testing
        class PolynomialGenerator: pass
        class NoiseInjector: pass
        class DatasetBuilder: pass
        class ModelFactory: pass
        class Trainer: pass
        class OptimizerConfig: pass
        class TrainingResults: pass
        class OptimizerFactory: pass
        class EarlyStopping: pass
        class CurvePlotter: pass
        class LossPlotter: pass
        class ComparisonPlotter: pass


@dataclass
class ExperimentResults:
    """Comprehensive experiment results tracking."""
    
    # Configuration and metadata
    config: ExperimentConfig
    experiment_id: str
    timestamp: datetime
    duration_seconds: float
    
    # Training results
    training_results: Optional[TrainingResults] = None
    
    # Performance metrics
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = float('inf')
    training_time: float = 0.0
    convergence_epoch: int = -1
    
    # Model information
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    model_size: int = 0
    
    # Data information
    data_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Generated files
    image_paths: List[str] = field(default_factory=list)
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    
    # Status and error information
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        result_dict = asdict(self)
        # Convert datetime to string for JSON serialization
        result_dict['timestamp'] = self.timestamp.isoformat()
        # Convert config to dict
        result_dict['config'] = self.config.to_dict()
        return result_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResults':
        """Create ExperimentResults from dictionary."""
        # Convert timestamp back to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        # Convert config back to ExperimentConfig
        data['config'] = ExperimentConfig.from_dict(data['config'])
        return cls(**data)
    
    @classmethod
    def create_mock_result(cls, 
                          config: ExperimentConfig,
                          final_train_loss: float = 0.001,
                          final_val_loss: float = 0.002,
                          training_time: float = 30.0,
                          status: str = "completed") -> 'ExperimentResults':
        """Create a mock experiment result for testing purposes."""
        import uuid
        
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        return cls(
            config=config,
            experiment_id=experiment_id,
            timestamp=timestamp,
            duration_seconds=training_time + 5.0,  # Add some overhead
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=final_val_loss * 0.95,  # Slightly better than final
            training_time=training_time,
            convergence_epoch=int(config.epochs * 0.7),  # Converged at 70% of epochs
            model_size=sum(config.hidden_dims) + 10 if config.hidden_dims else 2,
            status=status,
            data_stats={
                'num_train_samples': int(config.num_data_points * config.train_val_split),
                'num_val_samples': int(config.num_data_points * (1 - config.train_val_split)),
                'polynomial_degree': config.polynomial_degree,
                'noise_level': config.noise_level
            },
            model_parameters={
                'architecture': config.model_architecture,
                'hidden_dims': config.hidden_dims,
                'activation': config.activation_function,
                'total_params': sum(config.hidden_dims) + 10 if config.hidden_dims else 2
            }
        )


class ProgressTracker:
    """Thread-safe progress tracking for experiments."""
    
    def __init__(self):
        self._lock = Lock()
        self._progress = {}
        self._status = {}
    
    def update_progress(self, experiment_id: str, progress: float, status: str = "running"):
        """Update progress for an experiment."""
        with self._lock:
            self._progress[experiment_id] = progress
            self._status[experiment_id] = status
    
    def get_progress(self, experiment_id: str) -> Tuple[float, str]:
        """Get progress and status for an experiment."""
        with self._lock:
            return self._progress.get(experiment_id, 0.0), self._status.get(experiment_id, "pending")
    
    def get_all_progress(self) -> Dict[str, Tuple[float, str]]:
        """Get progress for all experiments."""
        with self._lock:
            return {exp_id: (self._progress.get(exp_id, 0.0), self._status.get(exp_id, "pending"))
                    for exp_id in set(list(self._progress.keys()) + list(self._status.keys()))}


class ExperimentRunner:
    """
    Manages complete experiment workflows for AI curve fitting research.
    
    This class orchestrates data generation, model training, visualization,
    and result storage with support for parallel execution and progress tracking.
    """
    
    def __init__(self, 
                 output_dir: str = "experiments",
                 images_dir: str = "images",
                 models_dir: str = "models",
                 enable_logging: bool = True):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Directory for experiment results
            images_dir: Directory for generated images
            models_dir: Directory for saved models
            enable_logging: Whether to enable logging
        """
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.progress_tracker = ProgressTracker()
        
        # Setup logging
        if enable_logging:
            self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
    
    def run_single_experiment(self, config: ExperimentConfig, experiment_id: Optional[str] = None) -> ExperimentResults:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config: Experiment configuration
            experiment_id: Optional experiment ID (generated if not provided)
            
        Returns:
            ExperimentResults containing all experiment data
        """
        if experiment_id is None:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.polynomial_degree}_{config.model_architecture}"
        
        start_time = time.time()
        
        # Initialize results
        results = ExperimentResults(
            config=config,
            experiment_id=experiment_id,
            timestamp=datetime.now(),
            duration_seconds=0.0,
            status="running"
        )
        
        try:
            self.logger.info(f"Starting experiment {experiment_id}")
            self.progress_tracker.update_progress(experiment_id, 0.0, "running")
            
            # Step 1: Generate data (20% progress)
            self.logger.info(f"Generating data for experiment {experiment_id}")
            train_data, val_data, data_stats = self._generate_data(config)
            results.data_stats = data_stats
            self.progress_tracker.update_progress(experiment_id, 0.2, "running")
            
            # Step 2: Create model (30% progress)
            self.logger.info(f"Creating model for experiment {experiment_id}")
            model = self._create_model(config)
            results.model_parameters = model.get_parameters()
            results.model_size = sum(p.numel() for p in model.parameters())
            self.progress_tracker.update_progress(experiment_id, 0.3, "running")
            
            # Step 3: Train model (70% progress)
            self.logger.info(f"Training model for experiment {experiment_id}")
            training_results = self._train_model(config, model, train_data, val_data, experiment_id)
            results.training_results = training_results
            results.final_train_loss = training_results.final_train_loss
            results.final_val_loss = training_results.final_val_loss
            results.best_val_loss = min(training_results.val_losses) if training_results.val_losses else float('inf')
            results.training_time = training_results.training_time
            results.convergence_epoch = training_results.best_epoch
            self.progress_tracker.update_progress(experiment_id, 0.7, "running")
            
            # Step 4: Generate visualizations (85% progress)
            self.logger.info(f"Generating visualizations for experiment {experiment_id}")
            image_paths = self._generate_visualizations(config, model, train_data, val_data, training_results, experiment_id)
            results.image_paths = image_paths
            self.progress_tracker.update_progress(experiment_id, 0.85, "running")
            
            # Step 5: Save results (95% progress)
            self.logger.info(f"Saving results for experiment {experiment_id}")
            model_path, config_path = self._save_experiment_artifacts(config, model, results, experiment_id)
            results.model_path = model_path
            results.config_path = config_path
            self.progress_tracker.update_progress(experiment_id, 0.95, "running")
            
            # Complete experiment
            results.duration_seconds = time.time() - start_time
            results.status = "completed"
            self.progress_tracker.update_progress(experiment_id, 1.0, "completed")
            
            self.logger.info(f"Experiment {experiment_id} completed successfully in {results.duration_seconds:.2f}s")
            
        except Exception as e:
            results.status = "failed"
            results.error_message = str(e)
            results.duration_seconds = time.time() - start_time
            self.progress_tracker.update_progress(experiment_id, 0.0, "failed")
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            raise
        
        return results
    
    def run_parallel_experiments(self, 
                                configs: List[ExperimentConfig], 
                                max_workers: Optional[int] = None,
                                use_processes: bool = False) -> List[ExperimentResults]:
        """
        Run multiple experiments in parallel.
        
        Args:
            configs: List of experiment configurations
            max_workers: Maximum number of parallel workers (defaults to CPU count)
            use_processes: Whether to use processes instead of threads
            
        Returns:
            List of ExperimentResults
        """
        if max_workers is None:
            max_workers = min(len(configs), mp.cpu_count())
        
        self.logger.info(f"Starting {len(configs)} experiments with {max_workers} workers")
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        results = []
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_config = {}
            for i, config in enumerate(configs):
                experiment_id = f"parallel_exp_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                future = executor.submit(self.run_single_experiment, config, experiment_id)
                future_to_config[future] = (config, experiment_id)
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config, experiment_id = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed experiment {experiment_id}")
                except Exception as e:
                    self.logger.error(f"Failed experiment {experiment_id}: {e}")
                    # Create failed result
                    failed_result = ExperimentResults(
                        config=config,
                        experiment_id=experiment_id,
                        timestamp=datetime.now(),
                        duration_seconds=0.0,
                        status="failed",
                        error_message=str(e)
                    )
                    results.append(failed_result)
        
        self.logger.info(f"Completed all {len(configs)} experiments")
        return results
    
    def _generate_data(self, config: ExperimentConfig) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, Any]]:
        """Generate training and validation data."""
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Initialize data generation components
        poly_gen = PolynomialGenerator(random_seed=config.random_seed)
        noise_injector = NoiseInjector(random_seed=config.random_seed)
        dataset_builder = DatasetBuilder(random_seed=config.random_seed)
        
        # Build complete dataset
        dataset = dataset_builder.build_dataset(
            degree=config.polynomial_degree,
            x_range=config.x_range,
            num_points=config.num_data_points,
            coeff_range=(-5.0, 5.0),
            noise_type="gaussian",
            noise_level=config.noise_level,
            train_ratio=config.train_val_split,
            normalize="standard",
            shuffle=True
        )
        
        # Create DataLoaders
        train_data, val_data = dataset_builder.create_dataloaders(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle_train=True,
            shuffle_val=False
        )
        
        # Collect data statistics from dataset metadata
        metadata = dataset['metadata']
        raw_data = dataset['raw_data']
        data_stats = {
            'num_points': metadata['num_points'],
            'polynomial_degree': metadata['degree'],
            'coefficients': metadata['coefficients'].tolist(),
            'x_range': metadata['x_range'],
            'y_range': (float(raw_data['y_noisy'].min()), float(raw_data['y_noisy'].max())),
            'noise_level': metadata['noise_level'],
            'train_size': len(train_data.dataset),
            'val_size': len(val_data.dataset)
        }
        
        return train_data, val_data, data_stats
    
    def _create_model(self, config: ExperimentConfig) -> torch.nn.Module:
        """Create model based on configuration."""
        model_config = {
            'input_dim': 1,  # Single input for curve fitting
            'hidden_dims': config.hidden_dims,
            'activation': config.activation_function,
            'dropout_rate': 0.0,  # Default dropout
            'batch_norm': False   # Default batch norm
        }
        
        model = ModelFactory.create_model(
            model_type=config.model_architecture,
            config=model_config
        )
        
        return model
    
    def _train_model(self, 
                    config: ExperimentConfig, 
                    model: torch.nn.Module, 
                    train_data: torch.utils.data.DataLoader, 
                    val_data: torch.utils.data.DataLoader,
                    experiment_id: str) -> TrainingResults:
        """Train the model."""
        # Create optimizer configuration
        optimizer_config = OptimizerConfig(
            optimizer_type=config.optimizer,
            learning_rate=config.learning_rate
        )
        
        # Create optimizer using OptimizerFactory
        optimizer_factory = OptimizerFactory()
        optimizer = optimizer_factory.create_optimizer(model, optimizer_config)
        
        # Initialize trainer
        trainer = Trainer(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            verbose=True
        )
        
        # Create early stopping if needed
        early_stopping = None
        if config.early_stopping_patience > 0:
            early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # Train model
        training_results = trainer.train(
            model=model,
            train_loader=train_data,
            val_loader=val_data,
            optimizer=optimizer,
            epochs=config.epochs,
            early_stopping=early_stopping
        )
        
        return training_results
    
    def _generate_visualizations(self, 
                               config: ExperimentConfig,
                               model: torch.nn.Module,
                               train_data: torch.utils.data.DataLoader,
                               val_data: torch.utils.data.DataLoader,
                               training_results: TrainingResults,
                               experiment_id: str) -> List[str]:
        """Generate visualization plots."""
        image_paths = []
        exp_image_dir = self.images_dir / experiment_id
        exp_image_dir.mkdir(exist_ok=True)
        
        # Initialize plotters
        curve_plotter = CurvePlotter()
        loss_plotter = LossPlotter()
        
        # Generate curve fitting plot
        # Get all data for plotting
        all_x = []
        all_y = []
        for batch_x, batch_y in train_data:
            all_x.append(batch_x.numpy())
            all_y.append(batch_y.numpy())
        for batch_x, batch_y in val_data:
            all_x.append(batch_x.numpy())
            all_y.append(batch_y.numpy())
        
        x_plot = np.concatenate(all_x).flatten()
        y_plot = np.concatenate(all_y).flatten()
        
        # Generate predictions on a denser grid for smooth curve
        x_pred = np.linspace(x_plot.min(), x_plot.max(), 200)
        model.eval()
        with torch.no_grad():
            x_pred_tensor = torch.FloatTensor(x_pred.reshape(-1, 1))
            y_pred = model(x_pred_tensor).numpy().flatten()
        
        # Create curve fitting plot
        fig = curve_plotter.plot_curve_fit(
            x_data=x_plot,
            y_data=y_plot,
            x_pred=x_pred,
            y_pred=y_pred,
            title=f"Curve Fitting - {config.model_architecture} (Degree {config.polynomial_degree})"
        )
        saved_paths = curve_plotter.save_figure(fig, "curve_fit", str(exp_image_dir))
        image_paths.extend(saved_paths)
        
        # Create loss curves plot
        fig = loss_plotter.plot_loss_curves(
            train_losses=training_results.train_losses,
            val_losses=training_results.val_losses,
            title=f"Training Progress - {config.optimizer.upper()}"
        )
        saved_paths = loss_plotter.save_figure(fig, "loss_curves", str(exp_image_dir))
        image_paths.extend(saved_paths)
        
        return image_paths
    
    def _save_experiment_artifacts(self, 
                                 config: ExperimentConfig,
                                 model: torch.nn.Module,
                                 results: ExperimentResults,
                                 experiment_id: str) -> Tuple[str, str]:
        """Save experiment artifacts."""
        exp_dir = self.output_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = str(exp_dir / "model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'config': config.to_dict()
        }, model_path)
        
        # Save configuration
        config_path = str(exp_dir / "config.json")
        config.to_json(config_path)
        
        # Save results
        results_path = str(exp_dir / "results.json")
        with open(results_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        return model_path, config_path
    
    def get_experiment_progress(self, experiment_id: str) -> Tuple[float, str]:
        """Get progress for a specific experiment."""
        return self.progress_tracker.get_progress(experiment_id)
    
    def get_all_experiment_progress(self) -> Dict[str, Tuple[float, str]]:
        """Get progress for all experiments."""
        return self.progress_tracker.get_all_progress()
    
    def load_experiment_results(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Load experiment results from disk."""
        results_path = self.output_dir / experiment_id / "results.json"
        
        if not results_path.exists():
            return None
        
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            return ExperimentResults.from_dict(data)
        except Exception as e:
            self.logger.error(f"Failed to load experiment results for {experiment_id}: {e}")
            return None