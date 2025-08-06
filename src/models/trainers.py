"""
Training utilities for neural network models in curve fitting experiments.

This module provides training orchestration, optimizer management, and
training progress tracking for AI curve fitting research.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from .architectures import BaseModel


@dataclass
class OptimizerConfig:
    """Configuration for optimizer creation."""
    optimizer_type: str = 'adam'
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9     # For Adam
    beta2: float = 0.999   # For Adam
    eps: float = 1e-8      # For Adam, RMSprop, AdaGrad
    alpha: float = 0.99    # For RMSprop
    centered: bool = False # For RMSprop
    lr_decay: float = 0.0  # For AdaGrad
    initial_accumulator_value: float = 0.0  # For AdaGrad
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'alpha': self.alpha,
            'centered': self.centered,
            'lr_decay': self.lr_decay,
            'initial_accumulator_value': self.initial_accumulator_value
        }


@dataclass
class TrainingResults:
    """Container for training results and metrics."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    training_time: float = 0.0
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    confidence_intervals: Optional[np.ndarray] = None
    epoch_times: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_train_loss': self.final_train_loss,
            'final_val_loss': self.final_val_loss,
            'training_time': self.training_time,
            'model_parameters': self.model_parameters,
            'predictions': self.predictions.tolist() if self.predictions is not None else None,
            'confidence_intervals': self.confidence_intervals.tolist() if self.confidence_intervals is not None else None,
            'epoch_times': self.epoch_times,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }


class LossTracker:
    """Tracks and manages training and validation losses during training."""
    
    def __init__(self, save_path: Optional[str] = None):
        """Initialize loss tracker.
        
        Args:
            save_path: Optional path to save loss history
        """
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        self.save_path = save_path
        self.start_time = None
        
    def start_epoch(self):
        """Mark the start of an epoch for timing."""
        self.start_time = time.time()
    
    def end_epoch(self, train_loss: float, val_loss: float):
        """Record losses and timing for completed epoch.
        
        Args:
            train_loss: Training loss for the epoch
            val_loss: Validation loss for the epoch
        """
        if self.start_time is not None:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if self.save_path:
            self.save_losses()
    
    def get_current_epoch(self) -> int:
        """Get current epoch number (0-indexed)."""
        return len(self.train_losses)
    
    def get_best_epoch(self) -> Tuple[int, float]:
        """Get epoch with best validation loss.
        
        Returns:
            Tuple of (best_epoch, best_val_loss)
        """
        if not self.val_losses:
            return 0, float('inf')
        
        best_epoch = np.argmin(self.val_losses)
        best_val_loss = self.val_losses[best_epoch]
        return best_epoch, best_val_loss
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """Get statistics about loss progression.
        
        Returns:
            Dictionary with loss statistics
        """
        if not self.train_losses or not self.val_losses:
            return {}
        
        best_epoch, best_val_loss = self.get_best_epoch()
        
        return {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'min_train_loss': min(self.train_losses),
            'max_train_loss': max(self.train_losses),
            'min_val_loss': min(self.val_losses),
            'max_val_loss': max(self.val_losses),
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0.0,
            'total_epochs': len(self.train_losses)
        }
    
    def save_losses(self):
        """Save loss history to file."""
        if not self.save_path:
            return
        
        os.makedirs(os.path.dirname(self.save_path) if os.path.dirname(self.save_path) else '.', exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        statistics = convert_numpy_types(self.get_loss_statistics())
        
        loss_data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch_times': self.epoch_times,
            'statistics': statistics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.save_path, 'w') as f:
            json.dump(loss_data, f, indent=2)
    
    def load_losses(self, file_path: str):
        """Load loss history from file.
        
        Args:
            file_path: Path to loss history file
        """
        with open(file_path, 'r') as f:
            loss_data = json.load(f)
        
        self.train_losses = loss_data.get('train_losses', [])
        self.val_losses = loss_data.get('val_losses', [])
        self.epoch_times = loss_data.get('epoch_times', [])


class Trainer:
    """Main training orchestration class for neural network models."""
    
    def __init__(self, 
                 device: Optional[torch.device] = None,
                 loss_fn: Optional[nn.Module] = None,
                 gradient_clipper: Optional['GradientClipper'] = None,
                 verbose: bool = True):
        """Initialize trainer.
        
        Args:
            device: Device to run training on (CPU/GPU)
            loss_fn: Loss function to use (default: MSELoss)
            gradient_clipper: Optional gradient clipper for training stability
            verbose: Whether to print training progress
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()
        self.gradient_clipper = gradient_clipper
        self.verbose = verbose
        
    def _compute_loss(self, model: BaseModel, data_loader: DataLoader) -> float:
        """Compute average loss over a dataset.
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader for the dataset
            
        Returns:
            Average loss over the dataset
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = model(batch_x)
                loss = self.loss_fn(predictions, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _train_epoch(self, model: BaseModel, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer) -> float:
        """Train model for one epoch.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer for parameter updates
            
        Returns:
            Average training loss for the epoch
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            predictions = model(batch_x)
            loss = self.loss_fn(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping if configured
            if self.gradient_clipper:
                self.gradient_clipper.clip_gradients(model)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, 
              model: BaseModel,
              train_loader: DataLoader,
              val_loader: DataLoader,
              optimizer: torch.optim.Optimizer,
              epochs: int,
              loss_tracker: Optional[LossTracker] = None,
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
              early_stopping: Optional['EarlyStopping'] = None,
              checkpoint_callback: Optional[callable] = None) -> TrainingResults:
        """Train model with specified configuration.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for parameter updates
            epochs: Number of training epochs
            loss_tracker: Optional loss tracker for monitoring
            scheduler: Optional learning rate scheduler
            early_stopping: Optional early stopping callback
            checkpoint_callback: Optional callback for model checkpointing
            
        Returns:
            TrainingResults object with training history and metrics
        """
        # Move model to device
        model.to(self.device)
        
        # Initialize loss tracker if not provided
        if loss_tracker is None:
            loss_tracker = LossTracker()
        
        # Record training start time
        training_start_time = time.time()
        
        if self.verbose:
            print(f"Starting training on {self.device}")
            print(f"Model: {model.__class__.__name__}")
            print(f"Parameters: {model.count_parameters():,}")
            print(f"Epochs: {epochs}")
            print("-" * 50)
        
        # Training loop
        for epoch in range(epochs):
            loss_tracker.start_epoch()
            
            # Train for one epoch
            train_loss = self._train_epoch(model, train_loader, optimizer)
            
            # Compute validation loss
            val_loss = self._compute_loss(model, val_loader)
            
            # Update learning rate scheduler
            if scheduler:
                scheduler.step()
            
            # Record losses
            loss_tracker.end_epoch(train_loss, val_loss)
            
            # Print progress
            if self.verbose:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {lr:.2e}")
            
            # Check early stopping
            if early_stopping:
                early_stopping.step(val_loss)
                if early_stopping.should_stop():
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Checkpoint callback
            if checkpoint_callback:
                checkpoint_callback(model, epoch, {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch
                })
        
        # Calculate total training time
        total_training_time = time.time() - training_start_time
        
        # Get final statistics
        stats = loss_tracker.get_loss_statistics()
        best_epoch, best_val_loss = loss_tracker.get_best_epoch()
        
        if self.verbose:
            print("-" * 50)
            print(f"Training completed in {total_training_time:.2f}s")
            print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
        
        # Create results object
        results = TrainingResults(
            train_losses=loss_tracker.train_losses,
            val_losses=loss_tracker.val_losses,
            final_train_loss=stats.get('final_train_loss', 0.0),
            final_val_loss=stats.get('final_val_loss', 0.0),
            training_time=total_training_time,
            model_parameters=model.get_parameters(),
            epoch_times=loss_tracker.epoch_times,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss
        )
        
        return results
    
    def train_with_full_pipeline(self,
                                model: BaseModel,
                                train_loader: DataLoader,
                                val_loader: DataLoader,
                                optimizer_config: OptimizerConfig,
                                epochs: int,
                                scheduler_type: str = 'none',
                                scheduler_kwargs: Optional[Dict[str, Any]] = None,
                                early_stopping_patience: Optional[int] = None,
                                gradient_clip_value: Optional[float] = None,
                                gradient_clip_type: str = 'norm',
                                save_checkpoints: bool = False,
                                checkpoint_dir: str = 'checkpoints') -> TrainingResults:
        """Train model with full pipeline including optimizer, scheduler, and early stopping.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer_config: Optimizer configuration
            epochs: Number of training epochs
            scheduler_type: Type of learning rate scheduler
            scheduler_kwargs: Scheduler-specific parameters
            early_stopping_patience: Patience for early stopping (None to disable)
            gradient_clip_value: Gradient clipping value (None to disable)
            gradient_clip_type: Type of gradient clipping ('norm' or 'value')
            save_checkpoints: Whether to save model checkpoints
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            TrainingResults with comprehensive training information
        """
        # Create optimizer
        optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
        
        # Create scheduler
        scheduler_kwargs = scheduler_kwargs or {}
        scheduler = LearningRateSchedulerFactory.create_scheduler(
            optimizer, scheduler_type, **scheduler_kwargs
        )
        
        # Create early stopping
        early_stopping = None
        if early_stopping_patience is not None:
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                verbose=self.verbose
            )
        
        # Create gradient clipper
        if gradient_clip_value is not None:
            self.gradient_clipper = GradientClipper(gradient_clip_type, gradient_clip_value)
        
        # Create checkpoint callback if requested
        checkpoint_callback = None
        if save_checkpoints:
            from .architectures import ModelCheckpoint
            checkpointer = ModelCheckpoint(checkpoint_dir)
            
            def checkpoint_callback(model, epoch, metrics):
                checkpointer.save_checkpoint(
                    model, epoch, metrics,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None
                )
        
        # Train model
        results = self.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=epochs,
            scheduler=scheduler,
            early_stopping=early_stopping,
            checkpoint_callback=checkpoint_callback
        )
        
        return results
    
    def evaluate(self, model: BaseModel, test_loader: DataLoader, 
                 calculate_confidence_intervals: bool = False) -> Dict[str, Any]:
        """Evaluate model on test dataset with comprehensive metrics.
        
        Args:
            model: Trained model to evaluate
            test_loader: Test data loader
            calculate_confidence_intervals: Whether to calculate prediction confidence intervals
            
        Returns:
            Dictionary with evaluation metrics and optional confidence intervals
        """
        model.to(self.device)
        model.eval()
        
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                batch_predictions = model(batch_x)
                loss = self.loss_fn(batch_predictions, batch_y)
                
                total_loss += loss.item()
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Calculate comprehensive metrics
        metrics = PerformanceMetrics.calculate_regression_metrics(targets, predictions)
        metrics['test_loss'] = total_loss / len(test_loader)
        metrics['num_samples'] = len(predictions)
        
        # Calculate confidence intervals if requested
        if calculate_confidence_intervals:
            residuals = targets - predictions
            confidence_intervals = PerformanceMetrics.calculate_confidence_intervals(
                predictions, residuals
            )
            metrics['confidence_intervals'] = confidence_intervals
        
        return metrics
    
    def predict(self, model: BaseModel, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for a dataset.
        
        Args:
            model: Trained model
            data_loader: Data loader for prediction
            
        Returns:
            Tuple of (predictions, targets)
        """
        model.to(self.device)
        model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                batch_predictions = model(batch_x)
                
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
        
        return np.array(predictions), np.array(targets)
    
    def predict_with_confidence(self, 
                               model: BaseModel, 
                               data_loader: DataLoader,
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with confidence intervals.
        
        Args:
            model: Trained model
            data_loader: Data loader for prediction
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (predictions, targets, confidence_intervals)
        """
        predictions, targets = self.predict(model, data_loader)
        
        # Calculate residuals for confidence interval estimation
        residuals = targets.flatten() - predictions.flatten()
        
        # Calculate confidence intervals
        confidence_intervals = PerformanceMetrics.calculate_confidence_intervals(
            predictions.flatten(), residuals, confidence_level
        )
        
        return predictions, targets, confidence_intervals


class OptimizerFactory:
    """Factory class for creating different optimizer instances."""
    
    SUPPORTED_OPTIMIZERS = ['sgd', 'adam', 'rmsprop', 'adagrad']
    
    @staticmethod
    def create_optimizer(model: BaseModel, config: OptimizerConfig) -> torch.optim.Optimizer:
        """Create optimizer instance based on configuration.
        
        Args:
            model: Model whose parameters to optimize
            config: Optimizer configuration
            
        Returns:
            Configured optimizer instance
            
        Raises:
            ValueError: If optimizer type is not supported
        """
        optimizer_type = config.optimizer_type.lower()
        
        if optimizer_type == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        
        elif optimizer_type == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay
            )
        
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(
                model.parameters(),
                lr=config.learning_rate,
                alpha=config.alpha,
                eps=config.eps,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                centered=config.centered
            )
        
        elif optimizer_type == 'adagrad':
            return optim.Adagrad(
                model.parameters(),
                lr=config.learning_rate,
                lr_decay=config.lr_decay,
                weight_decay=config.weight_decay,
                initial_accumulator_value=config.initial_accumulator_value,
                eps=config.eps
            )
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                           f"Supported types: {OptimizerFactory.SUPPORTED_OPTIMIZERS}")
    
    @staticmethod
    def get_supported_optimizers() -> List[str]:
        """Get list of supported optimizer types."""
        return OptimizerFactory.SUPPORTED_OPTIMIZERS.copy()
    
    @staticmethod
    def get_default_config(optimizer_type: str) -> OptimizerConfig:
        """Get default configuration for an optimizer type.
        
        Args:
            optimizer_type: Type of optimizer
            
        Returns:
            Default configuration for the optimizer
            
        Raises:
            ValueError: If optimizer type is not supported
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'sgd':
            return OptimizerConfig(
                optimizer_type='sgd',
                learning_rate=0.01,
                momentum=0.9,
                weight_decay=0.0
            )
        
        elif optimizer_type == 'adam':
            return OptimizerConfig(
                optimizer_type='adam',
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weight_decay=0.0
            )
        
        elif optimizer_type == 'rmsprop':
            return OptimizerConfig(
                optimizer_type='rmsprop',
                learning_rate=0.01,
                alpha=0.99,
                eps=1e-8,
                weight_decay=0.0,
                momentum=0.0,
                centered=False
            )
        
        elif optimizer_type == 'adagrad':
            return OptimizerConfig(
                optimizer_type='adagrad',
                learning_rate=0.01,
                lr_decay=0.0,
                weight_decay=0.0,
                initial_accumulator_value=0.0,
                eps=1e-10
            )
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. "
                           f"Supported types: {OptimizerFactory.SUPPORTED_OPTIMIZERS}")


class LearningRateSchedulerFactory:
    """Factory class for creating learning rate schedulers."""
    
    SUPPORTED_SCHEDULERS = ['step', 'exponential', 'cosine', 'plateau', 'none']
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer, 
                        scheduler_type: str,
                        **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler
            **kwargs: Scheduler-specific parameters
            
        Returns:
            Configured scheduler or None if scheduler_type is 'none'
            
        Raises:
            ValueError: If scheduler type is not supported
        """
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == 'none':
            return None
        
        elif scheduler_type == 'step':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_type == 'exponential':
            gamma = kwargs.get('gamma', 0.95)
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        
        elif scheduler_type == 'cosine':
            T_max = kwargs.get('T_max', 50)
            eta_min = kwargs.get('eta_min', 0)
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        elif scheduler_type == 'plateau':
            mode = kwargs.get('mode', 'min')
            factor = kwargs.get('factor', 0.1)
            patience = kwargs.get('patience', 10)
            threshold = kwargs.get('threshold', 1e-4)
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
            )
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}. "
                           f"Supported types: {LearningRateSchedulerFactory.SUPPORTED_SCHEDULERS}")
    
    @staticmethod
    def get_supported_schedulers() -> List[str]:
        """Get list of supported scheduler types."""
        return LearningRateSchedulerFactory.SUPPORTED_SCHEDULERS.copy()


class GradientClipper:
    """Utility class for gradient clipping during training."""
    
    def __init__(self, clip_type: str = 'norm', clip_value: float = 1.0):
        """Initialize gradient clipper.
        
        Args:
            clip_type: Type of clipping ('norm' or 'value')
            clip_value: Clipping threshold
        """
        self.clip_type = clip_type.lower()
        self.clip_value = clip_value
        
        if self.clip_type not in ['norm', 'value']:
            raise ValueError(f"Unsupported clip type: {clip_type}. Use 'norm' or 'value'")
    
    def clip_gradients(self, model: BaseModel) -> float:
        """Apply gradient clipping to model parameters.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Gradient norm before clipping
        """
        if self.clip_type == 'norm':
            return torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        else:  # clip_type == 'value'
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
            # Calculate and return gradient norm for monitoring
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** (1. / 2)


class EarlyStopping:
    """Early stopping utility to prevent overfitting during training."""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best_weights: bool = True,
                 verbose: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            restore_best_weights: Whether to restore model to best weights when stopping
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        if self.mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def step(self, current_value: float, model: Optional[BaseModel] = None, epoch: int = 0) -> bool:
        """Update early stopping state with current metric value.
        
        Args:
            current_value: Current value of the monitored metric
            model: Model to save weights from (if restore_best_weights is True)
            epoch: Current epoch number
            
        Returns:
            True if metric improved, False otherwise
        """
        improved = self._is_improvement(current_value)
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights if requested
            if self.restore_best_weights and model is not None:
                self.best_weights = {name: param.clone() for name, param in model.named_parameters()}
            
            if self.verbose:
                print(f"Early stopping: metric improved to {current_value:.6f}")
        else:
            self.wait += 1
            if self.verbose and self.wait == 1:
                print(f"Early stopping: no improvement for 1 epoch")
        
        return improved
    
    def should_stop(self) -> bool:
        """Check if training should be stopped.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.wait >= self.patience:
            self.stopped_epoch = self.best_epoch + self.wait
            return True
        return False
    
    def restore_best_model(self, model: BaseModel) -> bool:
        """Restore model to best weights.
        
        Args:
            model: Model to restore weights to
            
        Returns:
            True if weights were restored, False if no best weights available
        """
        if self.best_weights is not None:
            for name, param in model.named_parameters():
                if name in self.best_weights:
                    param.data.copy_(self.best_weights[name])
            return True
        return False
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about the best metric value.
        
        Returns:
            Dictionary with best value, epoch, and stopping information
        """
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch,
            'patience': self.patience,
            'wait': self.wait,
            'stopped': self.should_stop()
        }
    
    def _is_improvement(self, current_value: float) -> bool:
        """Check if current value is an improvement over best value."""
        if self.mode == 'min':
            return current_value < (self.best_value - self.min_delta)
        else:
            return current_value > (self.best_value + self.min_delta)


class PerformanceMetrics:
    """Utility class for calculating various performance metrics."""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary with various regression metrics
        """
        # Basic error metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Adjusted R-squared (assuming single feature for simplicity)
        n = len(y_true)
        p = 1  # number of predictors
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        # Explained variance score
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / var_y if var_y != 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'adjusted_r2': adj_r2,
            'mape': mape,
            'explained_variance': explained_var,
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_error': np.mean(y_true - y_pred)
        }
    
    @staticmethod
    def calculate_confidence_intervals(y_pred: np.ndarray, 
                                     residuals: np.ndarray,
                                     confidence_level: float = 0.95) -> np.ndarray:
        """Calculate prediction confidence intervals.
        
        Args:
            y_pred: Predicted values
            residuals: Residuals (y_true - y_pred)
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Array of confidence intervals with shape (n_samples, 2)
        """
        try:
            from scipy import stats
            
            # Calculate standard error of residuals
            residual_std = np.std(residuals)
            
            # Calculate t-value for confidence level
            alpha = 1 - confidence_level
            df = len(residuals) - 1  # degrees of freedom
            t_value = stats.t.ppf(1 - alpha/2, df)
            
            # Calculate confidence intervals
            margin_of_error = t_value * residual_std
            
        except ImportError:
            # Fallback to normal approximation if scipy is not available
            residual_std = np.std(residuals)
            
            # Use normal approximation (z-score) instead of t-distribution
            # For 95% confidence level, z = 1.96
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z_value = z_scores.get(confidence_level, 1.96)
            
            margin_of_error = z_value * residual_std
        
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        
        return np.column_stack([lower_bound, upper_bound])


class TrainingTimer:
    """Utility class for tracking training time and performance."""
    
    def __init__(self):
        """Initialize training timer."""
        self.start_time = None
        self.end_time = None
        self.epoch_times = []
        self.epoch_start_time = None
    
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
    
    def end_training(self):
        """Mark the end of training."""
        self.end_time = time.time()
    
    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
    
    def end_epoch(self):
        """Mark the end of an epoch and record time."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            return epoch_time
        return 0.0
    
    def get_total_time(self) -> float:
        """Get total training time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time
    
    def get_average_epoch_time(self) -> float:
        """Get average epoch time in seconds."""
        return np.mean(self.epoch_times) if self.epoch_times else 0.0
    
    def get_time_statistics(self) -> Dict[str, float]:
        """Get comprehensive time statistics.
        
        Returns:
            Dictionary with timing statistics
        """
        return {
            'total_time': self.get_total_time(),
            'average_epoch_time': self.get_average_epoch_time(),
            'min_epoch_time': min(self.epoch_times) if self.epoch_times else 0.0,
            'max_epoch_time': max(self.epoch_times) if self.epoch_times else 0.0,
            'total_epochs': len(self.epoch_times),
            'epochs_per_second': len(self.epoch_times) / self.get_total_time() if self.get_total_time() > 0 else 0.0
        }