"""
Plotting utilities for AI curve fitting research.

This module provides classes for creating publication-quality visualizations
of curve fitting results, training progress, and model performance comparisons.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend for thread safety
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import os


class CurvePlotter:
    """
    Creates publication-quality curve fitting visualizations.
    
    This class handles plotting original data points, fitted curves, and
    confidence intervals with customizable styling suitable for research papers.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the CurvePlotter with default styling.
        
        Args:
            style: Matplotlib style to use for plots
            figsize: Default figure size as (width, height)
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'data': '#2E86AB',
            'prediction': '#A23B72',
            'confidence': '#F18F01',
            'true_curve': '#C73E1D'
        }
        
        # Set publication-quality defaults
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def plot_curve_fit(self, 
                      x_data: np.ndarray, 
                      y_data: np.ndarray,
                      x_pred: np.ndarray,
                      y_pred: np.ndarray,
                      confidence_intervals: Optional[np.ndarray] = None,
                      true_curve: Optional[np.ndarray] = None,
                      title: str = "Curve Fitting Results",
                      xlabel: str = "x",
                      ylabel: str = "y") -> plt.Figure:
        """
        Create a curve fitting visualization with data points, predictions, and confidence intervals.
        
        Args:
            x_data: Original x data points
            y_data: Original y data points  
            x_pred: X values for predictions (typically more dense than x_data)
            y_pred: Model predictions
            confidence_intervals: Optional confidence intervals as (lower, upper) bounds
            true_curve: Optional true curve values for comparison
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot original data points
            ax.scatter(x_data, y_data, 
                      color=self.colors['data'], 
                      alpha=0.7, 
                      s=50,
                      label='Training Data',
                      zorder=3)
            
            # Plot true curve if provided
            if true_curve is not None:
                ax.plot(x_pred, true_curve, 
                       color=self.colors['true_curve'],
                       linestyle='--',
                       linewidth=2,
                       label='True Curve',
                       zorder=2)
            
            # Plot model predictions
            ax.plot(x_pred, y_pred, 
                   color=self.colors['prediction'],
                   linewidth=2.5,
                   label='Model Prediction',
                   zorder=4)
            
            # Plot confidence intervals if provided
            if confidence_intervals is not None:
                lower_bound, upper_bound = confidence_intervals
                ax.fill_between(x_pred, lower_bound, upper_bound,
                               color=self.colors['confidence'],
                               alpha=0.3,
                               label='95% Confidence Interval',
                               zorder=1)
            
            # Styling
            ax.set_xlabel(xlabel, fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add subtle background
            ax.set_facecolor('#fafafa')
            
            # Improve layout
            plt.tight_layout()
            
            return fig
    
    def plot_residuals(self,
                      x_data: np.ndarray,
                      residuals: np.ndarray,
                      title: str = "Residual Analysis") -> plt.Figure:
        """
        Create a residual plot to analyze model fit quality.
        
        Args:
            x_data: X values
            residuals: Residual values (y_true - y_pred)
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs fitted values
            ax1.scatter(x_data, residuals, 
                       color=self.colors['data'], 
                       alpha=0.7, 
                       s=50)
            ax1.axhline(y=0, color=self.colors['prediction'], 
                       linestyle='--', linewidth=2)
            ax1.set_xlabel('X Values', fontweight='bold')
            ax1.set_ylabel('Residuals', fontweight='bold')
            ax1.set_title('Residuals vs X Values', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#fafafa')
            
            # Histogram of residuals
            ax2.hist(residuals, bins=20, 
                    color=self.colors['confidence'], 
                    alpha=0.7, 
                    edgecolor='black')
            ax2.set_xlabel('Residuals', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title('Distribution of Residuals', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#fafafa')
            
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            return fig
    
    def save_figure(self, 
                   fig: plt.Figure, 
                   filename: str, 
                   output_dir: str = "images",
                   formats: List[str] = ['png', 'pdf']) -> List[str]:
        """
        Save figure in multiple formats with publication-quality settings.
        
        Args:
            fig: matplotlib Figure object to save
            filename: Base filename (without extension)
            output_dir: Directory to save files
            formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
            
        Returns:
            List of saved file paths
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for fmt in formats:
            filepath = os.path.join(output_dir, f"{filename}.{fmt}")
            
            # Format-specific settings
            save_kwargs = {
                'bbox_inches': 'tight',
                'pad_inches': 0.1,
                'facecolor': 'white',
                'edgecolor': 'none'
            }
            
            if fmt == 'png':
                save_kwargs['dpi'] = 300
            elif fmt == 'pdf':
                save_kwargs['dpi'] = 300
                save_kwargs['backend'] = 'pdf'
            elif fmt == 'svg':
                save_kwargs['format'] = 'svg'
            elif fmt == 'eps':
                save_kwargs['format'] = 'eps'
            
            fig.savefig(filepath, **save_kwargs)
            saved_paths.append(filepath)
        
        return saved_paths
    
    def create_subplot_grid(self, 
                           nrows: int, 
                           ncols: int, 
                           figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a grid of subplots with consistent styling.
        
        Args:
            nrows: Number of rows
            ncols: Number of columns
            figsize: Figure size, defaults to scaled version of self.figsize
            
        Returns:
            Tuple of (figure, axes array)
        """
        if figsize is None:
            figsize = (self.figsize[0] * ncols, self.figsize[1] * nrows)
        
        with plt.style.context(self.style):
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            
            # Ensure axes is always an array for consistent indexing
            if nrows == 1 and ncols == 1:
                axes = np.array([axes])
            elif nrows == 1 or ncols == 1:
                axes = axes.reshape(-1)
            
            # Apply consistent styling to all subplots
            for ax in axes.flat:
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_facecolor('#fafafa')
            
            plt.tight_layout()
            
            return fig, axes


class LossPlotter:
    """
    Creates visualizations for training progress and loss curves.
    
    This class handles plotting training and validation losses over time,
    with options for smoothing and multi-experiment comparisons.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize the LossPlotter.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            'train': '#2E86AB',
            'validation': '#A23B72',
            'smoothed': '#F18F01',
            'experiment_1': '#E74C3C',
            'experiment_2': '#3498DB',
            'experiment_3': '#2ECC71',
            'experiment_4': '#F39C12',
            'experiment_5': '#9B59B6'
        }
    
    def plot_loss_curves(self, 
                        train_losses: List[float],
                        val_losses: List[float],
                        title: str = "Training Progress",
                        smooth: bool = True,
                        smooth_window: int = 10) -> plt.Figure:
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            title: Plot title
            smooth: Whether to add smoothed curves
            smooth_window: Window size for smoothing
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            
            epochs = range(1, len(train_losses) + 1)
            
            # Plot raw loss curves
            ax.plot(epochs, train_losses, 
                   color=self.colors['train'], 
                   alpha=0.7,
                   linewidth=1.5,
                   label='Training Loss')
            
            ax.plot(epochs, val_losses, 
                   color=self.colors['validation'], 
                   alpha=0.7,
                   linewidth=1.5,
                   label='Validation Loss')
            
            # Add smoothed curves if requested
            if smooth and len(train_losses) > smooth_window:
                train_smooth = self._smooth_curve(train_losses, smooth_window)
                val_smooth = self._smooth_curve(val_losses, smooth_window)
                
                ax.plot(epochs, train_smooth,
                       color=self.colors['train'],
                       linewidth=3,
                       alpha=0.9,
                       label='Training (Smoothed)')
                
                ax.plot(epochs, val_smooth,
                       color=self.colors['validation'],
                       linewidth=3,
                       alpha=0.9,
                       label='Validation (Smoothed)')
            
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Loss', fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#fafafa')
            
            # Use log scale if losses span multiple orders of magnitude
            if max(train_losses + val_losses) / min(train_losses + val_losses) > 100:
                ax.set_yscale('log')
            
            plt.tight_layout()
            
            return fig
    
    def plot_multi_experiment_comparison(self,
                                        experiments_data: Dict[str, Dict[str, List[float]]],
                                        title: str = "Multi-Experiment Loss Comparison",
                                        smooth: bool = True,
                                        smooth_window: int = 10,
                                        show_std: bool = True) -> plt.Figure:
        """
        Plot loss curves for multiple experiments with statistical overlays.
        
        Args:
            experiments_data: Dictionary with experiment names as keys and 
                            {'train_losses': [...], 'val_losses': [...]} as values
            title: Plot title
            smooth: Whether to apply smoothing
            smooth_window: Window size for smoothing
            show_std: Whether to show standard deviation bands
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            experiment_names = list(experiments_data.keys())
            color_cycle = plt.cm.Set1(np.linspace(0, 1, len(experiment_names)))
            
            # Plot training losses
            for i, (exp_name, data) in enumerate(experiments_data.items()):
                train_losses = data['train_losses']
                epochs = range(1, len(train_losses) + 1)
                color = color_cycle[i]
                
                if smooth and len(train_losses) > smooth_window:
                    smoothed = self._smooth_curve(train_losses, smooth_window)
                    ax1.plot(epochs, smoothed, color=color, linewidth=2.5, 
                            label=f'{exp_name} (smoothed)', alpha=0.9)
                    ax1.plot(epochs, train_losses, color=color, linewidth=1, 
                            alpha=0.3)
                else:
                    ax1.plot(epochs, train_losses, color=color, linewidth=2, 
                            label=exp_name)
            
            ax1.set_xlabel('Epoch', fontweight='bold')
            ax1.set_ylabel('Training Loss', fontweight='bold')
            ax1.set_title('Training Loss Comparison', fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#fafafa')
            
            # Plot validation losses
            for i, (exp_name, data) in enumerate(experiments_data.items()):
                val_losses = data['val_losses']
                epochs = range(1, len(val_losses) + 1)
                color = color_cycle[i]
                
                if smooth and len(val_losses) > smooth_window:
                    smoothed = self._smooth_curve(val_losses, smooth_window)
                    ax2.plot(epochs, smoothed, color=color, linewidth=2.5, 
                            label=f'{exp_name} (smoothed)', alpha=0.9)
                    ax2.plot(epochs, val_losses, color=color, linewidth=1, 
                            alpha=0.3)
                else:
                    ax2.plot(epochs, val_losses, color=color, linewidth=2, 
                            label=exp_name)
            
            ax2.set_xlabel('Epoch', fontweight='bold')
            ax2.set_ylabel('Validation Loss', fontweight='bold')
            ax2.set_title('Validation Loss Comparison', fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#fafafa')
            
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            return fig
    
    def plot_loss_statistics(self,
                           loss_data: List[List[float]],
                           loss_type: str = "Training Loss",
                           title: str = "Loss Statistics Across Runs") -> plt.Figure:
        """
        Plot loss statistics (mean, std, min, max) across multiple runs.
        
        Args:
            loss_data: List of loss curves from different runs
            loss_type: Type of loss being plotted
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Convert to numpy array for easier computation
            max_epochs = max(len(losses) for losses in loss_data)
            
            # Pad shorter sequences with NaN
            padded_data = []
            for losses in loss_data:
                padded = list(losses) + [np.nan] * (max_epochs - len(losses))
                padded_data.append(padded)
            
            loss_array = np.array(padded_data)
            epochs = range(1, max_epochs + 1)
            
            # Calculate statistics (ignoring NaN values)
            mean_losses = np.nanmean(loss_array, axis=0)
            std_losses = np.nanstd(loss_array, axis=0)
            min_losses = np.nanmin(loss_array, axis=0)
            max_losses = np.nanmax(loss_array, axis=0)
            
            # Plot mean with standard deviation band
            ax.plot(epochs, mean_losses, color=self.colors['train'], 
                   linewidth=3, label='Mean', zorder=3)
            
            ax.fill_between(epochs, 
                           mean_losses - std_losses,
                           mean_losses + std_losses,
                           color=self.colors['train'], alpha=0.3,
                           label='±1 Std Dev', zorder=1)
            
            # Plot min/max envelope
            ax.fill_between(epochs, min_losses, max_losses,
                           color=self.colors['validation'], alpha=0.1,
                           label='Min-Max Range', zorder=0)
            
            # Plot individual runs with low alpha
            for i, losses in enumerate(loss_data):
                run_epochs = range(1, len(losses) + 1)
                ax.plot(run_epochs, losses, color='gray', 
                       alpha=0.2, linewidth=1, zorder=2)
            
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel(loss_type, fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#fafafa')
            
            plt.tight_layout()
            
            return fig
    
    def plot_convergence_analysis(self,
                                train_losses: List[float],
                                val_losses: List[float],
                                convergence_threshold: float = 0.001,
                                patience: int = 10) -> plt.Figure:
        """
        Analyze and visualize convergence behavior with early stopping indicators.
        
        Args:
            train_losses: Training loss values
            val_losses: Validation loss values
            convergence_threshold: Threshold for considering convergence
            patience: Number of epochs to wait for improvement
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            epochs = range(1, len(train_losses) + 1)
            
            # Main loss curves
            ax1.plot(epochs, train_losses, color=self.colors['train'], 
                    linewidth=2, label='Training Loss')
            ax1.plot(epochs, val_losses, color=self.colors['validation'], 
                    linewidth=2, label='Validation Loss')
            
            # Find convergence point
            convergence_epoch = self._find_convergence_point(
                val_losses, convergence_threshold, patience
            )
            
            if convergence_epoch is not None:
                ax1.axvline(x=convergence_epoch, color='red', linestyle='--', 
                           linewidth=2, alpha=0.7, 
                           label=f'Convergence (Epoch {convergence_epoch})')
            
            ax1.set_xlabel('Epoch', fontweight='bold')
            ax1.set_ylabel('Loss', fontweight='bold')
            ax1.set_title('Loss Curves with Convergence Analysis', fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#fafafa')
            
            # Loss difference analysis
            loss_diff = np.array(val_losses) - np.array(train_losses)
            ax2.plot(epochs, loss_diff, color=self.colors['smoothed'], 
                    linewidth=2, label='Validation - Training')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Highlight overfitting regions
            overfitting_mask = loss_diff > 0.1  # Threshold for overfitting
            if np.any(overfitting_mask):
                overfitting_epochs = np.array(epochs)[overfitting_mask]
                overfitting_values = loss_diff[overfitting_mask]
                ax2.scatter(overfitting_epochs, overfitting_values, 
                           color='red', alpha=0.6, s=30, 
                           label='Potential Overfitting')
            
            ax2.set_xlabel('Epoch', fontweight='bold')
            ax2.set_ylabel('Loss Difference', fontweight='bold')
            ax2.set_title('Overfitting Analysis', fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#fafafa')
            
            plt.tight_layout()
            
            return fig
    
    def _find_convergence_point(self, 
                               losses: List[float], 
                               threshold: float, 
                               patience: int) -> Optional[int]:
        """Find the epoch where training converged based on improvement threshold."""
        if len(losses) < patience + 1:
            return None
        
        for i in range(patience, len(losses)):
            recent_losses = losses[i-patience:i]
            current_loss = losses[i]
            
            # Check if improvement is below threshold for 'patience' epochs
            improvements = [abs(recent_losses[j] - recent_losses[j+1]) 
                          for j in range(len(recent_losses)-1)]
            
            if all(imp < threshold for imp in improvements):
                return i + 1  # Convert to 1-based indexing
        
        return None
    
    def _smooth_curve(self, values: List[float], window: int) -> np.ndarray:
        """Apply moving average smoothing to a curve."""
        if len(values) < window:
            return np.array(values)
        
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start_idx:end_idx]))
        
        return np.array(smoothed)
    
    def save_figure(self, 
                   fig: plt.Figure, 
                   filename: str, 
                   output_dir: str = "images",
                   formats: List[str] = ['png', 'pdf']) -> List[str]:
        """
        Save figure in multiple formats with publication-quality settings.
        
        Args:
            fig: matplotlib Figure object to save
            filename: Base filename (without extension)
            output_dir: Directory to save files
            formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
            
        Returns:
            List of saved file paths
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for fmt in formats:
            filepath = os.path.join(output_dir, f"{filename}.{fmt}")
            
            # Format-specific settings
            save_kwargs = {
                'bbox_inches': 'tight',
                'pad_inches': 0.1,
                'facecolor': 'white',
                'edgecolor': 'none'
            }
            
            if fmt == 'png':
                save_kwargs['dpi'] = 300
            elif fmt == 'pdf':
                save_kwargs['dpi'] = 300
                save_kwargs['backend'] = 'pdf'
            elif fmt == 'svg':
                save_kwargs['format'] = 'svg'
            elif fmt == 'eps':
                save_kwargs['format'] = 'eps'
            
            fig.savefig(filepath, **save_kwargs)
            saved_paths.append(filepath)
        
        return saved_paths


class ComparisonPlotter:
    """
    Creates visualizations for comparing model performance across experiments.
    
    This class handles performance comparison charts, heatmaps for hyperparameter
    analysis, and statistical significance testing visualizations.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the ComparisonPlotter.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Import scipy for statistical tests (optional dependency)
        try:
            from scipy import stats
            self.stats = stats
            self.has_scipy = True
        except ImportError:
            self.has_scipy = False
            print("Warning: scipy not available. Statistical significance tests will be disabled.")
    
    def plot_performance_comparison(self, 
                                  results_dict: Dict[str, Dict[str, float]],
                                  metrics: List[str] = ['train_loss', 'val_loss'],
                                  title: str = "Model Performance Comparison") -> plt.Figure:
        """
        Create bar charts comparing model performance across different metrics.
        
        Args:
            results_dict: Dictionary with model names as keys and metric dictionaries as values
            metrics: List of metrics to compare
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            n_metrics = len(metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
            
            if n_metrics == 1:
                axes = [axes]
            
            model_names = list(results_dict.keys())
            
            for i, metric in enumerate(metrics):
                values = [results_dict[model][metric] for model in model_names]
                
                bars = axes[i].bar(model_names, values, 
                                  color=sns.color_palette("husl", len(model_names)),
                                  alpha=0.8,
                                  edgecolor='black',
                                  linewidth=1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.4f}',
                               ha='center', va='bottom', fontweight='bold')
                
                axes[i].set_title(f'{metric.replace("_", " ").title()}', 
                                 fontweight='bold')
                axes[i].set_ylabel('Value', fontweight='bold')
                axes[i].grid(True, alpha=0.3, axis='y')
                axes[i].set_facecolor('#fafafa')
                
                # Rotate x-axis labels if they're long
                if max(len(name) for name in model_names) > 10:
                    axes[i].tick_params(axis='x', rotation=45)
            
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            return fig
    
    def plot_hyperparameter_heatmap(self,
                                   param_grid: Dict[str, List],
                                   performance_matrix: np.ndarray,
                                   param_names: List[str],
                                   metric_name: str = "Validation Loss",
                                   title: str = "Hyperparameter Analysis") -> plt.Figure:
        """
        Create a heatmap for hyperparameter analysis.
        
        Args:
            param_grid: Dictionary of parameter names and their tested values
            performance_matrix: 2D array of performance values
            param_names: Names of the two parameters being analyzed
            metric_name: Name of the performance metric
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create heatmap
            im = ax.imshow(performance_matrix, cmap='viridis', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(param_grid[param_names[0]])))
            ax.set_yticks(range(len(param_grid[param_names[1]])))
            ax.set_xticklabels(param_grid[param_names[0]])
            ax.set_yticklabels(param_grid[param_names[1]])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(metric_name, fontweight='bold')
            
            # Add text annotations
            for i in range(len(param_grid[param_names[1]])):
                for j in range(len(param_grid[param_names[0]])):
                    text = ax.text(j, i, f'{performance_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="white", 
                                 fontweight='bold')
            
            ax.set_xlabel(param_names[0], fontweight='bold')
            ax.set_ylabel(param_names[1], fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            return fig
    
    def plot_performance_comparison_with_ci(self,
                                          results_data: Dict[str, List[Dict[str, float]]],
                                          metrics: List[str] = ['train_loss', 'val_loss'],
                                          confidence_level: float = 0.95,
                                          title: str = "Model Performance Comparison with Confidence Intervals") -> plt.Figure:
        """
        Create performance comparison with confidence intervals and statistical significance.
        
        Args:
            results_data: Dictionary with model names as keys and list of result dictionaries as values
            metrics: List of metrics to compare
            confidence_level: Confidence level for intervals
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            n_metrics = len(metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 8))
            
            if n_metrics == 1:
                axes = [axes]
            
            model_names = list(results_data.keys())
            
            for i, metric in enumerate(metrics):
                # Extract metric values for each model
                model_values = {}
                for model in model_names:
                    values = [result[metric] for result in results_data[model]]
                    model_values[model] = values
                
                # Calculate statistics
                means = []
                stds = []
                cis = []
                
                for model in model_names:
                    values = np.array(model_values[model])
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                    
                    means.append(mean_val)
                    stds.append(std_val)
                    
                    # Calculate confidence interval
                    if len(values) > 1 and self.has_scipy:
                        ci = self.stats.t.interval(
                            confidence_level, len(values)-1, 
                            loc=mean_val, scale=std_val/np.sqrt(len(values))
                        )
                        cis.append((ci[1] - mean_val, mean_val - ci[0]))  # (upper_err, lower_err)
                    else:
                        cis.append((std_val, std_val))
                
                # Create bar plot with error bars
                x_pos = np.arange(len(model_names))
                colors = sns.color_palette("husl", len(model_names))
                
                bars = axes[i].bar(x_pos, means, 
                                  color=colors, alpha=0.8,
                                  edgecolor='black', linewidth=1,
                                  yerr=list(zip(*cis)) if cis else None,
                                  capsize=5)
                
                # Add value labels on bars
                for j, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + cis[j][0] + 0.01,
                               f'{mean_val:.4f}±{std_val:.4f}',
                               ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                # Perform statistical significance tests
                if self.has_scipy and len(model_names) > 1:
                    self._add_significance_annotations(axes[i], model_values, x_pos, means, cis)
                
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(model_names)
                axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                axes[i].set_ylabel('Value', fontweight='bold')
                axes[i].grid(True, alpha=0.3, axis='y')
                axes[i].set_facecolor('#fafafa')
                
                # Rotate x-axis labels if they're long
                if max(len(name) for name in model_names) > 10:
                    axes[i].tick_params(axis='x', rotation=45)
            
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            return fig
    
    def plot_scatter_comparison(self,
                              results_data: Dict[str, List[Dict[str, float]]],
                              x_metric: str = 'train_loss',
                              y_metric: str = 'val_loss',
                              title: str = "Model Performance Scatter Plot") -> plt.Figure:
        """
        Create scatter plot comparing two metrics across models.
        
        Args:
            results_data: Dictionary with model names and result lists
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            
            colors = sns.color_palette("husl", len(results_data))
            
            for i, (model_name, results) in enumerate(results_data.items()):
                x_values = [result[x_metric] for result in results]
                y_values = [result[y_metric] for result in results]
                
                ax.scatter(x_values, y_values, 
                          color=colors[i], alpha=0.7, s=100,
                          label=model_name, edgecolors='black', linewidth=1)
                
                # Add mean point with different marker
                mean_x = np.mean(x_values)
                mean_y = np.mean(y_values)
                ax.scatter(mean_x, mean_y, 
                          color=colors[i], s=200, marker='D',
                          edgecolors='black', linewidth=2, alpha=0.9)
                
                # Add confidence ellipse if we have enough points
                if len(x_values) > 2:
                    self._add_confidence_ellipse(ax, x_values, y_values, colors[i])
            
            # Add diagonal line for reference (perfect correlation)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            min_val = max(min(xlim), min(ylim))
            max_val = min(max(xlim), max(ylim))
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'k--', alpha=0.5, linewidth=1, label='Perfect Correlation')
            
            ax.set_xlabel(x_metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel(y_metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_title(title, fontweight='bold', pad=20)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#fafafa')
            
            plt.tight_layout()
            
            return fig
    
    def plot_ranking_analysis(self,
                            results_data: Dict[str, List[Dict[str, float]]],
                            metrics: List[str] = ['train_loss', 'val_loss'],
                            title: str = "Model Ranking Analysis") -> plt.Figure:
        """
        Create ranking analysis showing model performance across multiple metrics.
        
        Args:
            results_data: Dictionary with model names and result lists
            metrics: List of metrics to analyze
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        with plt.style.context(self.style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            model_names = list(results_data.keys())
            n_models = len(model_names)
            
            # Calculate average rankings
            rankings = {model: [] for model in model_names}
            
            for metric in metrics:
                # Get mean values for each model
                mean_values = {}
                for model in model_names:
                    values = [result[metric] for result in results_data[model]]
                    mean_values[model] = np.mean(values)
                
                # Rank models (lower is better for loss metrics)
                sorted_models = sorted(mean_values.items(), key=lambda x: x[1])
                for rank, (model, _) in enumerate(sorted_models):
                    rankings[model].append(rank + 1)
            
            # Calculate average ranking
            avg_rankings = {model: np.mean(ranks) for model, ranks in rankings.items()}
            
            # Plot 1: Ranking heatmap
            ranking_matrix = np.array([rankings[model] for model in model_names])
            
            im = ax1.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto')
            ax1.set_xticks(range(len(metrics)))
            ax1.set_yticks(range(len(model_names)))
            ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax1.set_yticklabels(model_names)
            
            # Add text annotations
            for i in range(len(model_names)):
                for j in range(len(metrics)):
                    text = ax1.text(j, i, f'{ranking_matrix[i, j]:.0f}',
                                   ha="center", va="center", color="black", 
                                   fontweight='bold')
            
            ax1.set_title('Ranking by Metric', fontweight='bold')
            
            # Add colorbar
            cbar1 = plt.colorbar(im, ax=ax1)
            cbar1.set_label('Rank (1=Best)', fontweight='bold')
            
            # Plot 2: Average ranking bar chart
            sorted_avg = sorted(avg_rankings.items(), key=lambda x: x[1])
            models_sorted = [item[0] for item in sorted_avg]
            ranks_sorted = [item[1] for item in sorted_avg]
            
            colors = sns.color_palette("RdYlGn_r", len(models_sorted))[::-1]  # Reverse for better colors
            bars = ax2.bar(range(len(models_sorted)), ranks_sorted, 
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar, rank in zip(bars, ranks_sorted):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rank:.2f}',
                        ha='center', va='bottom', fontweight='bold')
            
            ax2.set_xticks(range(len(models_sorted)))
            ax2.set_xticklabels(models_sorted)
            ax2.set_ylabel('Average Rank', fontweight='bold')
            ax2.set_title('Overall Model Ranking', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_facecolor('#fafafa')
            
            # Rotate labels if needed
            if max(len(name) for name in models_sorted) > 10:
                ax2.tick_params(axis='x', rotation=45)
            
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            return fig
    
    def _add_significance_annotations(self, ax, model_values, x_pos, means, cis):
        """Add statistical significance annotations to bar plot."""
        if not self.has_scipy:
            return
        
        model_names = list(model_values.keys())
        n_models = len(model_names)
        
        # Perform pairwise t-tests
        y_max = max(means[i] + cis[i][0] for i in range(n_models))
        y_offset = y_max * 0.05
        
        annotation_height = y_max + y_offset
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                values1 = model_values[model_names[i]]
                values2 = model_values[model_names[j]]
                
                if len(values1) > 1 and len(values2) > 1:
                    # Perform t-test
                    t_stat, p_value = self.stats.ttest_ind(values1, values2)
                    
                    # Add significance annotation
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    if sig_text != 'ns':
                        # Draw line and add text
                        x1, x2 = x_pos[i], x_pos[j]
                        ax.plot([x1, x2], [annotation_height, annotation_height], 
                               'k-', linewidth=1)
                        ax.text((x1 + x2) / 2, annotation_height + y_offset * 0.2, 
                               sig_text, ha='center', va='bottom', fontweight='bold')
                        
                        annotation_height += y_offset * 0.8
    
    def _add_confidence_ellipse(self, ax, x_data, y_data, color, confidence=0.95):
        """Add confidence ellipse to scatter plot."""
        if not self.has_scipy or len(x_data) < 3:
            return
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        # Calculate covariance matrix
        cov = np.cov(x_data, y_data)
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Calculate ellipse parameters
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        # Chi-square value for confidence level
        chi2_val = self.stats.chi2.ppf(confidence, df=2)
        
        # Scale eigenvalues
        width = 2 * np.sqrt(chi2_val * eigenvals[0])
        height = 2 * np.sqrt(chi2_val * eigenvals[1])
        
        # Create ellipse
        ellipse = patches.Ellipse(
            (np.mean(x_data), np.mean(y_data)),
            width, height, angle=angle,
            facecolor=color, alpha=0.2, edgecolor=color, linewidth=2
        )
        
        ax.add_patch(ellipse)
    
    def save_figure(self, 
                   fig: plt.Figure, 
                   filename: str, 
                   output_dir: str = "images",
                   formats: List[str] = ['png', 'pdf']) -> List[str]:
        """
        Save figure in multiple formats with publication-quality settings.
        
        Args:
            fig: matplotlib Figure object to save
            filename: Base filename (without extension)
            output_dir: Directory to save files
            formats: List of formats to save ('png', 'pdf', 'svg', 'eps')
            
        Returns:
            List of saved file paths
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for fmt in formats:
            filepath = os.path.join(output_dir, f"{filename}.{fmt}")
            
            # Format-specific settings
            save_kwargs = {
                'bbox_inches': 'tight',
                'pad_inches': 0.1,
                'facecolor': 'white',
                'edgecolor': 'none'
            }
            
            if fmt == 'png':
                save_kwargs['dpi'] = 300
            elif fmt == 'pdf':
                save_kwargs['dpi'] = 300
                save_kwargs['backend'] = 'pdf'
            elif fmt == 'svg':
                save_kwargs['format'] = 'svg'
            elif fmt == 'eps':
                save_kwargs['format'] = 'eps'
            
            fig.savefig(filepath, **save_kwargs)
            saved_paths.append(filepath)
        
        return saved_paths