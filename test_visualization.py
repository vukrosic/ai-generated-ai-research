"""
Tests for visualization module.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import os
import tempfile
import shutil

from src.visualization.plots import CurvePlotter, LossPlotter, ComparisonPlotter


class TestCurvePlotter:
    """Test cases for CurvePlotter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = CurvePlotter()
        self.temp_dir = tempfile.mkdtemp()
        
        # Generate sample data
        self.x_data = np.linspace(0, 10, 20)
        self.y_data = 2 * self.x_data + 1 + np.random.normal(0, 0.5, len(self.x_data))
        self.x_pred = np.linspace(0, 10, 100)
        self.y_pred = 2 * self.x_pred + 1
        
        # Generate confidence intervals
        self.confidence_intervals = (
            self.y_pred - 0.5,
            self.y_pred + 0.5
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_curve_plotter_initialization(self):
        """Test CurvePlotter initialization."""
        plotter = CurvePlotter()
        assert plotter.style == 'seaborn-v0_8'
        assert plotter.figsize == (10, 6)
        assert 'data' in plotter.colors
        assert 'prediction' in plotter.colors
    
    def test_plot_curve_fit_basic(self):
        """Test basic curve fitting plot."""
        fig = self.plotter.plot_curve_fit(
            self.x_data, self.y_data,
            self.x_pred, self.y_pred
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        assert ax.get_title() == 'Curve Fitting Results'
    
    def test_plot_curve_fit_with_confidence_intervals(self):
        """Test curve fitting plot with confidence intervals."""
        fig = self.plotter.plot_curve_fit(
            self.x_data, self.y_data,
            self.x_pred, self.y_pred,
            confidence_intervals=self.confidence_intervals
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        
        # Check that we have the expected number of plot elements
        # (scatter, line, fill_between)
        assert len(ax.collections) >= 1  # scatter plot and fill_between
        assert len(ax.lines) >= 1  # prediction line
    
    def test_plot_curve_fit_with_true_curve(self):
        """Test curve fitting plot with true curve."""
        true_curve = 2 * self.x_pred + 1
        
        fig = self.plotter.plot_curve_fit(
            self.x_data, self.y_data,
            self.x_pred, self.y_pred,
            true_curve=true_curve
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) >= 2  # prediction line and true curve line
    
    def test_plot_residuals(self):
        """Test residual plot creation."""
        residuals = self.y_data[:len(self.x_data)] - (2 * self.x_data + 1)
        
        fig = self.plotter.plot_residuals(self.x_data, residuals)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # residuals plot and histogram
    
    def test_save_figure(self):
        """Test figure saving functionality."""
        fig = self.plotter.plot_curve_fit(
            self.x_data, self.y_data,
            self.x_pred, self.y_pred
        )
        
        saved_paths = self.plotter.save_figure(
            fig, "test_plot", 
            output_dir=self.temp_dir,
            formats=['png']
        )
        
        assert len(saved_paths) == 1
        assert os.path.exists(saved_paths[0])
        assert saved_paths[0].endswith('.png')
    
    def test_create_subplot_grid(self):
        """Test subplot grid creation."""
        fig, axes = self.plotter.create_subplot_grid(2, 2)
        
        assert isinstance(fig, plt.Figure)
        assert axes.shape == (2, 2)
        
        # Test single subplot
        fig_single, axes_single = self.plotter.create_subplot_grid(1, 1)
        assert len(axes_single) == 1


class TestLossPlotter:
    """Test cases for LossPlotter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = LossPlotter()
        self.train_losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23]
        self.val_losses = [1.1, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.36, 0.35]
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close('all')
    
    def test_loss_plotter_initialization(self):
        """Test LossPlotter initialization."""
        plotter = LossPlotter()
        assert plotter.style == 'seaborn-v0_8'
        assert plotter.figsize == (12, 6)
        assert 'train' in plotter.colors
        assert 'validation' in plotter.colors
    
    def test_plot_loss_curves_basic(self):
        """Test basic loss curve plotting."""
        fig = self.plotter.plot_loss_curves(
            self.train_losses, self.val_losses
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Epoch'
        assert ax.get_ylabel() == 'Loss'
        assert len(ax.lines) >= 2  # train and validation lines
    
    def test_plot_loss_curves_with_smoothing(self):
        """Test loss curve plotting with smoothing."""
        fig = self.plotter.plot_loss_curves(
            self.train_losses, self.val_losses,
            smooth=True, smooth_window=3
        )
        
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Should have more lines when smoothing is enabled
        assert len(ax.lines) >= 2


class TestComparisonPlotter:
    """Test cases for ComparisonPlotter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = ComparisonPlotter()
        self.results_dict = {
            'Linear': {'train_loss': 0.5, 'val_loss': 0.6},
            'Shallow': {'train_loss': 0.3, 'val_loss': 0.4},
            'Deep': {'train_loss': 0.2, 'val_loss': 0.35}
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close('all')
    
    def test_comparison_plotter_initialization(self):
        """Test ComparisonPlotter initialization."""
        plotter = ComparisonPlotter()
        assert plotter.style == 'seaborn-v0_8'
        assert plotter.figsize == (12, 8)
    
    def test_plot_performance_comparison(self):
        """Test performance comparison plotting."""
        fig = self.plotter.plot_performance_comparison(
            self.results_dict,
            metrics=['train_loss', 'val_loss']
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # one subplot per metric
    
    def test_plot_hyperparameter_heatmap(self):
        """Test hyperparameter heatmap plotting."""
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64]
        }
        performance_matrix = np.random.rand(3, 3)
        
        fig = self.plotter.plot_hyperparameter_heatmap(
            param_grid, performance_matrix,
            param_names=['learning_rate', 'batch_size']
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1


if __name__ == "__main__":
    pytest.main([__file__])