"""
Simple test for visualization module without pytest dependency.
"""

import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import shutil

from src.visualization.plots import CurvePlotter, LossPlotter, ComparisonPlotter


def test_curve_plotter():
    """Test CurvePlotter functionality."""
    print("Testing CurvePlotter...")
    
    # Initialize plotter
    plotter = CurvePlotter()
    print("✓ CurvePlotter initialized successfully")
    
    # Generate sample data
    x_data = np.linspace(0, 10, 20)
    y_data = 2 * x_data + 1 + np.random.normal(0, 0.5, len(x_data))
    x_pred = np.linspace(0, 10, 100)
    y_pred = 2 * x_pred + 1
    
    # Test basic curve fitting plot
    fig = plotter.plot_curve_fit(x_data, y_data, x_pred, y_pred)
    assert isinstance(fig, plt.Figure)
    print("✓ Basic curve fitting plot created")
    
    # Test with confidence intervals
    confidence_intervals = (y_pred - 0.5, y_pred + 0.5)
    fig_ci = plotter.plot_curve_fit(
        x_data, y_data, x_pred, y_pred,
        confidence_intervals=confidence_intervals
    )
    assert isinstance(fig_ci, plt.Figure)
    print("✓ Curve fitting plot with confidence intervals created")
    
    # Test with true curve
    true_curve = 2 * x_pred + 1
    fig_true = plotter.plot_curve_fit(
        x_data, y_data, x_pred, y_pred,
        true_curve=true_curve
    )
    assert isinstance(fig_true, plt.Figure)
    print("✓ Curve fitting plot with true curve created")
    
    # Test residuals plot
    residuals = y_data[:len(x_data)] - (2 * x_data + 1)
    fig_res = plotter.plot_residuals(x_data, residuals)
    assert isinstance(fig_res, plt.Figure)
    print("✓ Residuals plot created")
    
    # Test saving functionality
    temp_dir = tempfile.mkdtemp()
    try:
        saved_paths = plotter.save_figure(
            fig, "test_plot", 
            output_dir=temp_dir,
            formats=['png']
        )
        assert len(saved_paths) == 1
        assert os.path.exists(saved_paths[0])
        print("✓ Figure saving works correctly")
    finally:
        shutil.rmtree(temp_dir)
    
    # Test subplot grid
    fig_grid, axes = plotter.create_subplot_grid(2, 2)
    assert isinstance(fig_grid, plt.Figure)
    assert axes.shape == (2, 2)
    print("✓ Subplot grid creation works")
    
    plt.close('all')
    print("CurvePlotter tests completed successfully!\n")


def test_loss_plotter():
    """Test LossPlotter functionality."""
    print("Testing LossPlotter...")
    
    plotter = LossPlotter()
    print("✓ LossPlotter initialized successfully")
    
    # Generate sample loss data
    train_losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23]
    val_losses = [1.1, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.36, 0.35]
    
    # Test basic loss curve plotting
    fig = plotter.plot_loss_curves(train_losses, val_losses)
    assert isinstance(fig, plt.Figure)
    print("✓ Basic loss curves created")
    
    # Test with smoothing
    fig_smooth = plotter.plot_loss_curves(
        train_losses, val_losses, smooth=True, smooth_window=3
    )
    assert isinstance(fig_smooth, plt.Figure)
    print("✓ Smoothed loss curves created")
    
    # Test multi-experiment comparison
    experiments_data = {
        'Experiment 1': {'train_losses': train_losses, 'val_losses': val_losses},
        'Experiment 2': {'train_losses': [x * 0.9 for x in train_losses], 
                        'val_losses': [x * 0.9 for x in val_losses]},
        'Experiment 3': {'train_losses': [x * 1.1 for x in train_losses], 
                        'val_losses': [x * 1.1 for x in val_losses]}
    }
    
    fig_multi = plotter.plot_multi_experiment_comparison(experiments_data)
    assert isinstance(fig_multi, plt.Figure)
    print("✓ Multi-experiment comparison plot created")
    
    # Test loss statistics
    loss_data = [train_losses, [x * 0.9 for x in train_losses], [x * 1.1 for x in train_losses]]
    fig_stats = plotter.plot_loss_statistics(loss_data)
    assert isinstance(fig_stats, plt.Figure)
    print("✓ Loss statistics plot created")
    
    # Test convergence analysis
    fig_conv = plotter.plot_convergence_analysis(train_losses, val_losses)
    assert isinstance(fig_conv, plt.Figure)
    print("✓ Convergence analysis plot created")
    
    plt.close('all')
    print("LossPlotter tests completed successfully!\n")


def test_comparison_plotter():
    """Test ComparisonPlotter functionality."""
    print("Testing ComparisonPlotter...")
    
    plotter = ComparisonPlotter()
    print("✓ ComparisonPlotter initialized successfully")
    
    # Sample results data for basic comparison
    results_dict = {
        'Linear': {'train_loss': 0.5, 'val_loss': 0.6},
        'Shallow': {'train_loss': 0.3, 'val_loss': 0.4},
        'Deep': {'train_loss': 0.2, 'val_loss': 0.35}
    }
    
    # Test performance comparison
    fig = plotter.plot_performance_comparison(
        results_dict, metrics=['train_loss', 'val_loss']
    )
    assert isinstance(fig, plt.Figure)
    print("✓ Performance comparison plot created")
    
    # Test hyperparameter heatmap
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64]
    }
    performance_matrix = np.random.rand(3, 3)
    
    fig_heatmap = plotter.plot_hyperparameter_heatmap(
        param_grid, performance_matrix,
        param_names=['learning_rate', 'batch_size']
    )
    assert isinstance(fig_heatmap, plt.Figure)
    print("✓ Hyperparameter heatmap created")
    
    # Sample results data with multiple runs for advanced features
    results_data_multi = {
        'Linear': [
            {'train_loss': 0.5, 'val_loss': 0.6},
            {'train_loss': 0.52, 'val_loss': 0.58},
            {'train_loss': 0.48, 'val_loss': 0.62}
        ],
        'Shallow': [
            {'train_loss': 0.3, 'val_loss': 0.4},
            {'train_loss': 0.32, 'val_loss': 0.38},
            {'train_loss': 0.28, 'val_loss': 0.42}
        ],
        'Deep': [
            {'train_loss': 0.2, 'val_loss': 0.35},
            {'train_loss': 0.22, 'val_loss': 0.33},
            {'train_loss': 0.18, 'val_loss': 0.37}
        ]
    }
    
    # Test performance comparison with confidence intervals
    fig_ci = plotter.plot_performance_comparison_with_ci(
        results_data_multi, metrics=['train_loss', 'val_loss']
    )
    assert isinstance(fig_ci, plt.Figure)
    print("✓ Performance comparison with confidence intervals created")
    
    # Test scatter comparison
    fig_scatter = plotter.plot_scatter_comparison(
        results_data_multi, x_metric='train_loss', y_metric='val_loss'
    )
    assert isinstance(fig_scatter, plt.Figure)
    print("✓ Scatter comparison plot created")
    
    # Test ranking analysis
    fig_ranking = plotter.plot_ranking_analysis(
        results_data_multi, metrics=['train_loss', 'val_loss']
    )
    assert isinstance(fig_ranking, plt.Figure)
    print("✓ Ranking analysis plot created")
    
    plt.close('all')
    print("ComparisonPlotter tests completed successfully!\n")


if __name__ == "__main__":
    print("Running visualization tests...\n")
    
    try:
        test_curve_plotter()
        test_loss_plotter()
        test_comparison_plotter()
        print("All visualization tests passed! ✅")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()