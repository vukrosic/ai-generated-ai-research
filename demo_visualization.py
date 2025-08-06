"""
Demonstration of the visualization system for AI curve fitting research.

This script shows how to use the CurvePlotter, LossPlotter, and ComparisonPlotter
classes to create publication-quality visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.visualization.plots import CurvePlotter, LossPlotter, ComparisonPlotter


def demo_curve_plotter():
    """Demonstrate CurvePlotter functionality."""
    print("=== CurvePlotter Demo ===")
    
    plotter = CurvePlotter()
    
    # Generate synthetic polynomial data
    np.random.seed(42)
    x_data = np.linspace(0, 10, 25)
    true_coeffs = [2, -0.5, 0.1]  # quadratic: 2 - 0.5x + 0.1x^2
    y_true = true_coeffs[0] + true_coeffs[1] * x_data + true_coeffs[2] * x_data**2
    y_data = y_true + np.random.normal(0, 0.8, len(x_data))
    
    # Generate predictions (denser grid)
    x_pred = np.linspace(0, 10, 100)
    y_pred = true_coeffs[0] + true_coeffs[1] * x_pred + true_coeffs[2] * x_pred**2
    y_true_pred = y_pred.copy()
    
    # Add some model error to predictions
    y_pred += np.random.normal(0, 0.2, len(x_pred))
    
    # Generate confidence intervals
    confidence_intervals = (y_pred - 0.5, y_pred + 0.5)
    
    # Create curve fitting plot
    fig = plotter.plot_curve_fit(
        x_data, y_data, x_pred, y_pred,
        confidence_intervals=confidence_intervals,
        true_curve=y_true_pred,
        title="Quadratic Curve Fitting Example",
        xlabel="Input Variable (x)",
        ylabel="Output Variable (y)"
    )
    
    # Save the plot
    saved_paths = plotter.save_figure(fig, "curve_fitting_demo", formats=['png', 'pdf'])
    print(f"Curve fitting plot saved to: {saved_paths}")
    
    # Create residuals plot
    residuals = y_data - (true_coeffs[0] + true_coeffs[1] * x_data + true_coeffs[2] * x_data**2)
    fig_res = plotter.plot_residuals(x_data, residuals, "Residual Analysis")
    plotter.save_figure(fig_res, "residuals_demo", formats=['png'])
    
    print("CurvePlotter demo completed!\n")


def demo_loss_plotter():
    """Demonstrate LossPlotter functionality."""
    print("=== LossPlotter Demo ===")
    
    plotter = LossPlotter()
    
    # Generate synthetic training data
    np.random.seed(42)
    epochs = 50
    
    # Simulate training curves with different characteristics
    train_losses_1 = [1.0 * np.exp(-0.1 * i) + 0.1 + np.random.normal(0, 0.02) for i in range(epochs)]
    val_losses_1 = [1.1 * np.exp(-0.08 * i) + 0.15 + np.random.normal(0, 0.03) for i in range(epochs)]
    
    train_losses_2 = [0.8 * np.exp(-0.12 * i) + 0.08 + np.random.normal(0, 0.015) for i in range(epochs)]
    val_losses_2 = [0.9 * np.exp(-0.1 * i) + 0.12 + np.random.normal(0, 0.025) for i in range(epochs)]
    
    train_losses_3 = [1.2 * np.exp(-0.09 * i) + 0.12 + np.random.normal(0, 0.025) for i in range(epochs)]
    val_losses_3 = [1.3 * np.exp(-0.07 * i) + 0.18 + np.random.normal(0, 0.035) for i in range(epochs)]
    
    # Basic loss curves
    fig1 = plotter.plot_loss_curves(train_losses_1, val_losses_1, 
                                   title="Training Progress - Model 1")
    plotter.save_figure(fig1, "loss_curves_demo", formats=['png'])
    
    # Multi-experiment comparison
    experiments_data = {
        'Model A': {'train_losses': train_losses_1, 'val_losses': val_losses_1},
        'Model B': {'train_losses': train_losses_2, 'val_losses': val_losses_2},
        'Model C': {'train_losses': train_losses_3, 'val_losses': val_losses_3}
    }
    
    fig2 = plotter.plot_multi_experiment_comparison(experiments_data)
    plotter.save_figure(fig2, "multi_experiment_demo", formats=['png'])
    
    # Loss statistics across multiple runs
    loss_data = [train_losses_1, train_losses_2, train_losses_3]
    fig3 = plotter.plot_loss_statistics(loss_data, "Training Loss", 
                                       "Training Loss Statistics Across Models")
    plotter.save_figure(fig3, "loss_statistics_demo", formats=['png'])
    
    # Convergence analysis
    fig4 = plotter.plot_convergence_analysis(train_losses_1, val_losses_1)
    plotter.save_figure(fig4, "convergence_analysis_demo", formats=['png'])
    
    print("LossPlotter demo completed!\n")


def demo_comparison_plotter():
    """Demonstrate ComparisonPlotter functionality."""
    print("=== ComparisonPlotter Demo ===")
    
    plotter = ComparisonPlotter()
    
    # Generate synthetic results data
    np.random.seed(42)
    
    # Multiple runs for each model
    results_data = {
        'Linear Model': [
            {'train_loss': 0.45 + np.random.normal(0, 0.02), 'val_loss': 0.52 + np.random.normal(0, 0.03), 'accuracy': 0.78 + np.random.normal(0, 0.02)},
            {'train_loss': 0.47 + np.random.normal(0, 0.02), 'val_loss': 0.54 + np.random.normal(0, 0.03), 'accuracy': 0.76 + np.random.normal(0, 0.02)},
            {'train_loss': 0.43 + np.random.normal(0, 0.02), 'val_loss': 0.50 + np.random.normal(0, 0.03), 'accuracy': 0.80 + np.random.normal(0, 0.02)},
            {'train_loss': 0.46 + np.random.normal(0, 0.02), 'val_loss': 0.53 + np.random.normal(0, 0.03), 'accuracy': 0.77 + np.random.normal(0, 0.02)},
            {'train_loss': 0.44 + np.random.normal(0, 0.02), 'val_loss': 0.51 + np.random.normal(0, 0.03), 'accuracy': 0.79 + np.random.normal(0, 0.02)}
        ],
        'Shallow Network': [
            {'train_loss': 0.28 + np.random.normal(0, 0.015), 'val_loss': 0.35 + np.random.normal(0, 0.025), 'accuracy': 0.85 + np.random.normal(0, 0.015)},
            {'train_loss': 0.30 + np.random.normal(0, 0.015), 'val_loss': 0.37 + np.random.normal(0, 0.025), 'accuracy': 0.83 + np.random.normal(0, 0.015)},
            {'train_loss': 0.26 + np.random.normal(0, 0.015), 'val_loss': 0.33 + np.random.normal(0, 0.025), 'accuracy': 0.87 + np.random.normal(0, 0.015)},
            {'train_loss': 0.29 + np.random.normal(0, 0.015), 'val_loss': 0.36 + np.random.normal(0, 0.025), 'accuracy': 0.84 + np.random.normal(0, 0.015)},
            {'train_loss': 0.27 + np.random.normal(0, 0.015), 'val_loss': 0.34 + np.random.normal(0, 0.025), 'accuracy': 0.86 + np.random.normal(0, 0.015)}
        ],
        'Deep Network': [
            {'train_loss': 0.18 + np.random.normal(0, 0.01), 'val_loss': 0.28 + np.random.normal(0, 0.02), 'accuracy': 0.91 + np.random.normal(0, 0.01)},
            {'train_loss': 0.20 + np.random.normal(0, 0.01), 'val_loss': 0.30 + np.random.normal(0, 0.02), 'accuracy': 0.89 + np.random.normal(0, 0.01)},
            {'train_loss': 0.16 + np.random.normal(0, 0.01), 'val_loss': 0.26 + np.random.normal(0, 0.02), 'accuracy': 0.93 + np.random.normal(0, 0.01)},
            {'train_loss': 0.19 + np.random.normal(0, 0.01), 'val_loss': 0.29 + np.random.normal(0, 0.02), 'accuracy': 0.90 + np.random.normal(0, 0.01)},
            {'train_loss': 0.17 + np.random.normal(0, 0.01), 'val_loss': 0.27 + np.random.normal(0, 0.02), 'accuracy': 0.92 + np.random.normal(0, 0.01)}
        ]
    }
    
    # Performance comparison with confidence intervals
    fig1 = plotter.plot_performance_comparison_with_ci(
        results_data, metrics=['train_loss', 'val_loss', 'accuracy']
    )
    plotter.save_figure(fig1, "performance_comparison_ci_demo", formats=['png'])
    
    # Scatter plot comparison
    fig2 = plotter.plot_scatter_comparison(
        results_data, x_metric='train_loss', y_metric='val_loss',
        title="Training vs Validation Loss Comparison"
    )
    plotter.save_figure(fig2, "scatter_comparison_demo", formats=['png'])
    
    # Ranking analysis
    fig3 = plotter.plot_ranking_analysis(
        results_data, metrics=['train_loss', 'val_loss', 'accuracy']
    )
    plotter.save_figure(fig3, "ranking_analysis_demo", formats=['png'])
    
    # Hyperparameter heatmap
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1, 0.5],
        'batch_size': [16, 32, 64, 128]
    }
    # Simulate performance matrix (lower is better)
    performance_matrix = np.array([
        [0.45, 0.38, 0.42, 0.48],
        [0.35, 0.28, 0.32, 0.39],
        [0.52, 0.41, 0.36, 0.44],
        [0.68, 0.55, 0.49, 0.51]
    ])
    
    fig4 = plotter.plot_hyperparameter_heatmap(
        param_grid, performance_matrix,
        param_names=['learning_rate', 'batch_size'],
        title="Hyperparameter Optimization Results"
    )
    plotter.save_figure(fig4, "hyperparameter_heatmap_demo", formats=['png'])
    
    print("ComparisonPlotter demo completed!\n")


if __name__ == "__main__":
    print("AI Curve Fitting Research - Visualization System Demo")
    print("=" * 55)
    
    # Create images directory
    import os
    os.makedirs("images", exist_ok=True)
    
    # Run demonstrations
    demo_curve_plotter()
    demo_loss_plotter()
    demo_comparison_plotter()
    
    print("All visualization demos completed!")
    print("Check the 'images' directory for generated plots.")
    
    # Close all figures to free memory
    plt.close('all')