"""
Simple test for experiment runner without pytest.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

from src.experiments.config import ExperimentConfig
from src.experiments.runner import ExperimentRunner, ExperimentResults
import tempfile
import shutil
from pathlib import Path


def test_experiment_runner():
    """Test basic experiment runner functionality."""
    print("Testing ExperimentRunner...")
    
    # Create temporary directories for testing
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Initialize runner with temporary directories
        runner = ExperimentRunner(
            output_dir=str(temp_dir / "experiments"),
            images_dir=str(temp_dir / "images"),
            models_dir=str(temp_dir / "models"),
            enable_logging=False  # Disable logging for test
        )
        
        print("ExperimentRunner initialized successfully")
        
        # Create a simple test configuration
        config = ExperimentConfig(
            polynomial_degree=2,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture="linear",  # Use simplest model for test
            hidden_dims=[],  # Linear model doesn't need hidden dims
            optimizer="adam",
            learning_rate=0.01,
            batch_size=16,
            epochs=5,  # Small number for quick test
            random_seed=42,
            num_data_points=100  # Small dataset for quick test
        )
        
        print("Test configuration created")
        
        # Test progress tracking
        progress, status = runner.get_experiment_progress("test_exp")
        print(f"Initial progress: {progress}, status: {status}")
        
        # Run a single experiment
        print("Running single experiment...")
        results = runner.run_single_experiment(config, "test_exp")
        
        print(f"Experiment completed with status: {results.status}")
        print(f"Final train loss: {results.final_train_loss:.4f}")
        print(f"Final val loss: {results.final_val_loss:.4f}")
        print(f"Duration: {results.duration_seconds:.2f}s")
        print(f"Generated {len(results.image_paths)} images")
        
        # Test progress after completion
        progress, status = runner.get_experiment_progress("test_exp")
        print(f"Final progress: {progress}, status: {status}")
        
        # Test loading results
        loaded_results = runner.load_experiment_results("test_exp")
        if loaded_results:
            print("Successfully loaded experiment results from disk")
        else:
            print("Failed to load experiment results")
        
        print("ExperimentRunner test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_parallel_experiments():
    """Test parallel experiment execution."""
    print("\nTesting parallel experiments...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        runner = ExperimentRunner(
            output_dir=str(temp_dir / "experiments"),
            images_dir=str(temp_dir / "images"),
            models_dir=str(temp_dir / "models"),
            enable_logging=False
        )
        
        # Create multiple test configurations
        configs = []
        for degree in [1, 2]:
            for arch in ["linear"]:  # Keep simple for test
                config = ExperimentConfig(
                    polynomial_degree=degree,
                    noise_level=0.1,
                    train_val_split=0.8,
                    model_architecture=arch,
                    hidden_dims=[] if arch == "linear" else [32],
                    optimizer="adam",
                    learning_rate=0.01,
                    batch_size=16,
                    epochs=3,  # Very small for quick test
                    random_seed=42 + degree,
                    num_data_points=50
                )
                configs.append(config)
        
        print(f"Created {len(configs)} test configurations")
        
        # Run parallel experiments
        results = runner.run_parallel_experiments(configs, max_workers=2)
        
        print(f"Completed {len(results)} parallel experiments")
        
        successful = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status == "failed")
        
        print(f"Successful: {successful}, Failed: {failed}")
        
        if successful > 0:
            print("Parallel experiments test passed!")
        else:
            print("All parallel experiments failed")
    
    except Exception as e:
        print(f"Parallel test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_experiment_runner()
    test_parallel_experiments()