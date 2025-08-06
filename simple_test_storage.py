"""
Simple test for experiment storage system.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

from src.experiments.config import ExperimentConfig
from src.experiments.runner import ExperimentRunner, ExperimentResults
from src.experiments.storage import ExperimentStorage, QueryFilter, StorageBackend
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta


def test_storage_backends():
    """Test different storage backends."""
    print("Testing storage backends...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create a test experiment result
        config = ExperimentConfig(
            polynomial_degree=2,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture="linear",
            hidden_dims=[],
            optimizer="adam",
            learning_rate=0.01,
            batch_size=16,
            epochs=5,
            random_seed=42,
            num_data_points=100
        )
        
        # Create mock experiment result
        result = ExperimentResults(
            config=config,
            experiment_id="test_storage_exp",
            timestamp=datetime.now(),
            duration_seconds=10.5,
            final_train_loss=0.5,
            final_val_loss=0.6,
            best_val_loss=0.55,
            training_time=8.0,
            convergence_epoch=3,
            model_size=100,
            status="completed"
        )
        
        # Test JSON backend
        print("Testing JSON backend...")
        json_storage = ExperimentStorage(
            storage_dir=str(temp_dir / "json_storage"),
            backend=StorageBackend.JSON
        )
        
        success = json_storage.store_experiment(result)
        print(f"JSON storage: {'Success' if success else 'Failed'}")
        
        loaded_result = json_storage.load_experiment("test_storage_exp")
        if loaded_result:
            print(f"JSON load: Success - {loaded_result.experiment_id}")
        else:
            print("JSON load: Failed")
        
        # Test SQLite backend
        print("Testing SQLite backend...")
        sqlite_storage = ExperimentStorage(
            storage_dir=str(temp_dir / "sqlite_storage"),
            backend=StorageBackend.SQLITE
        )
        
        success = sqlite_storage.store_experiment(result)
        print(f"SQLite storage: {'Success' if success else 'Failed'}")
        
        loaded_result = sqlite_storage.load_experiment("test_storage_exp")
        if loaded_result:
            print(f"SQLite load: Success - {loaded_result.experiment_id}")
        else:
            print("SQLite load: Failed")
        
        # Test Pickle backend
        print("Testing Pickle backend...")
        pickle_storage = ExperimentStorage(
            storage_dir=str(temp_dir / "pickle_storage"),
            backend=StorageBackend.PICKLE
        )
        
        success = pickle_storage.store_experiment(result)
        print(f"Pickle storage: {'Success' if success else 'Failed'}")
        
        loaded_result = pickle_storage.load_experiment("test_storage_exp")
        if loaded_result:
            print(f"Pickle load: Success - {loaded_result.experiment_id}")
        else:
            print("Pickle load: Failed")
        
        print("Storage backends test completed!")
        
    except Exception as e:
        print(f"Storage test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_querying_and_filtering():
    """Test experiment querying and filtering."""
    print("\nTesting querying and filtering...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        storage = ExperimentStorage(
            storage_dir=str(temp_dir / "query_storage"),
            backend=StorageBackend.SQLITE  # Use SQLite for advanced querying
        )
        
        # Create multiple test experiments
        base_time = datetime.now()
        
        test_experiments = [
            {
                "id": "exp_1",
                "degree": 1,
                "arch": "linear",
                "opt": "adam",
                "val_loss": 0.5,
                "time_offset": 0
            },
            {
                "id": "exp_2", 
                "degree": 2,
                "arch": "shallow",
                "opt": "sgd",
                "val_loss": 0.3,
                "time_offset": 1
            },
            {
                "id": "exp_3",
                "degree": 3,
                "arch": "deep",
                "opt": "adam",
                "val_loss": 0.7,
                "time_offset": 2
            },
            {
                "id": "exp_4",
                "degree": 2,
                "arch": "shallow",
                "opt": "rmsprop",
                "val_loss": 0.4,
                "time_offset": 3
            }
        ]
        
        # Store test experiments
        for exp in test_experiments:
            config = ExperimentConfig(
                polynomial_degree=exp["degree"],
                noise_level=0.1,
                train_val_split=0.8,
                model_architecture=exp["arch"],
                hidden_dims=[64] if exp["arch"] != "linear" else [],
                optimizer=exp["opt"],
                learning_rate=0.01,
                batch_size=32,
                epochs=10,
                random_seed=42
            )
            
            result = ExperimentResults(
                config=config,
                experiment_id=exp["id"],
                timestamp=base_time + timedelta(hours=exp["time_offset"]),
                duration_seconds=15.0,
                final_train_loss=exp["val_loss"] - 0.1,
                final_val_loss=exp["val_loss"],
                best_val_loss=exp["val_loss"] - 0.05,
                training_time=12.0,
                convergence_epoch=5,
                model_size=200,
                status="completed"
            )
            
            storage.store_experiment(result)
        
        print(f"Stored {len(test_experiments)} test experiments")
        
        # Test basic query (all experiments)
        all_results = storage.query_experiments()
        print(f"All experiments: {len(all_results)}")
        
        # Test filtering by polynomial degree
        degree_filter = QueryFilter(polynomial_degrees=[2])
        degree_results = storage.query_experiments(degree_filter)
        print(f"Degree 2 experiments: {len(degree_results)}")
        
        # Test filtering by model architecture
        arch_filter = QueryFilter(model_architectures=["shallow"])
        arch_results = storage.query_experiments(arch_filter)
        print(f"Shallow architecture experiments: {len(arch_results)}")
        
        # Test filtering by performance
        perf_filter = QueryFilter(max_final_val_loss=0.5)
        perf_results = storage.query_experiments(perf_filter)
        print(f"Low validation loss experiments: {len(perf_results)}")
        
        # Test combined filtering
        combined_filter = QueryFilter(
            polynomial_degrees=[2, 3],
            optimizers=["adam", "sgd"],
            max_final_val_loss=0.6
        )
        combined_results = storage.query_experiments(combined_filter)
        print(f"Combined filter experiments: {len(combined_results)}")
        
        # Test sorting
        sorted_results = storage.query_experiments(
            sort_by="final_val_loss",
            ascending=True
        )
        print(f"Best validation loss: {sorted_results[0].final_val_loss:.3f}")
        
        # Test best experiments
        best_results = storage.get_best_experiments(n=2)
        print(f"Top 2 experiments: {[r.experiment_id for r in best_results]}")
        
        # Test summary statistics
        summary = storage.get_experiment_summary()
        print(f"Summary - Total: {summary['total_experiments']}, Success rate: {summary['success_rate']:.2f}")
        
        # Test CSV export
        csv_path = temp_dir / "experiments.csv"
        success = storage.export_to_csv(str(csv_path))
        print(f"CSV export: {'Success' if success else 'Failed'}")
        
        if csv_path.exists():
            print(f"CSV file size: {csv_path.stat().st_size} bytes")
        
        print("Querying and filtering test completed!")
        
    except Exception as e:
        print(f"Querying test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_integration_with_runner():
    """Test integration between storage and experiment runner."""
    print("\nTesting integration with experiment runner...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Initialize storage
        storage = ExperimentStorage(
            storage_dir=str(temp_dir / "integration_storage"),
            backend=StorageBackend.JSON
        )
        
        # Initialize runner
        runner = ExperimentRunner(
            output_dir=str(temp_dir / "experiments"),
            images_dir=str(temp_dir / "images"),
            models_dir=str(temp_dir / "models"),
            enable_logging=False
        )
        
        # Run a few experiments
        configs = []
        for degree in [1, 2]:
            config = ExperimentConfig(
                polynomial_degree=degree,
                noise_level=0.1,
                train_val_split=0.8,
                model_architecture="linear",
                hidden_dims=[],
                optimizer="adam",
                learning_rate=0.01,
                batch_size=16,
                epochs=3,
                random_seed=42 + degree,
                num_data_points=50
            )
            configs.append(config)
        
        # Run experiments
        results = runner.run_parallel_experiments(configs, max_workers=2)
        print(f"Ran {len(results)} experiments")
        
        # Store results in storage system
        stored_count = 0
        for result in results:
            if storage.store_experiment(result):
                stored_count += 1
        
        print(f"Stored {stored_count} experiments in storage system")
        
        # Query stored experiments
        stored_results = storage.query_experiments()
        print(f"Retrieved {len(stored_results)} experiments from storage")
        
        # Verify data integrity
        for original, stored in zip(results, stored_results):
            if original.experiment_id == stored.experiment_id:
                print(f"Data integrity check passed for {original.experiment_id}")
            else:
                print(f"Data integrity check failed for {original.experiment_id}")
        
        print("Integration test completed!")
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_storage_backends()
    test_querying_and_filtering()
    test_integration_with_runner()