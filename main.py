#!/usr/bin/env python3
"""
AI Curve Fitting Research - Main Entry Point

This module serves as the main entry point for the AI curve fitting research project.
It provides command-line interface for running experiments with different configurations.
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def parse_arguments():
    """Parse command-line arguments for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="AI Curve Fitting Research - Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config configs/experiment1.json
  python main.py --polynomial-degree 3 --optimizer adam --epochs 100
  python main.py --batch-run configs/
        """
    )
    
    # Configuration file options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to experiment configuration JSON file'
    )
    
    parser.add_argument(
        '--batch-run', '-b',
        type=str,
        help='Directory containing multiple configuration files for batch execution'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        help='Maximum number of concurrent experiments for batch execution (default: CPU count)'
    )
    
    parser.add_argument(
        '--use-processes',
        action='store_true',
        help='Use processes instead of threads for parallel execution'
    )
    
    # Individual experiment parameters
    parser.add_argument(
        '--polynomial-degree',
        type=int,
        choices=range(1, 7),
        default=3,
        help='Degree of polynomial to fit (1-6, default: 3)'
    )
    
    parser.add_argument(
        '--noise-level',
        type=float,
        default=0.1,
        help='Noise level for synthetic data generation (default: 0.1)'
    )
    
    parser.add_argument(
        '--optimizer',
        choices=['sgd', 'adam', 'rmsprop', 'adagrad'],
        default='adam',
        help='Optimizer to use for training (default: adam)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer (default: 0.001)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--model-architecture',
        choices=['linear', 'shallow', 'deep'],
        default='shallow',
        help='Neural network architecture to use (default: shallow)'
    )
    
    # Output and visualization options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments',
        help='Directory to save experiment results (default: experiments)'
    )
    
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--generate-readme',
        action='store_true',
        help='Generate README documentation'
    )
    
    parser.add_argument(
        '--generate-latex',
        action='store_true',
        help='Generate LaTeX research paper'
    )
    
    # Utility options
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running experiments'
    )
    
    return parser.parse_args()


def check_system_requirements():
    """Check system requirements and dependencies."""
    errors = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        errors.append("Python 3.7 or higher is required")
    
    # Check required packages
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn')
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
        except ImportError:
            errors.append(f"{name} is not installed (pip install {package})")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            warnings.append(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        else:
            warnings.append("CUDA not available, using CPU")
    except ImportError:
        pass
    
    return errors, warnings


def validate_arguments(args):
    """Validate parsed arguments and check for required dependencies."""
    errors = []
    
    # Check system requirements first
    sys_errors, sys_warnings = check_system_requirements()
    errors.extend(sys_errors)
    
    if sys_warnings and args.verbose:
        print("System information:")
        for warning in sys_warnings:
            print(f"  - {warning}")
    
    # Check if config file exists
    if args.config and not Path(args.config).exists():
        errors.append(f"Configuration file not found: {args.config}")
    
    # Check if batch run directory exists
    if args.batch_run and not Path(args.batch_run).is_dir():
        errors.append(f"Batch run directory not found: {args.batch_run}")
    
    # Validate output directory
    output_path = Path(args.output_dir)
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory {args.output_dir}: {e}")
    
    # Validate parameter ranges
    if args.noise_level < 0 or args.noise_level > 1:
        errors.append("Noise level must be between 0 and 1")
    
    if args.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    if args.epochs <= 0:
        errors.append("Number of epochs must be positive")
    
    if args.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    # Validate concurrent execution parameters
    if args.max_concurrent is not None and args.max_concurrent <= 0:
        errors.append("Max concurrent experiments must be positive")
    
    return errors


def create_config_from_args(args) -> 'ExperimentConfig':
    """Create ExperimentConfig from command-line arguments with validation."""
    from src.experiments.config import ExperimentConfig, ConfigValidator
    
    # Determine hidden dimensions based on architecture
    if args.model_architecture == "linear":
        hidden_dims = []
    elif args.model_architecture == "shallow":
        hidden_dims = [64, 32]
    else:  # deep
        hidden_dims = [128, 64, 32, 16]
    
    try:
        config = ExperimentConfig(
            polynomial_degree=args.polynomial_degree,
            noise_level=args.noise_level,
            train_val_split=0.8,  # Default split
            model_architecture=args.model_architecture,
            hidden_dims=hidden_dims,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            random_seed=args.random_seed
        )
        
        # Validate configuration and show warnings
        warnings = ConfigValidator.validate_config_compatibility(config)
        if warnings and args.verbose:
            print("\nConfiguration warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        suggestions = ConfigValidator.suggest_improvements(config)
        if suggestions and args.verbose:
            print("\nConfiguration suggestions:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
        
        return config
        
    except Exception as e:
        print(f"Error creating configuration: {e}")
        raise


def run_single_experiment_from_args(runner, args):
    """Run a single experiment using CLI arguments with error handling."""
    print("\nRunning single experiment with CLI parameters...")
    
    try:
        # Create configuration from arguments
        config = create_config_from_args(args)
        
        if args.verbose:
            print(f"Configuration: {config}")
        
        # Run experiment with progress indication
        print("Starting experiment execution...")
        result = runner.run_single_experiment(config)
        
        if result.status == "completed":
            # Generate additional outputs if requested
            generate_additional_outputs(runner, [result], args)
            
            print(f"\n✓ Experiment completed successfully: {result.experiment_id}")
            print(f"  Final validation loss: {result.final_val_loss:.6f}")
            print(f"  Training time: {result.training_time:.1f}s")
            print(f"  Results saved to: {args.output_dir}/{result.experiment_id}")
            
            if result.image_paths:
                print(f"  Generated {len(result.image_paths)} visualization(s)")
        else:
            print(f"\n✗ Experiment failed: {result.error_message}")
            
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        raise
    except Exception as e:
        print(f"\n✗ Error during experiment execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise


def run_experiment_from_config(runner, args):
    """Run a single experiment from configuration file with error handling."""
    print(f"\nRunning experiment from config file: {args.config}")
    
    try:
        # Load and validate configuration
        from src.experiments.config import ExperimentConfig, ConfigValidator
        config = ExperimentConfig.from_json(args.config)
        
        if args.verbose:
            print(f"Loaded configuration: {config}")
            
            # Show configuration warnings and suggestions
            warnings = ConfigValidator.validate_config_compatibility(config)
            if warnings:
                print("\nConfiguration warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
        
        # Run experiment with progress indication
        print("Starting experiment execution...")
        result = runner.run_single_experiment(config)
        
        if result.status == "completed":
            # Generate additional outputs if requested
            generate_additional_outputs(runner, [result], args)
            
            print(f"\n✓ Experiment completed successfully: {result.experiment_id}")
            print(f"  Final validation loss: {result.final_val_loss:.6f}")
            print(f"  Training time: {result.training_time:.1f}s")
            print(f"  Results saved to: {args.output_dir}/{result.experiment_id}")
            
            if result.image_paths:
                print(f"  Generated {len(result.image_paths)} visualization(s)")
        else:
            print(f"\n✗ Experiment failed: {result.error_message}")
            
    except FileNotFoundError:
        print(f"✗ Configuration file not found: {args.config}")
        raise
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in configuration file: {e}")
        raise
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        raise
    except Exception as e:
        print(f"\n✗ Error during experiment execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise


def run_batch_experiments(runner, args):
    """Run batch experiments from directory of config files."""
    print(f"\nRunning batch experiments from directory: {args.batch_run}")
    
    # Import batch runner
    from src.experiments.batch_runner import BatchExperimentRunner
    
    # Create batch runner with enhanced capabilities
    batch_runner = BatchExperimentRunner(
        base_runner=runner,
        max_concurrent=args.max_concurrent,  # Use CLI argument or default
        use_processes=args.use_processes     # Use CLI argument
    )
    
    try:
        # Run batch experiments with progress monitoring
        results = batch_runner.run_from_directory(args.batch_run)
        
        # Generate additional outputs if requested
        generate_additional_outputs(runner, results, args)
        
        # Save detailed progress report
        progress_report_path = Path(args.output_dir) / "batch_progress_report.json"
        batch_runner.save_progress_report(str(progress_report_path))
        
        # Print summary
        successful = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]
        
        print(f"\nBatch execution summary:")
        print(f"  Total experiments: {len(results)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        if successful:
            avg_val_loss = sum(r.final_val_loss for r in successful) / len(successful)
            best_val_loss = min(r.final_val_loss for r in successful)
            print(f"  Average validation loss: {avg_val_loss:.6f}")
            print(f"  Best validation loss: {best_val_loss:.6f}")
        
        print(f"  Progress report saved: {progress_report_path}")
        
    except Exception as e:
        print(f"Error during batch execution: {e}")
        raise


def generate_additional_outputs(runner, results, args):
    """Generate additional outputs like README and LaTeX if requested."""
    if not (args.generate_readme or args.generate_latex):
        return
    
    successful_results = [r for r in results if r.status == "completed"]
    if not successful_results:
        print("No successful experiments to generate documentation for")
        return
    
    try:
        if args.generate_readme:
            print("Generating README documentation...")
            from src.documentation.readme_generator import ReadmeGenerator
            readme_gen = ReadmeGenerator()
            readme_content = readme_gen.generate_readme(successful_results)
            
            readme_path = Path(args.output_dir) / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            print(f"README generated: {readme_path}")
        
        if args.generate_latex:
            print("Generating LaTeX research paper...")
            from src.documentation.latex_generator import LatexGenerator
            latex_gen = LatexGenerator()
            latex_content = latex_gen.generate_paper(successful_results)
            
            papers_dir = Path("papers")
            papers_dir.mkdir(exist_ok=True)
            latex_path = papers_dir / "research_paper.tex"
            with open(latex_path, 'w') as f:
                f.write(latex_content)
            print(f"LaTeX paper generated: {latex_path}")
            
    except Exception as e:
        print(f"Error generating additional outputs: {e}")


def run_integration_test():
    """Run a quick integration test to verify all components work together."""
    print("Running integration test...")
    
    try:
        # Test imports
        from src.experiments.config import ExperimentConfig
        from src.experiments.runner import ExperimentRunner
        from src.data.generators import PolynomialGenerator
        from src.models.architectures import ModelFactory
        
        # Create minimal test config
        config = ExperimentConfig(
            polynomial_degree=2,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture="linear",
            hidden_dims=[],
            optimizer="adam",
            learning_rate=0.01,
            batch_size=16,
            epochs=5,  # Very short for testing
            random_seed=42,
            num_data_points=100  # Small dataset for testing
        )
        
        # Test data generation
        poly_gen = PolynomialGenerator(random_seed=42)
        x_data, y_data, coeffs = poly_gen.generate_polynomial_data(
            degree=2, x_range=(-2, 2), num_points=50
        )
        
        # Test model creation
        model = ModelFactory.create_model("linear", {"input_dim": 1, "hidden_dims": []})
        
        print("✓ Integration test passed - all components working")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Main entry point for the application."""
    print("AI Curve Fitting Research - Starting...")
    
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Validate arguments and system
        validation_errors = validate_arguments(args)
        if validation_errors:
            print("✗ Validation errors found:")
            for error in validation_errors:
                print(f"  - {error}")
            sys.exit(1)
        
        # Display configuration
        if args.verbose:
            print("\nExperiment Configuration:")
            print(f"  Polynomial Degree: {args.polynomial_degree}")
            print(f"  Noise Level: {args.noise_level}")
            print(f"  Optimizer: {args.optimizer}")
            print(f"  Learning Rate: {args.learning_rate}")
            print(f"  Epochs: {args.epochs}")
            print(f"  Batch Size: {args.batch_size}")
            print(f"  Model Architecture: {args.model_architecture}")
            print(f"  Output Directory: {args.output_dir}")
            print(f"  Random Seed: {args.random_seed}")
            
            if args.max_concurrent:
                print(f"  Max Concurrent: {args.max_concurrent}")
            if args.use_processes:
                print(f"  Use Processes: {args.use_processes}")
        
        if args.dry_run:
            print("\n🔍 Dry run mode - validating setup without execution")
            
            # Run integration test in dry run mode
            if run_integration_test():
                print("✓ All systems ready for experiment execution")
            else:
                print("✗ System validation failed")
                sys.exit(1)
            return
        
        # Import experiment components with detailed error handling
        try:
            from src.experiments.config import ExperimentConfig, ConfigValidator
            from src.experiments.runner import ExperimentRunner
            from src.experiments.batch_runner import BatchExperimentRunner
            from src.documentation.readme_generator import ReadmeGenerator
            from src.documentation.latex_generator import LatexGenerator
        except ImportError as e:
            print(f"Error importing experiment components: {e}")
            print("\nThis could be due to:")
            print("  - Missing dependencies (run: pip install -r requirements.txt)")
            print("  - Incorrect Python path")
            print("  - Missing source files")
            print("\nPlease check your installation and try again.")
            sys.exit(1)
        
        # Run quick integration test if verbose
        if args.verbose:
            print("\n🔧 Running system integration test...")
            if not run_integration_test():
                print("✗ Integration test failed - please check your installation")
                sys.exit(1)
        
        # Initialize experiment runner
        runner = ExperimentRunner(
            output_dir=args.output_dir,
            images_dir="images",
            models_dir="models"
        )
        
        try:
            import time
            start_time = time.time()
            
            if args.config:
                # Run single experiment from config file
                run_experiment_from_config(runner, args)
            elif args.batch_run:
                # Run batch experiments from directory
                run_batch_experiments(runner, args)
            else:
                # Run single experiment with CLI parameters
                run_single_experiment_from_args(runner, args)
            
            total_time = time.time() - start_time
            print(f"\n🎉 All experiments completed successfully!")
            print(f"   Total execution time: {total_time:.1f}s")
            print(f"   Results saved to: {args.output_dir}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Execution interrupted by user")
            print("   Partial results may be available in the output directory")
            sys.exit(130)  # Standard exit code for SIGINT
            
        except Exception as e:
            print(f"\n💥 Error during experiment execution: {e}")
            if args.verbose:
                print("\nFull error traceback:")
                import traceback
                traceback.print_exc()
            print(f"\nFor help, run: {sys.argv[0]} --help")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()