#!/usr/bin/env python3
"""
Simple test script for documentation generation system.

This script demonstrates the automated documentation generation capabilities
including README generation, LaTeX paper creation, and version management.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.experiments.storage import ExperimentStorage
    from src.experiments.config import ExperimentConfig
    from src.experiments.runner import ExperimentRunner, ExperimentResults
    from src.documentation import ReadmeGenerator, LatexGenerator, DocumentationVersionManager
except ImportError:
    # Fallback for direct execution
    from experiments.storage import ExperimentStorage
    from experiments.config import ExperimentConfig
    from experiments.runner import ExperimentRunner, ExperimentResults
    from documentation import ReadmeGenerator, LatexGenerator, DocumentationVersionManager


def create_sample_experiments():
    """Create sample experiment results for testing documentation."""
    print("Creating sample experiment results...")
    
    storage = ExperimentStorage("test_experiments")
    
    # Create sample configurations
    configs = [
        ExperimentConfig(
            polynomial_degree=2,
            noise_level=0.1,
            train_val_split=0.8,
            model_architecture="shallow",
            hidden_dims=[32, 16],
            optimizer="adam",
            learning_rate=0.001,
            batch_size=32,
            epochs=50,
            random_seed=42
        ),
        ExperimentConfig(
            polynomial_degree=3,
            noise_level=0.05,
            train_val_split=0.8,
            model_architecture="deep",
            hidden_dims=[64, 32, 16],
            optimizer="sgd",
            learning_rate=0.01,
            batch_size=64,
            epochs=100,
            random_seed=123
        ),
        ExperimentConfig(
            polynomial_degree=4,
            noise_level=0.2,
            train_val_split=0.8,
            model_architecture="linear",
            hidden_dims=[],
            optimizer="rmsprop",
            learning_rate=0.005,
            batch_size=16,
            epochs=75,
            random_seed=456
        )
    ]
    
    # Create sample results
    sample_results = []
    for i, config in enumerate(configs):
        # Simulate experiment results
        result = ExperimentResults.create_mock_result(
            config=config,
            final_train_loss=0.001 + i * 0.0005,
            final_val_loss=0.002 + i * 0.001,
            training_time=30.0 + i * 10.0,
            status="completed"
        )
        
        # Store result
        storage.store_experiment(result)
        sample_results.append(result)
        print(f"  Created experiment {i+1}: {result.experiment_id[:8]}")
    
    return storage, sample_results


def test_readme_generation():
    """Test README generation functionality."""
    print("\n=== Testing README Generation ===")
    
    storage, _ = create_sample_experiments()
    
    # Create README generator
    readme_gen = ReadmeGenerator(storage, output_dir="test_output")
    
    # Generate README
    print("Generating README...")
    readme_content = readme_gen.generate_readme()
    
    print(f"README generated successfully ({len(readme_content)} characters)")
    print("README sections include:")
    
    # Check for key sections
    sections = [
        "# AI Curve Fitting Research",
        "## Overview", 
        "## Results",
        "## Key Findings",
        "## Usage"
    ]
    
    for section in sections:
        if section in readme_content:
            print(f"  ✓ {section}")
        else:
            print(f"  ✗ {section} (missing)")
    
    return readme_content


def test_latex_generation():
    """Test LaTeX paper generation functionality."""
    print("\n=== Testing LaTeX Generation ===")
    
    storage, _ = create_sample_experiments()
    
    # Create LaTeX generator
    latex_gen = LatexGenerator(storage, output_dir="test_output/papers")
    
    # Generate LaTeX paper
    print("Generating LaTeX paper...")
    latex_content = latex_gen.generate_paper(
        title="Test Paper: Neural Network Curve Fitting Analysis",
        authors=["Test Author"],
        affiliations=["Test Institution"]
    )
    
    print(f"LaTeX paper generated successfully ({len(latex_content)} characters)")
    
    # Check for key LaTeX elements
    elements = [
        "\\documentclass",
        "\\begin{document}",
        "\\title{",
        "\\section{Introduction}",
        "\\section{Results and Analysis}",
        "\\end{document}"
    ]
    
    print("LaTeX structure check:")
    for element in elements:
        if element in latex_content:
            print(f"  ✓ {element}")
        else:
            print(f"  ✗ {element} (missing)")
    
    return latex_content


def test_version_management():
    """Test documentation version management."""
    print("\n=== Testing Version Management ===")
    
    storage, _ = create_sample_experiments()
    
    # Create version manager
    version_manager = DocumentationVersionManager(
        storage, 
        output_dir="test_output",
        version_dir="test_output/.doc_versions"
    )
    
    # Check for updates
    print("Checking for documentation updates...")
    update_check = version_manager.check_for_updates()
    print(f"Update needed: README={update_check['readme']}, LaTeX={update_check['latex']}")
    if update_check['reason']:
        print(f"Reasons: {', '.join(update_check['reason'])}")
    
    # Perform initial update
    print("Performing initial documentation update...")
    update_results = version_manager.update_documentation()
    
    print("Update results:")
    for doc_type in ['readme', 'latex']:
        result = update_results[doc_type]
        if result['updated']:
            print(f"  ✓ {doc_type}: Updated to version {result['version']}")
            print(f"    Changes: {result['changes']}")
        else:
            print(f"  - {doc_type}: No update needed")
    
    # Test version history
    print("\nVersion history:")
    history = version_manager.get_version_history()
    for doc_type, versions in history.items():
        print(f"  {doc_type}: {len(versions)} versions")
        if versions:
            latest = versions[-1]
            print(f"    Latest: v{latest['version']} ({latest['experiment_count']} experiments)")
    
    # Test validation
    print("\nValidating documentation...")
    validation = version_manager.validate_documentation()
    
    for doc_type in ['readme', 'latex']:
        result = validation[doc_type]
        if result['valid']:
            print(f"  ✓ {doc_type}: Valid")
        else:
            print(f"  ✗ {doc_type}: Issues found")
            for issue in result['issues']:
                print(f"    - {issue}")
    
    return version_manager


def test_incremental_updates():
    """Test incremental documentation updates."""
    print("\n=== Testing Incremental Updates ===")
    
    storage, _ = create_sample_experiments()
    version_manager = DocumentationVersionManager(
        storage, 
        output_dir="test_output",
        version_dir="test_output/.doc_versions"
    )
    
    # Initial update
    print("Performing initial update...")
    initial_update = version_manager.update_documentation()
    initial_readme_version = initial_update['readme']['version']
    
    # Add a new experiment
    print("Adding new experiment...")
    new_config = ExperimentConfig(
        polynomial_degree=5,
        noise_level=0.15,
        train_val_split=0.8,
        model_architecture="deep",
        hidden_dims=[128, 64, 32, 16],
        optimizer="adam",
        learning_rate=0.0005,
        batch_size=32,
        epochs=150,
        random_seed=789
    )
    
    new_result = ExperimentResults.create_mock_result(
        config=new_config,
        final_train_loss=0.0008,
        final_val_loss=0.0015,
        training_time=45.0,
        status="completed"
    )
    
    storage.store_experiment(new_result)
    
    # Check for updates
    print("Checking for updates after new experiment...")
    update_check = version_manager.check_for_updates()
    print(f"Update needed: {update_check['readme']}")
    
    # Perform incremental update
    print("Performing incremental update...")
    incremental_update = version_manager.update_documentation()
    new_readme_version = incremental_update['readme']['version']
    
    if new_readme_version != initial_readme_version:
        print(f"  ✓ README updated: {initial_readme_version} → {new_readme_version}")
        print(f"    Changes: {incremental_update['readme']['changes']}")
    else:
        print("  - No version change detected")
    
    # Generate change log
    print("\nGenerating change log...")
    changelog = version_manager.generate_change_log()
    changelog_lines = len(changelog.split('\n'))
    print(f"Change log generated ({changelog_lines} lines)")
    
    # Save change log
    changelog_path = Path("test_output") / "CHANGELOG.md"
    with open(changelog_path, 'w') as f:
        f.write(changelog)
    print(f"Change log saved to {changelog_path}")


def main():
    """Run all documentation tests."""
    print("AI Curve Fitting Research - Documentation System Test")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Test individual components
        test_readme_generation()
        test_latex_generation()
        test_version_management()
        test_incremental_updates()
        
        print("\n" + "=" * 60)
        print("✓ All documentation tests completed successfully!")
        print(f"\nGenerated files can be found in: {output_dir.absolute()}")
        
        # List generated files
        print("\nGenerated files:")
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(output_dir)
                size = file_path.stat().st_size
                print(f"  {rel_path} ({size:,} bytes)")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())