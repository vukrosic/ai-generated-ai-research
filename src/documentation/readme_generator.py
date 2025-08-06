"""
README generation for AI curve fitting research project.

This module provides comprehensive README generation with experiment summaries,
embedded images, performance metrics, and key findings extraction.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

try:
    from ..experiments.storage import ExperimentStorage, QueryFilter
    from ..experiments.runner import ExperimentResults
except ImportError:
    from experiments.storage import ExperimentStorage, QueryFilter
    from experiments.runner import ExperimentResults


class ReadmeGenerator:
    """
    Generates comprehensive README documentation for curve fitting experiments.
    
    This class creates structured README files with methodology sections,
    results summaries, embedded images, and key findings extraction.
    """
    
    def __init__(self, 
                 storage: ExperimentStorage,
                 output_dir: str = ".",
                 images_dir: str = "images"):
        """
        Initialize the README generator.
        
        Args:
            storage: ExperimentStorage instance for accessing experiment data
            output_dir: Directory to save README file
            images_dir: Directory containing generated images
        """
        self.storage = storage
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_readme(self, 
                       filter_criteria: Optional[QueryFilter] = None,
                       include_failed: bool = False) -> str:
        """
        Generate comprehensive README with experiment summaries and analysis.
        
        Args:
            filter_criteria: Optional filter for selecting experiments
            include_failed: Whether to include failed experiments in analysis
            
        Returns:
            Generated README content as string
        """
        # Get experiment data
        if not include_failed:
            if filter_criteria is None:
                filter_criteria = QueryFilter()
            filter_criteria.statuses = ["completed"]
        
        experiments = self.storage.query_experiments(
            filter_criteria=filter_criteria,
            sort_by="timestamp",
            ascending=False
        )
        
        if not experiments:
            return self._generate_empty_readme()
        
        # Generate README sections
        readme_content = []
        
        # Header and introduction
        readme_content.append(self._generate_header())
        readme_content.append(self._generate_introduction())
        
        # Methodology section
        readme_content.append(self._generate_methodology(experiments))
        
        # Results section
        readme_content.append(self._generate_results_section(experiments))
        
        # Key findings
        readme_content.append(self._generate_key_findings(experiments))
        
        # Experiment details
        readme_content.append(self._generate_experiment_details(experiments))
        
        # Images section
        readme_content.append(self._generate_images_section())
        
        # Usage instructions
        readme_content.append(self._generate_usage_section())
        
        # Footer
        readme_content.append(self._generate_footer())
        
        # Join all sections
        full_readme = "\n\n".join(readme_content)
        
        # Save to file
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(full_readme)
        
        return full_readme
    
    def _generate_empty_readme(self) -> str:
        """Generate README for when no experiments are available."""
        return f"""# AI Curve Fitting Research

## Overview

This repository contains an AI research project focused on curve fitting using neural networks with different polynomial curves and optimization techniques.

## Status

No completed experiments found. Run experiments using the main.py script to generate results and documentation.

## Usage

```bash
python main.py --config configs/example_config.json
```

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    def _generate_header(self) -> str:
        """Generate README header section."""
        return """# AI Curve Fitting Research

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This repository contains a comprehensive AI research project focused on analyzing the effectiveness of different neural network architectures and optimization techniques for polynomial curve fitting tasks. The system systematically explores various combinations of model architectures, optimizers, and hyperparameters to understand their impact on curve fitting performance."""
    
    def _generate_introduction(self) -> str:
        """Generate introduction section."""
        return """## Introduction

Curve fitting is a fundamental problem in machine learning and scientific computing. This research project investigates how different neural network architectures (linear, shallow, and deep networks) perform when fitting polynomial curves of varying complexity. The study examines the effectiveness of different optimization algorithms (SGD, Adam, RMSprop, AdaGrad) and analyzes their convergence behavior, generalization performance, and computational efficiency.

### Research Questions

1. How do different neural network architectures compare in their ability to fit polynomial curves?
2. Which optimization algorithms provide the best convergence and generalization for curve fitting tasks?
3. How does polynomial complexity (degree) affect the performance of different model architectures?
4. What are the optimal hyperparameter configurations for different curve fitting scenarios?"""
    
    def _generate_methodology(self, experiments: List[ExperimentResults]) -> str:
        """Generate methodology section based on experiments."""
        # Extract unique configurations
        architectures = set(exp.config.model_architecture for exp in experiments)
        optimizers = set(exp.config.optimizer for exp in experiments)
        degrees = set(exp.config.polynomial_degree for exp in experiments)
        
        methodology = """## Methodology

### Experimental Setup

The research employs a systematic experimental approach to evaluate different combinations of neural network architectures and optimization techniques for polynomial curve fitting.

#### Data Generation
- **Polynomial Degrees**: """ + f"{sorted(degrees)}" + """
- **Synthetic Data**: Generated using polynomial functions with configurable noise levels
- **Dataset Split**: Training/validation splits with configurable ratios
- **Noise Injection**: Gaussian and uniform noise options for robustness testing

#### Model Architectures
- **Architectures Tested**: """ + f"{sorted(architectures)}" + """
- **Linear Model**: Simple linear regression baseline
- **Shallow Networks**: 1-2 hidden layers with configurable width
- **Deep Networks**: 3+ hidden layers with various activation functions

#### Optimization Algorithms
- **Optimizers Tested**: """ + f"{sorted(optimizers)}" + """
- **Hyperparameter Tuning**: Systematic exploration of learning rates, batch sizes, and epochs
- **Early Stopping**: Validation-based early stopping to prevent overfitting

#### Evaluation Metrics
- **Training Loss**: Mean squared error on training data
- **Validation Loss**: Mean squared error on validation data
- **Convergence Analysis**: Training time and convergence behavior
- **Model Complexity**: Parameter count and computational requirements"""
        
        return methodology
    
    def _generate_results_section(self, experiments: List[ExperimentResults]) -> str:
        """Generate results section with performance analysis."""
        # Calculate summary statistics
        summary = self._calculate_experiment_summary(experiments)
        
        results = f"""## Results

### Experiment Summary

- **Total Experiments**: {summary['total_experiments']}
- **Success Rate**: {summary['success_rate']:.1%}
- **Average Training Time**: {summary['avg_training_time']:.2f} seconds

### Performance Overview

#### Best Performing Models
"""
        
        # Get best models by validation loss
        best_models = sorted(experiments, key=lambda x: x.final_val_loss)[:5]
        
        results += "| Rank | Architecture | Optimizer | Degree | Val Loss | Train Loss | Time (s) |\n"
        results += "|------|-------------|-----------|---------|----------|------------|----------|\n"
        
        for i, exp in enumerate(best_models, 1):
            results += f"| {i} | {exp.config.model_architecture} | {exp.config.optimizer} | {exp.config.polynomial_degree} | {exp.final_val_loss:.6f} | {exp.final_train_loss:.6f} | {exp.training_time:.2f} |\n"
        
        # Architecture comparison
        arch_stats = self._calculate_architecture_stats(experiments)
        results += f"""
### Architecture Comparison

"""
        
        for arch, stats in arch_stats.items():
            results += f"""#### {arch.title()} Networks
- **Average Validation Loss**: {stats['avg_val_loss']:.6f} ± {stats['std_val_loss']:.6f}
- **Average Training Time**: {stats['avg_time']:.2f}s ± {stats['std_time']:.2f}s
- **Success Rate**: {stats['success_rate']:.1%}
- **Best Performance**: {stats['best_val_loss']:.6f}

"""
        
        # Optimizer comparison
        opt_stats = self._calculate_optimizer_stats(experiments)
        results += "### Optimizer Comparison\n\n"
        
        for opt, stats in opt_stats.items():
            results += f"""#### {opt.upper()}
- **Average Validation Loss**: {stats['avg_val_loss']:.6f} ± {stats['std_val_loss']:.6f}
- **Average Training Time**: {stats['avg_time']:.2f}s ± {stats['std_time']:.2f}s
- **Convergence Rate**: {stats['convergence_rate']:.1%}
- **Best Performance**: {stats['best_val_loss']:.6f}

"""
        
        return results
    
    def _generate_key_findings(self, experiments: List[ExperimentResults]) -> str:
        """Extract and generate key findings from experiments."""
        findings = """## Key Findings

### Performance Insights

"""
        
        # Architecture insights
        arch_stats = self._calculate_architecture_stats(experiments)
        best_arch = min(arch_stats.items(), key=lambda x: x[1]['avg_val_loss'])
        findings += f"1. **Best Architecture**: {best_arch[0].title()} networks achieved the lowest average validation loss ({best_arch[1]['avg_val_loss']:.6f})\n\n"
        
        # Optimizer insights
        opt_stats = self._calculate_optimizer_stats(experiments)
        best_opt = min(opt_stats.items(), key=lambda x: x[1]['avg_val_loss'])
        findings += f"2. **Best Optimizer**: {best_opt[0].upper()} demonstrated superior performance with average validation loss of {best_opt[1]['avg_val_loss']:.6f}\n\n"
        
        # Complexity insights
        degree_stats = self._calculate_degree_stats(experiments)
        findings += "3. **Polynomial Complexity Impact**:\n"
        for degree in sorted(degree_stats.keys()):
            stats = degree_stats[degree]
            findings += f"   - Degree {degree}: Avg validation loss {stats['avg_val_loss']:.6f}\n"
        findings += "\n"
        
        # Training efficiency
        fastest_arch = min(arch_stats.items(), key=lambda x: x[1]['avg_time'])
        findings += f"4. **Training Efficiency**: {fastest_arch[0].title()} networks were fastest to train (avg: {fastest_arch[1]['avg_time']:.2f}s)\n\n"
        
        # Convergence analysis
        convergence_rates = {arch: stats['success_rate'] for arch, stats in arch_stats.items()}
        most_stable = max(convergence_rates.items(), key=lambda x: x[1])
        findings += f"5. **Stability**: {most_stable[0].title()} networks showed highest success rate ({most_stable[1]:.1%})\n\n"
        
        findings += """### Recommendations

Based on the experimental results:

- **For High Accuracy**: Use the best performing architecture-optimizer combination identified above
- **For Fast Training**: Consider linear or shallow networks for simple polynomials
- **For Stability**: Choose architectures with high success rates for production use
- **For Complex Curves**: Deep networks may be necessary for high-degree polynomials

### Future Work

- Investigate regularization techniques to improve generalization
- Explore ensemble methods combining multiple architectures
- Analyze performance on real-world datasets beyond synthetic polynomials
- Study the effect of different activation functions and initialization strategies"""
        
        return findings
    
    def _generate_experiment_details(self, experiments: List[ExperimentResults]) -> str:
        """Generate detailed experiment information."""
        details = """## Experiment Details

### Configuration Summary

The following table summarizes all experiment configurations and their results:

| ID | Architecture | Optimizer | Degree | LR | Batch | Epochs | Val Loss | Status |
|----|-------------|-----------|---------|----|----|--------|----------|--------|
"""
        
        for exp in experiments[:20]:  # Limit to first 20 for readability
            details += f"| {exp.experiment_id[:8]} | {exp.config.model_architecture} | {exp.config.optimizer} | {exp.config.polynomial_degree} | {exp.config.learning_rate} | {exp.config.batch_size} | {exp.config.epochs} | {exp.final_val_loss:.6f} | {exp.status} |\n"
        
        if len(experiments) > 20:
            details += f"\n*Showing first 20 of {len(experiments)} experiments*\n"
        
        details += f"""
### Experiment Statistics

- **Date Range**: {min(exp.timestamp for exp in experiments).strftime('%Y-%m-%d')} to {max(exp.timestamp for exp in experiments).strftime('%Y-%m-%d')}
- **Total Runtime**: {sum(exp.duration_seconds for exp in experiments):.2f} seconds
- **Average Experiment Duration**: {np.mean([exp.duration_seconds for exp in experiments]):.2f} seconds
- **Configuration Diversity**: {len(set((exp.config.model_architecture, exp.config.optimizer, exp.config.polynomial_degree) for exp in experiments))} unique combinations tested
"""
        
        return details
    
    def _generate_images_section(self) -> str:
        """Generate images section with embedded visualizations."""
        images_section = """## Visualizations

This section contains key visualizations generated from the experiments:

"""
        
        # Look for common image files
        image_files = []
        if self.images_dir.exists():
            for pattern in ['*.png', '*.pdf']:
                image_files.extend(self.images_dir.glob(pattern))
        
        if image_files:
            # Group images by type
            curve_images = [f for f in image_files if 'curve' in f.name.lower()]
            loss_images = [f for f in image_files if 'loss' in f.name.lower()]
            comparison_images = [f for f in image_files if 'comparison' in f.name.lower() or 'heatmap' in f.name.lower()]
            
            if curve_images:
                images_section += "### Curve Fitting Results\n\n"
                for img in sorted(curve_images)[:5]:  # Limit to 5 images
                    rel_path = os.path.relpath(img, self.output_dir)
                    images_section += f"![Curve Fitting]({rel_path})\n\n"
            
            if loss_images:
                images_section += "### Training Progress\n\n"
                for img in sorted(loss_images)[:5]:
                    rel_path = os.path.relpath(img, self.output_dir)
                    images_section += f"![Training Progress]({rel_path})\n\n"
            
            if comparison_images:
                images_section += "### Performance Comparisons\n\n"
                for img in sorted(comparison_images)[:5]:
                    rel_path = os.path.relpath(img, self.output_dir)
                    images_section += f"![Performance Comparison]({rel_path})\n\n"
        else:
            images_section += "*No visualization images found. Run experiments to generate plots.*\n"
        
        return images_section
    
    def _generate_usage_section(self) -> str:
        """Generate usage instructions section."""
        return """## Usage

### Running Experiments

1. **Single Experiment**:
   ```bash
   python main.py --config configs/example_config.json
   ```

2. **Batch Experiments**:
   ```bash
   python main.py --batch-config configs/batch_experiments.json
   ```

3. **Custom Configuration**:
   ```python
   from src.experiments.config import ExperimentConfig
   from src.experiments.runner import ExperimentRunner
   
   config = ExperimentConfig(
       polynomial_degree=3,
       model_architecture="shallow",
       optimizer="adam",
       learning_rate=0.001,
       # ... other parameters
   )
   
   runner = ExperimentRunner()
   results = runner.run_experiment(config)
   ```

### Generating Documentation

```python
from src.documentation.readme_generator import ReadmeGenerator
from src.experiments.storage import ExperimentStorage

storage = ExperimentStorage()
generator = ReadmeGenerator(storage)
readme_content = generator.generate_readme()
```

### Analyzing Results

```python
from src.experiments.storage import ExperimentStorage, QueryFilter

storage = ExperimentStorage()

# Get best performing experiments
best_experiments = storage.get_best_experiments(metric="final_val_loss", n=10)

# Filter by architecture
filter_criteria = QueryFilter(model_architectures=["deep"])
deep_experiments = storage.query_experiments(filter_criteria)
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Seaborn
- Pandas (optional, for CSV export)
- SciPy (optional, for statistical tests)

Install dependencies:
```bash
pip install -r requirements.txt
```"""
    
    def _generate_footer(self) -> str:
        """Generate README footer."""
        return f"""## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{{ai-curve-fitting-research,
  title={{AI Curve Fitting Research: A Systematic Study of Neural Network Architectures and Optimization Techniques}},
  author={{Your Name}},
  year={{{datetime.now().year}}},
  url={{https://github.com/yourusername/ai-curve-fitting-research}}
}}
```

---
*README generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
    
    def _calculate_experiment_summary(self, experiments: List[ExperimentResults]) -> Dict[str, Any]:
        """Calculate summary statistics for experiments."""
        completed = [exp for exp in experiments if exp.status == "completed"]
        
        return {
            'total_experiments': len(experiments),
            'completed_experiments': len(completed),
            'success_rate': len(completed) / len(experiments) if experiments else 0,
            'avg_training_time': np.mean([exp.training_time for exp in completed]) if completed else 0,
            'avg_val_loss': np.mean([exp.final_val_loss for exp in completed]) if completed else 0,
        }
    
    def _calculate_architecture_stats(self, experiments: List[ExperimentResults]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics grouped by architecture."""
        completed = [exp for exp in experiments if exp.status == "completed"]
        arch_groups = {}
        
        for exp in completed:
            arch = exp.config.model_architecture
            if arch not in arch_groups:
                arch_groups[arch] = []
            arch_groups[arch].append(exp)
        
        stats = {}
        for arch, exps in arch_groups.items():
            val_losses = [exp.final_val_loss for exp in exps]
            times = [exp.training_time for exp in exps]
            
            stats[arch] = {
                'avg_val_loss': np.mean(val_losses),
                'std_val_loss': np.std(val_losses),
                'best_val_loss': np.min(val_losses),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'success_rate': len(exps) / len([e for e in experiments if e.config.model_architecture == arch]),
                'count': len(exps)
            }
        
        return stats
    
    def _calculate_optimizer_stats(self, experiments: List[ExperimentResults]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics grouped by optimizer."""
        completed = [exp for exp in experiments if exp.status == "completed"]
        opt_groups = {}
        
        for exp in completed:
            opt = exp.config.optimizer
            if opt not in opt_groups:
                opt_groups[opt] = []
            opt_groups[opt].append(exp)
        
        stats = {}
        for opt, exps in opt_groups.items():
            val_losses = [exp.final_val_loss for exp in exps]
            times = [exp.training_time for exp in exps]
            converged = [exp for exp in exps if exp.convergence_epoch is not None]
            
            stats[opt] = {
                'avg_val_loss': np.mean(val_losses),
                'std_val_loss': np.std(val_losses),
                'best_val_loss': np.min(val_losses),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'convergence_rate': len(converged) / len(exps) if exps else 0,
                'count': len(exps)
            }
        
        return stats
    
    def _calculate_degree_stats(self, experiments: List[ExperimentResults]) -> Dict[int, Dict[str, float]]:
        """Calculate statistics grouped by polynomial degree."""
        completed = [exp for exp in experiments if exp.status == "completed"]
        degree_groups = {}
        
        for exp in completed:
            degree = exp.config.polynomial_degree
            if degree not in degree_groups:
                degree_groups[degree] = []
            degree_groups[degree].append(exp)
        
        stats = {}
        for degree, exps in degree_groups.items():
            val_losses = [exp.final_val_loss for exp in exps]
            
            stats[degree] = {
                'avg_val_loss': np.mean(val_losses),
                'std_val_loss': np.std(val_losses),
                'best_val_loss': np.min(val_losses),
                'count': len(exps)
            }
        
        return stats
    
    def update_readme(self, 
                     new_experiments: List[ExperimentResults],
                     incremental: bool = True) -> str:
        """
        Update existing README with new experiment results.
        
        Args:
            new_experiments: List of new experiment results to include
            incremental: If True, append to existing results; if False, regenerate completely
            
        Returns:
            Updated README content
        """
        if incremental:
            # Load existing experiments and combine with new ones
            all_experiments = self.storage.query_experiments()
            # Filter out duplicates based on experiment_id
            existing_ids = {exp.experiment_id for exp in all_experiments}
            unique_new = [exp for exp in new_experiments if exp.experiment_id not in existing_ids]
            
            if unique_new:
                # Store new experiments
                for exp in unique_new:
                    self.storage.store_experiment(exp)
                
                # Regenerate README with all experiments
                return self.generate_readme()
            else:
                # No new experiments, return existing README
                readme_path = self.output_dir / "README.md"
                if readme_path.exists():
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return self.generate_readme()
        else:
            # Complete regeneration
            return self.generate_readme()
    
    def extract_key_findings(self, experiments: List[ExperimentResults]) -> Dict[str, Any]:
        """
        Extract key findings from experiments for use in other documentation.
        
        Args:
            experiments: List of experiment results to analyze
            
        Returns:
            Dictionary containing key findings and statistics
        """
        if not experiments:
            return {}
        
        completed = [exp for exp in experiments if exp.status == "completed"]
        
        # Best performing models
        best_overall = min(completed, key=lambda x: x.final_val_loss) if completed else None
        
        # Architecture analysis
        arch_stats = self._calculate_architecture_stats(experiments)
        best_arch = min(arch_stats.items(), key=lambda x: x[1]['avg_val_loss']) if arch_stats else None
        
        # Optimizer analysis
        opt_stats = self._calculate_optimizer_stats(experiments)
        best_opt = min(opt_stats.items(), key=lambda x: x[1]['avg_val_loss']) if opt_stats else None
        
        # Complexity analysis
        degree_stats = self._calculate_degree_stats(experiments)
        
        return {
            'total_experiments': len(experiments),
            'completed_experiments': len(completed),
            'success_rate': len(completed) / len(experiments) if experiments else 0,
            'best_overall_model': {
                'experiment_id': best_overall.experiment_id if best_overall else None,
                'architecture': best_overall.config.model_architecture if best_overall else None,
                'optimizer': best_overall.config.optimizer if best_overall else None,
                'val_loss': best_overall.final_val_loss if best_overall else None,
            } if best_overall else None,
            'best_architecture': {
                'name': best_arch[0] if best_arch else None,
                'avg_val_loss': best_arch[1]['avg_val_loss'] if best_arch else None,
                'success_rate': best_arch[1]['success_rate'] if best_arch else None,
            } if best_arch else None,
            'best_optimizer': {
                'name': best_opt[0] if best_opt else None,
                'avg_val_loss': best_opt[1]['avg_val_loss'] if best_opt else None,
                'convergence_rate': best_opt[1]['convergence_rate'] if best_opt else None,
            } if best_opt else None,
            'architecture_stats': arch_stats,
            'optimizer_stats': opt_stats,
            'degree_stats': degree_stats,
            'date_range': {
                'start': min(exp.timestamp for exp in experiments).isoformat() if experiments else None,
                'end': max(exp.timestamp for exp in experiments).isoformat() if experiments else None,
            }
        }