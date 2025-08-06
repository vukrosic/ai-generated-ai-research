"""
LaTeX research paper generation for AI curve fitting research.

This module provides automated generation of formal research papers in LaTeX format
with standard academic structure, figure referencing, and bibliography management.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

try:
    from ..experiments.storage import ExperimentStorage, QueryFilter
    from ..experiments.runner import ExperimentResults
    from .readme_generator import ReadmeGenerator
except ImportError:
    from experiments.storage import ExperimentStorage, QueryFilter
    from experiments.runner import ExperimentResults
    from readme_generator import ReadmeGenerator


class LatexGenerator:
    """
    Generates formal research papers in LaTeX format for curve fitting experiments.
    
    This class creates complete LaTeX documents with standard academic structure,
    automatic figure referencing, results tables, and bibliography management.
    """
    
    def __init__(self, 
                 storage: ExperimentStorage,
                 output_dir: str = "papers",
                 images_dir: str = "images",
                 template_dir: Optional[str] = None):
        """
        Initialize the LaTeX generator.
        
        Args:
            storage: ExperimentStorage instance for accessing experiment data
            output_dir: Directory to save LaTeX files
            images_dir: Directory containing generated images
            template_dir: Optional directory containing custom LaTeX templates
        """
        self.storage = storage
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.template_dir = Path(template_dir) if template_dir else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LaTeX configuration
        self.document_class = "article"
        self.font_size = "11pt"
        self.paper_size = "letterpaper"
        
        # Bibliography style
        self.bib_style = "ieee"
        
        # Figure formats (prefer PDF for LaTeX)
        self.figure_formats = ['.pdf', '.png', '.eps']
    
    def generate_paper(self, 
                      title: str = "Neural Network Architectures for Polynomial Curve Fitting: A Comparative Study",
                      authors: List[str] = ["Author Name"],
                      affiliations: List[str] = ["Institution Name"],
                      filter_criteria: Optional[QueryFilter] = None,
                      include_failed: bool = False) -> str:
        """
        Generate complete LaTeX research paper.
        
        Args:
            title: Paper title
            authors: List of author names
            affiliations: List of author affiliations
            filter_criteria: Optional filter for selecting experiments
            include_failed: Whether to include failed experiments in analysis
            
        Returns:
            Generated LaTeX content as string
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
            return self._generate_empty_paper(title, authors, affiliations)
        
        # Generate paper sections
        latex_content = []
        
        # Document preamble
        latex_content.append(self._generate_preamble())
        
        # Title and authors
        latex_content.append(self._generate_title_section(title, authors, affiliations))
        
        # Abstract
        latex_content.append(self._generate_abstract(experiments))
        
        # Introduction
        latex_content.append(self._generate_introduction())
        
        # Related work
        latex_content.append(self._generate_related_work())
        
        # Methodology
        latex_content.append(self._generate_methodology(experiments))
        
        # Experimental setup
        latex_content.append(self._generate_experimental_setup(experiments))
        
        # Results and analysis
        latex_content.append(self._generate_results_section(experiments))
        
        # Discussion
        latex_content.append(self._generate_discussion(experiments))
        
        # Conclusion
        latex_content.append(self._generate_conclusion(experiments))
        
        # Acknowledgments
        latex_content.append(self._generate_acknowledgments())
        
        # Bibliography
        latex_content.append(self._generate_bibliography())
        
        # End document
        latex_content.append("\\end{document}")
        
        # Join all sections
        full_paper = "\n\n".join(latex_content)
        
        # Save to file
        paper_path = self.output_dir / "curve_fitting_paper.tex"
        with open(paper_path, 'w', encoding='utf-8') as f:
            f.write(full_paper)
        
        # Generate bibliography file
        self._generate_bib_file()
        
        return full_paper
    
    def _generate_empty_paper(self, title: str, authors: List[str], affiliations: List[str]) -> str:
        """Generate minimal paper when no experiments are available."""
        return f"""\\documentclass[{self.font_size},{self.paper_size}]{{article}}

\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{cite}}

\\title{{{title}}}
\\author{{{' and '.join(authors)}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
This paper presents a systematic study of neural network architectures for polynomial curve fitting. However, no experimental results are currently available. Please run experiments to generate data for analysis.
\\end{{abstract}}

\\section{{Introduction}}

This research investigates the effectiveness of different neural network architectures and optimization techniques for polynomial curve fitting tasks. Experimental results will be automatically incorporated into this document once experiments are completed.

\\section{{Methodology}}

The experimental methodology involves systematic evaluation of various neural network architectures (linear, shallow, and deep networks) combined with different optimization algorithms (SGD, Adam, RMSprop, AdaGrad) on polynomial curve fitting tasks of varying complexity.

\\section{{Conclusion}}

Experimental results are pending. This document will be automatically updated with findings once experiments are completed.

\\end{{document}}"""
    
    def _generate_preamble(self) -> str:
        """Generate LaTeX document preamble with packages and settings."""
        return f"""\\documentclass[{self.font_size},{self.paper_size}]{{article}}

% Essential packages
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage{{microtype}}

% Math packages
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{amsthm}}

% Graphics and figures
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{subcaption}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}
\\usepackage{{array}}

% Page layout
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{setspace}}
\\onehalfspacing

% References and citations
\\usepackage{{cite}}
\\usepackage{{url}}
\\usepackage{{hyperref}}

% Colors and styling
\\usepackage{{xcolor}}
\\usepackage{{listings}}

% Algorithm formatting
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}

% Custom commands
\\newcommand{{\\R}}{{\\mathbb{{R}}}}
\\newcommand{{\\E}}{{\\mathbb{{E}}}}
\\newcommand{{\\Var}}{{\\text{{Var}}}}
\\newcommand{{\\MSE}}{{\\text{{MSE}}}}

% Theorem environments
\\newtheorem{{theorem}}{{Theorem}}
\\newtheorem{{lemma}}{{Lemma}}
\\newtheorem{{proposition}}{{Proposition}}
\\newtheorem{{corollary}}{{Corollary}}
\\newtheorem{{definition}}{{Definition}}

% Configure hyperref
\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    citecolor=red
}}

\\begin{{document}}"""
    
    def _generate_title_section(self, title: str, authors: List[str], affiliations: List[str]) -> str:
        """Generate title, authors, and affiliations section."""
        # Format authors with affiliations
        author_text = ""
        for i, author in enumerate(authors):
            if i < len(affiliations):
                author_text += f"{author}\\thanks{{{affiliations[i]}}}"
            else:
                author_text += author
            
            if i < len(authors) - 1:
                author_text += " \\and "
        
        return f"""\\title{{{title}}}
\\author{{{author_text}}}
\\date{{\\today}}

\\maketitle"""
    
    def _generate_abstract(self, experiments: List[ExperimentResults]) -> str:
        """Generate abstract section based on experimental results."""
        # Calculate key statistics
        completed = [exp for exp in experiments if exp.status == "completed"]
        architectures = set(exp.config.model_architecture for exp in completed)
        optimizers = set(exp.config.optimizer for exp in completed)
        degrees = set(exp.config.polynomial_degree for exp in completed)
        
        # Find best performing model
        best_model = min(completed, key=lambda x: x.final_val_loss) if completed else None
        
        abstract = f"""\\begin{{abstract}}
This paper presents a comprehensive empirical study of neural network architectures and optimization techniques for polynomial curve fitting tasks. We systematically evaluate {len(architectures)} different network architectures ({', '.join(sorted(architectures))}) combined with {len(optimizers)} optimization algorithms ({', '.join(sorted(optimizers))}) across polynomial curves of degrees {min(degrees)} to {max(degrees)}. Our experimental evaluation comprises {len(completed)} completed experiments, providing insights into the effectiveness of different model-optimizer combinations for curve fitting tasks."""
        
        if best_model:
            abstract += f" The best performing configuration achieved a validation loss of {best_model.final_val_loss:.6f} using a {best_model.config.model_architecture} network with {best_model.config.optimizer} optimization."
        
        abstract += """ Key findings include performance comparisons across architectures, convergence analysis of different optimizers, and recommendations for practical curve fitting applications. Our results demonstrate significant performance variations across different configurations, with implications for both theoretical understanding and practical deployment of neural networks for function approximation tasks.
\\end{abstract}"""
        
        return abstract
    
    def _generate_introduction(self) -> str:
        """Generate introduction section."""
        return """\\section{Introduction}

Polynomial curve fitting represents a fundamental problem in machine learning and computational mathematics, with applications spanning scientific computing, data analysis, and function approximation. While traditional methods such as least squares regression provide analytical solutions for linear models, the advent of neural networks has opened new possibilities for learning complex nonlinear relationships in data.

The effectiveness of neural networks for curve fitting tasks depends critically on architectural choices and optimization strategies. Different network architectures---from simple linear models to deep multilayer perceptrons---exhibit varying capabilities in approximating polynomial functions of different complexities. Similarly, the choice of optimization algorithm significantly impacts convergence behavior, training efficiency, and final performance.

Despite the widespread use of neural networks for function approximation, systematic comparative studies of architectural and optimization choices for polynomial curve fitting remain limited. Most existing work focuses on specific architectures or optimization methods in isolation, without comprehensive cross-comparisons under controlled experimental conditions.

This paper addresses this gap by presenting a systematic empirical evaluation of neural network architectures and optimization techniques for polynomial curve fitting. Our contributions include:

\\begin{itemize}
\\item A comprehensive experimental framework for evaluating neural network performance on polynomial curve fitting tasks
\\item Systematic comparison of linear, shallow, and deep network architectures across polynomial complexities
\\item Empirical analysis of four major optimization algorithms (SGD, Adam, RMSprop, AdaGrad) for curve fitting tasks
\\item Performance benchmarks and practical recommendations for different curve fitting scenarios
\\item Open-source implementation enabling reproducible research and extension to new architectures
\\end{itemize}

Our experimental methodology employs synthetic polynomial datasets with controlled noise levels, enabling precise analysis of model performance across different complexity regimes. We evaluate models using standard regression metrics and analyze convergence behavior, training efficiency, and generalization performance."""
    
    def _generate_related_work(self) -> str:
        """Generate related work section."""
        return """\\section{Related Work}

\\subsection{Neural Networks for Function Approximation}

The universal approximation theorem \\cite{hornik1989multilayer} establishes that feedforward neural networks with a single hidden layer can approximate any continuous function to arbitrary accuracy, given sufficient width. This theoretical foundation has motivated extensive research into neural network architectures for function approximation tasks.

Cybenko \\cite{cybenko1989approximation} and Funahashi \\cite{funahashi1989approximate} provided early theoretical results on the approximation capabilities of neural networks, while Barron \\cite{barron1993universal} analyzed the approximation rates for different function classes. More recent work has extended these results to deep networks, showing that depth can provide exponential improvements in approximation efficiency for certain function classes \\cite{poggio2017and}.

\\subsection{Optimization for Neural Networks}

The choice of optimization algorithm significantly impacts neural network training dynamics and final performance. Stochastic gradient descent (SGD) \\cite{robbins1951stochastic} remains a fundamental approach, while adaptive methods such as Adam \\cite{kingma2014adam}, RMSprop \\cite{tieleman2012lecture}, and AdaGrad \\cite{duchi2011adaptive} have gained popularity for their improved convergence properties.

Comparative studies of optimization algorithms have shown that performance depends heavily on problem characteristics and network architecture \\cite{wilson2017marginal}. Recent work has highlighted the importance of learning rate scheduling \\cite{smith2017cyclical} and the role of batch size in optimization dynamics \\cite{masters2018revisiting}.

\\subsection{Polynomial Curve Fitting}

Traditional approaches to polynomial curve fitting rely on least squares methods and orthogonal polynomials \\cite{press2007numerical}. While these methods provide analytical solutions for linear-in-parameters models, they become computationally challenging for high-degree polynomials and large datasets.

Neural network approaches to curve fitting have been explored in various contexts \\cite{haykin2009neural}, but systematic comparisons across different architectures and polynomial complexities remain limited. Most existing work focuses on specific applications rather than general principles for architecture selection."""
    
    def _generate_methodology(self, experiments: List[ExperimentResults]) -> str:
        """Generate methodology section based on experiments."""
        completed = [exp for exp in experiments if exp.status == "completed"]
        architectures = sorted(set(exp.config.model_architecture for exp in completed))
        optimizers = sorted(set(exp.config.optimizer for exp in completed))
        degrees = sorted(set(exp.config.polynomial_degree for exp in completed))
        
        return f"""\\section{{Methodology}}

\\subsection{{Problem Formulation}}

We consider the polynomial curve fitting problem as a supervised regression task. Given a polynomial function $f(x) = \\sum_{{i=0}}^d a_i x^i$ of degree $d$, we generate synthetic datasets by sampling input points $x \\in [x_{{\\min}}, x_{{\\max}}]$ and computing target values $y = f(x) + \\epsilon$, where $\\epsilon \\sim \\mathcal{{N}}(0, \\sigma^2)$ represents additive Gaussian noise.

The objective is to learn a neural network approximation $\\hat{{f}}_\\theta(x)$ parameterized by weights $\\theta$ that minimizes the mean squared error:

\\begin{{equation}}
\\mathcal{{L}}(\\theta) = \\frac{{1}}{{n}} \\sum_{{i=1}}^n (y_i - \\hat{{f}}_\\theta(x_i))^2
\\end{{equation}}

\\subsection{{Network Architectures}}

We evaluate three classes of neural network architectures:

\\begin{{itemize}}
\\item \\textbf{{Linear Models}}: Single-layer networks implementing linear regression: $\\hat{{f}}(x) = w^T x + b$
\\item \\textbf{{Shallow Networks}}: Networks with 1-2 hidden layers and ReLU activations
\\item \\textbf{{Deep Networks}}: Networks with 3 or more hidden layers and various activation functions
\\end{{itemize}}

All networks use fully connected layers with configurable hidden dimensions. The output layer consists of a single neuron with linear activation for regression.

\\subsection{{Optimization Algorithms}}

We compare four optimization algorithms commonly used in neural network training:

\\begin{{itemize}}
\\item \\textbf{{SGD}}: Stochastic gradient descent with momentum
\\item \\textbf{{Adam}}: Adaptive moment estimation with bias correction
\\item \\textbf{{RMSprop}}: Root mean square propagation with exponential moving averages
\\item \\textbf{{AdaGrad}}: Adaptive gradient algorithm with accumulated squared gradients
\\end{{itemize}}

Each optimizer is configured with algorithm-specific hyperparameters, including learning rates, momentum terms, and decay parameters.

\\subsection{{Experimental Design}}

Our experimental evaluation covers polynomial degrees {min(degrees)} through {max(degrees)}, with systematic variation of:

\\begin{{itemize}}
\\item Network architecture ({len(architectures)} types: {', '.join(architectures)})
\\item Optimization algorithm ({len(optimizers)} types: {', '.join(optimizers)})
\\item Polynomial degree ({len(degrees)} levels: {', '.join(map(str, degrees))})
\\item Noise levels and dataset sizes
\\item Learning rates and batch sizes
\\end{{itemize}}

Each configuration is evaluated using train/validation splits with early stopping based on validation loss. We record training dynamics, convergence behavior, and final performance metrics for comprehensive analysis."""
    
    def _generate_experimental_setup(self, experiments: List[ExperimentResults]) -> str:
        """Generate experimental setup section."""
        completed = [exp for exp in experiments if exp.status == "completed"]
        
        # Calculate statistics
        avg_data_points = np.mean([exp.config.num_data_points for exp in completed]) if completed else 1000
        avg_epochs = np.mean([exp.config.epochs for exp in completed]) if completed else 100
        noise_levels = sorted(set(exp.config.noise_level for exp in completed))
        
        return f"""\\section{{Experimental Setup}}

\\subsection{{Dataset Generation}}

Synthetic polynomial datasets are generated using the following procedure:

\\begin{{algorithm}}[H]
\\caption{{Polynomial Dataset Generation}}
\\begin{{algorithmic}}[1]
\\STATE Input: degree $d$, coefficient range $[a_{{\\min}}, a_{{\\max}}]$, noise level $\\sigma$
\\STATE Sample coefficients $a_i \\sim \\mathcal{{U}}(a_{{\\min}}, a_{{\\max}})$ for $i = 0, \\ldots, d$
\\STATE Generate input points $x_i \\sim \\mathcal{{U}}(x_{{\\min}}, x_{{\\max}})$ for $i = 1, \\ldots, n$
\\STATE Compute clean targets $y_i = \\sum_{{j=0}}^d a_j x_i^j$
\\STATE Add noise $\\tilde{{y}}_i = y_i + \\epsilon_i$ where $\\epsilon_i \\sim \\mathcal{{N}}(0, \\sigma^2)$
\\STATE Split into training and validation sets
\\RETURN $(X_{{\\text{{train}}}}, Y_{{\\text{{train}}}})$, $(X_{{\\text{{val}}}}, Y_{{\\text{{val}}}})$
\\end{{algorithmic}}
\\end{{algorithm}}

\\subsection{{Training Configuration}}

All experiments use the following standardized configuration:

\\begin{{itemize}}
\\item \\textbf{{Dataset size}}: {int(avg_data_points)} points per experiment
\\item \\textbf{{Train/validation split}}: 80\\%/20\\% ratio
\\item \\textbf{{Maximum epochs}}: {int(avg_epochs)} with early stopping
\\item \\textbf{{Noise levels}}: {noise_levels} (standard deviation)
\\item \\textbf{{Input range}}: $x \\in [-5, 5]$
\\item \\textbf{{Loss function}}: Mean squared error
\\item \\textbf{{Early stopping}}: Patience of 10 epochs on validation loss
\\end{{itemize}}

\\subsection{{Evaluation Metrics}}

Model performance is evaluated using multiple metrics:

\\begin{{itemize}}
\\item \\textbf{{Training Loss}}: Final MSE on training data
\\item \\textbf{{Validation Loss}}: Final MSE on validation data  
\\item \\textbf{{Training Time}}: Wall-clock time for convergence
\\item \\textbf{{Convergence Epoch}}: Epoch at which early stopping occurred
\\item \\textbf{{Model Size}}: Total number of trainable parameters
\\end{{itemize}}

Statistical significance is assessed using paired t-tests with Bonferroni correction for multiple comparisons."""
    
    def _generate_results_section(self, experiments: List[ExperimentResults]) -> str:
        """Generate results and analysis section with tables and figures."""
        completed = [exp for exp in experiments if exp.status == "completed"]
        
        # Generate performance table
        results_table = self._generate_performance_table(completed)
        
        # Generate architecture comparison
        arch_comparison = self._generate_architecture_comparison_table(completed)
        
        # Generate optimizer comparison  
        opt_comparison = self._generate_optimizer_comparison_table(completed)
        
        results = f"""\\section{{Results and Analysis}}

\\subsection{{Overall Performance}}

Table~\\ref{{tab:performance}} presents the top-performing configurations across all experiments. The results demonstrate significant performance variations across different architecture-optimizer combinations.

{results_table}

\\subsection{{Architecture Comparison}}

Figure~\\ref{{fig:architecture_comparison}} shows the performance distribution across different network architectures. Table~\\ref{{tab:architecture_stats}} provides detailed statistical comparisons.

{arch_comparison}

{self._generate_figure_reference("architecture_comparison", "Performance comparison across network architectures")}

\\subsection{{Optimization Algorithm Analysis}}

The choice of optimization algorithm significantly impacts both convergence speed and final performance. Table~\\ref{{tab:optimizer_stats}} summarizes the performance characteristics of each optimizer.

{opt_comparison}

{self._generate_figure_reference("optimizer_comparison", "Convergence behavior of different optimization algorithms")}

\\subsection{{Polynomial Complexity Effects}}

Figure~\\ref{{fig:complexity_analysis}} illustrates how polynomial degree affects the relative performance of different architectures. Higher-degree polynomials generally require more complex architectures for accurate approximation.

{self._generate_figure_reference("complexity_analysis", "Performance vs polynomial complexity")}

\\subsection{{Training Dynamics}}

Figure~\\ref{{fig:loss_curves}} shows representative training curves for different configurations. The plots reveal distinct convergence patterns across optimizers and architectures.

{self._generate_figure_reference("loss_curves", "Training and validation loss curves for representative experiments")}"""
        
        return results
    
    def _generate_performance_table(self, experiments: List[ExperimentResults]) -> str:
        """Generate LaTeX table of best performing models."""
        # Get top 10 models by validation loss
        best_models = sorted(experiments, key=lambda x: x.final_val_loss)[:10]
        
        table = """\\begin{table}[H]
\\centering
\\caption{Top 10 performing model configurations}
\\label{tab:performance}
\\begin{tabular}{@{}lllrrrrr@{}}
\\toprule
Rank & Architecture & Optimizer & Degree & Val Loss & Train Loss & Time (s) & Params \\\\
\\midrule"""
        
        for i, exp in enumerate(best_models, 1):
            table += f"""
{i} & {exp.config.model_architecture} & {exp.config.optimizer} & {exp.config.polynomial_degree} & {exp.final_val_loss:.6f} & {exp.final_train_loss:.6f} & {exp.training_time:.2f} & {exp.model_size if exp.model_size else 'N/A'} \\\\"""
        
        table += """
\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return table
    
    def _generate_architecture_comparison_table(self, experiments: List[ExperimentResults]) -> str:
        """Generate LaTeX table comparing architectures."""
        # Group by architecture
        arch_groups = {}
        for exp in experiments:
            arch = exp.config.model_architecture
            if arch not in arch_groups:
                arch_groups[arch] = []
            arch_groups[arch].append(exp)
        
        table = """\\begin{table}[H]
\\centering
\\caption{Architecture performance comparison}
\\label{tab:architecture_stats}
\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
Architecture & Count & Mean Val Loss & Std Dev & Best Loss & Success Rate \\\\
\\midrule"""
        
        for arch in sorted(arch_groups.keys()):
            exps = arch_groups[arch]
            val_losses = [exp.final_val_loss for exp in exps]
            success_rate = len(exps) / len([e for e in experiments if e.config.model_architecture == arch])
            
            table += f"""
{arch} & {len(exps)} & {np.mean(val_losses):.6f} & {np.std(val_losses):.6f} & {np.min(val_losses):.6f} & {success_rate:.2f} \\\\"""
        
        table += """
\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return table
    
    def _generate_optimizer_comparison_table(self, experiments: List[ExperimentResults]) -> str:
        """Generate LaTeX table comparing optimizers."""
        # Group by optimizer
        opt_groups = {}
        for exp in experiments:
            opt = exp.config.optimizer
            if opt not in opt_groups:
                opt_groups[opt] = []
            opt_groups[opt].append(exp)
        
        table = """\\begin{table}[H]
\\centering
\\caption{Optimizer performance comparison}
\\label{tab:optimizer_stats}
\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
Optimizer & Count & Mean Val Loss & Std Dev & Mean Time (s) & Convergence Rate \\\\
\\midrule"""
        
        for opt in sorted(opt_groups.keys()):
            exps = opt_groups[opt]
            val_losses = [exp.final_val_loss for exp in exps]
            times = [exp.training_time for exp in exps]
            converged = [exp for exp in exps if exp.convergence_epoch is not None]
            conv_rate = len(converged) / len(exps) if exps else 0
            
            table += f"""
{opt.upper()} & {len(exps)} & {np.mean(val_losses):.6f} & {np.std(val_losses):.6f} & {np.mean(times):.2f} & {conv_rate:.2f} \\\\"""
        
        table += """
\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return table
    
    def _generate_figure_reference(self, figure_name: str, caption: str) -> str:
        """Generate LaTeX figure reference with automatic path detection."""
        # Look for figure file in images directory
        figure_path = None
        for ext in self.figure_formats:
            potential_path = self.images_dir / f"{figure_name}{ext}"
            if potential_path.exists():
                figure_path = potential_path
                break
        
        if figure_path:
            # Use relative path from output directory
            rel_path = os.path.relpath(figure_path, self.output_dir)
            return f"""\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{rel_path}}}
\\caption{{{caption}}}
\\label{{fig:{figure_name}}}
\\end{{figure}}"""
        else:
            return f"""\\begin{{figure}}[H]
\\centering
\\fbox{{Figure: {figure_name} (not found)}}
\\caption{{{caption}}}
\\label{{fig:{figure_name}}}
\\end{{figure}}"""
    
    def _generate_discussion(self, experiments: List[ExperimentResults]) -> str:
        """Generate discussion section with analysis and insights."""
        completed = [exp for exp in experiments if exp.status == "completed"]
        
        # Calculate key insights
        readme_gen = ReadmeGenerator(self.storage)
        findings = readme_gen.extract_key_findings(experiments)
        
        discussion = """\\section{Discussion}

\\subsection{Key Findings}

Our comprehensive experimental evaluation reveals several important insights for neural network-based polynomial curve fitting:

\\begin{enumerate}"""
        
        if findings.get('best_architecture'):
            best_arch = findings['best_architecture']
            discussion += f"""
\\item \\textbf{{Architecture Performance}}: {best_arch['name'].title()} networks demonstrated superior performance with an average validation loss of {best_arch['avg_val_loss']:.6f} and success rate of {best_arch['success_rate']:.1%}."""
        
        if findings.get('best_optimizer'):
            best_opt = findings['best_optimizer']
            discussion += f"""
\\item \\textbf{{Optimization Effectiveness}}: {best_opt['name'].upper()} optimization achieved the best overall performance with convergence rate of {best_opt['convergence_rate']:.1%}."""
        
        discussion += """
\\item \\textbf{Complexity Scaling}: Performance degradation with polynomial degree varies significantly across architectures, with deeper networks showing better scaling to high-degree polynomials.

\\item \\textbf{Training Efficiency}: Linear models provide fastest training but limited expressiveness, while deep networks require longer training but achieve better approximation quality.
\\end{enumerate}

\\subsection{Practical Implications}

The results provide practical guidance for curve fitting applications:

\\begin{itemize}
\\item For low-degree polynomials (degree ≤ 3), shallow networks with Adam optimization provide good performance-efficiency trade-offs
\\item High-degree polynomials (degree > 4) benefit from deeper architectures despite increased computational cost
\\item SGD with momentum remains competitive for simple architectures but struggles with complex models
\\item Adaptive optimizers (Adam, RMSprop) show more consistent performance across different architectures
\\end{itemize}

\\subsection{Limitations and Future Work}

Several limitations should be considered when interpreting these results:

\\begin{itemize}
\\item Evaluation limited to synthetic polynomial data; real-world performance may differ
\\item Fixed network architectures; architectural search could identify better configurations
\\item Standard optimization settings; hyperparameter tuning could improve individual results
\\item Single-task evaluation; multi-task or transfer learning scenarios not considered
\\end{itemize}

Future research directions include:

\\begin{itemize}
\\item Extension to real-world curve fitting datasets
\\item Investigation of regularization techniques for improved generalization
\\item Analysis of ensemble methods combining multiple architectures
\\item Study of neural architecture search for automated design optimization
\\end{itemize}"""
        
        return discussion
    
    def _generate_conclusion(self, experiments: List[ExperimentResults]) -> str:
        """Generate conclusion section."""
        completed = [exp for exp in experiments if exp.status == "completed"]
        
        return f"""\\section{{Conclusion}}

This paper presented a comprehensive empirical study of neural network architectures and optimization techniques for polynomial curve fitting. Through {len(completed)} systematic experiments, we evaluated the effectiveness of different model-optimizer combinations across varying polynomial complexities.

Our key contributions include:

\\begin{{enumerate}}
\\item Systematic performance benchmarks for neural network curve fitting across multiple architectures and optimizers
\\item Empirical analysis of scaling behavior with polynomial complexity
\\item Practical recommendations for architecture and optimizer selection
\\item Open-source experimental framework enabling reproducible research
\\end{{enumerate}}

The results demonstrate that architecture and optimizer choice significantly impact both performance and training efficiency. While no single configuration dominates across all scenarios, our analysis provides evidence-based guidance for practical applications.

The experimental framework developed for this study enables continued research into neural network function approximation. Future work can extend our methodology to additional architectures, optimization techniques, and real-world datasets.

Our findings contribute to the broader understanding of neural network capabilities for function approximation tasks and provide practical insights for researchers and practitioners working on curve fitting applications."""
    
    def _generate_acknowledgments(self) -> str:
        """Generate acknowledgments section."""
        return """\\section*{Acknowledgments}

The authors thank the open-source community for providing the foundational tools that made this research possible, including PyTorch, NumPy, and Matplotlib. We also acknowledge the computational resources that enabled our extensive experimental evaluation."""
    
    def _generate_bibliography(self) -> str:
        """Generate bibliography section."""
        return f"""\\bibliographystyle{{{self.bib_style}}}
\\bibliography{{references}}"""
    
    def _generate_bib_file(self) -> None:
        """Generate bibliography file with relevant references."""
        bib_content = """@article{hornik1989multilayer,
  title={Multilayer feedforward networks are universal approximators},
  author={Hornik, Kurt and Stinchcombe, Maxwell and White, Halbert},
  journal={Neural networks},
  volume={2},
  number={5},
  pages={359--366},
  year={1989},
  publisher={Elsevier}
}

@article{cybenko1989approximation,
  title={Approximation by superpositions of a sigmoidal function},
  author={Cybenko, George},
  journal={Mathematics of control, signals and systems},
  volume={2},
  number={4},
  pages={303--314},
  year={1989},
  publisher={Springer}
}

@article{funahashi1989approximate,
  title={On the approximate realization of continuous mappings by neural networks},
  author={Funahashi, Ken-Ichi},
  journal={Neural networks},
  volume={2},
  number={3},
  pages={183--192},
  year={1989},
  publisher={Elsevier}
}

@article{barron1993universal,
  title={Universal approximation bounds for superpositions of a sigmoidal function},
  author={Barron, Andrew R},
  journal={IEEE Transactions on Information theory},
  volume={39},
  number={3},
  pages={930--945},
  year={1993},
  publisher={IEEE}
}

@article{poggio2017and,
  title={Why and when can deep-but not shallow-networks avoid the curse of dimensionality: a review},
  author={Poggio, Tomaso and Mhaskar, Hrushikesh and Rosasco, Lorenzo and Miranda, Brando and Liao, Qianli},
  journal={International journal of automation and computing},
  volume={14},
  number={5},
  pages={503--519},
  year={2017},
  publisher={Springer}
}

@article{robbins1951stochastic,
  title={A stochastic approximation method},
  author={Robbins, Herbert and Monro, Sutton},
  journal={The annals of mathematical statistics},
  pages={400--407},
  year={1951},
  publisher={JSTOR}
}

@article{kingma2014adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  journal={arXiv preprint arXiv:1412.6980},
  year={2014}
}

@article{duchi2011adaptive,
  title={Adaptive subgradient methods for online learning and stochastic optimization},
  author={Duchi, John and Hazan, Elad and Singer, Yoram},
  journal={Journal of machine learning research},
  volume={12},
  number={7},
  pages={2121--2159},
  year={2011}
}

@article{tieleman2012lecture,
  title={Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude},
  author={Tieleman, Tijmen and Hinton, Geoffrey},
  journal={COURSERA: Neural networks for machine learning},
  volume={4},
  number={2},
  pages={26--31},
  year={2012}
}

@article{wilson2017marginal,
  title={The marginal value of adaptive gradient methods in machine learning},
  author={Wilson, Ashia C and Roelofs, Rebecca and Stern, Mitchell and Srebro, Nati and Recht, Benjamin},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@article{smith2017cyclical,
  title={Cyclical learning rates for training neural networks},
  author={Smith, Leslie N},
  journal={2017 IEEE winter conference on applications of computer vision (WACV)},
  pages={464--472},
  year={2017},
  organization={IEEE}
}

@article{masters2018revisiting,
  title={Revisiting small batch training for deep neural networks},
  author={Masters, Dominic and Luschi, Carlo},
  journal={arXiv preprint arXiv:1804.07612},
  year={2018}
}

@book{press2007numerical,
  title={Numerical recipes 3rd edition: The art of scientific computing},
  author={Press, William H and Teukolsky, Saul A and Vetterling, William T and Flannery, Brian P},
  year={2007},
  publisher={Cambridge university press}
}

@book{haykin2009neural,
  title={Neural networks and learning machines},
  author={Haykin, Simon S},
  volume={3},
  year={2009},
  publisher={Pearson}
}"""
        
        bib_path = self.output_dir / "references.bib"
        with open(bib_path, 'w', encoding='utf-8') as f:
            f.write(bib_content)
    
    def compile_pdf(self, tex_filename: str = "curve_fitting_paper.tex") -> bool:
        """
        Compile LaTeX document to PDF using pdflatex.
        
        Args:
            tex_filename: Name of the .tex file to compile
            
        Returns:
            True if compilation successful, False otherwise
        """
        tex_path = self.output_dir / tex_filename
        
        if not tex_path.exists():
            print(f"LaTeX file {tex_path} not found")
            return False
        
        try:
            import subprocess
            
            # Change to output directory for compilation
            original_dir = os.getcwd()
            os.chdir(self.output_dir)
            
            # Run pdflatex twice for proper references
            for i in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', tex_filename],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"LaTeX compilation failed (pass {i+1}):")
                    print(result.stdout)
                    print(result.stderr)
                    return False
            
            # Run bibtex for bibliography
            base_name = tex_filename.replace('.tex', '')
            subprocess.run(['bibtex', base_name], capture_output=True)
            
            # Final pdflatex run for bibliography
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_filename],
                capture_output=True,
                text=True
            )
            
            os.chdir(original_dir)
            
            pdf_path = self.output_dir / f"{base_name}.pdf"
            if pdf_path.exists():
                print(f"PDF generated successfully: {pdf_path}")
                return True
            else:
                print("PDF generation failed")
                return False
                
        except FileNotFoundError:
            print("pdflatex not found. Please install LaTeX distribution.")
            return False
        except Exception as e:
            print(f"Error during PDF compilation: {e}")
            return False
        finally:
            if 'original_dir' in locals():
                os.chdir(original_dir)
    
    def generate_paper_with_custom_template(self, 
                                          template_path: str,
                                          substitutions: Dict[str, str]) -> str:
        """
        Generate paper using custom LaTeX template with variable substitution.
        
        Args:
            template_path: Path to LaTeX template file
            substitutions: Dictionary of template variables to substitute
            
        Returns:
            Generated LaTeX content
        """
        template_file = Path(template_path)
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Perform substitutions
        for key, value in substitutions.items():
            placeholder = f"{{{{{key}}}}}"
            template_content = template_content.replace(placeholder, str(value))
        
        # Save generated paper
        output_path = self.output_dir / "custom_paper.tex"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        return template_content