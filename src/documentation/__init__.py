"""
Documentation generation module for AI curve fitting research.

This module provides automated generation of comprehensive documentation
including README files and LaTeX research papers, with version control
and incremental updating capabilities.
"""

from .readme_generator import ReadmeGenerator
from .latex_generator import LatexGenerator
from .version_manager import DocumentationVersionManager, DocumentVersion, DocumentationState

__all__ = [
    'ReadmeGenerator', 
    'LatexGenerator', 
    'DocumentationVersionManager',
    'DocumentVersion',
    'DocumentationState'
]