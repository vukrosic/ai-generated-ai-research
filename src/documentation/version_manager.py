"""
Documentation version control and updating system.

This module provides incremental documentation updates, version tracking,
and validation for generated documentation files.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import difflib

try:
    from ..experiments.storage import ExperimentStorage, QueryFilter
    from ..experiments.runner import ExperimentResults
    from .readme_generator import ReadmeGenerator
    from .latex_generator import LatexGenerator
except ImportError:
    from experiments.storage import ExperimentStorage, QueryFilter
    from experiments.runner import ExperimentResults
    from readme_generator import ReadmeGenerator
    from latex_generator import LatexGenerator


@dataclass
class DocumentVersion:
    """Represents a version of a documentation file."""
    
    version: str
    timestamp: datetime
    file_path: str
    content_hash: str
    experiment_count: int
    experiment_ids: List[str]
    changes_summary: str
    file_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'file_path': self.file_path,
            'content_hash': self.content_hash,
            'experiment_count': self.experiment_count,
            'experiment_ids': self.experiment_ids,
            'changes_summary': self.changes_summary,
            'file_size': self.file_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentVersion':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class DocumentationState:
    """Tracks the current state of all documentation files."""
    
    last_update: datetime
    readme_version: Optional[str]
    latex_version: Optional[str]
    total_experiments: int
    last_experiment_id: Optional[str]
    versions: Dict[str, List[DocumentVersion]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'last_update': self.last_update.isoformat(),
            'readme_version': self.readme_version,
            'latex_version': self.latex_version,
            'total_experiments': self.total_experiments,
            'last_experiment_id': self.last_experiment_id,
            'versions': {
                file_type: [v.to_dict() for v in versions]
                for file_type, versions in self.versions.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentationState':
        """Create from dictionary."""
        data = data.copy()
        data['last_update'] = datetime.fromisoformat(data['last_update'])
        data['versions'] = {
            file_type: [DocumentVersion.from_dict(v) for v in versions]
            for file_type, versions in data['versions'].items()
        }
        return cls(**data)


class DocumentationVersionManager:
    """
    Manages documentation versions, incremental updates, and validation.
    
    This class provides comprehensive version control for generated documentation,
    including change tracking, incremental updates, and validation checks.
    """
    
    def __init__(self, 
                 storage: ExperimentStorage,
                 output_dir: str = ".",
                 version_dir: str = ".doc_versions"):
        """
        Initialize the documentation version manager.
        
        Args:
            storage: ExperimentStorage instance for accessing experiment data
            output_dir: Directory containing documentation files
            version_dir: Directory for storing version information
        """
        self.storage = storage
        self.output_dir = Path(output_dir)
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize generators
        self.readme_generator = ReadmeGenerator(storage, str(self.output_dir))
        self.latex_generator = LatexGenerator(storage, str(self.output_dir / "papers"))
        
        # State file path
        self.state_file = self.version_dir / "documentation_state.json"
        
        # Load or initialize state
        self.state = self._load_state()
    
    def _load_state(self) -> DocumentationState:
        """Load documentation state from file or create new state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return DocumentationState.from_dict(data)
            except Exception as e:
                print(f"Warning: Could not load documentation state: {e}")
        
        # Create new state
        return DocumentationState(
            last_update=datetime.now(),
            readme_version=None,
            latex_version=None,
            total_experiments=0,
            last_experiment_id=None,
            versions={}
        )
    
    def _save_state(self) -> None:
        """Save current documentation state to file."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save documentation state: {e}")
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _generate_version_string(self, file_type: str) -> str:
        """Generate version string for a file type."""
        existing_versions = self.state.versions.get(file_type, [])
        if not existing_versions:
            return "1.0.0"
        
        # Parse latest version and increment
        latest_version = existing_versions[-1].version
        try:
            major, minor, patch = map(int, latest_version.split('.'))
            return f"{major}.{minor}.{patch + 1}"
        except ValueError:
            # Fallback if version format is unexpected
            return f"1.0.{len(existing_versions)}"
    
    def _detect_changes(self, 
                       old_content: str, 
                       new_content: str) -> Tuple[bool, str]:
        """
        Detect changes between old and new content.
        
        Args:
            old_content: Previous content
            new_content: New content
            
        Returns:
            Tuple of (has_changes, changes_summary)
        """
        if old_content == new_content:
            return False, "No changes"
        
        # Calculate basic statistics
        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')
        
        # Use difflib to find differences
        diff = list(difflib.unified_diff(
            old_lines, new_lines, 
            fromfile='old', tofile='new', 
            lineterm='', n=0
        ))
        
        # Count additions and deletions
        additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        
        # Generate summary
        changes = []
        if additions > 0:
            changes.append(f"{additions} additions")
        if deletions > 0:
            changes.append(f"{deletions} deletions")
        
        if not changes:
            changes.append("content modified")
        
        return True, ", ".join(changes)
    
    def check_for_updates(self) -> Dict[str, bool]:
        """
        Check if documentation needs updating based on new experiments.
        
        Returns:
            Dictionary indicating which documents need updates
        """
        # Get current experiments
        current_experiments = self.storage.query_experiments()
        current_count = len(current_experiments)
        
        # Check if we have new experiments
        needs_update = {
            'readme': False,
            'latex': False,
            'reason': []
        }
        
        if current_count > self.state.total_experiments:
            needs_update['readme'] = True
            needs_update['latex'] = True
            needs_update['reason'].append(f"New experiments: {current_count - self.state.total_experiments}")
        
        # Check if latest experiment is different
        if current_experiments:
            latest_exp_id = current_experiments[0].experiment_id
            if latest_exp_id != self.state.last_experiment_id:
                needs_update['readme'] = True
                needs_update['latex'] = True
                needs_update['reason'].append("Latest experiment changed")
        
        return needs_update
    
    def update_documentation(self, 
                           force_update: bool = False,
                           update_readme: bool = True,
                           update_latex: bool = True) -> Dict[str, Any]:
        """
        Update documentation with version tracking.
        
        Args:
            force_update: Force update even if no changes detected
            update_readme: Whether to update README
            update_latex: Whether to update LaTeX paper
            
        Returns:
            Dictionary with update results
        """
        results = {
            'readme': {'updated': False, 'version': None, 'changes': None},
            'latex': {'updated': False, 'version': None, 'changes': None},
            'timestamp': datetime.now().isoformat()
        }
        
        # Get current experiments
        current_experiments = self.storage.query_experiments()
        
        if not current_experiments and not force_update:
            results['message'] = "No experiments found, skipping update"
            return results
        
        # Update README
        if update_readme:
            readme_result = self._update_readme(current_experiments, force_update)
            results['readme'] = readme_result
        
        # Update LaTeX paper
        if update_latex:
            latex_result = self._update_latex(current_experiments, force_update)
            results['latex'] = latex_result
        
        # Update state
        self.state.last_update = datetime.now()
        self.state.total_experiments = len(current_experiments)
        if current_experiments:
            self.state.last_experiment_id = current_experiments[0].experiment_id
        
        self._save_state()
        
        return results
    
    def _update_readme(self, 
                      experiments: List[ExperimentResults], 
                      force_update: bool) -> Dict[str, Any]:
        """Update README with version tracking."""
        readme_path = self.output_dir / "README.md"
        
        # Generate new content
        new_content = self.readme_generator.generate_readme()
        
        # Check for existing content
        old_content = ""
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                old_content = f.read()
        
        # Detect changes
        has_changes, changes_summary = self._detect_changes(old_content, new_content)
        
        if not has_changes and not force_update:
            return {
                'updated': False,
                'version': self.state.readme_version,
                'changes': 'No changes detected'
            }
        
        # Create new version
        version = self._generate_version_string('readme')
        content_hash = self._calculate_content_hash(new_content)
        
        # Save content
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # Create version record
        doc_version = DocumentVersion(
            version=version,
            timestamp=datetime.now(),
            file_path=str(readme_path),
            content_hash=content_hash,
            experiment_count=len(experiments),
            experiment_ids=[exp.experiment_id for exp in experiments[:10]],  # Limit to first 10
            changes_summary=changes_summary,
            file_size=len(new_content.encode('utf-8'))
        )
        
        # Update state
        if 'readme' not in self.state.versions:
            self.state.versions['readme'] = []
        self.state.versions['readme'].append(doc_version)
        self.state.readme_version = version
        
        # Save version backup
        self._save_version_backup('readme', version, new_content)
        
        return {
            'updated': True,
            'version': version,
            'changes': changes_summary,
            'file_size': doc_version.file_size,
            'experiment_count': len(experiments)
        }
    
    def _update_latex(self, 
                     experiments: List[ExperimentResults], 
                     force_update: bool) -> Dict[str, Any]:
        """Update LaTeX paper with version tracking."""
        latex_path = self.output_dir / "papers" / "curve_fitting_paper.tex"
        
        # Generate new content
        new_content = self.latex_generator.generate_paper()
        
        # Check for existing content
        old_content = ""
        if latex_path.exists():
            with open(latex_path, 'r', encoding='utf-8') as f:
                old_content = f.read()
        
        # Detect changes
        has_changes, changes_summary = self._detect_changes(old_content, new_content)
        
        if not has_changes and not force_update:
            return {
                'updated': False,
                'version': self.state.latex_version,
                'changes': 'No changes detected'
            }
        
        # Create new version
        version = self._generate_version_string('latex')
        content_hash = self._calculate_content_hash(new_content)
        
        # Content is already saved by latex_generator.generate_paper()
        
        # Create version record
        doc_version = DocumentVersion(
            version=version,
            timestamp=datetime.now(),
            file_path=str(latex_path),
            content_hash=content_hash,
            experiment_count=len(experiments),
            experiment_ids=[exp.experiment_id for exp in experiments[:10]],
            changes_summary=changes_summary,
            file_size=len(new_content.encode('utf-8'))
        )
        
        # Update state
        if 'latex' not in self.state.versions:
            self.state.versions['latex'] = []
        self.state.versions['latex'].append(doc_version)
        self.state.latex_version = version
        
        # Save version backup
        self._save_version_backup('latex', version, new_content)
        
        return {
            'updated': True,
            'version': version,
            'changes': changes_summary,
            'file_size': doc_version.file_size,
            'experiment_count': len(experiments)
        }
    
    def _save_version_backup(self, file_type: str, version: str, content: str) -> None:
        """Save a backup of the versioned content."""
        backup_dir = self.version_dir / file_type
        backup_dir.mkdir(exist_ok=True)
        
        backup_file = backup_dir / f"{file_type}_v{version}.{'tex' if file_type == 'latex' else 'md'}"
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Warning: Could not save version backup: {e}")
    
    def get_version_history(self, file_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get version history for documentation files.
        
        Args:
            file_type: Specific file type ('readme' or 'latex'), or None for all
            
        Returns:
            Dictionary with version history
        """
        if file_type:
            if file_type in self.state.versions:
                return {file_type: [v.to_dict() for v in self.state.versions[file_type]]}
            else:
                return {file_type: []}
        else:
            return {
                file_type: [v.to_dict() for v in versions]
                for file_type, versions in self.state.versions.items()
            }
    
    def rollback_to_version(self, file_type: str, version: str) -> bool:
        """
        Rollback a documentation file to a specific version.
        
        Args:
            file_type: Type of file ('readme' or 'latex')
            version: Version to rollback to
            
        Returns:
            True if successful, False otherwise
        """
        if file_type not in self.state.versions:
            print(f"No version history found for {file_type}")
            return False
        
        # Find the version
        target_version = None
        for v in self.state.versions[file_type]:
            if v.version == version:
                target_version = v
                break
        
        if not target_version:
            print(f"Version {version} not found for {file_type}")
            return False
        
        # Load backup content
        backup_dir = self.version_dir / file_type
        backup_file = backup_dir / f"{file_type}_v{version}.{'tex' if file_type == 'latex' else 'md'}"
        
        if not backup_file.exists():
            print(f"Backup file not found: {backup_file}")
            return False
        
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Restore to current location
            if file_type == 'readme':
                current_path = self.output_dir / "README.md"
            else:  # latex
                current_path = self.output_dir / "papers" / "curve_fitting_paper.tex"
            
            with open(current_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Successfully rolled back {file_type} to version {version}")
            return True
            
        except Exception as e:
            print(f"Error during rollback: {e}")
            return False
    
    def validate_documentation(self) -> Dict[str, Any]:
        """
        Validate current documentation files for consistency and correctness.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'readme': {'valid': True, 'issues': []},
            'latex': {'valid': True, 'issues': []},
            'overall': {'valid': True, 'summary': []}
        }
        
        # Validate README
        readme_path = self.output_dir / "README.md"
        if readme_path.exists():
            readme_issues = self._validate_readme(readme_path)
            validation_results['readme']['issues'] = readme_issues
            validation_results['readme']['valid'] = len(readme_issues) == 0
        else:
            validation_results['readme']['valid'] = False
            validation_results['readme']['issues'] = ['README.md file not found']
        
        # Validate LaTeX
        latex_path = self.output_dir / "papers" / "curve_fitting_paper.tex"
        if latex_path.exists():
            latex_issues = self._validate_latex(latex_path)
            validation_results['latex']['issues'] = latex_issues
            validation_results['latex']['valid'] = len(latex_issues) == 0
        else:
            validation_results['latex']['valid'] = False
            validation_results['latex']['issues'] = ['LaTeX file not found']
        
        # Overall validation
        overall_valid = validation_results['readme']['valid'] and validation_results['latex']['valid']
        validation_results['overall']['valid'] = overall_valid
        
        if not overall_valid:
            total_issues = len(validation_results['readme']['issues']) + len(validation_results['latex']['issues'])
            validation_results['overall']['summary'] = [f"Found {total_issues} validation issues"]
        
        return validation_results
    
    def _validate_readme(self, readme_path: Path) -> List[str]:
        """Validate README file for common issues."""
        issues = []
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic structure
            required_sections = ['# AI Curve Fitting Research', '## Overview', '## Results']
            for section in required_sections:
                if section not in content:
                    issues.append(f"Missing required section: {section}")
            
            # Check for broken image links
            import re
            image_links = re.findall(r'!\[.*?\]\((.*?)\)', content)
            for link in image_links:
                if not link.startswith('http'):  # Local file
                    image_path = self.output_dir / link
                    if not image_path.exists():
                        issues.append(f"Broken image link: {link}")
            
            # Check for empty sections
            if '## Results' in content:
                results_section = content.split('## Results')[1].split('##')[0]
                if len(results_section.strip()) < 100:
                    issues.append("Results section appears to be empty or too short")
            
        except Exception as e:
            issues.append(f"Error reading README file: {e}")
        
        return issues
    
    def _validate_latex(self, latex_path: Path) -> List[str]:
        """Validate LaTeX file for common issues."""
        issues = []
        
        try:
            with open(latex_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic LaTeX structure
            required_elements = [
                '\\documentclass',
                '\\begin{document}',
                '\\end{document}',
                '\\title{',
                '\\author{',
                '\\maketitle'
            ]
            
            for element in required_elements:
                if element not in content:
                    issues.append(f"Missing required LaTeX element: {element}")
            
            # Check for unmatched braces (basic check)
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces != close_braces:
                issues.append(f"Unmatched braces: {open_braces} open, {close_braces} close")
            
            # Check for figure references
            import re
            fig_refs = re.findall(r'\\includegraphics.*?\{(.*?)\}', content)
            for fig_path in fig_refs:
                # Check if figure file exists (relative to latex file)
                full_path = latex_path.parent / fig_path
                if not full_path.exists():
                    issues.append(f"Missing figure file: {fig_path}")
            
            # Check for bibliography
            if '\\bibliography{' in content:
                bib_match = re.search(r'\\bibliography\{(.*?)\}', content)
                if bib_match:
                    bib_file = bib_match.group(1) + '.bib'
                    bib_path = latex_path.parent / bib_file
                    if not bib_path.exists():
                        issues.append(f"Missing bibliography file: {bib_file}")
            
        except Exception as e:
            issues.append(f"Error reading LaTeX file: {e}")
        
        return issues
    
    def generate_change_log(self, file_type: Optional[str] = None) -> str:
        """
        Generate a change log for documentation versions.
        
        Args:
            file_type: Specific file type or None for all
            
        Returns:
            Formatted change log as string
        """
        changelog = "# Documentation Change Log\n\n"
        changelog += f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        file_types = [file_type] if file_type else list(self.state.versions.keys())
        
        for ftype in file_types:
            if ftype not in self.state.versions:
                continue
            
            changelog += f"## {ftype.title()} Documentation\n\n"
            
            versions = sorted(self.state.versions[ftype], 
                            key=lambda x: x.timestamp, reverse=True)
            
            for version in versions:
                changelog += f"### Version {version.version}\n"
                changelog += f"- **Date**: {version.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                changelog += f"- **Experiments**: {version.experiment_count}\n"
                changelog += f"- **Changes**: {version.changes_summary}\n"
                changelog += f"- **File Size**: {version.file_size:,} bytes\n\n"
        
        return changelog
    
    def cleanup_old_versions(self, keep_versions: int = 10) -> Dict[str, int]:
        """
        Clean up old version backups, keeping only the most recent versions.
        
        Args:
            keep_versions: Number of versions to keep for each file type
            
        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {}
        
        for file_type in self.state.versions.keys():
            versions = self.state.versions[file_type]
            
            if len(versions) <= keep_versions:
                cleanup_stats[file_type] = 0
                continue
            
            # Sort by timestamp and keep only recent versions
            sorted_versions = sorted(versions, key=lambda x: x.timestamp, reverse=True)
            versions_to_remove = sorted_versions[keep_versions:]
            
            # Remove backup files
            backup_dir = self.version_dir / file_type
            removed_count = 0
            
            for version in versions_to_remove:
                backup_file = backup_dir / f"{file_type}_v{version.version}.{'tex' if file_type == 'latex' else 'md'}"
                if backup_file.exists():
                    try:
                        backup_file.unlink()
                        removed_count += 1
                    except Exception as e:
                        print(f"Warning: Could not remove {backup_file}: {e}")
            
            # Update state to keep only recent versions
            self.state.versions[file_type] = sorted_versions[:keep_versions]
            cleanup_stats[file_type] = removed_count
        
        # Save updated state
        self._save_state()
        
        return cleanup_stats