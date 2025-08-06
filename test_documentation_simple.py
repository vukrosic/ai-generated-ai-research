#!/usr/bin/env python3
"""
Simple test for documentation generation functionality.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_basic_functionality():
    """Test basic documentation functionality."""
    print("Testing documentation system...")
    
    try:
        # Test imports
        from experiments.config import ExperimentConfig
        from experiments.storage import ExperimentStorage
        from documentation.readme_generator import ReadmeGenerator
        from documentation.latex_generator import LatexGenerator
        from documentation.version_manager import DocumentationVersionManager
        
        print("✓ All imports successful")
        
        # Test configuration
        config = ExperimentConfig.get_default_config()
        print("✓ Configuration creation successful")
        
        # Test storage
        storage = ExperimentStorage("test_experiments")
        print("✓ Storage initialization successful")
        
        # Test README generator
        readme_gen = ReadmeGenerator(storage, output_dir="test_output")
        print("✓ README generator creation successful")
        
        # Test LaTeX generator
        latex_gen = LatexGenerator(storage, output_dir="test_output/papers")
        print("✓ LaTeX generator creation successful")
        
        # Test version manager
        version_mgr = DocumentationVersionManager(storage, output_dir="test_output")
        print("✓ Version manager creation successful")
        
        # Generate empty documentation (no experiments)
        print("\nGenerating documentation with no experiments...")
        
        readme_content = readme_gen.generate_readme()
        print(f"✓ README generated ({len(readme_content)} characters)")
        
        latex_content = latex_gen.generate_paper()
        print(f"✓ LaTeX paper generated ({len(latex_content)} characters)")
        
        # Test version management
        update_check = version_mgr.check_for_updates()
        print(f"✓ Update check completed: {update_check}")
        
        validation = version_mgr.validate_documentation()
        print(f"✓ Validation completed: overall valid = {validation['overall']['valid']}")
        
        print("\n" + "="*50)
        print("✓ All basic tests passed successfully!")
        
        # Show generated files
        output_dir = Path("test_output")
        if output_dir.exists():
            print(f"\nGenerated files in {output_dir}:")
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(output_dir)
                    size = file_path.stat().st_size
                    print(f"  {rel_path} ({size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)