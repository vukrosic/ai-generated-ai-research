#!/usr/bin/env python3
"""
Integration test for the main application.
Tests all major functionality without running full experiments.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, expect_failure=False):
    """Run a command and check its exit code."""
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if expect_failure:
            if result.returncode != 0:
                print("✓ PASSED (expected failure)")
                return True
            else:
                print("✗ FAILED (expected failure but succeeded)")
                return False
        else:
            if result.returncode == 0:
                print("✓ PASSED")
                return True
            else:
                print("✗ FAILED")
                print(f"Exit code: {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return False
            
    except subprocess.TimeoutExpired:
        print("✗ FAILED (timeout)")
        return False
    except Exception as e:
        print(f"✗ FAILED (exception): {e}")
        return False
    finally:
        print("-" * 50)


def main():
    """Run integration tests."""
    print("AI Curve Fitting Research - Integration Tests")
    print("=" * 50)
    
    tests = [
        # Basic help and version tests
        (["python", "main.py", "--help"], "Help display", False),
        
        # Dry run tests
        (["python", "main.py", "--dry-run"], "Basic dry run", False),
        (["python", "main.py", "--dry-run", "--verbose"], "Verbose dry run", False),
        
        # Parameter validation tests
        (["python", "main.py", "--dry-run", "--polynomial-degree", "4", "--optimizer", "sgd"], 
         "Custom parameters dry run", False),
        
        # Config file tests (if configs exist)
        (["python", "main.py", "--config", "configs/experiment_linear_sgd.json", "--dry-run"], 
         "Config file dry run", False),
        
        # Batch run tests (if configs directory exists)
        (["python", "main.py", "--batch-run", "configs", "--dry-run"], 
         "Batch run dry run", False),
        
        # Error handling tests (these should fail)
        (["python", "main.py", "--config", "nonexistent.json", "--dry-run"], 
         "Nonexistent config file (should fail)", True),
        (["python", "main.py", "--batch-run", "nonexistent_dir", "--dry-run"], 
         "Nonexistent batch directory (should fail)", True),
    ]
    
    passed = 0
    total = len(tests)
    
    for cmd, description, expect_failure in tests:
        if run_command(cmd, description, expect_failure):
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())