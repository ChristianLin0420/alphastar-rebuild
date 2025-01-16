#!/usr/bin/env python3
"""
Script to run all tests for the AlphaStar implementation.
"""

import os
import sys
import pytest
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run AlphaStar tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML coverage report')
    parser.add_argument('--test-path', type=str, default='tests',
                       help='Path to test directory')
    parser.add_argument('--pattern', type=str, default='test_*',
                       help='Pattern to match test files')
    parser.add_argument('--workers', '-n', type=str, default='auto',
                       help='Number of workers for parallel execution (auto or int)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Build pytest arguments
    pytest_args = [args.test_path]
    
    if args.verbose:
        pytest_args.extend(['-v', '--tb=long'])
    
    if args.coverage:
        pytest_args.extend([
            '--cov=models',
            '--cov=train',
            '--cov=utils',
            '--cov-report=term-missing'
        ])
        
        if args.html:
            pytest_args.append('--cov-report=html')
    
    # Add parallel execution only if explicitly requested
    if args.workers and args.workers != '1':
        pytest_args.extend(['-n', args.workers])
    
    # Add test pattern
    if args.pattern:
        pytest_args.extend(['-k', args.pattern])
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main() 