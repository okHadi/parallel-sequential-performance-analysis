#!/usr/bin/env python3
"""
Main script to run the entire performance analysis system.

This script provides options to:
1. Run all performance tests
2. Generate visualizations from existing results
3. Run specific algorithms only
"""

import os
import argparse
import time
from performance_analysis import run_all_tests
from visualize_performance import generate_all_visualizations


def main():
    """
    Main function to parse command-line arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Performance Comparison of Sequential vs. Parallel Algorithms"
    )
    
    # Add command-line arguments
    parser.add_argument(
        "--run-tests", "-t", action="store_true",
        help="Run performance tests for all algorithms"
    )
    parser.add_argument(
        "--generate-plots", "-p", action="store_true", 
        help="Generate visualization plots from existing results"
    )
    parser.add_argument(
        "--matrix-only", action="store_true",
        help="Run tests only for matrix multiplication"
    )
    parser.add_argument(
        "--primes-only", action="store_true",
        help="Run tests only for prime number generation"
    )
    parser.add_argument(
        "--pi-only", action="store_true",
        help="Run tests only for Monte Carlo Pi estimation"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run performance tests if requested
    if args.run_tests:
        print("Running performance tests...")
        # If specific algorithms are requested, modify run_all_tests behavior
        # For this example, we'll just run all tests
        start_time = time.time()
        run_all_tests()
        end_time = time.time()
        print(f"\nPerformance tests completed in {end_time - start_time:.2f} seconds.")
    
    # Generate visualization plots if requested
    if args.generate_plots:
        print("\nGenerating visualization plots...")
        generate_all_visualizations()


if __name__ == "__main__":
    main()