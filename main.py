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
import multiprocessing as mp
from performance_analysis import (
    test_matrix_multiplication, 
    test_prime_generation, 
    test_monte_carlo_pi,
    calculate_speedup_efficiency,
    save_results
)
from visualize_performance import generate_all_visualizations


def run_selected_tests(run_matrix=True, run_primes=True, run_monte_carlo=True):
    """
    Run performance tests for selected algorithms and save results.
    
    Args:
        run_matrix (bool): Whether to run matrix multiplication tests
        run_primes (bool): Whether to run prime number generation tests
        run_monte_carlo (bool): Whether to run Monte Carlo Pi estimation tests
    """
    # Get available CPU count
    cpu_count = mp.cpu_count()
    
    # Test with different numbers of processes
    num_processes_list = [1, 2, 4]
    if cpu_count > 4:
        num_processes_list.extend([cpu_count // 2, cpu_count])
    # Remove duplicates and sort
    num_processes_list = sorted(list(set(num_processes_list)))
    
    print(f"Running tests with process counts: {num_processes_list}")
    print(f"Available CPU cores: {cpu_count}")
    
    # 1. Matrix Multiplication tests
    if run_matrix:
        print("\n=== Matrix Multiplication Performance Tests ===")
        matrix_sizes = [500, 1000, 1500]
        matrix_results = test_matrix_multiplication(matrix_sizes, num_processes_list)
        matrix_metrics = calculate_speedup_efficiency(matrix_results)
        matrix_filename = save_results(matrix_results, matrix_metrics, "matrix_multiplication")
        print(f"- Matrix multiplication results saved to: {matrix_filename}")
    
    # 2. Prime Number Generation tests
    if run_primes:
        print("\n=== Prime Number Generation Performance Tests ===")
        prime_ranges = [(1, 100000), (1, 500000), (1, 1000000)]
        prime_results = test_prime_generation(prime_ranges, num_processes_list)
        prime_metrics = calculate_speedup_efficiency(prime_results)
        prime_filename = save_results(prime_results, prime_metrics, "prime_generation")
        print(f"- Prime generation results saved to: {prime_filename}")
    
    # 3. Monte Carlo Pi Estimation tests
    if run_monte_carlo:
        print("\n=== Monte Carlo Pi Estimation Performance Tests ===")
        sample_sizes = [1000000, 5000000, 10000000]
        pi_results = test_monte_carlo_pi(sample_sizes, num_processes_list)
        pi_metrics = calculate_speedup_efficiency(pi_results)
        pi_filename = save_results(pi_results, pi_metrics, "monte_carlo_pi")
        print(f"- Monte Carlo Pi results saved to: {pi_filename}")
    
    print("\nSelected performance tests completed.")


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
        # Determine which algorithms to run based on the flags
        run_matrix = not (args.primes_only or args.pi_only) or args.matrix_only
        run_primes = not (args.matrix_only or args.pi_only) or args.primes_only
        run_monte_carlo = not (args.matrix_only or args.primes_only) or args.pi_only
        
        start_time = time.time()
        run_selected_tests(run_matrix, run_primes, run_monte_carlo)
        end_time = time.time()
        print(f"\nPerformance tests completed in {end_time - start_time:.2f} seconds.")
    
    # Generate visualization plots if requested
    if args.generate_plots:
        print("\nGenerating visualization plots...")
        generate_all_visualizations()


if __name__ == "__main__":
    main()