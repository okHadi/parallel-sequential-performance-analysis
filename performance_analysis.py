#!/usr/bin/env python3
"""
Performance Analysis for Sequential vs. Parallel Algorithms

This script runs performance tests for sequential and parallel implementations of:
1. Matrix Multiplication
2. Prime Number Generation
3. Monte Carlo Pi Estimation

It measures execution times for different input sizes and numbers of processes,
analyzing speedup, efficiency, and scalability.
"""

import numpy as np
import time
import multiprocessing as mp
import os
import json
from datetime import datetime

# Import sequential implementations
from sequential_algorithms import (
    sequential_matrix_multiply, 
    numpy_matrix_multiply,
    generate_random_matrix,
    sequential_find_primes,
    sequential_monte_carlo_pi
)

# Import parallel implementations
from parallel_algorithms import (
    parallel_matrix_multiply,
    parallel_find_primes,
    parallel_monte_carlo_pi
)


def test_matrix_multiplication(sizes, num_processes_list, repetitions=3):
    """
    Test matrix multiplication performance for different matrix sizes and process counts.
    
    Args:
        sizes (list): List of matrix sizes to test
        num_processes_list (list): List of process counts to test
        repetitions (int): Number of repetitions for each test
        
    Returns:
        dict: Results containing execution times for different configurations
    """
    results = {
        "sizes": sizes,
        "num_processes": num_processes_list,
        "sequential_times": [],
        "numpy_times": [],
        "parallel_times": {p: [] for p in num_processes_list}
    }
    
    for size in sizes:
        print(f"Testing matrix multiplication with size {size}x{size}")
        
        # Store times for this size
        sequential_times = []
        numpy_times = []
        parallel_times = {p: [] for p in num_processes_list}
        
        for _ in range(repetitions):
            # Generate random matrices
            A = generate_random_matrix(size, size)
            B = generate_random_matrix(size, size)
            
            # Test sequential implementation
            _, time_sequential = sequential_matrix_multiply(A, B)
            sequential_times.append(time_sequential)
            
            # Test NumPy implementation
            _, time_numpy = numpy_matrix_multiply(A, B)
            numpy_times.append(time_numpy)
            
            # Test parallel implementation with different process counts
            for num_processes in num_processes_list:
                _, time_parallel = parallel_matrix_multiply(A, B, num_processes)
                parallel_times[num_processes].append(time_parallel)
        
        # Calculate average times
        results["sequential_times"].append(sum(sequential_times) / repetitions)
        results["numpy_times"].append(sum(numpy_times) / repetitions)
        for num_processes in num_processes_list:
            results["parallel_times"][num_processes].append(
                sum(parallel_times[num_processes]) / repetitions
            )
    
    return results


def test_prime_generation(ranges, num_processes_list, repetitions=3):
    """
    Test prime number generation performance for different ranges and process counts.
    
    Args:
        ranges (list): List of (start, end) ranges to test
        num_processes_list (list): List of process counts to test
        repetitions (int): Number of repetitions for each test
        
    Returns:
        dict: Results containing execution times for different configurations
    """
    results = {
        "ranges": ranges,
        "num_processes": num_processes_list,
        "sequential_times": [],
        "parallel_times": {p: [] for p in num_processes_list},
        "prime_counts": []
    }
    
    for start, end in ranges:
        print(f"Testing prime number generation in range [{start}, {end}]")
        
        # Store times for this range
        sequential_times = []
        parallel_times = {p: [] for p in num_processes_list}
        
        # Store number of primes found
        prime_count = None
        
        for _ in range(repetitions):
            # Test sequential implementation
            primes_sequential, time_sequential = sequential_find_primes(start, end)
            sequential_times.append(time_sequential)
            
            if prime_count is None:
                prime_count = len(primes_sequential)
            
            # Test parallel implementation with different process counts
            for num_processes in num_processes_list:
                primes_parallel, time_parallel = parallel_find_primes(start, end, num_processes)
                parallel_times[num_processes].append(time_parallel)
                
                # Verify correctness (only needed once)
                if _ == 0 and num_processes == num_processes_list[0]:
                    assert len(primes_sequential) == len(primes_parallel), "Parallel implementation found a different number of primes!"
                    assert sorted(primes_sequential) == sorted(primes_parallel), "Parallel implementation found different primes!"
        
        # Calculate average times
        results["sequential_times"].append(sum(sequential_times) / repetitions)
        for num_processes in num_processes_list:
            results["parallel_times"][num_processes].append(
                sum(parallel_times[num_processes]) / repetitions
            )
        
        results["prime_counts"].append(prime_count)
    
    return results


def test_monte_carlo_pi(sample_sizes, num_processes_list, repetitions=3):
    """
    Test Monte Carlo Pi estimation performance for different sample sizes and process counts.
    
    Args:
        sample_sizes (list): List of sample sizes to test
        num_processes_list (list): List of process counts to test
        repetitions (int): Number of repetitions for each test
        
    Returns:
        dict: Results containing execution times for different configurations
    """
    results = {
        "sample_sizes": sample_sizes,
        "num_processes": num_processes_list,
        "sequential_times": [],
        "parallel_times": {p: [] for p in num_processes_list},
        "sequential_errors": [],
        "parallel_errors": {p: [] for p in num_processes_list}
    }
    
    for samples in sample_sizes:
        print(f"Testing Monte Carlo Pi estimation with {samples} samples")
        
        # Store times and errors for this sample size
        sequential_times = []
        sequential_errors = []
        parallel_times = {p: [] for p in num_processes_list}
        parallel_errors = {p: [] for p in num_processes_list}
        
        for _ in range(repetitions):
            # Test sequential implementation
            pi_sequential, time_sequential = sequential_monte_carlo_pi(samples)
            sequential_times.append(time_sequential)
            sequential_errors.append(abs(pi_sequential - np.pi))
            
            # Test parallel implementation with different process counts
            for num_processes in num_processes_list:
                pi_parallel, time_parallel = parallel_monte_carlo_pi(samples, num_processes)
                parallel_times[num_processes].append(time_parallel)
                parallel_errors[num_processes].append(abs(pi_parallel - np.pi))
        
        # Calculate average times and errors
        results["sequential_times"].append(sum(sequential_times) / repetitions)
        results["sequential_errors"].append(sum(sequential_errors) / repetitions)
        
        for num_processes in num_processes_list:
            results["parallel_times"][num_processes].append(
                sum(parallel_times[num_processes]) / repetitions
            )
            results["parallel_errors"][num_processes].append(
                sum(parallel_errors[num_processes]) / repetitions
            )
    
    return results


def calculate_speedup_efficiency(results):
    """
    Calculate speedup and efficiency for the results.
    
    Args:
        results (dict): Dictionary containing execution times
        
    Returns:
        dict: Dictionary with speedup and efficiency calculations
    """
    metrics = {
        "speedup": {},
        "efficiency": {}
    }
    
    # Get the list of process counts
    num_processes_list = results["num_processes"]
    
    # Calculate speedup and efficiency for each process count
    for p in num_processes_list:
        # Calculate speedup: T_sequential / T_parallel
        speedup = [seq / par for seq, par in zip(results["sequential_times"], results["parallel_times"][p])]
        metrics["speedup"][p] = speedup
        
        # Calculate efficiency: speedup / p
        efficiency = [s / p for s in speedup]
        metrics["efficiency"][p] = efficiency
    
    return metrics


def save_results(results, metrics, algorithm_name):
    """
    Save performance test results to a JSON file.
    
    Args:
        results (dict): Test results
        metrics (dict): Calculated metrics (speedup, efficiency)
        algorithm_name (str): Name of the algorithm
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{algorithm_name}_results_{timestamp}.json"
    
    # Combine results and metrics
    data = {
        "results": results,
        "metrics": metrics,
        "timestamp": timestamp,
        "cpu_count": mp.cpu_count()
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename


def run_all_tests():
    """
    Run performance tests for all three algorithms and save results.
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
    print("\n=== Matrix Multiplication Performance Tests ===")
    matrix_sizes = [50, 100, 150]
    matrix_results = test_matrix_multiplication(matrix_sizes, num_processes_list)
    matrix_metrics = calculate_speedup_efficiency(matrix_results)
    matrix_filename = save_results(matrix_results, matrix_metrics, "matrix_multiplication")
    
    # 2. Prime Number Generation tests
    print("\n=== Prime Number Generation Performance Tests ===")
    prime_ranges = [(1, 100000), (1, 500000), (1, 1000000)]
    prime_results = test_prime_generation(prime_ranges, num_processes_list)
    prime_metrics = calculate_speedup_efficiency(prime_results)
    prime_filename = save_results(prime_results, prime_metrics, "prime_generation")
    
    # 3. Monte Carlo Pi Estimation tests
    print("\n=== Monte Carlo Pi Estimation Performance Tests ===")
    sample_sizes = [1000000, 5000000, 10000000]
    pi_results = test_monte_carlo_pi(sample_sizes, num_processes_list)
    pi_metrics = calculate_speedup_efficiency(pi_results)
    pi_filename = save_results(pi_results, pi_metrics, "monte_carlo_pi")
    
    print("\nAll performance tests completed. Results saved to:")
    print(f"- {matrix_filename}")
    print(f"- {prime_filename}")
    print(f"- {pi_filename}")


if __name__ == "__main__":
    run_all_tests()