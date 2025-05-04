#!/usr/bin/env python3
"""
Parallel Implementations of Computationally Intensive Tasks

This module provides parallel implementations using Python's multiprocessing of:
1. Matrix Multiplication
2. Prime Number Generation
3. Monte Carlo Pi Estimation (Numerical Integration)

These implementations demonstrate parallel computing techniques for performance comparison.
"""

import numpy as np
import time
import random
import multiprocessing as mp
from functools import partial
import math


# ===== Matrix Multiplication =====

def _multiply_row_chunk(args):
    """
    Multiply a chunk of rows of matrix A with matrix B.
    Helper function for process-based matrix multiplication.
    
    Args:
        args (tuple): (start_row, end_row, A, B)
            - start_row (int): Starting row index
            - end_row (int): Ending row index (exclusive)
            - A (numpy.ndarray): First matrix
            - B (numpy.ndarray): Second matrix
    
    Returns:
        numpy.ndarray: Partial result matrix
    """
    start_row, end_row, A, B = args
    result = np.zeros((end_row - start_row, B.shape[1]))
    
    for i in range(end_row - start_row):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i + start_row, k] * B[k, j]
    
    return result

def parallel_matrix_multiply(A, B, num_processes=None):
    """
    Perform matrix multiplication using multiprocessing.
    
    Args:
        A (numpy.ndarray): First matrix
        B (numpy.ndarray): Second matrix
        num_processes (int, optional): Number of processes to use. 
                                      If None, uses all available cores.
    
    Returns:
        numpy.ndarray: Result of A * B
        float: Execution time in seconds
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Matrix dimensions don't match for multiplication: {A.shape} and {B.shape}")
    
    # Use all available cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Limit processes to the number of rows in A
    num_processes = min(num_processes, A.shape[0])
    
    # Calculate rows per process and prepare chunks
    chunk_size = A.shape[0] // num_processes
    chunks = []
    for i in range(num_processes):
        start_row = i * chunk_size
        end_row = start_row + chunk_size if i < num_processes - 1 else A.shape[0]
        chunks.append((start_row, end_row, A, B))
    
    # Start timing
    start_time = time.time()
    
    # Create process pool and compute in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(_multiply_row_chunk, chunks)
    
    # Combine results
    C = np.vstack(results)
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return C, execution_time


# ===== Prime Number Generation =====

def _find_primes_chunk(args):
    """
    Find prime numbers in a specific range.
    Helper function for parallel prime finding.
    
    Args:
        args (tuple): (start, end)
            - start (int): Start of range (inclusive)
            - end (int): End of range (inclusive)
    
    Returns:
        list: List of prime numbers in the range
    """
    start, end = args
    
    primes = []
    for num in range(start, end + 1):
        # Check if prime using the function defined in sequential_algorithms.py
        if is_prime(num):
            primes.append(num)
    
    return primes

def is_prime(n):
    """
    Check if a number is prime.
    
    Args:
        n (int): Number to check
    
    Returns:
        bool: True if n is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True

def parallel_find_primes(start, end, num_processes=None):
    """
    Find all prime numbers in the range [start, end] using parallel processing.
    
    Args:
        start (int): Start of range (inclusive)
        end (int): End of range (inclusive)
        num_processes (int, optional): Number of processes to use. 
                                      If None, uses all available cores.
    
    Returns:
        list: List of prime numbers in the range
        float: Execution time in seconds
    """
    # Use all available cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Calculate the chunk size for each process
    total_numbers = end - start + 1
    chunk_size = total_numbers // num_processes
    
    # Prepare chunks
    chunks = []
    for i in range(num_processes):
        chunk_start = start + i * chunk_size
        chunk_end = chunk_start + chunk_size - 1 if i < num_processes - 1 else end
        chunks.append((chunk_start, chunk_end))
    
    # Start timing
    start_time = time.time()
    
    # Create process pool and compute in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(_find_primes_chunk, chunks)
    
    # Combine results
    all_primes = []
    for prime_list in results:
        all_primes.extend(prime_list)
    
    # Sort the combined list (optional, but ensures same output order as sequential)
    all_primes.sort()
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return all_primes, execution_time


# ===== Monte Carlo Pi Estimation (Numerical Integration) =====

def _monte_carlo_sample(num_samples):
    """
    Estimate Pi using a subset of random samples.
    Helper function for parallel Monte Carlo Pi estimation.
    
    Args:
        num_samples (int): Number of random samples to use
    
    Returns:
        int: Number of points inside the circle
    """
    inside_circle = 0
    for _ in range(num_samples):
        # Generate random point within the unit square
        x = random.random()
        y = random.random()
        
        # Check if the point is inside the circle (distance from origin <= 1)
        if x*x + y*y <= 1:
            inside_circle += 1
    
    return inside_circle

def parallel_monte_carlo_pi(num_samples, num_processes=None):
    """
    Estimate the value of Pi using Monte Carlo method with parallel computation.
    
    Args:
        num_samples (int): Total number of random samples to use
        num_processes (int, optional): Number of processes to use. 
                                      If None, uses all available cores.
    
    Returns:
        float: Estimated value of Pi
        float: Execution time in seconds
    """
    # Use all available cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Calculate samples per process
    samples_per_process = num_samples // num_processes
    
    # Prepare arguments for each process
    args = [samples_per_process] * num_processes
    # Distribute any remaining samples to the last process
    if num_samples % num_processes != 0:
        args[-1] += num_samples % num_processes
    
    # Start timing
    start_time = time.time()
    
    # Create process pool and compute in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(_monte_carlo_sample, args)
    
    # Combine results
    total_inside = sum(results)
    total_samples = sum(args)
    
    # Estimate Pi: (points inside circle / total points) * 4
    estimated_pi = (total_inside / total_samples) * 4
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return estimated_pi, execution_time


# Test functions if run directly
if __name__ == "__main__":
    # Import for comparison
    from sequential_algorithms import (
        sequential_matrix_multiply, numpy_matrix_multiply, generate_random_matrix,
        sequential_find_primes, sequential_monte_carlo_pi
    )
    
    print(f"Number of CPU cores available: {mp.cpu_count()}\n")
    
    # Test matrix multiplication
    print("Testing Matrix Multiplication:")
    A = generate_random_matrix(500, 500)
    B = generate_random_matrix(500, 500)
    
    _, time_sequential = sequential_matrix_multiply(A, B)
    result_parallel, time_parallel = parallel_matrix_multiply(A, B)
    result_numpy, time_numpy = numpy_matrix_multiply(A, B)
    
    # Verify correctness
    is_correct = np.allclose(result_parallel, result_numpy)
    speedup = time_sequential / time_parallel
    
    print(f"  Sequential execution time: {time_sequential:.6f} seconds")
    print(f"  Parallel execution time: {time_parallel:.6f} seconds")
    print(f"  NumPy execution time: {time_numpy:.6f} seconds")
    print(f"  Speedup (sequential/parallel): {speedup:.2f}x")
    print(f"  Results match NumPy: {is_correct}\n")
    
    # Test prime number generation
    print("Testing Prime Number Generation:")
    start, end = 1, 100000
    
    _, time_sequential = sequential_find_primes(start, end)
    primes_parallel, time_parallel = parallel_find_primes(start, end)
    
    speedup = time_sequential / time_parallel
    
    print(f"  Sequential execution time: {time_sequential:.6f} seconds")
    print(f"  Parallel execution time: {time_parallel:.6f} seconds")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Found {len(primes_parallel)} primes between {start} and {end}\n")
    
    # Test Monte Carlo Pi estimation
    print("Testing Monte Carlo Pi Estimation:")
    samples = 10000000
    
    pi_sequential, time_sequential = sequential_monte_carlo_pi(samples)
    pi_parallel, time_parallel = parallel_monte_carlo_pi(samples)
    
    speedup = time_sequential / time_parallel
    
    print(f"  Sequential execution time: {time_sequential:.6f} seconds")
    print(f"  Parallel execution time: {time_parallel:.6f} seconds")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Sequential Pi estimate: {pi_sequential}")
    print(f"  Parallel Pi estimate: {pi_parallel}")
    print(f"  True Pi value: {math.pi}")
    print(f"  Sequential error: {abs(pi_sequential - math.pi)}")
    print(f"  Parallel error: {abs(pi_parallel - math.pi)}")