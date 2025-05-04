#!/usr/bin/env python3
"""
Sequential Implementations of Computationally Intensive Tasks

This module provides standard sequential implementations of:
1. Matrix Multiplication
2. Prime Number Generation
3. Monte Carlo Pi Estimation (Numerical Integration)

These implementations serve as a baseline for comparison with parallel versions.
"""

import numpy as np
import time
import random
import math

# ===== Matrix Multiplication =====

def sequential_matrix_multiply(A, B):
    """
    Perform matrix multiplication of A and B using standard sequential algorithm.
    
    Args:
        A (numpy.ndarray): First matrix
        B (numpy.ndarray): Second matrix
    
    Returns:
        numpy.ndarray: Result of A * B
        float: Execution time in seconds
    """
    # Check if matrices can be multiplied
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Matrix dimensions don't match for multiplication: {A.shape} and {B.shape}")
    
    # Initialize result matrix
    C = np.zeros((A.shape[0], B.shape[1]))
    
    # Start timing
    start_time = time.time()
    
    # Perform multiplication
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return C, execution_time

def numpy_matrix_multiply(A, B):
    """
    Perform matrix multiplication using NumPy's optimized implementation.
    This serves as a baseline for comparison.
    
    Args:
        A (numpy.ndarray): First matrix
        B (numpy.ndarray): Second matrix
    
    Returns:
        numpy.ndarray: Result of A * B
        float: Execution time in seconds
    """
    # Start timing
    start_time = time.time()
    
    # Use NumPy's optimized matrix multiplication
    C = np.matmul(A, B)
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return C, execution_time

def generate_random_matrix(rows, cols, min_val=0, max_val=10):
    """
    Generate a random matrix of specified dimensions.
    
    Args:
        rows (int): Number of rows
        cols (int): Number of columns
        min_val (int): Minimum value for random elements
        max_val (int): Maximum value for random elements
        
    Returns:
        numpy.ndarray: Random matrix
    """
    return np.random.randint(min_val, max_val, size=(rows, cols))


# ===== Prime Number Generation =====

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

def sequential_find_primes(start, end):
    """
    Find all prime numbers in the range [start, end] using a sequential approach.
    
    Args:
        start (int): Start of range (inclusive)
        end (int): End of range (inclusive)
    
    Returns:
        list: List of prime numbers in the range
        float: Execution time in seconds
    """
    # Start timing
    start_time = time.time()
    
    # Find prime numbers
    primes = []
    for num in range(start, end + 1):
        if is_prime(num):
            primes.append(num)
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return primes, execution_time


# ===== Monte Carlo Pi Estimation (Numerical Integration) =====

def sequential_monte_carlo_pi(num_samples):
    """
    Estimate the value of Pi using Monte Carlo method with sequential computation.
    
    This method uses random points within a square to estimate the area of a circle,
    which can be used to calculate Pi.
    
    Args:
        num_samples (int): Number of random samples to use
    
    Returns:
        float: Estimated value of Pi
        float: Execution time in seconds
    """
    # Start timing
    start_time = time.time()
    
    # Count points inside the circle
    inside_circle = 0
    
    for _ in range(num_samples):
        # Generate random point within the unit square
        x = random.random()
        y = random.random()
        
        # Check if the point is inside the circle (distance from origin <= 1)
        if x*x + y*y <= 1:
            inside_circle += 1
    
    # Estimate Pi: (points inside circle / total points) * 4
    estimated_pi = (inside_circle / num_samples) * 4
    
    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    
    return estimated_pi, execution_time


# Test functions if run directly
if __name__ == "__main__":
    # Test matrix multiplication
    print("Testing Matrix Multiplication:")
    A = generate_random_matrix(200, 200)
    B = generate_random_matrix(200, 200)
    
    result_sequential, time_sequential = sequential_matrix_multiply(A, B)
    result_numpy, time_numpy = numpy_matrix_multiply(A, B)
    
    is_correct = np.allclose(result_sequential, result_numpy)
    print(f"  Sequential algorithm execution time: {time_sequential:.6f} seconds")
    print(f"  NumPy implementation execution time: {time_numpy:.6f} seconds")
    print(f"  Results match: {is_correct}\n")
    
    # Test prime number generation
    print("Testing Prime Number Generation:")
    start, end = 1, 10000
    primes, time_primes = sequential_find_primes(start, end)
    print(f"  Found {len(primes)} primes between {start} and {end}")
    print(f"  Execution time: {time_primes:.6f} seconds\n")
    
    # Test Monte Carlo Pi estimation
    print("Testing Monte Carlo Pi Estimation:")
    samples = 1000000
    pi_estimate, time_pi = sequential_monte_carlo_pi(samples)
    print(f"  Pi estimate with {samples} samples: {pi_estimate}")
    print(f"  True Pi value: {math.pi}")
    print(f"  Error: {abs(pi_estimate - math.pi)}")
    print(f"  Execution time: {time_pi:.6f} seconds")