# Performance Comparison of Sequential vs. Parallel Algorithms

This project implements and analyzes the performance characteristics of sequential and parallel algorithms for three computationally intensive tasks:

1. **Matrix Multiplication**: Comparing naive sequential implementation with parallel row-based distribution
2. **Prime Number Generation**: Finding prime numbers within a range sequentially vs in parallel
3. **Monte Carlo Pi Estimation**: Numerical integration to approximate Pi using random sampling

The project demonstrates speedup, efficiency, and scalability of parallel computing techniques and applies theoretical models (Amdahl's Law and Gustafson's Law) to understand performance improvements.

## Project Structure

```
matrix-multiply-performance-analysis/
├── sequential_algorithms.py   # Sequential implementations of all three algorithms
├── parallel_algorithms.py     # Parallel implementations using multiprocessing
├── performance_analysis.py    # Scripts to run performance tests and collect metrics
├── visualize_performance.py   # Visualization tools for creating performance charts
├── main.py                    # Main script to run the entire analysis
├── results/                   # Generated test results (JSON files)
└── plots/                     # Generated visualization plots
```

## Algorithms Implementation Details

### 1. Matrix Multiplication

#### Sequential Implementation

The sequential matrix multiplication algorithm uses three nested loops with O(n³) time complexity:

```python
def sequential_matrix_multiply(A, B):
    # Initialize result matrix C of size A.rows × B.columns with zeros
    C = np.zeros((A.shape[0], B.shape[1]))

    # Triple nested loop to compute the product
    for i in range(A.shape[0]):        # For each row in A
        for j in range(B.shape[1]):    # For each column in B
            for k in range(A.shape[1]): # For each element in row i of A and column j of B
                C[i, j] += A[i, k] * B[k, j]

    return C
```

#### Parallel Implementation

The parallel implementation divides the work by distributing rows of the result matrix across multiple processes:

1. The input matrix A is divided into chunks based on the number of available processors
2. Each process computes its assigned chunk of rows of the result matrix
3. Results from all processes are combined to form the final matrix

This approach minimizes inter-process communication as each process can compute its assigned rows independently.

#### NumPy Baseline

For comparison, we also use NumPy's highly optimized matrix multiplication which leverages:

- BLAS (Basic Linear Algebra Subprograms) libraries
- Vectorized operations
- Cache-optimized algorithms

### 2. Prime Number Generation

#### Sequential Implementation

The sequential prime number algorithm checks each number in a range:

```python
def sequential_find_primes(start, end):
    primes = []
    for num in range(start, end + 1):
        if is_prime(num):
            primes.append(num)
    return primes

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False

    # Check all potential factors of form 6k±1 up to sqrt(n)
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

#### Parallel Implementation

The parallel implementation:

1. Divides the range of numbers into chunks
2. Assigns each chunk to a separate process
3. Each process identifies prime numbers in its chunk
4. Results from all processes are merged and sorted

This approach is an example of "embarrassingly parallel" computation since each number can be checked independently.

### 3. Monte Carlo Pi Estimation

#### Sequential Implementation

The sequential Monte Carlo Pi estimation uses random sampling:

```python
def sequential_monte_carlo_pi(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        # Generate random point (x,y) in the unit square [0,1] × [0,1]
        x = random.random()
        y = random.random()

        # Check if the point is inside the quarter circle
        if x*x + y*y <= 1:
            inside_circle += 1

    # Calculate Pi estimate: (points inside / total points) * 4
    pi_estimate = (inside_circle / num_samples) * 4
    return pi_estimate
```

#### Parallel Implementation

The parallel implementation:

1. Divides the total number of samples across processes
2. Each process generates its portion of random points and counts those inside the circle
3. The counts from all processes are summed
4. The final Pi estimate is calculated from the total

This algorithm demonstrates how Monte Carlo methods can be effectively parallelized since each sample is independent.

## How to Run the Code

### Prerequisites

First, ensure you have all required dependencies installed:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy matplotlib
```

### Running Performance Tests

```bash
# Set executable permissions if needed
chmod +x main.py

# Run all tests and generate visualizations (takes time but provides complete analysis)
python main.py --run-tests --generate-plots

# Run tests with specific problem sizes (edit performance_analysis.py to customize parameters)
python main.py --run-tests

# Run only specific algorithm tests (faster for testing)
python main.py --run-tests --matrix-only
python main.py --run-tests --primes-only
python main.py --run-tests --pi-only

# Generate visualizations from previous test results without re-running tests
python main.py --generate-plots
```

### Customizing Tests

To modify test parameters (problem sizes, repetitions, etc.), edit the `main.py`:

```python
# Example: Change matrix sizes to test
matrix_sizes = [200, 500, 1000]  # Default: [500, 1000, 1500]

# Example: Change prime number ranges to test
prime_ranges = [(1, 50000), (1, 100000)]  # Default: [(1, 100000), (1, 500000), (1, 1000000)]

# Example: Change Monte Carlo sample sizes
sample_sizes = [1000000, 5000000]  # Default: [1000000, 5000000, 10000000]
```

## Understanding the Output and Analysis Results

### Test Results (JSON files)

After running performance tests, JSON files are generated in the `results/` directory with timestamps, containing:

- Execution times for each algorithm, input size, and process count
- Calculated speedup and efficiency metrics
- System information (available CPU cores)

### Visualization Plots

Running with `--generate-plots` creates visualizations in the `plots/` directory:

#### 1. Execution Time Plots

- **Filename pattern**: `*_execution_times.png`
- **What they show**: Execution time vs. input size for sequential and parallel implementations
- **How to interpret**: Lower execution times are better. Look for downward curves that show the algorithm scales well as the number of processes increases.
- **Log-log scale**: These plots use logarithmic scales for both axes to better visualize large differences.

#### 2. Speedup Plots

- **Filename pattern**: `*_speedup.png`
- **What they show**: Speedup (T_sequential/T_parallel) vs. input size for different process counts
- **How to interpret**:
  - Higher values indicate better parallel performance
  - Compare with the "ideal" dashed lines to see how close the implementation gets to theoretical speedup
  - If speedup increases with problem size, it indicates good scaling characteristics

#### 3. Efficiency Plots

- **Filename pattern**: `*_efficiency.png`
- **What they show**: Efficiency (speedup/number of processes) vs. input size
- **How to interpret**:
  - Values close to 1.0 indicate optimal efficiency
  - Decreasing efficiency with more processes indicates overhead costs
  - Higher efficiency for larger problem sizes shows good scaling behavior

#### 4. Strong Scaling Plots

- **Filename pattern**: `*_strong_scaling.png`
- **What they show**: Speedup vs. number of processes for fixed problem sizes
- **How to interpret**:
  - Linear scaling (following the "ideal" line) indicates perfect parallelization
  - Plateaus indicate diminishing returns from adding more processors
  - Different curves for different problem sizes show how scaling behavior changes

#### 5. Amdahl's Law Analysis

- **Filename pattern**: `*_amdahl_analysis.png`
- **What they show**: Observed speedup vs. theoretical maximum speedup based on estimated serial fraction
- **How to interpret**:
  - The blue curve shows Amdahl's Law prediction based on estimated serial fraction
  - Red points show observed speedups
  - The horizontal line (if present) shows the maximum possible speedup according to Amdahl's Law
  - If observed points follow the curve closely, it validates the serial fraction estimate

#### 6. Gustafson's Law Analysis

- **Filename pattern**: `*_gustafson_analysis.png`
- **What they show**: Scaled speedup for different problem sizes as processes increase
- **How to interpret**:
  - Unlike Amdahl's Law, Gustafson's Law shows how parallelization scales as both problem size and processors increase
  - Multiple curves show different scaled problem sizes
  - Higher curves indicate problems that scale better with increasing processors

#### 7. Algorithm Comparison Summary

- **Filename**: `algorithm_comparison_summary.png`
- **What it shows**: Side-by-side comparison of all algorithms on key metrics
- **How to interpret**:
  - "Maximum Speedup": The highest speedup achieved for each algorithm
  - "Average Efficiency": Overall parallel efficiency across all configurations
  - "Estimated Serial Fraction": Proportion of code that cannot be parallelized (lower is better)
  - "Execution Time Improvement": Overall performance gain from parallelization

## Theoretical Concepts and Interpretation

### Amdahl's Law

Amdahl's Law predicts the theoretical maximum speedup when parallelizing a program:

```
Speedup(N) = 1 / (s + (1-s)/N)
```

Where:

- `s` is the serial fraction (portion that cannot be parallelized)
- `N` is the number of processors

**Key insights**:

- Even a small serial fraction significantly limits maximum possible speedup
- As N approaches infinity, maximum speedup approaches 1/s
- This explains why some algorithms show diminishing returns with more processors

### Gustafson's Law

Gustafson's Law provides an alternative view of scaled speedup:

```
Scaled Speedup(N) = N + (1-N) * s
```

**Key insights**:

- Larger problems can achieve better speedup even with the same serial fraction
- Focuses on how increasing both problem size and processors enables solving larger problems
- Better represents real-world usage where larger resources handle larger problems

### Interpreting Serial Fraction

The estimated serial fraction (from the Amdahl analysis plots) reveals important algorithm characteristics:

- **Low serial fraction (<0.1)**: Highly parallelizable, scales well with more processors
- **Medium serial fraction (0.1-0.5)**: Moderate parallelization potential, diminishing returns beyond 8-16 processors
- **High serial fraction (>0.5)**: Limited parallelization potential, best with few processors

### Performance Metrics Explained

1. **Speedup**: T_sequential / T_parallel

   - Linear speedup (equal to N) is ideal but rarely achieved
   - Sub-linear speedup (less than N) is typical due to overhead
   - Super-linear speedup (greater than N) is possible with cache effects but uncommon

2. **Efficiency**: Speedup / N

   - Values close to 1.0 indicate efficient parallelization
   - Decreasing efficiency with more processors indicates overhead
   - Helps determine the optimal number of processors

3. **Isoefficiency**: How problem size must grow to maintain efficiency
   - Can be estimated from efficiency curves
   - Algorithms with better isoefficiency scale better to more processors

## Expected Results and Interpretations

### Matrix Multiplication

Typically demonstrates:

- Good speedup for large matrices
- Poor speedup for small matrices due to overhead
- Medium serial fraction (~0.1-0.3)
- Dramatically worse performance than optimized libraries like NumPy
- Communication overhead increases with matrix size

### Prime Number Generation

Typically demonstrates:

- Excellent speedup (nearly linear)
- Very low serial fraction (~0.01-0.05)
- Good efficiency even with many processors
- Classic example of "embarrassingly parallel" computation

### Monte Carlo Pi Estimation

Typically demonstrates:

- Excellent speedup (nearly linear)
- Very low serial fraction
- Consistent performance across different sample sizes
- Example of a stochastic algorithm that parallelizes well

## Conclusion

This project demonstrates fundamental concepts in parallel computing by comparing sequential and parallel implementations of three representative algorithms. The visualizations and metrics provide insights into:

1. When parallelization is beneficial and when it's not
2. How different algorithm types respond to parallelization
3. The practical limitations imposed by Amdahl's Law
4. How to measure and interpret parallel performance

The tools and methodology used can be applied to analyze other algorithms and understand their parallelization potential.

## Contribution Guidelines

To contribute to this project:

1. Fork the repository
2. Add new algorithms or improve existing ones
3. Ensure all tests pass by running `python main.py --run-tests`
4. Submit a pull request with a clear description of your changes

We welcome improvements to:

- Algorithm implementations
- Visualization techniques
- Performance metrics
- Documentation and examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.
