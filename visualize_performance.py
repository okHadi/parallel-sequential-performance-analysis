#!/usr/bin/env python3
"""
Visualization Module for Performance Analysis

This module creates visualizations of performance metrics for:
1. Matrix Multiplication
2. Prime Number Generation
3. Monte Carlo Pi Estimation

It generates various charts for speedup, efficiency, and scalability analysis
and creates a comprehensive report using Amdahl's Law and Gustafson's Law.
"""

import os
import json
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import multiprocessing as mp


def plot_execution_times(results, algorithm_name, output_dir="plots"):
    """
    Plot execution times for different input sizes and process counts.
    
    Args:
        results (dict): Results data from performance tests
        algorithm_name (str): Name of the algorithm
        output_dir (str): Directory to save plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine x-axis data depending on algorithm
    if algorithm_name == "matrix_multiplication":
        x_data = results["sizes"]
        x_label = "Matrix Size (n × n)"
        title_input = "Matrix Size"
    elif algorithm_name == "prime_generation":
        x_data = [end for _, end in results["ranges"]]
        x_label = "Range End Value"
        title_input = "Range Size"
    elif algorithm_name == "monte_carlo_pi":
        x_data = results["sample_sizes"]
        x_label = "Number of Samples"
        title_input = "Sample Size"
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.title(f"{algorithm_name.replace('_', ' ').title()} Execution Time vs {title_input}")
    
    # Sequential times
    plt.plot(x_data, results["sequential_times"], marker='o', linewidth=2, markersize=8, label="Sequential")
    
    # NumPy times (for matrix multiplication only)
    if "numpy_times" in results:
        plt.plot(x_data, results["numpy_times"], marker='s', linewidth=2, markersize=8, label="NumPy (Optimized)")
    
    # Parallel times for each process count
    for p in results["num_processes"]:
        # Check if the key exists in the parallel_times dictionary
        if str(p) in results["parallel_times"]:
            plt.plot(x_data, results["parallel_times"][str(p)], marker='*', linewidth=2, markersize=8, 
                     label=f"Parallel ({p} processes)")
        elif p in results["parallel_times"]:
            plt.plot(x_data, results["parallel_times"][p], marker='*', linewidth=2, markersize=8, 
                     label=f"Parallel ({p} processes)")
        else:
            print(f"Warning: No data for process count {p} in {algorithm_name}")
    
    # Configure plot
    plt.xlabel(x_label)
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Use log-log scale for better visualization of large differences
    plt.xscale('log')
    plt.yscale('log')
    
    # Save figure
    plot_filename = f"{output_dir}/{algorithm_name}_execution_times.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Execution time plot saved as {plot_filename}")
    return plot_filename


def plot_speedup(results, metrics, algorithm_name, output_dir="plots"):
    """
    Plot speedup for different input sizes and process counts.
    
    Args:
        results (dict): Results data from performance tests
        metrics (dict): Calculated performance metrics
        algorithm_name (str): Name of the algorithm
        output_dir (str): Directory to save plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine x-axis data depending on algorithm
    if algorithm_name == "matrix_multiplication":
        x_data = results["sizes"]
        x_label = "Matrix Size (n × n)"
        title_input = "Matrix Size"
    elif algorithm_name == "prime_generation":
        x_data = [end for _, end in results["ranges"]]
        x_label = "Range End Value"
        title_input = "Range Size"
    elif algorithm_name == "monte_carlo_pi":
        x_data = results["sample_sizes"]
        x_label = "Number of Samples"
        title_input = "Sample Size"
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.title(f"{algorithm_name.replace('_', ' ').title()} Speedup vs {title_input}")
    
    # Plot ideal speedup for reference
    for p in results["num_processes"]:
        if p > 1:  # Don't plot ideal for p=1 (would just be a straight line at y=1)
            plt.plot(x_data, [p] * len(x_data), linestyle='--', color='gray', alpha=0.5, label=f"Ideal ({p} processes)" if p == results["num_processes"][-1] else "")
    
    # Plot actual speedup for each process count
    for p in results["num_processes"]:
        if p > 1:  # Don't plot speedup for p=1 (would just be a straight line at y=1)
            # Check if the key exists in the metrics dictionary as string or int
            if str(p) in metrics["speedup"]:
                plt.plot(x_data, metrics["speedup"][str(p)], marker='*', linewidth=2, markersize=8, 
                         label=f"Actual ({p} processes)")
            elif p in metrics["speedup"]:
                plt.plot(x_data, metrics["speedup"][p], marker='*', linewidth=2, markersize=8, 
                         label=f"Actual ({p} processes)")
            else:
                print(f"Warning: No speedup data for process count {p} in {algorithm_name}")
    
    # Configure plot
    plt.xlabel(x_label)
    plt.ylabel("Speedup (T_sequential / T_parallel)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    plot_filename = f"{output_dir}/{algorithm_name}_speedup.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Speedup plot saved as {plot_filename}")
    return plot_filename


def plot_efficiency(results, metrics, algorithm_name, output_dir="plots"):
    """
    Plot parallel efficiency for different input sizes and process counts.
    
    Args:
        results (dict): Results data from performance tests
        metrics (dict): Calculated performance metrics
        algorithm_name (str): Name of the algorithm
        output_dir (str): Directory to save plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine x-axis data depending on algorithm
    if algorithm_name == "matrix_multiplication":
        x_data = results["sizes"]
        x_label = "Matrix Size (n × n)"
        title_input = "Matrix Size"
    elif algorithm_name == "prime_generation":
        x_data = [end for _, end in results["ranges"]]
        x_label = "Range End Value"
        title_input = "Range Size"
    elif algorithm_name == "monte_carlo_pi":
        x_data = results["sample_sizes"]
        x_label = "Number of Samples"
        title_input = "Sample Size"
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.title(f"{algorithm_name.replace('_', ' ').title()} Efficiency vs {title_input}")
    
    # Plot ideal efficiency (always 1.0)
    plt.plot(x_data, [1.0] * len(x_data), linestyle='--', color='gray', alpha=0.5, label="Ideal")
    
    # Plot actual efficiency for each process count
    for p in results["num_processes"]:
        if p > 1:  # Don't plot efficiency for p=1 (would just be a straight line at y=1)
            # Check if the key exists in the metrics dictionary as string or int
            if str(p) in metrics["efficiency"]:
                plt.plot(x_data, metrics["efficiency"][str(p)], marker='*', linewidth=2, markersize=8, label=f"{p} processes")
            elif p in metrics["efficiency"]:
                plt.plot(x_data, metrics["efficiency"][p], marker='*', linewidth=2, markersize=8, label=f"{p} processes")
            else:
                print(f"Warning: No efficiency data for process count {p} in {algorithm_name}")
    
    # Configure plot
    plt.xlabel(x_label)
    plt.ylabel("Efficiency (Speedup / Number of Processes)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Use y-axis scale from 0 to 1.2
    plt.ylim(0, 1.2)
    
    # Save figure
    plot_filename = f"{output_dir}/{algorithm_name}_efficiency.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Efficiency plot saved as {plot_filename}")
    return plot_filename


def plot_scalability(results, algorithm_name, output_dir="plots"):
    """
    Plot strong scaling (speedup vs. number of processes) for different input sizes.
    
    Args:
        results (dict): Results data from performance tests
        algorithm_name (str): Name of the algorithm
        output_dir (str): Directory to save plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine input sizes and labels depending on algorithm
    if algorithm_name == "matrix_multiplication":
        input_sizes = results["sizes"]
        input_label = "Matrix Size"
    elif algorithm_name == "prime_generation":
        input_sizes = [end for _, end in results["ranges"]]
        input_label = "Range End"
    elif algorithm_name == "monte_carlo_pi":
        input_sizes = results["sample_sizes"]
        input_label = "Sample Size"
    
    # Get process counts
    process_counts = results["num_processes"]
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.title(f"{algorithm_name.replace('_', ' ').title()} Strong Scaling (Speedup vs. Number of Processes)")
    
    # Plot ideal scaling
    plt.plot(process_counts, process_counts, linestyle='--', color='gray', alpha=0.5, label="Ideal Scaling")
    
    # Calculate and plot actual scaling for each input size
    for i, size in enumerate(input_sizes):
        # Calculate speedup for each process count
        speedup = []
        for p in process_counts:
            # T_sequential / T_parallel for this size and process count
            # Check if the key exists in the parallel_times dictionary
            if str(p) in results["parallel_times"]:
                s = results["sequential_times"][i] / results["parallel_times"][str(p)][i]
            elif p in results["parallel_times"]:
                s = results["sequential_times"][i] / results["parallel_times"][p][i]
            else:
                print(f"Warning: No data for process count {p} in {algorithm_name}")
                s = 1.0  # Default to no speedup if data is missing
            speedup.append(s)
        
        # Plot the speedup
        plt.plot(process_counts, speedup, marker='*', linewidth=2, markersize=8, label=f"{input_label}={size}")
    
    # Configure plot
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup (T_sequential / T_parallel)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    plot_filename = f"{output_dir}/{algorithm_name}_strong_scaling.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Strong scaling plot saved as {plot_filename}")
    return plot_filename


def plot_amdahl_analysis(results, algorithm_name, output_dir="plots"):
    """
    Plot Amdahl's Law analysis for the algorithm.
    
    Args:
        results (dict): Results data from performance tests
        algorithm_name (str): Name of the algorithm
        output_dir (str): Directory to save plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the largest input size for the analysis
    if algorithm_name == "matrix_multiplication":
        input_sizes = results["sizes"]
    elif algorithm_name == "prime_generation":
        input_sizes = [end for _, end in results["ranges"]]
    elif algorithm_name == "monte_carlo_pi":
        input_sizes = results["sample_sizes"]
    
    # Use the largest input size for Amdahl's Law analysis
    largest_idx = input_sizes.index(max(input_sizes))
    
    # Get sequential time and parallel times for the largest input size
    seq_time = results["sequential_times"][largest_idx]
    
    # Handle both string and integer keys for parallel_times
    par_times = {}
    for p in results["num_processes"]:
        if str(p) in results["parallel_times"]:
            par_times[p] = results["parallel_times"][str(p)][largest_idx]
        elif p in results["parallel_times"]:
            par_times[p] = results["parallel_times"][p][largest_idx]
        else:
            print(f"Warning: No data for process count {p} in {algorithm_name}")
            # Use sequential time as fallback (no speedup)
            par_times[p] = seq_time
    
    # Calculate observed speedups
    observed_speedups = {p: seq_time / par_times[p] for p in results["num_processes"]}
    
    # Estimate the serial fraction using Amdahl's Law
    # For each process count p > 1, estimate f from the observed speedup S:
    # S = 1 / (f + (1-f)/p)
    # Solve for f: f = (1 - S/p) / (1 - 1/p)
    serial_fractions = {}
    for p in results["num_processes"]:
        if p > 1:
            S = observed_speedups[p]
            f = (1 - S/p) / (1 - 1/p)
            serial_fractions[p] = max(0, min(1, f))  # Ensure f is between 0 and 1
    
    # Average the estimated serial fractions
    avg_serial_fraction = sum(serial_fractions.values()) / len(serial_fractions) if serial_fractions else 0
    
    # Generate Amdahl's Law curve
    p_values = np.linspace(1, max(results["num_processes"]) * 2, 100)
    amdahl_speedups = [1 / (avg_serial_fraction + (1 - avg_serial_fraction) / p) for p in p_values]
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.title(f"{algorithm_name.replace('_', ' ').title()} - Amdahl's Law Analysis")
    
    # Plot Amdahl's Law prediction
    plt.plot(p_values, amdahl_speedups, linestyle='-', color='blue', 
             label=f"Amdahl's Law (serial fraction ≈ {avg_serial_fraction:.3f})")
    
    # Plot ideal speedup
    plt.plot(p_values, p_values, linestyle='--', color='gray', alpha=0.5, label="Ideal (linear) Speedup")
    
    # Plot observed speedups
    plt.scatter(results["num_processes"], [observed_speedups[p] for p in results["num_processes"]], 
                color='red', marker='o', s=100, label="Observed Speedups")
    
    # Add horizontal line showing maximum possible speedup according to Amdahl's Law
    max_possible = 1 / avg_serial_fraction if avg_serial_fraction > 0 else float('inf')
    if max_possible < 100:  # Only show if it's within a reasonable range
        plt.axhline(y=max_possible, color='red', linestyle='-.', alpha=0.5,
                   label=f"Max Possible Speedup ≈ {max_possible:.2f}")
    
    # Configure plot
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    plot_filename = f"{output_dir}/{algorithm_name}_amdahl_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Amdahl's Law analysis plot saved as {plot_filename}")
    return plot_filename, avg_serial_fraction


def plot_gustafson_analysis(results, algorithm_name, output_dir="plots"):
    """
    Plot Gustafson's Law analysis for weak scaling.
    
    Args:
        results (dict): Results data from performance tests
        algorithm_name (str): Name of the algorithm
        output_dir (str): Directory to save plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # For Gustafson's Law, we focus on how performance scales with problem size
    # Assume the problem size increases linearly with the number of processors
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.title(f"{algorithm_name.replace('_', ' ').title()} - Gustafson's Law Analysis")
    
    # Get the list of process counts and determine input sizes based on algorithm
    process_counts = results["num_processes"]
    
    if algorithm_name == "matrix_multiplication":
        input_sizes = results["sizes"]
        input_label = "Matrix Size"
    elif algorithm_name == "prime_generation":
        input_sizes = [end for _, end in results["ranges"]]
        input_label = "Range End"
    elif algorithm_name == "monte_carlo_pi":
        input_sizes = results["sample_sizes"]
        input_label = "Sample Size"
    
    # For each input size, calculate the scaled speedup
    for i, size in enumerate(input_sizes):
        # Get sequential time for this size
        seq_time = results["sequential_times"][i]
        
        # Calculate scaled speedup for each process count
        # Gustafson's Law: Scaled Speedup = p + (1 - p) * serial_fraction
        gustafson_speedups = []
        observed_speedups = []
        
        for p in process_counts:
            # Get parallel time for this size and process count
            if str(p) in results["parallel_times"]:
                par_time = results["parallel_times"][str(p)][i]
            elif p in results["parallel_times"]:
                par_time = results["parallel_times"][p][i]
            else:
                print(f"Warning: No data for process count {p} in {algorithm_name}")
                par_time = seq_time  # Default to sequential time if data is missing
            
            # Calculate observed speedup
            speedup = seq_time / par_time
            observed_speedups.append(speedup)
            
            # Estimate serial fraction for Gustafson's Law
            # If we have only one data point, we can't estimate accurately
            if p > 1:
                # Solving for serial_fraction from observed speedup:
                # speedup = p + (1 - p) * serial_fraction
                # serial_fraction = (speedup - p) / (1 - p)
                serial_fraction = (speedup - p) / (1 - p)
                # Ensure it's between 0 and 1
                serial_fraction = max(0, min(1, serial_fraction))
                
                # Calculate Gustafson's prediction
                gustafson_speedup = p + (1 - p) * serial_fraction
                gustafson_speedups.append((p, gustafson_speedup, serial_fraction))
        
        # Plot observed speedups
        plt.plot(process_counts, observed_speedups, marker='o', linewidth=2, markersize=8, 
                 label=f"Observed ({input_label}={size})")
    
    # Also plot ideal scaling (linear speedup)
    plt.plot([1, max(process_counts)], [1, max(process_counts)], linestyle='--', 
             color='gray', alpha=0.5, label="Ideal (linear) Scaling")
    
    # Configure plot
    plt.xlabel("Number of Processes (p)")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure
    plot_filename = f"{output_dir}/{algorithm_name}_gustafson_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Gustafson's Law analysis plot saved as {plot_filename}")
    return plot_filename


def create_comparison_chart(algorithms, output_dir="plots"):
    """
    Create a summary comparison chart of all algorithms.
    
    Args:
        algorithms (list): List of algorithm results to compare
        output_dir (str): Directory to save plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up a figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Subplot 1: Maximum speedup comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Maximum Speedup Comparison")
    
    # Subplot 2: Efficiency comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Average Efficiency Comparison")
    
    # Subplot 3: Estimated serial fraction comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Estimated Serial Fraction Comparison")
    
    # Subplot 4: Execution time improvement
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Execution Time Improvement (Sequential vs. Best Parallel)")
    
    # Prepare data for each subplot
    algorithm_names = []
    max_speedups = []
    avg_efficiencies = []
    serial_fractions = []
    time_improvements = []
    
    # Extract data from each algorithm's results
    for alg_name, results, metrics in algorithms:
        algorithm_names.append(alg_name.replace('_', ' ').title())
        
        # Find maximum speedup across all process counts and input sizes
        max_speedup = 0
        for p in results["num_processes"]:
            if p > 1:  # Skip p=1 as speedup would be 1
                # Check if the key exists in metrics dictionary as string or int
                if str(p) in metrics["speedup"]:
                    max_speedup = max(max_speedup, max(metrics["speedup"][str(p)]))
                elif p in metrics["speedup"]:
                    max_speedup = max(max_speedup, max(metrics["speedup"][p]))
        max_speedups.append(max_speedup)
        
        # Calculate average efficiency across all process counts and input sizes
        avg_efficiency = 0
        count = 0
        for p in results["num_processes"]:
            if p > 1:  # Skip p=1 as efficiency would be 1
                # Check if the key exists in metrics dictionary as string or int
                if str(p) in metrics["efficiency"]:
                    avg_efficiency += sum(metrics["efficiency"][str(p)])
                    count += len(metrics["efficiency"][str(p)])
                elif p in metrics["efficiency"]:
                    avg_efficiency += sum(metrics["efficiency"][p])
                    count += len(metrics["efficiency"][p])
        avg_efficiency = avg_efficiency / count if count > 0 else 0
        avg_efficiencies.append(avg_efficiency)
        
        # Use serial fraction from Amdahl's analysis
        # For simplicity, we'll estimate it directly from the maximum input size and maximum process count
        largest_idx = -1  # Use the largest input size
        max_p = max(results["num_processes"])
        if max_p > 1:
            # Safely access parallel times with string or int key
            if str(max_p) in results["parallel_times"]:
                max_p_time = results["parallel_times"][str(max_p)][largest_idx]
            elif max_p in results["parallel_times"]:
                max_p_time = results["parallel_times"][max_p][largest_idx]
            else:
                # Find the highest process count that exists in the data
                available_p = [p for p in results["num_processes"] if p > 1 and 
                              (p in results["parallel_times"] or str(p) in results["parallel_times"])]
                
                if available_p:
                    max_p = max(available_p)
                    if str(max_p) in results["parallel_times"]:
                        max_p_time = results["parallel_times"][str(max_p)][largest_idx]
                    else:
                        max_p_time = results["parallel_times"][max_p][largest_idx]
                else:
                    # No parallel data available
                    max_p_time = results["sequential_times"][largest_idx]
                    
            observed_speedup = results["sequential_times"][largest_idx] / max_p_time
            serial_fraction = (1 - observed_speedup/max_p) / (1 - 1/max_p)
            serial_fraction = max(0, min(1, serial_fraction))  # Ensure it's between 0 and 1
        else:
            serial_fraction = 1.0
        serial_fractions.append(serial_fraction)
        
        # Calculate execution time improvement (sequential vs. best parallel)
        seq_time = results["sequential_times"][largest_idx]
        
        # Find the best parallel time safely
        par_times = []
        for p in results["num_processes"]:
            if p > 1:  # Only consider parallel processes (p > 1)
                if str(p) in results["parallel_times"]:
                    par_times.append(results["parallel_times"][str(p)][largest_idx])
                elif p in results["parallel_times"]:
                    par_times.append(results["parallel_times"][p][largest_idx])
        
        best_par_time = min(par_times) if par_times else seq_time
        time_improvement = seq_time / best_par_time
        time_improvements.append(time_improvement)
    
    # Plot data in each subplot
    # 1. Maximum speedup
    ax1.bar(algorithm_names, max_speedups, color='skyblue')
    ax1.set_ylabel("Maximum Speedup")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(max_speedups):
        ax1.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    # 2. Average efficiency
    ax2.bar(algorithm_names, avg_efficiencies, color='lightgreen')
    ax2.set_ylabel("Average Efficiency")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(avg_efficiencies):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # 3. Serial fraction
    ax3.bar(algorithm_names, serial_fractions, color='salmon')
    ax3.set_ylabel("Estimated Serial Fraction")
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(serial_fractions):
        ax3.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # 4. Execution time improvement
    ax4.bar(algorithm_names, time_improvements, color='violet')
    ax4.set_ylabel("Speedup (Sequential / Best Parallel)")
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(time_improvements):
        ax4.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    # Add a title for the entire figure
    plt.suptitle("Comparative Performance Analysis of Sequential vs. Parallel Algorithms", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
    
    # Save figure
    plot_filename = f"{output_dir}/algorithm_comparison_summary.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    print(f"Comparison chart saved as {plot_filename}")
    return plot_filename


def load_latest_results(pattern):
    """
    Load the most recent results file matching the pattern.
    
    Args:
        pattern (str): Glob pattern to match result files
        
    Returns:
        dict: Loaded results data or None if no files found
    """
    # Find all files matching the pattern
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time (most recent last)
    files.sort(key=os.path.getmtime)
    
    # Load the most recent file
    with open(files[-1], 'r') as f:
        data = json.load(f)
    
    return data


def generate_all_visualizations():
    """
    Generate all performance visualization plots based on the latest results.
    """
    print("Generating performance visualizations...")
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Load the latest results for each algorithm
    matrix_data = load_latest_results("results/matrix_multiplication_*.json")
    prime_data = load_latest_results("results/prime_generation_*.json")
    pi_data = load_latest_results("results/monte_carlo_pi_*.json")
    
    generated_plots = []
    algorithm_data = []
    
    # Generate visualizations for each algorithm
    for algorithm_name, data in [
        ("matrix_multiplication", matrix_data),
        ("prime_generation", prime_data),
        ("monte_carlo_pi", pi_data)
    ]:
        if data:
            results = data["results"]
            metrics = data["metrics"]
            
            # Store data for comparison chart
            algorithm_data.append((algorithm_name, results, metrics))
            
            # Generate plots
            execution_plot = plot_execution_times(results, algorithm_name)
            speedup_plot = plot_speedup(results, metrics, algorithm_name)
            efficiency_plot = plot_efficiency(results, metrics, algorithm_name)
            scalability_plot = plot_scalability(results, algorithm_name)
            amdahl_plot, serial_fraction = plot_amdahl_analysis(results, algorithm_name)
            gustafson_plot = plot_gustafson_analysis(results, algorithm_name)
            
            generated_plots.extend([
                execution_plot, speedup_plot, efficiency_plot, 
                scalability_plot, amdahl_plot, gustafson_plot
            ])
            
            print(f"\nAnalysis for {algorithm_name}:")
            print(f"  Estimated serial fraction: {serial_fraction:.4f}")
            print(f"  Maximum theoretical speedup: {1/serial_fraction if serial_fraction > 0 else 'Infinity':.2f}x")
    
    # Generate comparison chart if we have multiple algorithms
    if len(algorithm_data) > 1:
        comparison_plot = create_comparison_chart(algorithm_data)
        generated_plots.append(comparison_plot)
    
    print("\nAll visualizations generated successfully!")
    return generated_plots


if __name__ == "__main__":
    generate_all_visualizations()