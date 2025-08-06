#!/usr/bin/env python3
"""
Analyze win rate matrices from LLM judge results to verify they sum correctly.
"""

import json
import numpy as np
from pathlib import Path
import argparse

def analyze_win_matrices(json_file):
    """Analyze win matrices from a judge results file."""

    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"\nAnalyzing: {json_file}")
    print("="*60)

    # Extract basic info
    conditions = data['condition_keys']
    n_conditions = len(conditions)
    win_rates = data['win_rates_avg']

    print(f"Number of conditions: {n_conditions}")
    print(f"Number of examples: {data.get('num_examples', 'unknown')}")

    # Check win matrix if available
    if 'win_matrix_avg' in data:
        win_matrix = np.array(data['win_matrix_avg'])
        print(f"\nWin matrix shape: {win_matrix.shape}")

        # Check diagonal (should be 0 or 0.5)
        diagonal = np.diag(win_matrix)
        print(f"Diagonal values (self-comparison): {diagonal[:5]}..." if len(diagonal) > 5 else f"Diagonal: {diagonal}")

        # Check if matrix + transpose = 1 (for off-diagonal)
        # For a proper win matrix, w[i,j] + w[j,i] should equal 1
        asymmetry = win_matrix + win_matrix.T
        off_diagonal_mask = ~np.eye(n_conditions, dtype=bool)
        off_diagonal_sums = asymmetry[off_diagonal_mask]

        print(f"\nMatrix symmetry check (should be ~1.0 for off-diagonal):")
        print(f"  Mean: {np.mean(off_diagonal_sums):.4f}")
        print(f"  Std: {np.std(off_diagonal_sums):.4f}")
        print(f"  Min: {np.min(off_diagonal_sums):.4f}")
        print(f"  Max: {np.max(off_diagonal_sums):.4f}")

        # Sum each row (total wins for each condition)
        row_sums = np.sum(win_matrix, axis=1)
        print(f"\nRow sums (total wins per condition):")
        print(f"  Mean: {np.mean(row_sums):.4f}")
        print(f"  Std: {np.std(row_sums):.4f}")
        print(f"  Min: {np.min(row_sums):.4f}")
        print(f"  Max: {np.max(row_sums):.4f}")
        print(f"  Expected (if no self-play): {n_conditions - 1}")

        # Check total sum
        total_sum = np.sum(win_matrix)
        expected_sum = n_conditions * (n_conditions - 1) / 2  # Each pair plays once
        print(f"\nTotal sum of matrix: {total_sum:.2f}")
        print(f"Expected sum (if each pair plays once): {expected_sum:.2f}")

    # Analyze win rates (derived from matrix)
    print(f"\nWin rates analysis:")
    print(f"  Length: {len(win_rates)}")
    print(f"  Sum: {sum(win_rates):.4f}")
    print(f"  Mean: {np.mean(win_rates):.4f}")
    print(f"  Expected mean (balanced): {(n_conditions - 1) / (2 * (n_conditions - 1)):.4f} = 0.5")

    # Show top and bottom performers
    sorted_indices = np.argsort(win_rates)
    print(f"\nBottom 3 conditions (lowest win rate):")
    for i in sorted_indices[:3]:
        print(f"  {conditions[i]}: {win_rates[i]:.4f}")

    print(f"\nTop 3 conditions (highest win rate):")
    for i in sorted_indices[-3:]:
        print(f"  {conditions[i]}: {win_rates[i]:.4f}")

    # Check for win rate anomalies
    win_rates_array = np.array(win_rates)
    anomalies = np.where((win_rates_array < 0) | (win_rates_array > 1))[0]
    if len(anomalies) > 0:
        print(f"\n⚠️  WARNING: Found {len(anomalies)} win rates outside [0,1]:")
        for idx in anomalies:
            print(f"  {conditions[idx]}: {win_rates[idx]}")

    return {
        'total_sum': np.sum(win_matrix) if 'win_matrix_avg' in data else None,
        'mean_win_rate': np.mean(win_rates),
        'n_conditions': n_conditions
    }

def compare_results(file1, file2):
    """Compare two judge result files."""

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    results1 = analyze_win_matrices(file1)
    results2 = analyze_win_matrices(file2)

    if results1['n_conditions'] != results2['n_conditions']:
        print(f"\n⚠️  WARNING: Different number of conditions! {results1['n_conditions']} vs {results2['n_conditions']}")

    if results1['total_sum'] and results2['total_sum']:
        sum_diff = results2['total_sum'] - results1['total_sum']
        print(f"\nTotal sum difference: {sum_diff:.2f}")
        print(f"  File 1: {results1['total_sum']:.2f}")
        print(f"  File 2: {results2['total_sum']:.2f}")

        if abs(sum_diff) > 0.1:
            print("  ⚠️  Significant difference in total wins!")

    mean_diff = results2['mean_win_rate'] - results1['mean_win_rate']
    print(f"\nMean win rate difference: {mean_diff:.4f}")
    print(f"  File 1: {results1['mean_win_rate']:.4f}")
    print(f"  File 2: {results2['mean_win_rate']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze win rate matrices from LLM judge results')
    parser.add_argument('files', nargs='+', help='Judge result JSON files to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare the first two files')

    args = parser.parse_args()

    # Analyze each file
    for file_path in args.files:
        if Path(file_path).exists():
            analyze_win_matrices(file_path)
        else:
            print(f"File not found: {file_path}")

    # Compare if requested and we have at least 2 files
    if args.compare and len(args.files) >= 2:
        if Path(args.files[0]).exists() and Path(args.files[1]).exists():
            compare_results(args.files[0], args.files[1])

if __name__ == "__main__":
    main()