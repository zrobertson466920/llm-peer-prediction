#!/usr/bin/env python3
"""
Test pre-registered hypotheses for the peer prediction study.
Reads aggregated results from run_all_binary_analyses.py and tests H1-H3.

IMPORTANT: Only MI, GPPM, and TVD-MI are true mechanisms with incentive guarantees.
Baseline (ROUGE/BLEU) and LLM Judge are evaluation metrics without guarantees.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Define mechanism categories
TRUE_MECHANISMS = ['mi', 'gppm', 'tvd_mi']  # Have incentive guarantees
EVALUATION_METRICS = ['baseline', 'llm_judge_with', 'llm_judge_without']  # No guarantees

MECHANISM_NAMES = {
    'mi': 'MI (DoE)',
    'gppm': 'GPPM',
    'tvd_mi': 'TVD-MI',
    'llm_judge_with': 'LLM Judge (with context)',
    'llm_judge_without': 'LLM Judge (without context)',
    'baseline': 'Baseline (ROUGE/BLEU)'
}

def load_aggregated_results(results_file):
    """Load the aggregated results from all domains."""
    with open(results_file, 'r') as f:
        return json.load(f)

def test_h1a_effect_sizes(results):
    """
    H1a: All mechanisms will distinguish information-degrading (Problematic) 
    from information-preserving (Good Faith) agents with effect size d > 0.5

    NOTE: Only tests TRUE MECHANISMS (MI, GPPM, TVD-MI)
    """
    print("\n" + "="*70)
    print("H1a: Information Preservation Detection (d > 0.5)")
    print("Testing only TRUE MECHANISMS with incentive guarantees")
    print("="*70)

    h1a_results = {}

    # Test true mechanisms
    for mechanism in TRUE_MECHANISMS:
        effect_sizes = []
        ci_lowers = []
        ci_uppers = []
        domains = []

        for result in results:
            if mechanism in result['stats_results']:
                stats_data = result['stats_results'][mechanism]
                d = abs(stats_data['cohens_d'])
                effect_sizes.append(d)

                # Get CI if available
                ci = stats_data.get('cohens_d_ci', [None, None])
                if ci[0] is not None:
                    ci_lowers.append(abs(ci[0]))
                    ci_uppers.append(abs(ci[1]))
                else:
                    ci_lowers.append(d - 0.2)  # Approximate if not available
                    ci_uppers.append(d + 0.2)

                domains.append(result.get('display_name', result['dataset_name']))

        if effect_sizes:
            # Test if all effect sizes > 0.5
            all_above_threshold = all(d > 0.5 for d in effect_sizes)
            min_d = min(effect_sizes)
            max_d = max(effect_sizes)
            mean_d = np.mean(effect_sizes)

            # Count how many are above threshold
            n_above = sum(1 for d in effect_sizes if d > 0.5)

            h1a_results[mechanism] = {
                'pass': all_above_threshold,
                'min_d': min_d,
                'max_d': max_d,
                'mean_d': mean_d,
                'n_domains': len(effect_sizes),
                'n_above_threshold': n_above,
                'effect_sizes': effect_sizes,
                'ci_lowers': ci_lowers,
                'ci_uppers': ci_uppers,
                'domains': domains
            }

            print(f"\n{MECHANISM_NAMES[mechanism]}:")
            print(f"  Hypothesis: {'SUPPORTED' if all_above_threshold else 'NOT SUPPORTED'}")
            print(f"  Domains above d>0.5: {n_above}/{len(effect_sizes)}")
            print(f"  Effect sizes: min={min_d:.3f}, mean={mean_d:.3f}, max={max_d:.3f}")

            if not all_above_threshold:
                # Show which domains failed
                failed_domains = [(domains[i], effect_sizes[i]) 
                                for i in range(len(domains)) 
                                if effect_sizes[i] <= 0.5]
                print(f"  Failed domains:")
                for domain, d in failed_domains:
                    print(f"    - {domain}: d={d:.3f}")

    # Also show evaluation metrics for comparison (but don't count for hypothesis)
    print("\n" + "-"*70)
    print("EVALUATION METRICS (for comparison only, not part of H1a):")
    for metric in EVALUATION_METRICS:
        effect_sizes = []
        for result in results:
            if metric in result['stats_results']:
                effect_sizes.append(abs(result['stats_results'][metric]['cohens_d']))

        if effect_sizes:
            n_above = sum(1 for d in effect_sizes if d > 0.5)
            print(f"{MECHANISM_NAMES.get(metric, metric)}: {n_above}/{len(effect_sizes)} domains above d>0.5")

    # Overall H1a conclusion
    print("\n" + "-"*70)
    fully_supported = [m for m, r in h1a_results.items() if r['pass']]
    print(f"H1a OVERALL: {len(fully_supported)}/{len(TRUE_MECHANISMS)} mechanisms fully support hypothesis")
    print(f"Fully supported mechanisms: {', '.join([MECHANISM_NAMES[m] for m in fully_supported])}")

    return h1a_results

def test_h1b_compression_effects(results):
    """
    H1b: Detection ability will decrease with compression ratio across domains

    NOTE: Only tests TRUE MECHANISMS
    """
    print("\n" + "="*70)
    print("H1b: Compression Ratio Effects")
    print("Testing only TRUE MECHANISMS")
    print("="*70)

    h1b_results = {}

    # Create figure with 2x2 layout (3 mechanisms + 1 summary)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, mechanism in enumerate(TRUE_MECHANISMS):
        compressions = []
        effect_sizes = []

        for result in results:
            if mechanism in result['stats_results'] and result.get('compression_ratio'):
                compressions.append(result['compression_ratio'])
                effect_sizes.append(abs(result['stats_results'][mechanism]['cohens_d']))

        if len(compressions) > 3:  # Need at least 4 points for meaningful regression
            # Linear regression
            X = np.array(compressions).reshape(-1, 1)
            y = np.array(effect_sizes)

            model = LinearRegression()
            model.fit(X, y)

            slope = model.coef_[0]
            r_squared = model.score(X, y)

            # Test if slope is significantly negative
            # Calculate standard error of slope
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            se_slope = np.sqrt(mse / (np.sum((X - X.mean()) ** 2)))
            t_stat = slope / se_slope
            df = len(compressions) - 2
            # Correct two-tailed p-value calculation
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            # Also test for non-linear relationship (log-compression)
            X_log = np.log(X)
            model_log = LinearRegression()
            model_log.fit(X_log, y)
            slope_log = model_log.coef_[0]
            r_squared_log = model_log.score(X_log, y)

            h1b_results[mechanism] = {
                'slope': slope,
                'r_squared': r_squared,
                'p_value': p_value,
                'n_points': len(compressions),
                'significant_negative': slope < 0 and p_value < 0.05,
                'slope_log': slope_log,
                'r_squared_log': r_squared_log
            }

            # Plot
            ax = axes[idx]
            ax.scatter(compressions, effect_sizes, alpha=0.6, s=100)
            ax.plot(X, y_pred, 'r-', linewidth=2, label='Linear')
            
            # Add log fit if it's better
            if r_squared_log > r_squared + 0.1:  # Substantially better
                X_sorted = np.sort(X.flatten())
                y_pred_log = model_log.predict(np.log(X_sorted).reshape(-1, 1))
                ax.plot(X_sorted, y_pred_log, 'g--', linewidth=2, label='Log fit')
                ax.legend()
            
            ax.set_xlabel('Compression Ratio')
            ax.set_ylabel('Effect Size |d|')
            ax.set_title(f'{MECHANISM_NAMES[mechanism]}\nSlope={slope:.3f}, R²={r_squared:.3f}, p={p_value:.3f}')
            ax.grid(True, alpha=0.3)

            print(f"\n{MECHANISM_NAMES[mechanism]}:")
            print(f"  Linear: Slope={slope:.3f}, R²={r_squared:.3f}, p={p_value:.3f}")
            print(f"  Log: Slope={slope_log:.3f}, R²={r_squared_log:.3f}")
            print(f"  Conclusion: {'SUPPORTED' if slope < 0 and p_value < 0.05 else 'NOT SUPPORTED'}")

    # Use last subplot for summary
    ax = axes[3]
    ax.axis('off')
    summary_text = "H1b Summary:\n\n"
    for m in TRUE_MECHANISMS:
        if m in h1b_results:
            r = h1b_results[m]
            summary_text += f"{MECHANISM_NAMES[m]}:\n"
            summary_text += f"  Linear: Slope={r['slope']:.3f}, R²={r['r_squared']:.3f}\n"
            summary_text += f"  Log: R²={r['r_squared_log']:.3f}\n"
            summary_text += f"  Significant: {'Yes' if r['significant_negative'] else 'No'}\n\n"
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    plt.savefig('h1b_compression_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Overall H1b conclusion
    print("\n" + "-"*70)
    supported = [m for m, r in h1b_results.items() if r.get('significant_negative', False)]
    print(f"H1b OVERALL: {len(supported)}/{len(TRUE_MECHANISMS)} mechanisms show significant negative slope")
    print(f"Supported mechanisms: {', '.join([MECHANISM_NAMES[m] for m in supported]) if supported else 'None'}")

    return h1b_results

def test_h1c_tvd_mi_robustness(results, h1b_results):
    """
    H1c: TVD-MI will show the most robust detection across compression levels

    Robustness = maintaining performance across different compression ratios
    """
    print("\n" + "="*70)
    print("H1c: TVD-MI Robustness")
    print("Comparing among TRUE MECHANISMS only")
    print("="*70)
    
    # Group results by compression bins
    compression_bins = [(0, 5), (5, 10), (10, 15), (15, 25)]

    mechanism_stability = {}

    for mechanism in TRUE_MECHANISMS:
        bin_performances = {bin_range: [] for bin_range in compression_bins}

        for result in results:
            if mechanism in result['stats_results'] and result.get('compression_ratio'):
                comp_ratio = result['compression_ratio']
                effect_size = abs(result['stats_results'][mechanism]['cohens_d'])

                # Assign to bin
                for bin_range in compression_bins:
                    if bin_range[0] <= comp_ratio < bin_range[1]:
                        bin_performances[bin_range].append(effect_size)
                        break

        # Calculate stability metrics
        bin_means = [np.mean(perfs) for perfs in bin_performances.values() if perfs]

        if len(bin_means) >= 2:
            # Key metric: coefficient of variation across bins
            cross_bin_cv = np.std(bin_means) / np.mean(bin_means) if np.mean(bin_means) > 0 else float('inf')
            # Alternative: max deviation from overall mean
            all_effects = [e for perfs in bin_performances.values() for e in perfs if perfs]
            if all_effects:
                overall_mean = np.mean(all_effects)
                max_deviation = max(abs(m - overall_mean) / overall_mean for m in bin_means) if overall_mean > 0 else float('inf')
            else:
                max_deviation = float('inf')

            mechanism_stability[mechanism] = {
                'cross_bin_cv': cross_bin_cv,
                'max_deviation': max_deviation,
                'n_bins': len(bin_means),
                'bin_means': bin_means
            }

    # Rank by stability (lower CV = more stable)
    ranked = sorted(mechanism_stability.items(), key=lambda x: x[1]['cross_bin_cv'])
    tvd_mi_rank = next((i for i, (m, _) in enumerate(ranked, 1) if m == 'tvd_mi'), len(ranked))

    print("\nRobustness Metrics Comparison:")
    print("-"*70)
    print(f"{'Mechanism':<15} {'Cross-Bin CV':<12} {'Max Dev':<12} {'# Bins':<8} {'Rank':<8}")
    print("-"*70)
    
    for rank, (mechanism, metrics) in enumerate(ranked, 1):
        print(f"{MECHANISM_NAMES[mechanism]:<15} {metrics['cross_bin_cv']:<12.3f} {metrics['max_deviation']:<12.3f} {metrics['n_bins']:<8} {rank:<8}")
    
    print("\n" + "-"*70)
    print(f"TVD-MI Robustness Rank: #{tvd_mi_rank}/{len(TRUE_MECHANISMS)}")
    print("-"*70)
    print(f"H1c CONCLUSION: {'SUPPORTED' if tvd_mi_rank == 1 else 'NOT SUPPORTED'}")
    print(f"TVD-MI ranks #{tvd_mi_rank} out of {len(TRUE_MECHANISMS)} mechanisms in cross-bin stability")

    return tvd_mi_rank == 1, mechanism_stability

def test_h2a_bounded_consistency(results, variance_results):
    """
    H2a: Mechanisms with bounded outputs (TVD-MI) will show more consistent performance
    
    NOTE: Since we don't have raw scores, we'll use the variance_results from H1c
    which contains CV information across domains.
    """
    print("\n" + "="*70)
    print("H2a: Bounded Output Consistency")
    print("="*70)
    
    # Get CVs from variance results
    bounded_cv = variance_results.get('tvd_mi', {}).get('cross_bin_cv', None)
    unbounded_cvs = []
    
    for mechanism in ['mi', 'gppm']:
        if mechanism in variance_results:
            unbounded_cvs.append(variance_results[mechanism]['cross_bin_cv'])
    
    if bounded_cv is not None and unbounded_cvs:
        mean_unbounded_cv = np.mean(unbounded_cvs)
        
        print(f"Bounded mechanism (TVD-MI): CV = {bounded_cv:.3f}")
        print(f"Unbounded mechanisms (MI, GPPM): mean CV = {mean_unbounded_cv:.3f}")
        
        # H2a is supported if bounded CV is lower (more consistent)
        h2a_supported = bounded_cv < mean_unbounded_cv
        ratio = bounded_cv / mean_unbounded_cv
        
        print(f"Difference: {bounded_cv - mean_unbounded_cv:.3f} ({ratio:.1f}x {'less' if ratio < 1 else 'more'} variable)")
        
        print(f"\nH2a CONCLUSION: {'SUPPORTED' if h2a_supported else 'NOT SUPPORTED'}")
        print(f"TVD-MI (bounded) is {'more' if h2a_supported else 'not more'} consistent than unbounded mechanisms")
        
        return h2a_supported
    else:
        print("Insufficient data for H2a test")
        return False

def test_h2b_log_prob_degradation(results, h1b_results):
    """
    H2b: Log-probability based mechanisms will degrade more in high-compression domains
    
    NOTE: Compares MI/GPPM (log-prob) vs TVD-MI (non-log-prob) only
    """
    print("\n" + "="*70)
    print("H2b: Log-Probability Degradation")
    print("="*70)
    
    # Among true mechanisms: MI and GPPM are log-probability based, TVD-MI is not
    log_prob_mechanisms = ['mi', 'gppm']
    non_log_prob_mechanisms = ['tvd_mi']
    
    # Compare slopes from H1b results
    log_prob_slopes = [h1b_results[m]['slope'] for m in log_prob_mechanisms if m in h1b_results]
    non_log_prob_slopes = [h1b_results[m]['slope'] for m in non_log_prob_mechanisms if m in h1b_results]
    
    if log_prob_slopes and non_log_prob_slopes:
        mean_log_prob_slope = np.mean(log_prob_slopes)
        mean_non_log_prob_slope = np.mean(non_log_prob_slopes)
        
        print(f"Log-probability mechanisms (MI, GPPM):")
        for m in log_prob_mechanisms:
            if m in h1b_results:
                print(f"  {MECHANISM_NAMES[m]}: slope = {h1b_results[m]['slope']:.3f}")
        print(f"  Mean slope: {mean_log_prob_slope:.3f}")
        
        print(f"\nNon-log-probability mechanism (TVD-MI):")
        print(f"  {MECHANISM_NAMES['tvd_mi']}: slope = {mean_non_log_prob_slope:.3f}")
        
        # H2b: Log-prob mechanisms should degrade MORE (have MORE NEGATIVE slopes)
        # If slopes are positive, then LESS POSITIVE means more degradation
        # The key is that degradation = decrease in performance with compression
        
        # Check if log-prob mechanisms degrade more
        if mean_log_prob_slope < 0 and mean_non_log_prob_slope < 0:
            # Both negative: more negative = more degradation
            h2b_supported = mean_log_prob_slope < mean_non_log_prob_slope
        elif mean_log_prob_slope > 0 and mean_non_log_prob_slope > 0:
            # Both positive: less positive = more degradation
            h2b_supported = mean_log_prob_slope < mean_non_log_prob_slope
        else:
            # Mixed signs: negative slope shows degradation
            h2b_supported = mean_log_prob_slope < mean_non_log_prob_slope
        
        # Calculate difference
        slope_diff = mean_log_prob_slope - mean_non_log_prob_slope
        print(f"\nSlope difference: {slope_diff:.3f}")
        
        if h2b_supported:
            print("Log-prob mechanisms show more degradation (lower slope)")
        else:
            print("Log-prob mechanisms show less degradation (higher slope)")
        
    else:
        h2b_supported = False
        print("Insufficient data for comparison")
    
    print(f"\nH2b CONCLUSION: {'SUPPORTED' if h2b_supported else 'NOT SUPPORTED'}")
    print(f"Log-prob mechanisms {'do' if h2b_supported else 'do not'} degrade more with compression")
    
    return h2b_supported

def create_summary_visualization(results, h1a_results, h1b_results, variance_results):
    """Create a comprehensive visualization summarizing all hypothesis tests."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. H1a: Effect sizes by domain
    ax1 = fig.add_subplot(gs[0, :])
    mechanisms = TRUE_MECHANISMS
    domains = []
    
    # Get unique domains
    for result in results:
        domain = result.get('display_name', result['dataset_name'])
        if domain not in domains:
            domains.append(domain)
    
    # Sort domains by compression ratio
    domain_compressions = {}
    for result in results:
        domain = result.get('display_name', result['dataset_name'])
        if result.get('compression_ratio'):
            domain_compressions[domain] = result['compression_ratio']
    
    domains.sort(key=lambda x: domain_compressions.get(x, 0))
    
    # Plot effect sizes
    x = np.arange(len(domains))
    width = 0.25
    
    for i, mechanism in enumerate(mechanisms):
        effect_sizes = []
        for domain in domains:
            # Find the result for this domain
            for result in results:
                if result.get('display_name', result['dataset_name']) == domain:
                    if mechanism in result['stats_results']:
                        effect_sizes.append(abs(result['stats_results'][mechanism]['cohens_d']))
                        break
            else:
                effect_sizes.append(0)
        
        ax1.bar(x + i*width, effect_sizes, width, label=MECHANISM_NAMES[mechanism])
    
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='d=0.5 threshold')
    ax1.set_xlabel('Domain (ordered by compression ratio)')
    ax1.set_ylabel('Effect Size |d|')
    ax1.set_title('H1a: Effect Sizes Across Domains')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([d[:15] + '...' if len(d) > 15 else d for d in domains], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 2. H1b: Compression effects scatter
    ax2 = fig.add_subplot(gs[1, :2])
    colors = ['blue', 'orange', 'green']
    for i, mechanism in enumerate(mechanisms):
        compressions = []
        effect_sizes = []
        
        for result in results:
            if mechanism in result['stats_results'] and result.get('compression_ratio'):
                compressions.append(result['compression_ratio'])
                effect_sizes.append(abs(result['stats_results'][mechanism]['cohens_d']))
        
        if compressions:
            ax2.scatter(compressions, effect_sizes, 
                       label=MECHANISM_NAMES[mechanism], 
                       color=colors[i], s=100, alpha=0.6)
            
            # Add regression line
            if mechanism in h1b_results:
                X = np.array(compressions).reshape(-1, 1)
                y_pred = h1b_results[mechanism]['slope'] * X + np.mean(effect_sizes)
                sort_idx = np.argsort(compressions)
                ax2.plot(np.array(compressions)[sort_idx], 
                        y_pred[sort_idx].flatten(), 
                        color=colors[i], linestyle='--', alpha=0.8)
    
    ax2.set_xlabel('Compression Ratio')
    ax2.set_ylabel('Effect Size |d|')
    ax2.set_title('H1b: Compression Effects on Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. H1c & H2a: Consistency comparison
    ax3 = fig.add_subplot(gs[1, 2])
    mechanisms_sorted = sorted(variance_results.items(), key=lambda x: x[1]['cross_bin_cv'])
    names = [MECHANISM_NAMES[m[0]] for m in mechanisms_sorted]
    cvs = [m[1]['cross_bin_cv'] for m in mechanisms_sorted]
    colors_bar = ['green' if m[0] == 'tvd_mi' else 'gray' for m in mechanisms_sorted]

    bars = ax3.bar(names, cvs, color=colors_bar, alpha=0.7)
    ax3.set_ylabel('Cross-Bin CV')
    ax3.set_title('H1c & H2a: Mechanism Consistency')
    ax3.tick_params(axis='x', rotation=45)

    # Add text annotation
    for i, (bar, cv) in enumerate(zip(bars, cvs)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{cv:.2f}', ha='center', va='bottom')
    
    # 4. Summary table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_data = []
    summary_data.append(['Hypothesis', 'Result', 'Details'])
    summary_data.append(['H1a: All mechanisms d > 0.5', 
                        f'{"SUPPORTED" if all(r["pass"] for r in h1a_results.values()) else "NOT SUPPORTED"}',
                        f'{sum(r["pass"] for r in h1a_results.values())}/{len(TRUE_MECHANISMS)} mechanisms pass'])
    
    h1b_supported = sum(1 for r in h1b_results.values() if r.get('significant_negative', False))
    summary_data.append(['H1b: Negative compression effect',
                        f'{"SUPPORTED" if h1b_supported > len(TRUE_MECHANISMS)/2 else "NOT SUPPORTED"}',
                        f'{h1b_supported}/{len(TRUE_MECHANISMS)} show significant negative slope'])
    
    summary_data.append(['H1c: TVD-MI most robust',
                        'See ranking →',
                        f'TVD-MI CV rank: {next(i for i, (m, _) in enumerate(mechanisms_sorted, 1) if m == "tvd_mi")}/{len(TRUE_MECHANISMS)}'])
    
    # Create table
    table = ax4.table(cellText=summary_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Hypothesis Testing Summary: True Mechanisms Only', fontsize=16, fontweight='bold')
    plt.savefig('hypothesis_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_latex_summary(h1a_results, h1b_results, variance_results, h1c_supported, h2a_supported, h2b_supported):
    """Generate LaTeX-formatted summary for paper."""
    
    latex_lines = [
        "% Hypothesis Testing Results",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Summary of Hypothesis Testing Results}",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Hypothesis & Result & Evidence \\\\",
        "\\midrule"
    ]
    
    # H1a
    h1a_pass = sum(r['pass'] for r in h1a_results.values())
    latex_lines.append(
        f"H1a: All mechanisms $d > 0.5$ & "
        f"{'Supported' if h1a_pass == len(TRUE_MECHANISMS) else 'Not Supported'} & "
        f"{h1a_pass}/{len(TRUE_MECHANISMS)} mechanisms \\\\"
    )
    
    # H1b
    h1b_sig = sum(1 for r in h1b_results.values() if r.get('significant_negative', False))
    latex_lines.append(
        f"H1b: Compression degrades detection & "
        f"{'Supported' if h1b_sig > len(TRUE_MECHANISMS)/2 else 'Not Supported'} & "
        f"{h1b_sig}/{len(TRUE_MECHANISMS)} significant \\\\"
    )
    
    # H1c
    latex_lines.append(
        f"H1c: TVD-MI most robust & "
        f"{'Supported' if h1c_supported else 'Not Supported'} & "
        f"Lowest CV among mechanisms \\\\"
    )
    
    # H2a
    latex_lines.append(
        f"H2a: Bounded more consistent & "
        f"{'Supported' if h2a_supported else 'Not Supported'} & "
        f"TVD-MI vs MI/GPPM \\\\"
    )
    
    # H2b
    latex_lines.append(
        f"H2b: Log-prob degrade more & "
        f"{'Supported' if h2b_supported else 'Not Supported'} & "
        f"Slope comparison \\\\"
    )
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    latex_summary = "\n".join(latex_lines)
    
    with open('hypothesis_results_table.tex', 'w') as f:
        f.write(latex_summary)
    
    print("\nLaTeX table saved to hypothesis_results_table.tex")
    
    return latex_summary

def main():
    parser = argparse.ArgumentParser(description='Test pre-registered hypotheses')
    parser.add_argument('--results-file', type=str, 
                        default='aggregated_results/all_domains_results.json',
                        help='Path to aggregated results JSON file')
    parser.add_argument('--output-dir', type=str, default='hypothesis_tests',
                        help='Output directory for figures and results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_aggregated_results(args.results_file)
    print(f"Loaded {len(results)} domain results")

    # Test hypotheses
    print("\n" + "="*70)
    print("TESTING PRE-REGISTERED HYPOTHESES")
    print("Only TRUE MECHANISMS (MI, GPPM, TVD-MI) count for hypothesis testing")
    print("="*70)

    # H1a: Effect sizes > 0.5
    h1a_results = test_h1a_effect_sizes(results)

    # H1b: Compression effects
    h1b_results = test_h1b_compression_effects(results)

    # H1c: TVD-MI robustness
    h1c_supported, variance_results = test_h1c_tvd_mi_robustness(results, h1b_results)

    # H2a: Bounded consistency
    h2a_supported = test_h2a_bounded_consistency(results, variance_results)

    # H2b: Log-prob degradation
    h2b_supported = test_h2b_log_prob_degradation(results, h1b_results)

    # Create summary visualization
    create_summary_visualization(results, h1a_results, h1b_results, variance_results)

    # Generate LaTeX summary
    generate_latex_summary(h1a_results, h1b_results, variance_results, 
                          h1c_supported, h2a_supported, h2b_supported)

    # Final summary
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING COMPLETE")
    print("="*70)
    print(f"Results saved to {args.output_dir}")
    print("\nFinal Summary:")
    print(f"  H1a (d > 0.5): {'SUPPORTED' if all(r['pass'] for r in h1a_results.values()) else 'NOT SUPPORTED'}")
    print(f"  H1b (compression effect): {'SUPPORTED' if sum(1 for r in h1b_results.values() if r.get('significant_negative', False)) > len(TRUE_MECHANISMS)/2 else 'NOT SUPPORTED'}")
    print(f"  H1c (TVD-MI robustness): {'SUPPORTED' if h1c_supported else 'NOT SUPPORTED'}")
    print(f"  H2a (bounded consistency): {'SUPPORTED' if h2a_supported else 'NOT SUPPORTED'}")
    print(f"  H2b (log-prob degradation): {'SUPPORTED' if h2b_supported else 'NOT SUPPORTED'}")

if __name__ == "__main__":
    main()
