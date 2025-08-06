#!/usr/bin/env python3
"""
Explore non-linear relationships between compression ratio and effect sizes.
Based on the observation that effect sizes might follow an inverted-U pattern.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Define mechanism categories
TRUE_MECHANISMS = ['mi', 'gppm', 'tvd_mi']
EVALUATION_METRICS = ['baseline', 'llm_judge_with', 'llm_judge_without']

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

def fit_linear_and_quadratic(compressions, effect_sizes):
    """Fit both linear and quadratic models, return statistics."""
    X = np.array(compressions).reshape(-1, 1)
    y = np.array(effect_sizes)

    # Linear model
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    y_pred_linear = linear_model.predict(X)
    r2_linear = r2_score(y, y_pred_linear)

    # Quadratic model
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    quad_model = LinearRegression()
    quad_model.fit(X_poly, y)
    y_pred_quad = quad_model.predict(X_poly)
    r2_quad = r2_score(y, y_pred_quad)

    # Get quadratic coefficients (a*x^2 + b*x + c)
    # X_poly columns are: [1, x, x^2]
    c, b, a = quad_model.coef_[0], quad_model.coef_[1], quad_model.coef_[2]

    # Find vertex of parabola (maximum point if a < 0)
    if a != 0:
        vertex_x = -b / (2 * a)
        vertex_y = quad_model.predict(poly.transform([[vertex_x]]))[0]
    else:
        vertex_x, vertex_y = None, None

    # Statistical test for quadratic term
    # Use F-test to compare nested models
    rss_linear = np.sum((y - y_pred_linear) ** 2)
    rss_quad = np.sum((y - y_pred_quad) ** 2)

    n = len(y)
    f_stat = ((rss_linear - rss_quad) / 1) / (rss_quad / (n - 3))
    p_value = 1 - stats.f.cdf(f_stat, 1, n - 3)

    return {
        'linear': {
            'r2': r2_linear,
            'slope': linear_model.coef_[0],
            'intercept': linear_model.intercept_,
            'predictions': y_pred_linear
        },
        'quadratic': {
            'r2': r2_quad,
            'a': a,
            'b': b,
            'c': c,
            'vertex_x': vertex_x,
            'vertex_y': vertex_y,
            'predictions': y_pred_quad,
            'p_value_vs_linear': p_value,
            'significant': p_value < 0.05
        },
        'X_poly': X_poly,
        'poly': poly,
        'quad_model': quad_model
    }

def plot_compression_relationships(results, output_dir):
    """Create comprehensive plots of compression vs effect size relationships."""

    # Create figure with subplots for each mechanism
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    all_mechanisms = TRUE_MECHANISMS + EVALUATION_METRICS
    model_results = {}

    for idx, mechanism in enumerate(all_mechanisms):
        ax = axes[idx]

        # Collect data
        compressions = []
        effect_sizes = []
        domains = []

        for result in results:
            if mechanism in result['stats_results'] and result.get('compression_ratio'):
                compressions.append(result['compression_ratio'])
                effect_sizes.append(abs(result['stats_results'][mechanism]['cohens_d']))
                domains.append(result.get('display_name', result['dataset_name']))

        if len(compressions) > 3:
            # Fit models
            models = fit_linear_and_quadratic(compressions, effect_sizes)
            model_results[mechanism] = models

            # Sort for plotting
            sort_idx = np.argsort(compressions)
            X_sorted = np.array(compressions)[sort_idx]

            # Plot data points
            scatter = ax.scatter(compressions, effect_sizes, s=100, alpha=0.6, c='blue')

            # Plot linear fit
            ax.plot(X_sorted, models['linear']['predictions'][sort_idx], 
                   'g--', linewidth=2, label=f"Linear (R²={models['linear']['r2']:.3f})")

            # Plot quadratic fit
            X_fine = np.linspace(min(compressions), max(compressions), 100).reshape(-1, 1)
            X_poly_fine = models['poly'].transform(X_fine)
            y_quad_fine = models['quad_model'].predict(X_poly_fine)

            quad_label = f"Quadratic (R²={models['quadratic']['r2']:.3f})"
            if models['quadratic']['significant']:
                quad_label += "*"
            ax.plot(X_fine.flatten(), y_quad_fine, 'r-', linewidth=2, label=quad_label)

            # Mark vertex if it exists and is within data range
            if (models['quadratic']['vertex_x'] is not None and 
                min(compressions) <= models['quadratic']['vertex_x'] <= max(compressions)):
                ax.plot(models['quadratic']['vertex_x'], models['quadratic']['vertex_y'], 
                       'ro', markersize=10, label=f"Peak at {models['quadratic']['vertex_x']:.1f}:1")

            # Add domain labels for outliers
            for i, (x, y, domain) in enumerate(zip(compressions, effect_sizes, domains)):
                if abs(y - np.mean(effect_sizes)) > 1.5 * np.std(effect_sizes):
                    ax.annotate(domain[:10], (x, y), fontsize=8, alpha=0.7)

            ax.set_xlabel('Compression Ratio')
            ax.set_ylabel('Effect Size |d|')
            ax.set_title(f'{MECHANISM_NAMES[mechanism]}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Linear vs Quadratic Relationships: Compression Ratio and Effect Size', fontsize=16)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'compression_nonlinear_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return model_results

def create_summary_table(model_results, output_dir):
    """Create a summary table of linear vs quadratic model fits."""

    summary_data = []

    for mechanism in TRUE_MECHANISMS + EVALUATION_METRICS:
        if mechanism in model_results:
            models = model_results[mechanism]

            row = {
                'Mechanism': MECHANISM_NAMES[mechanism],
                'Linear R²': f"{models['linear']['r2']:.3f}",
                'Quad R²': f"{models['quadratic']['r2']:.3f}",
                'R² Improvement': f"{models['quadratic']['r2'] - models['linear']['r2']:.3f}",
                'Quad Significant': '✓' if models['quadratic']['significant'] else '✗',
                'p-value': f"{models['quadratic']['p_value_vs_linear']:.3f}",
                'Peak Location': f"{models['quadratic']['vertex_x']:.1f}" if models['quadratic']['vertex_x'] else 'N/A',
                'Inverted-U': '✓' if models['quadratic']['a'] < 0 else '✗'
            }
            summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Save as CSV
    csv_path = output_dir / 'model_comparison_summary.csv'
    df.to_csv(csv_path, index=False)

    # Create nice formatted text summary
    summary_text = [
        "MODEL COMPARISON SUMMARY",
        "=" * 80,
        "",
        df.to_string(index=False),
        "",
        "=" * 80,
        "Legend:",
        "  - Quad Significant: ✓ if quadratic term significantly improves fit (p < 0.05)",
        "  - Inverted-U: ✓ if quadratic coefficient is negative (effect peaks at middle compression)",
        "  - Peak Location: Compression ratio where effect size is maximized (if inverted-U)",
        ""
    ]

    text_path = output_dir / 'model_comparison_summary.txt'
    with open(text_path, 'w') as f:
        f.write('\n'.join(summary_text))

    # Print summary
    print('\n'.join(summary_text))

    return df

def analyze_inverted_u_hypothesis(model_results, results):
    """Specifically test the inverted-U hypothesis for true mechanisms."""

    print("\n" + "="*80)
    print("INVERTED-U HYPOTHESIS TEST")
    print("="*80)

    for mechanism in TRUE_MECHANISMS:
        if mechanism not in model_results:
            continue

        models = model_results[mechanism]

        print(f"\n{MECHANISM_NAMES[mechanism]}:")
        print("-" * 40)

        # Check if quadratic is better than linear
        r2_improvement = models['quadratic']['r2'] - models['linear']['r2']
        print(f"  R² improvement: {r2_improvement:.3f}")
        print(f"  Quadratic term significant: {models['quadratic']['significant']}")
        print(f"  Quadratic coefficient (a): {models['quadratic']['a']:.4f}")

        # Check for inverted-U shape
        if models['quadratic']['a'] < 0:
            print(f"  ✓ Shows inverted-U pattern")
            if models['quadratic']['vertex_x']:
                print(f"  Peak at compression ratio: {models['quadratic']['vertex_x']:.1f}:1")

                # Find which domains are near the peak
                peak_domains = []
                for result in results:
                    if mechanism in result['stats_results'] and result.get('compression_ratio'):
                        comp = result['compression_ratio']
                        if abs(comp - models['quadratic']['vertex_x']) < 2:
                            peak_domains.append((result.get('display_name', result['dataset_name']), comp))

                if peak_domains:
                    print(f"  Domains near peak:")
                    for domain, comp in sorted(peak_domains, key=lambda x: x[1]):
                        print(f"    - {domain} ({comp:.1f}:1)")
        else:
            print(f"  ✗ Does not show inverted-U pattern")
            print(f"  Pattern appears to be: {'increasing' if models['quadratic']['a'] > 0 else 'linear'}")

def main():
    parser = argparse.ArgumentParser(description='Explore non-linear compression relationships')
    parser.add_argument('--results-file', type=str, 
                        default='aggregated_results/all_domains_results.json',
                        help='Path to aggregated results JSON file')
    parser.add_argument('--output-dir', type=str, default='nonlinear_analysis',
                        help='Output directory for figures and results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_aggregated_results(args.results_file)
    print(f"Loaded {len(results)} domain results")

    # Create visualizations
    print("\nCreating compression relationship plots...")
    model_results = plot_compression_relationships(results, output_dir)

    # Create summary table
    print("\nGenerating summary table...")
    summary_df = create_summary_table(model_results, output_dir)

    # Test inverted-U hypothesis specifically
    analyze_inverted_u_hypothesis(model_results, results)

    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nKey files:")
    print(f"  - {output_dir}/compression_nonlinear_analysis.png")
    print(f"  - {output_dir}/model_comparison_summary.csv")
    print(f"  - {output_dir}/model_comparison_summary.txt")

if __name__ == "__main__":
    main()