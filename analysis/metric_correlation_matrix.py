"""
Correlation matrix analysis for all available metrics
Creates heatmaps and identifies significant correlation pairs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from pathlib import Path
import argparse

def load_results(results_dir):
    """Load all validation, mechanism, and LLM judge results from the results directory."""
    results_dir = Path(results_dir)
    datasets = {}

    for validation_file in results_dir.glob("*_validation.json"):
        base_name = validation_file.stem.replace("_validation", "")
        mechanism_file = results_dir / f"{base_name}_mi_gppm.json"
        tvd_mi_file = results_dir / f"{base_name}_tvd_mi.json"
        llm_judge_with_context_file = results_dir / f"{base_name}_judge_with_context.json"
        llm_judge_without_context_file = results_dir / f"{base_name}_judge_without_context.json"

        if mechanism_file.exists():
            with open(validation_file, 'r') as f:
                validation_data = json.load(f)
            with open(mechanism_file, 'r') as f:
                mechanism_data = json.load(f)

            dataset_entry = {
                'validation': validation_data,
                'mechanism': mechanism_data
            }

            if tvd_mi_file.exists():
                with open(tvd_mi_file, 'r') as f:
                    tvd_mi_data = json.load(f)
                dataset_entry['tvd_mi'] = tvd_mi_data

            if llm_judge_with_context_file.exists():
                with open(llm_judge_with_context_file, 'r') as f:
                    llm_judge_with_context_data = json.load(f)
                dataset_entry['llm_judge_with_context'] = llm_judge_with_context_data

            if llm_judge_without_context_file.exists():
                with open(llm_judge_without_context_file, 'r') as f:
                    llm_judge_without_context_data = json.load(f)
                dataset_entry['llm_judge_without_context'] = llm_judge_without_context_data

            datasets[base_name] = dataset_entry
            print(f"Loaded dataset: {base_name}")

    return datasets

def extract_all_metrics(dataset_data):
    """Extract all available metrics into a DataFrame."""
    validation = dataset_data['validation']
    mechanism = dataset_data['mechanism']

    conditions = mechanism['condition_keys']

    # Extract baseline scores
    baseline_scores = {}
    baseline_type = None

    for condition in conditions:
        if condition in validation['baseline_scores']:
            scores = validation['baseline_scores'][condition]
            if 'bleu' in scores:
                baseline_type = 'BLEU'
                baseline_scores[condition] = scores['bleu']['corpus_score']
            elif 'rouge1_f1' in scores:
                baseline_type = 'ROUGE-1'
                baseline_scores[condition] = scores['rouge1_f1']['mean']

    # Build dataframe
    data = []
    for i, condition in enumerate(conditions):
        if condition in baseline_scores and i < len(mechanism.get('response_lengths_avg', [])):
            row = {
                'condition': condition,
                'baseline': baseline_scores[condition],
                'mi': mechanism['combined_avgs_avg'][i],
                'gppm': mechanism['gppm_normalized_avg'][i],
                'length': mechanism['response_lengths_avg'][i]
            }

            # Add TVD-MI if available
            if 'tvd_mi' in dataset_data:
                tvd_mi_data = dataset_data['tvd_mi']
                tvd_mi_scores = tvd_mi_data.get('tvd_mi_bidirectional_avg', tvd_mi_data.get('tvd_mi_scores_avg'))
                if tvd_mi_scores and i < len(tvd_mi_scores):
                    row['tvd_mi'] = tvd_mi_scores[i]

            # Add LLM judge scores if available
            if 'llm_judge_with_context' in dataset_data:
                llm_with_scores = dataset_data['llm_judge_with_context'].get('win_rates_avg')
                if llm_with_scores and i < len(llm_with_scores):
                    row['llm_with_context'] = llm_with_scores[i]

            if 'llm_judge_without_context' in dataset_data:
                llm_without_scores = dataset_data['llm_judge_without_context'].get('win_rates_avg')
                if llm_without_scores and i < len(llm_without_scores):
                    row['llm_without_context'] = llm_without_scores[i]

            data.append(row)

    df = pd.DataFrame(data)

    return df, baseline_type

def create_correlation_matrix(df, baseline_type, dataset_name, output_dir, significance_threshold=0.05):
    """Create correlation matrix heatmap and identify significant pairs."""

    # Define metric columns (exclude 'condition')
    metric_columns = [col for col in df.columns if col != 'condition']

    # Create nice labels
    label_mapping = {
        'baseline': baseline_type,
        'mi': 'MI',
        'gppm': 'GPPM',
        'tvd_mi': 'TVD-MI',
        'llm_with_context': 'LLM-Judge (w/ context)',
        'llm_without_context': 'LLM-Judge (w/o context)',
        'length': 'Response Length'
    }

    # Get available metrics
    available_metrics = [col for col in metric_columns if col in df.columns and not df[col].isna().all()]

    # Create BOTH correlation matrices
    pearson_corr_matrix = df[available_metrics].corr(method='pearson')
    spearman_corr_matrix = df[available_metrics].corr(method='spearman')

    # Create p-value matrices for both
    n_metrics = len(available_metrics)
    pearson_p_values = np.zeros((n_metrics, n_metrics))
    spearman_p_values = np.zeros((n_metrics, n_metrics))

    for i, metric1 in enumerate(available_metrics):
        for j, metric2 in enumerate(available_metrics):
            if i != j:
                try:
                    # Pearson
                    _, p_val = stats.pearsonr(df[metric1].dropna(), df[metric2].dropna())
                    pearson_p_values[i, j] = p_val

                    # Spearman
                    _, p_val_rank = stats.spearmanr(df[metric1].dropna(), df[metric2].dropna())
                    spearman_p_values[i, j] = p_val_rank
                except:
                    pearson_p_values[i, j] = 1.0
                    spearman_p_values[i, j] = 1.0
            else:
                pearson_p_values[i, j] = 0.0
                spearman_p_values[i, j] = 0.0

    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 18))

    labels = [label_mapping.get(col, col) for col in available_metrics]

    # Pearson correlation heatmap
    sns.heatmap(pearson_corr_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax1,
                cbar_kws={'label': 'Pearson Correlation'})
    ax1.set_title(f'Pearson Correlation Matrix', fontsize=12, fontweight='bold')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Pearson p-values
    p_values_df = pd.DataFrame(pearson_p_values, index=available_metrics, columns=available_metrics)
    diagonal_mask = np.eye(len(available_metrics), dtype=bool)

    sns.heatmap(p_values_df,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn_r',
                vmin=0, vmax=0.1,
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax2,
                cbar_kws={'label': 'p-value'},
                mask=diagonal_mask)
    ax2.set_title(f'Pearson p-values (green = p < {significance_threshold})', fontsize=12, fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Spearman correlation heatmap
    sns.heatmap(spearman_corr_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax3,
                cbar_kws={'label': 'Spearman Correlation'})
    ax3.set_title(f'Spearman Rank Correlation Matrix', fontsize=12, fontweight='bold')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Spearman p-values
    spearman_p_values_df = pd.DataFrame(spearman_p_values, index=available_metrics, columns=available_metrics)

    sns.heatmap(spearman_p_values_df,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn_r',
                vmin=0, vmax=0.1,
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax4,
                cbar_kws={'label': 'p-value'},
                mask=diagonal_mask)
    ax4.set_title(f'Spearman p-values (green = p < {significance_threshold})', fontsize=12, fontweight='bold')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add overall title
    fig.suptitle(f'{dataset_name}: Correlation Analysis\n({baseline_type} as baseline metric)', 
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"{dataset_name}_correlation_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save both correlation matrices as CSV
    pearson_with_labels = pearson_corr_matrix.copy()
    pearson_with_labels.index = labels
    pearson_with_labels.columns = labels
    pearson_csv_path = output_dir / f"{dataset_name}_pearson_correlation.csv"
    pearson_with_labels.to_csv(pearson_csv_path)

    spearman_with_labels = spearman_corr_matrix.copy()
    spearman_with_labels.index = labels
    spearman_with_labels.columns = labels
    spearman_csv_path = output_dir / f"{dataset_name}_spearman_correlation.csv"
    spearman_with_labels.to_csv(spearman_csv_path)

    # Find significant correlations for BOTH methods
    pearson_significant_pairs = []
    spearman_significant_pairs = []

    for i, metric1 in enumerate(available_metrics):
        for j, metric2 in enumerate(available_metrics):
            if i < j:  # Only upper triangle
                # Pearson
                pearson_corr = pearson_corr_matrix.iloc[i, j]
                pearson_p = pearson_p_values[i, j]
                if pearson_p < significance_threshold:
                    pearson_significant_pairs.append({
                        'metric1': label_mapping.get(metric1, metric1),
                        'metric2': label_mapping.get(metric2, metric2),
                        'correlation': pearson_corr,
                        'p_value': pearson_p,
                        'abs_correlation': abs(pearson_corr),
                        'method': 'Pearson'
                    })

                # Spearman
                spearman_corr = spearman_corr_matrix.iloc[i, j]
                spearman_p = spearman_p_values[i, j]
                if spearman_p < significance_threshold:
                    spearman_significant_pairs.append({
                        'metric1': label_mapping.get(metric1, metric1),
                        'metric2': label_mapping.get(metric2, metric2),
                        'correlation': spearman_corr,
                        'p_value': spearman_p,
                        'abs_correlation': abs(spearman_corr),
                        'method': 'Spearman'
                    })

    # Sort by absolute correlation
    pearson_significant_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
    spearman_significant_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)

    # Combine for return (you might want to keep them separate)
    all_significant_pairs = pearson_significant_pairs + spearman_significant_pairs

    return spearman_corr_matrix, all_significant_pairs, available_metrics, labels
def print_significant_correlations(dataset_name, significant_pairs, threshold=0.05):
    """Print significant correlation pairs to console."""

    print(f"\n{'='*80}")
    print(f"SIGNIFICANT CORRELATIONS FOR {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"Significance threshold: p < {threshold}")

    # Separate by method
    pearson_pairs = [p for p in significant_pairs if p['method'] == 'Pearson']
    spearman_pairs = [p for p in significant_pairs if p['method'] == 'Spearman']

    print(f"\nPEARSON correlations: {len(pearson_pairs)} significant pairs")
    print(f"SPEARMAN correlations: {len(spearman_pairs)} significant pairs")
    print()

    def print_method_pairs(pairs, method_name):
        if not pairs:
            print(f"No significant {method_name} correlations found!")
            return

        strong_threshold = 0.7
        moderate_threshold = 0.4

        strong_pairs = [p for p in pairs if p['abs_correlation'] >= strong_threshold]
        moderate_pairs = [p for p in pairs if moderate_threshold <= p['abs_correlation'] < strong_threshold]
        weak_pairs = [p for p in pairs if p['abs_correlation'] < moderate_threshold]

        print(f"\n{method_name.upper()} CORRELATIONS:")
        print("-" * 40)

        def print_pairs(pairs, category):
            if pairs:
                print(f"\n{category.upper()} CORRELATIONS:")
                for pair in pairs:
                    direction = "positive" if pair['correlation'] > 0 else "negative"
                    print(f"  {pair['metric1']:25} ↔ {pair['metric2']:25} "
                          f"r={pair['correlation']:6.3f} (p={pair['p_value']:.4f}) [{direction}]")

        print_pairs(strong_pairs, "Strong (|r| ≥ 0.7)")
        print_pairs(moderate_pairs, "Moderate (0.4 ≤ |r| < 0.7)")
        print_pairs(weak_pairs, "Weak (|r| < 0.4)")

    print_method_pairs(pearson_pairs, "Pearson")
    print_method_pairs(spearman_pairs, "Spearman")

    # Compare methods
    print(f"\n{'='*50}")
    print("METHOD COMPARISON")
    print("="*50)

    # Find pairs significant in both methods
    pearson_set = {(p['metric1'], p['metric2']) for p in pearson_pairs}
    spearman_set = {(p['metric1'], p['metric2']) for p in spearman_pairs}

    both_methods = pearson_set & spearman_set
    pearson_only = pearson_set - spearman_set
    spearman_only = spearman_set - pearson_set

    print(f"\nSignificant in BOTH methods: {len(both_methods)} pairs")
    print(f"Significant in Pearson ONLY: {len(pearson_only)} pairs")
    print(f"Significant in Spearman ONLY: {len(spearman_only)} pairs")

    if spearman_only:
        print("\nPairs significant ONLY in Spearman (indicates non-linear relationships):")
        for m1, m2 in list(spearman_only)[:5]:  # Show top 5
            print(f"  - {m1} ↔ {m2}")

def create_summary_report(all_correlations, all_significant_pairs, output_dir):
    """Create a summary report across all datasets."""

    # Combine all significant pairs
    all_pairs = []
    for dataset, pairs in all_significant_pairs.items():
        for pair in pairs:
            pair_copy = pair.copy()
            pair_copy['dataset'] = dataset
            all_pairs.append(pair_copy)

    # Create summary DataFrame
    df_summary = pd.DataFrame(all_pairs)

    if len(df_summary) > 0:
        # Save detailed summary
        summary_path = output_dir / "correlation_summary_all_datasets.csv"
        df_summary.to_csv(summary_path, index=False)
        # Cross-dataset analysis
        print(f"\n{'='*100}")
        print("CROSS-DATASET CORRELATION SUMMARY")
        print(f"{'='*100}")

        # Find consistent patterns across datasets
        pair_consistency = {}
        for _, row in df_summary.iterrows():
            pair_key = tuple(sorted([row['metric1'], row['metric2']]))
            if pair_key not in pair_consistency:
                pair_consistency[pair_key] = []
            pair_consistency[pair_key].append({
                'dataset': row['dataset'],
                'correlation': row['correlation'],
                'p_value': row['p_value']
            })

        # Find pairs that appear in multiple datasets
        consistent_pairs = {k: v for k, v in pair_consistency.items() if len(v) > 1}

        if consistent_pairs:
            print(f"\nPAIRS SIGNIFICANT ACROSS MULTIPLE DATASETS:")
            print("-" * 60)
            for pair_key, occurrences in consistent_pairs.items():
                metric1, metric2 = pair_key
                print(f"\n{metric1} ↔ {metric2}:")
                correlations = [occ['correlation'] for occ in occurrences]
                mean_r = np.mean(correlations)
                std_r = np.std(correlations)

                for occ in occurrences:
                    print(f"  {occ['dataset']:20} r={occ['correlation']:6.3f} (p={occ['p_value']:.4f})")
                print(f"  {'Mean ± Std:':20} r={mean_r:6.3f} ± {std_r:.3f}")

                # Consistency check
                if std_r < 0.2:
                    print(f"  → Consistent pattern across datasets")
                else:
                    print(f"  → Variable pattern across datasets")

        # Most frequent metric pairs
        print(f"\nMOST FREQUENTLY SIGNIFICANT PAIRS:")
        print("-" * 40)
        pair_counts = df_summary.groupby(['metric1', 'metric2']).size().sort_values(ascending=False)
        for (m1, m2), count in pair_counts.head(10).items():
            avg_r = df_summary[(df_summary['metric1'] == m1) & (df_summary['metric2'] == m2)]['correlation'].mean()
            print(f"{m1:25} ↔ {m2:25} ({count} datasets, avg r={avg_r:.3f})")

        # Strongest correlations across all datasets
        print(f"\nSTRONGEST CORRELATIONS ACROSS ALL DATASETS:")
        print("-" * 50)
        strongest = df_summary.nlargest(10, 'abs_correlation')
        for _, row in strongest.iterrows():
            print(f"{row['dataset']:20} {row['metric1']:20} ↔ {row['metric2']:20} r={row['correlation']:6.3f}")

    else:
        print("No significant correlations found across datasets!")

def generate_metric_insights(all_correlations, output_dir):
    """Generate insights about each metric's behavior."""

    insights_path = output_dir / "metric_insights.txt"

    with open(insights_path, 'w') as f:
        f.write("METRIC BEHAVIOR INSIGHTS\n")
        f.write("="*50 + "\n\n")

        # Analyze each metric type
        metric_types = {
            'Baseline Metrics': ['BLEU', 'ROUGE-1'],
            'Peer Prediction Mechanisms': ['MI', 'GPPM', 'TVD-MI'],
            'LLM Judge Metrics': ['LLM-Judge (w/ context)', 'LLM-Judge (w/o context)'],
            'Control Metrics': ['Response Length']
        }

        for category, metrics in metric_types.items():
            f.write(f"{category.upper()}\n")
            f.write("-" * len(category) + "\n")

            # This would need to be implemented based on the correlation patterns
            # For now, placeholder structure
            f.write(f"Metrics in this category: {', '.join(metrics)}\n")
            f.write("Key patterns:\n")
            f.write("• [Pattern analysis would go here]\n")
            f.write("• [Cross-correlations within category]\n")
            f.write("• [Correlations with other categories]\n\n")

    print(f"Detailed metric insights saved to: {insights_path}")

def main():
    parser = argparse.ArgumentParser(description='Create comprehensive correlation matrix analysis')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing all result files')
    parser.add_argument('--figures-dir', type=str, default='results/figures',
                        help='Output directory for figures and analysis')
    parser.add_argument('--significance', type=float, default=0.05,
                        help='Significance threshold for correlations (default: 0.05)')

    args = parser.parse_args()

    # Create output directory
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True, parents=True)

    # Load all results
    datasets = load_results(args.results_dir)

    if not datasets:
        print("No datasets found!")
        return

    all_correlations = {}
    all_significant_pairs = {}

    # Process each dataset
    for dataset_name, dataset_data in datasets.items():
        print(f"\nProcessing {dataset_name}...")

        try:
            # Extract all metrics
            df, baseline_type = extract_all_metrics(dataset_data)

            if len(df) < 3:
                print(f"  Insufficient data for correlation analysis (n={len(df)})")
                continue

            # Create correlation matrix
            corr_matrix, significant_pairs, available_metrics, labels = create_correlation_matrix(
                df, baseline_type, dataset_name, figures_dir, args.significance
            )

            all_correlations[dataset_name] = {
                'correlation_matrix': corr_matrix,
                'available_metrics': available_metrics,
                'labels': labels,
                'baseline_type': baseline_type
            }

            all_significant_pairs[dataset_name] = significant_pairs

            # Print significant correlations for this dataset
            print_significant_correlations(dataset_name, significant_pairs, args.significance)

            print(f"  Correlation matrix saved: {dataset_name}_correlation_matrix.png")
            print(f"  Correlation data saved: {dataset_name}_correlation_matrix.csv")

        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Create cross-dataset summary
    if all_correlations:
        create_summary_report(all_correlations, all_significant_pairs, figures_dir)
        generate_metric_insights(all_correlations, figures_dir)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {figures_dir}")
        print(f"- Individual correlation matrices: *_correlation_matrix.png")
        print(f"- Correlation data: *_correlation_matrix.csv") 
        print(f"- Cross-dataset summary: correlation_summary_all_datasets.csv")
        print(f"- Metric insights: metric_insights.txt")

        # Summary statistics
        total_significant = sum(len(pairs) for pairs in all_significant_pairs.values())
        total_datasets = len(all_correlations)

        print(f"\nSUMMARY STATISTICS:")
        print(f"- Datasets processed: {total_datasets}")
        print(f"- Total significant correlations: {total_significant}")
        print(f"- Average significant correlations per dataset: {total_significant/total_datasets:.1f}")

        # Most common metrics across datasets
        all_metrics = set()
        for data in all_correlations.values():
            all_metrics.update(data['labels'])

        print(f"- Unique metrics found: {len(all_metrics)}")
        print(f"- Metrics: {', '.join(sorted(all_metrics))}")

if __name__ == "__main__":
    main()