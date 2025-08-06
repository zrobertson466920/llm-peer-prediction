import json
import numpy as np
from scipy import stats
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir, agent_name):
    """Load all mechanism results for a given agent."""
    results = {}
    results_dir = Path(results_dir)

    # Find the actual files in the directory by suffix pattern
    for file_path in results_dir.glob("*.json"):
        filename = file_path.name

        if filename.endswith('_validation.json'):
            with open(file_path, 'r') as f:
                results['validation'] = json.load(f)
        elif filename.endswith('_mi_gppm.json'):
            with open(file_path, 'r') as f:
                results['mi_gppm'] = json.load(f)
        elif filename.endswith('_tvd_mi.json'):
            with open(file_path, 'r') as f:
                results['tvd_mi'] = json.load(f)
        elif filename.endswith('_judge_with_context.json'):
            with open(file_path, 'r') as f:
                results['judge_with_context'] = json.load(f)
        elif filename.endswith('_judge_without_context.json'):
            with open(file_path, 'r') as f:
                results['judge_without_context'] = json.load(f)

    # Debug: print what we found
    print(f"Found in {results_dir}:")
    for key in results:
        print(f"  - {key}")

    return results

def extract_individual_scores(results_dir, agent_name, mechanism, condition=None):
    """Extract individual example scores for paired testing.
    
    If condition is None, returns overall mechanism scores for each example.
    If condition is specified, returns scores for that specific condition.
    """
    scores = []
    results_dir = Path(results_dir)

    if mechanism == 'mi':
        archive_dir = results_dir / "log_individual_examples"
        if archive_dir.exists():
            for example_file in sorted(archive_dir.glob("*_example_*.json")):
                with open(example_file, 'r') as f:
                    example_data = json.load(f)
                    if "combined_avgs" in example_data:
                        if condition is None:
                            # Overall MI score: average across all conditions
                            scores.append(np.mean(example_data["combined_avgs"]))
                        else:
                            # Specific condition score
                            if "condition_keys" in example_data and condition in example_data["condition_keys"]:
                                idx = example_data["condition_keys"].index(condition)
                                if idx < len(example_data["combined_avgs"]):
                                    scores.append(example_data["combined_avgs"][idx])

    elif mechanism == 'gppm':
        archive_dir = results_dir / "log_individual_examples"
        if archive_dir.exists():
            for example_file in sorted(archive_dir.glob("*_example_*.json")):
                with open(example_file, 'r') as f:
                    example_data = json.load(f)
                    if "gppm_normalized" in example_data:
                        if condition is None:
                            # Overall GPPM score: average across all conditions
                            # Note: GPPM scores are stored as negative, flip sign
                            scores.append(-np.mean(example_data["gppm_normalized"]))
                        else:
                            if "condition_keys" in example_data and condition in example_data["condition_keys"]:
                                idx = example_data["condition_keys"].index(condition)
                                if idx < len(example_data["gppm_normalized"]):
                                    scores.append(-example_data["gppm_normalized"][idx])

    elif mechanism == 'tvd_mi':
        archive_dir = results_dir / "tvd_mi_individual_examples"
        if archive_dir.exists():
            for example_file in sorted(archive_dir.glob("*_example_*.json")):
                with open(example_file, 'r') as f:
                    example_data = json.load(f)
                    # Check for bidirectional first, then regular
                    tvd_key = "tvd_mi_bidirectional" if "tvd_mi_bidirectional" in example_data else "tvd_mi_scores"
                    if tvd_key in example_data:
                        if condition is None:
                            # Overall TVD-MI score: average across all conditions
                            scores.append(np.mean(example_data[tvd_key]))
                        else:
                            if "condition_keys" in example_data and condition in example_data["condition_keys"]:
                                idx = example_data["condition_keys"].index(condition)
                                if idx < len(example_data[tvd_key]):
                                    scores.append(example_data[tvd_key][idx])

    elif mechanism == 'judge_with_context':
        archive_dir = results_dir / "llm_context_individual_examples"
        if archive_dir.exists():
            for example_file in sorted(archive_dir.glob("*_example_*.json")):
                with open(example_file, 'r') as f:
                    example_data = json.load(f)
                    if "win_rates" in example_data:
                        if condition is None:
                            # Overall judge score: average win rate across all conditions
                            scores.append(np.mean(example_data["win_rates"]))
                        else:
                            if "condition_keys" in example_data and condition in example_data["condition_keys"]:
                                idx = example_data["condition_keys"].index(condition)
                                if idx < len(example_data["win_rates"]):
                                    scores.append(example_data["win_rates"][idx])

    elif mechanism == 'judge_without_context':
        archive_dir = results_dir / "llm_without_context_individual_examples"
        if archive_dir.exists():
            for example_file in sorted(archive_dir.glob("*_example_*.json")):
                with open(example_file, 'r') as f:
                    example_data = json.load(f)
                    if "win_rates" in example_data:
                        if condition is None:
                            # Overall judge score: average win rate across all conditions
                            scores.append(np.mean(example_data["win_rates"]))
                        else:
                            if "condition_keys" in example_data and condition in example_data["condition_keys"]:
                                idx = example_data["condition_keys"].index(condition)
                                if idx < len(example_data["win_rates"]):
                                    scores.append(example_data["win_rates"][idx])

    return scores

def calculate_effect_size(before, after):
    """Calculate Cohen's d effect size."""
    # Use pooled standard deviation
    n1, n2 = len(before), len(after)
    var1, var2 = np.var(before, ddof=1), np.var(after, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0

    return (np.mean(after) - np.mean(before)) / pooled_std

def load_condition_categories():
    """Load condition categories from binary_cat_analysis.py"""
    # These are the exact categories used in binary_cat_analysis.py
    
    # Faithful conditions
    faithful_conditions = {
        'Faithful', 'Comprehensive', 'Objective', 'Neutral Tone'
    }
    
    # Style conditions
    style_conditions = {
        'Academic Style', 'Sensationalist', 'Bureaucratic', 'Casual Conversational',
        'Historical Perspective', 'Poetic', 'Technical Jargon', 'Euphemistic',
        'Overly Technical', 'Minimalist'
    }
    
    # Strategic conditions
    strategic_conditions = {
        'Fact Manipulation', 'Selective Omission', 'Misleading Emphasis',
        'Sentiment Flip', 'Conspiracy Theory', 'Contradictory', 'Context Removal',
        'False Attribution', 'Agenda Push', 'Cherry Pick'
    }
    
    # Low Effort conditions
    low_effort_conditions = {
        'Low Effort', 'Ultra Concise', 'Template Response', 'Surface Skim', 'Minimal Detail'
    }
    
    # Good Faith = Faithful + Style
    good_faith_conditions = faithful_conditions | style_conditions
    
    # Problematic = Strategic + Low Effort
    problematic_conditions = strategic_conditions | low_effort_conditions
    
    return good_faith_conditions, problematic_conditions


def calculate_category_effect_size_individual(results_dir, agent_name, mechanism, good_faith_conditions, problematic_conditions):
    """Calculate effect size between good faith and problematic conditions using individual example scores."""
    good_faith_scores = []
    problematic_scores = []

    # Get individual scores for each example
    scores = extract_individual_scores(results_dir, agent_name, mechanism, condition=None)

    # For each example, we need to calculate the mean score for good faith vs problematic conditions
    if mechanism == 'mi':
        archive_dir = Path(results_dir) / "log_individual_examples"
    elif mechanism == 'gppm':
        archive_dir = Path(results_dir) / "log_individual_examples"
    elif mechanism == 'tvd_mi':
        archive_dir = Path(results_dir) / "tvd_mi_individual_examples"
    elif mechanism == 'judge_with_context':
        archive_dir = Path(results_dir) / "llm_context_individual_examples"
    elif mechanism == 'judge_without_context':
        archive_dir = Path(results_dir) / "llm_without_context_individual_examples"
    else:
        return None

    if not archive_dir.exists():
        return None

    # Process each example file
    for example_file in sorted(archive_dir.glob("*_example_*.json")):
        with open(example_file, 'r') as f:
            example_data = json.load(f)

        # Get condition keys
        if "condition_keys" not in example_data:
            continue

        condition_keys = example_data["condition_keys"]

        # Get scores for this example
        if mechanism == 'mi':
            if "combined_avgs" not in example_data:
                continue
            scores = example_data["combined_avgs"]
        elif mechanism == 'gppm':
            if "gppm_normalized" not in example_data:
                continue
            scores = [-s for s in example_data["gppm_normalized"]]  # Flip sign
        elif mechanism == 'tvd_mi':
            tvd_key = "tvd_mi_bidirectional" if "tvd_mi_bidirectional" in example_data else "tvd_mi_scores"
            if tvd_key not in example_data:
                continue
            scores = example_data[tvd_key]
        elif mechanism.startswith('judge'):
            if "win_rates" not in example_data:
                continue
            scores = example_data["win_rates"]
        else:
            continue

        # Calculate mean scores for good faith and problematic conditions
        good_faith_scores_example = []
        problematic_scores_example = []

        for i, condition in enumerate(condition_keys):
            if i < len(scores):
                if condition in good_faith_conditions:
                    good_faith_scores_example.append(scores[i])
                elif condition in problematic_conditions:
                    problematic_scores_example.append(scores[i])

        # Add the mean scores for this example
        if good_faith_scores_example and problematic_scores_example:
            good_faith_scores.append(np.mean(good_faith_scores_example))
            problematic_scores.append(np.mean(problematic_scores_example))

    if len(good_faith_scores) > 0 and len(problematic_scores) > 0:
        # Calculate paired differences
        differences = np.array(good_faith_scores) - np.array(problematic_scores)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)

        if std_diff > 0:
            # Paired samples Cohen's d
            effect_size = mean_diff / std_diff
        else:
            effect_size = 0

        return {
            'effect_size': effect_size,
            'mean_good_faith': np.mean(good_faith_scores),
            'mean_problematic': np.mean(problematic_scores),
            'n_examples': len(differences)
        }

    return None

def paired_analysis(original_dir, transformed_dir, original_name, transformed_name, output_dir):
    """Perform paired analysis between original and transformed results."""

    # Load results
    original_results = load_results(original_dir, original_name)
    transformed_results = load_results(transformed_dir, transformed_name)
    
    # Load condition categories
    good_faith_conditions, problematic_conditions = load_condition_categories()
    
    # Debug: Check for any missing conditions
    all_categorized = good_faith_conditions | problematic_conditions
    print(f"\nCategorized conditions ({len(all_categorized)}): {sorted(all_categorized)}")

    # Validate that we have comparable datasets
    if 'mi_gppm' not in original_results or 'mi_gppm' not in transformed_results:
        raise ValueError("Both datasets must have MI/GPPM results for comparison")

    # Extract conditions - using the correct structure from the files
    original_conditions = set(original_results['mi_gppm']['condition_keys'])
    transformed_conditions = set(transformed_results['mi_gppm']['condition_keys'])

    # Debug print
    print(f"\nOriginal conditions ({len(original_conditions)}): {sorted(original_conditions)}")
    print(f"Transformed conditions ({len(transformed_conditions)}): {sorted(transformed_conditions)}")

    # Find common conditions
    common_conditions = original_conditions & transformed_conditions
    print(f"Common conditions ({len(common_conditions)}): {sorted(common_conditions)}")

    if not common_conditions:
        print(f"ERROR: No common conditions found between datasets!")
        print(f"Original only: {sorted(original_conditions - transformed_conditions)}")
        print(f"Transformed only: {sorted(transformed_conditions - original_conditions)}")
        raise ValueError("Cannot compare datasets with no common conditions.")

    conditions = list(common_conditions)

    # Prepare analysis results
    analysis_results = {
        'transformation': {
            'original': original_name,
            'transformed': transformed_name
        },
        'mechanisms': {},
        'paired_tests': {},  # Store paired test results
        'category_effect_sizes': {}  # Store effect sizes between strategic/non-strategic
    }

    # Analyze each mechanism
    mechanisms = [
        ('mi', 'MI (DoE)'),
        ('gppm', 'GPPM'),
        ('tvd_mi', 'TVD-MI'),
        ('judge_with_context', 'Judge (with context)'),
        ('judge_without_context', 'Judge (without context)')
    ]

    for mech_key, mech_name in mechanisms:
        print(f"\nAnalyzing {mech_name}...")

        mechanism_results = {
            'conditions': {}
        }

        # Get the index for each condition
        for i, condition in enumerate(conditions):
            # Get aggregate scores based on the correct data structure
            if mech_key == 'mi':
                if 'mi_gppm' in original_results and 'mi_gppm' in transformed_results:
                    orig_idx = original_results['mi_gppm']['condition_keys'].index(condition)
                    trans_idx = transformed_results['mi_gppm']['condition_keys'].index(condition)

                    orig_mean = original_results['mi_gppm']['combined_avgs_avg'][orig_idx]
                    trans_mean = transformed_results['mi_gppm']['combined_avgs_avg'][trans_idx]

                    # Get confidence intervals if available - they're lists parallel to the avg lists
                    orig_ci_list = original_results['mi_gppm'].get('combined_avgs_ci', [])
                    trans_ci_list = transformed_results['mi_gppm'].get('combined_avgs_ci', [])
                    orig_ci = orig_ci_list[orig_idx] if orig_idx < len(orig_ci_list) else [None, None]
                    trans_ci = trans_ci_list[trans_idx] if trans_idx < len(trans_ci_list) else [None, None]
                else:
                    continue

            elif mech_key == 'gppm':
                if 'mi_gppm' in original_results and 'mi_gppm' in transformed_results:
                    orig_idx = original_results['mi_gppm']['condition_keys'].index(condition)
                    trans_idx = transformed_results['mi_gppm']['condition_keys'].index(condition)

                    # Note: GPPM scores are negative in the files, but binary_cat_analysis flips the sign
                    orig_mean = -original_results['mi_gppm']['gppm_normalized_avg'][orig_idx]
                    trans_mean = -transformed_results['mi_gppm']['gppm_normalized_avg'][trans_idx]

                    orig_ci_list = original_results['mi_gppm'].get('gppm_normalized_ci', [])
                    trans_ci_list = transformed_results['mi_gppm'].get('gppm_normalized_ci', [])
                    orig_ci = orig_ci_list[orig_idx] if orig_idx < len(orig_ci_list) else [None, None]
                    trans_ci = trans_ci_list[trans_idx] if trans_idx < len(trans_ci_list) else [None, None]
                else:
                    continue

            elif mech_key == 'tvd_mi':
                if 'tvd_mi' in original_results and 'tvd_mi' in transformed_results:
                    orig_idx = original_results['tvd_mi']['condition_keys'].index(condition)
                    trans_idx = transformed_results['tvd_mi']['condition_keys'].index(condition)

                    # Try bidirectional first, fall back to regular
                    tvd_key = 'tvd_mi_bidirectional_avg' if 'tvd_mi_bidirectional_avg' in original_results['tvd_mi'] else 'tvd_mi_scores_avg'
                    orig_mean = original_results['tvd_mi'][tvd_key][orig_idx]
                    trans_mean = transformed_results['tvd_mi'][tvd_key][trans_idx]

                    ci_key = tvd_key.replace('_avg', '_ci')
                    orig_ci_list = original_results['tvd_mi'].get(ci_key, [])
                    trans_ci_list = transformed_results['tvd_mi'].get(ci_key, [])
                    orig_ci = orig_ci_list[orig_idx] if orig_idx < len(orig_ci_list) else [None, None]
                    trans_ci = trans_ci_list[trans_idx] if trans_idx < len(trans_ci_list) else [None, None]
                else:
                    continue

            elif mech_key.startswith('judge'):
                if mech_key in original_results and mech_key in transformed_results:
                    orig_idx = original_results[mech_key]['condition_keys'].index(condition)
                    trans_idx = transformed_results[mech_key]['condition_keys'].index(condition)

                    orig_mean = original_results[mech_key]['win_rates_avg'][orig_idx]
                    trans_mean = transformed_results[mech_key]['win_rates_avg'][trans_idx]

                    orig_ci_list = original_results[mech_key].get('win_rates_ci', [])
                    trans_ci_list = transformed_results[mech_key].get('win_rates_ci', [])
                    orig_ci = orig_ci_list[orig_idx] if orig_idx < len(orig_ci_list) else [None, None]
                    trans_ci = trans_ci_list[trans_idx] if trans_idx < len(trans_ci_list) else [None, None]
                else:
                    continue

            # Calculate changes
            absolute_change = trans_mean - orig_mean
            if orig_mean != 0:
                relative_change = (absolute_change / abs(orig_mean)) * 100
            else:
                relative_change = float('inf') if absolute_change != 0 else 0

            # Store the results
            mechanism_results['conditions'][condition] = {
                'original_mean': orig_mean,
                'transformed_mean': trans_mean,
                'absolute_change': absolute_change,
                'relative_change_percent': relative_change,
                'has_paired_test': False
            }

        analysis_results['mechanisms'][mech_key] = mechanism_results

    # Calculate category effect sizes using individual-level data
    print("\n" + "="*80)
    print("CATEGORY EFFECT SIZES (Good Faith vs Problematic)")
    print("="*80)
    
    for mech_key, mech_name in mechanisms:
        print(f"\n{mech_name}:")
        
        # Calculate effect sizes using individual example scores
        orig_effect = calculate_category_effect_size_individual(
            original_dir, original_name, mech_key,
            good_faith_conditions, problematic_conditions
        )
        trans_effect = calculate_category_effect_size_individual(
            transformed_dir, transformed_name, mech_key,
            good_faith_conditions, problematic_conditions
        )
        
        if orig_effect and trans_effect:
            effect_size_change = trans_effect['effect_size'] - orig_effect['effect_size']
            
            analysis_results['category_effect_sizes'][mech_key] = {
                'original_effect_size': orig_effect['effect_size'],
                'transformed_effect_size': trans_effect['effect_size'],
                'effect_size_change': effect_size_change,
                'original_means': {
                    'problematic': orig_effect['mean_problematic'],
                    'good_faith': orig_effect['mean_good_faith']
                },
                'transformed_means': {
                    'problematic': trans_effect['mean_problematic'],
                    'good_faith': trans_effect['mean_good_faith']
                },
                'n_examples': orig_effect['n_examples']
            }
            
            print(f"  Original effect size: {orig_effect['effect_size']:.3f}")
            print(f"    Problematic mean: {orig_effect['mean_problematic']:.4f}")
            print(f"    Good faith mean: {orig_effect['mean_good_faith']:.4f}")
            print(f"    N examples: {orig_effect['n_examples']}")
            print(f"  Transformed effect size: {trans_effect['effect_size']:.3f}")
            print(f"    Problematic mean: {trans_effect['mean_problematic']:.4f}")
            print(f"    Good faith mean: {trans_effect['mean_good_faith']:.4f}")
            print(f"    N examples: {trans_effect['n_examples']}")
            print(f"  Change in effect size: {effect_size_change:.3f}")
            
            # Interpret the change
            if abs(effect_size_change) < 0.1:
                interpretation = "minimal change"
            elif effect_size_change < -0.2:
                interpretation = "substantial reduction"
            elif effect_size_change > 0.2:
                interpretation = "substantial increase"
            else:
                interpretation = "moderate change"
            print(f"  Interpretation: {interpretation}")

    # Perform paired t-tests using individual scores if available
    print("\n" + "="*80)
    print("PAIRED T-TESTS")
    print("="*80)
    
    for mech_key, mech_name in mechanisms:
        print(f"\n{mech_name}:")
        
        # Collect all paired differences for this mechanism
        all_orig_scores = []
        all_trans_scores = []
        
        # Get overall mechanism scores for each example (not condition-specific)
        all_orig_scores = extract_individual_scores(original_dir, original_name, mech_key, condition=None)
        all_trans_scores = extract_individual_scores(transformed_dir, transformed_name, mech_key, condition=None)
        
        
        if len(all_orig_scores) > 0 and len(all_orig_scores) == len(all_trans_scores):
            # Perform paired t-test
            differences = np.array(all_trans_scores) - np.array(all_orig_scores)
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            n = len(differences)
            
            # Calculate t-statistic and p-value
            t_stat, p_value = stats.ttest_rel(all_trans_scores, all_orig_scores)
            
            # Calculate 95% CI for mean difference
            t_critical = stats.t.ppf(0.975, n-1)
            ci_lower = mean_diff - t_critical * (std_diff / np.sqrt(n))
            ci_upper = mean_diff + t_critical * (std_diff / np.sqrt(n))
            
            # Calculate effect size
            effect_size = calculate_effect_size(all_orig_scores, all_trans_scores)
            
            analysis_results['paired_tests'][mech_key] = {
                'n_pairs': int(n),
                'mean_difference': float(mean_diff),
                'std_difference': float(std_diff),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'ci_95': [float(ci_lower), float(ci_upper)],
                'effect_size_cohens_d': float(effect_size),
                'significant': bool(p_value < 0.05)
            }
            
            print(f"  Paired samples: {n}")
            print(f"  Mean original: {np.mean(all_orig_scores):.4f}")
            print(f"  Mean transformed: {np.mean(all_trans_scores):.4f}")
            print(f"  Mean difference: {mean_diff:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
            print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
            print(f"  Effect size (Cohen's d): {effect_size:.3f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        else:
            # Fall back to using aggregate statistics
            orig_means = []
            trans_means = []
            
            for condition in conditions:
                if condition in analysis_results['mechanisms'][mech_key]['conditions']:
                    orig_means.append(analysis_results['mechanisms'][mech_key]['conditions'][condition]['original_mean'])
                    trans_means.append(analysis_results['mechanisms'][mech_key]['conditions'][condition]['transformed_mean'])
            
            if len(orig_means) > 1:
                # Perform paired t-test on condition means
                t_stat, p_value = stats.ttest_rel(trans_means, orig_means)
                differences = np.array(trans_means) - np.array(orig_means)
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)
                n = len(differences)
                
                t_critical = stats.t.ppf(0.975, n-1)
                ci_lower = mean_diff - t_critical * (std_diff / np.sqrt(n))
                ci_upper = mean_diff + t_critical * (std_diff / np.sqrt(n))
                
                effect_size = calculate_effect_size(orig_means, trans_means)
                
                analysis_results['paired_tests'][mech_key] = {
                    'n_pairs': int(n),
                    'mean_difference': float(mean_diff),
                    'std_difference': float(std_diff),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'ci_95': [float(ci_lower), float(ci_upper)],
                    'effect_size_cohens_d': float(effect_size),
                    'significant': bool(p_value < 0.05),
                    'note': 'Based on condition-level means'
                }
                
                print(f"  Paired conditions: {n}")
                print(f"  Mean original: {np.mean(orig_means):.4f}")
                print(f"  Mean transformed: {np.mean(trans_means):.4f}")
                print(f"  Mean difference: {mean_diff:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
                print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
                print(f"  Effect size (Cohen's d): {effect_size:.3f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                print(f"  Note: Based on condition-level means, not individual examples")
            else:
                print(f"  Insufficient data for paired test")

    # Create visualization
    create_comparison_plot(analysis_results, output_dir)

    # Save detailed results
    output_path = Path(output_dir) / 'paired_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    # Print summary
    print_summary(analysis_results)

    return analysis_results

def create_comparison_plot(analysis_results, output_dir):
    """Create visualization of the changes."""
    # Load condition categories
    good_faith_conditions, problematic_conditions = load_condition_categories()

    # Prepare data for plotting
    plot_data = []

    for mech_key, mech_data in analysis_results['mechanisms'].items():
        for condition, stats in mech_data['conditions'].items():
            # Determine category
            if condition in good_faith_conditions:
                category = 'Good Faith'
            elif condition in problematic_conditions:
                category = 'Problematic'
            else:
                category = 'Unknown'

            plot_data.append({
                'Mechanism': mech_key.replace('_', ' ').title(),
                'Condition': condition,
                'Category': category,
                'Original': stats['original_mean'],
                'Transformed': stats['transformed_mean'],
                'Change': stats['absolute_change'],
                'Relative_Change': stats['relative_change_percent'],
                'Significant': stats.get('p_value', 1) < 0.05 if stats.get('has_paired_test') else False
            })

    # Check if we have any data to plot
    if not plot_data:
        print("Warning: No data available for plotting. Skipping visualization.")
        return

    df = pd.DataFrame(plot_data)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Before/After comparison by mechanism
    ax = axes[0, 0]
    mechanisms = df['Mechanism'].unique()
    x = np.arange(len(mechanisms))
    width = 0.35

    for i, mech in enumerate(mechanisms):
        mech_df = df[df['Mechanism'] == mech]
        original_mean = mech_df['Original'].mean()
        transformed_mean = mech_df['Transformed'].mean()
        ax.bar(i - width/2, original_mean, width, label='Original' if i == 0 else '', color='#1f77b4')
        ax.bar(i + width/2, transformed_mean, width, label='Transformed' if i == 0 else '', color='#ff7f0e')

    ax.set_xlabel('Mechanism')
    ax.set_ylabel('Mean Score')
    ax.set_title('Average Scores: Original vs Transformed')
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Category-based change analysis (replacing the all-zero heatmap)
    ax = axes[0, 1]

    # Calculate average changes by category and mechanism
    category_changes = []
    for mech in df['Mechanism'].unique():
        for category in ['Good Faith', 'Problematic']:
            cat_df = df[(df['Mechanism'] == mech) & (df['Category'] == category)]
            if len(cat_df) > 0:
                avg_change = cat_df['Change'].mean()
                avg_relative = cat_df['Relative_Change'].mean()
                category_changes.append({
                    'Mechanism': mech,
                    'Category': category,
                    'Absolute_Change': avg_change,
                    'Relative_Change': avg_relative
                })

    cat_change_df = pd.DataFrame(category_changes)

    # Create grouped bar plot for category changes
    categories = ['Good Faith', 'Problematic']
    x = np.arange(len(mechanisms))
    width = 0.35

    colors = {'Good Faith': '#2ca02c', 'Problematic': '#d62728'}

    for i, cat in enumerate(categories):
        cat_data = cat_change_df[cat_change_df['Category'] == cat]
        changes = []
        for mech in mechanisms:
            mech_cat_data = cat_data[cat_data['Mechanism'] == mech]
            if len(mech_cat_data) > 0:
                changes.append(mech_cat_data['Absolute_Change'].values[0])
            else:
                changes.append(0)

        ax.bar(x + i*width - width/2, changes, width, label=cat, color=colors[cat], alpha=0.8)

    ax.set_xlabel('Mechanism')
    ax.set_ylabel('Average Absolute Change')
    ax.set_title('Changes by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # 3. Sorted changes by condition (grouped by category)
    ax = axes[1, 0]

    # Sort conditions by category, then by name
    df_sorted = df.sort_values(['Category', 'Condition'])
    conditions_sorted = df_sorted['Condition'].unique()

    # Create visual separation between categories
    good_faith_conds = [c for c in conditions_sorted if c in good_faith_conditions]
    problematic_conds = [c for c in conditions_sorted if c in problematic_conditions]

    # Combine with a separator
    all_conditions = good_faith_conds + [''] + problematic_conds

    # Calculate positions with gap
    positions = []
    pos = 0
    for i, cond in enumerate(all_conditions):
        if cond == '':  # Skip separator
            pos += 0.5  # Add gap
            continue
        positions.append(pos)
        pos += 1

    # Plot each mechanism
    width = 0.15
    mechanism_colors = plt.cm.Set3(np.linspace(0, 1, len(mechanisms)))

    for i, mech in enumerate(mechanisms):
        mech_df = df[df['Mechanism'] == mech]
        changes = []
        pos_idx = 0

        for cond in all_conditions:
            if cond == '':  # Skip separator
                continue
            cond_data = mech_df[mech_df['Condition'] == cond]
            if len(cond_data) > 0:
                changes.append(cond_data['Change'].values[0])
            else:
                changes.append(0)

        ax.bar(np.array(positions) + i * width, changes, width, 
               label=mech, color=mechanism_colors[i])

    # Add category labels
    ax.axvline(x=len(good_faith_conds) - 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(len(good_faith_conds)/2, ax.get_ylim()[1]*0.95, 'Good Faith', 
            ha='center', va='top', fontsize=10, weight='bold', alpha=0.7)
    ax.text(len(good_faith_conds) + 0.5 + len(problematic_conds)/2, ax.get_ylim()[1]*0.95, 
            'Problematic', ha='center', va='top', fontsize=10, weight='bold', alpha=0.7)

    ax.set_xlabel('Condition')
    ax.set_ylabel('Absolute Change')
    ax.set_title('Changes by Condition (Grouped by Category)')

    # Set x-ticks only for actual conditions
    actual_positions = []
    actual_labels = []
    pos_idx = 0
    for cond in all_conditions:
        if cond != '':
            actual_positions.append(positions[pos_idx] + width * (len(mechanisms) - 1) / 2)
            actual_labels.append(cond)
            pos_idx += 1

    ax.set_xticks(actual_positions)
    ax.set_xticklabels(actual_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Category effect sizes comparison (keep as is)
    ax = axes[1, 1]

    if 'category_effect_sizes' in analysis_results and analysis_results['category_effect_sizes']:
        # Create effect size comparison plot
        effect_data = []
        for mech_key, effect_results in analysis_results['category_effect_sizes'].items():
            mech_name = mech_key.replace('_', ' ').title()
            effect_data.append({
                'Mechanism': mech_name,
                'Original': effect_results['original_effect_size'],
                'Transformed': effect_results['transformed_effect_size'],
                'Change': effect_results['effect_size_change']
            })

        if effect_data:
            effect_df = pd.DataFrame(effect_data)

            # Create grouped bar plot
            x = np.arange(len(effect_df))
            width = 0.35

            bars1 = ax.bar(x - width/2, effect_df['Original'], width, label='Original', alpha=0.8, color='#1f77b4')
            bars2 = ax.bar(x + width/2, effect_df['Transformed'], width, label='Transformed', alpha=0.8, color='#ff7f0e')

            # Add change values as text
            for i, (orig, trans, change) in enumerate(zip(effect_df['Original'], 
                                                          effect_df['Transformed'], 
                                                          effect_df['Change'])):
                y_pos = max(orig, trans) + 0.05
                color = 'red' if change < -0.3 else 'black'
                ax.text(i, y_pos, f'Δ={change:.2f}', ha='center', va='bottom', fontsize=9, color=color)

            ax.set_xlabel('Mechanism')
            ax.set_ylabel('Effect Size (Cohen\'s d)')
            ax.set_title('Good Faith vs Problematic Discrimination')
            ax.set_xticks(x)
            ax.set_xticklabels(effect_df['Mechanism'], rotation=45, ha='right')
            ax.legend()
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3, label='Small effect')
            ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.3, label='Large effect')
            ax.grid(True, alpha=0.3)

            # Add interpretation text
            ax.text(0.02, 0.98, 'Red Δ = substantial reduction (>0.3)', 
                   transform=ax.transAxes, fontsize=8, va='top', alpha=0.7)

    plt.tight_layout()
    output_path = Path(output_dir) / 'paired_analysis_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_summary(analysis_results):
    """Print a summary of the analysis."""
    print("\n" + "="*80)
    print("PAIRED ANALYSIS SUMMARY")
    print("="*80)
    
    # Show category effect size changes
    if 'category_effect_sizes' in analysis_results and analysis_results['category_effect_sizes']:
        print("\nCategory Effect Size Changes (Good Faith vs Problematic):")
        print("-" * 40)
        for mech_key, effect_data in analysis_results['category_effect_sizes'].items():
            mech_name = mech_key.replace('_', ' ').title()
            print(f"\n{mech_name}:")
            print(f"  Original effect size: {effect_data['original_effect_size']:.3f}")
            print(f"  Transformed effect size: {effect_data['transformed_effect_size']:.3f}")
            print(f"  Change: {effect_data['effect_size_change']:.3f}")
            
            # Check if discrimination is preserved
            if effect_data['original_effect_size'] > 0.2:  # Originally showed discrimination
                if effect_data['transformed_effect_size'] < 0.1:
                    print(f"  ⚠️  WARNING: Lost ability to discriminate strategic conditions")
                elif effect_data['effect_size_change'] < -0.3:
                    print(f"  ⚠️  WARNING: Substantial reduction in discrimination")

    # First show paired test results if available
    if 'paired_tests' in analysis_results and analysis_results['paired_tests']:
        print("\n\nOverall Paired Test Results:")
        print("-" * 40)
        for mech_key, test_results in analysis_results['paired_tests'].items():
            mech_name = mech_key.replace('_', ' ').title()
            print(f"\n{mech_name}:")
            print(f"  Mean change: {test_results['mean_difference']:.4f} "
                  f"(95% CI: [{test_results['ci_95'][0]:.4f}, {test_results['ci_95'][1]:.4f}])")
            print(f"  p-value: {test_results['p_value']:.4f} "
                  f"({'significant' if test_results['significant'] else 'not significant'})")
            print(f"  Effect size: {test_results['effect_size_cohens_d']:.3f}")
            if 'note' in test_results:
                print(f"  Note: {test_results['note']}")

    print("\n\nCondition-Level Summary:")
    print("-" * 40)
    
    for mech_key, mech_data in analysis_results['mechanisms'].items():
        mech_name = mech_key.replace('_', ' ').title()
        print(f"\n{mech_name}:")
        print("-" * 40)

        # Calculate aggregate statistics
        all_changes = []
        all_effect_sizes = []
        sig_count = 0

        for condition, stats in mech_data['conditions'].items():
            all_changes.append(stats['relative_change_percent'])
            if stats.get('has_paired_test') and stats.get('effect_size_cohens_d') is not None:
                all_effect_sizes.append(stats['effect_size_cohens_d'])
                if stats.get('p_value', 1) < 0.05:
                    sig_count += 1

        if all_changes:
            print(f"  Average relative change: {np.mean(all_changes):.1f}%")
            print(f"  Range of changes: [{min(all_changes):.1f}%, {max(all_changes):.1f}%]")

        if all_effect_sizes:
            print(f"  Average effect size: {np.mean(all_effect_sizes):.3f}")
            print(f"  Significant changes: {sig_count}/{len(mech_data['conditions'])} conditions")

        # Show largest changes
        if mech_data['conditions']:
            sorted_conditions = sorted(mech_data['conditions'].items(), 
                                     key=lambda x: abs(x[1]['absolute_change']), 
                                     reverse=True)
            print(f"  Largest absolute change: {sorted_conditions[0][0]} "
                  f"({sorted_conditions[0][1]['absolute_change']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Paired analysis of mechanism scores before/after transformation")
    parser.add_argument("--original-dir", required=True, help="Results directory for original data")
    parser.add_argument("--transformed-dir", required=True, help="Results directory for transformed data")
    parser.add_argument("--output-dir", help="Output directory for analysis results")

    args = parser.parse_args()

    # Extract agent names from directory names
    original_name = Path(args.original_dir).name.replace("_results", "")
    transformed_name = Path(args.transformed_dir).name.replace("_results", "")

    # Default output directory
    if not args.output_dir:
        # Extract transformation type from transformed name
        transform_type = "unknown"
        for t in ["repetition", "pattern", "format", "case_flip", "padding"]:
            if f"_{t}_transformed" in transformed_name:
                transform_type = t
                break
        args.output_dir = f"ablation_analysis_{transform_type}"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Comparing:")
    print(f"  Original: {original_name}")
    print(f"  Transformed: {transformed_name}")
    print(f"  Output: {args.output_dir}")

    # Run analysis
    results = paired_analysis(
        args.original_dir, 
        args.transformed_dir,
        original_name,
        transformed_name,
        args.output_dir
    )

if __name__ == "__main__":
    main()
