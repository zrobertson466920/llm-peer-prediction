"""
Binary Category Analysis for Advisor Meeting
Compares "Good Faith" (Faithful + Style) vs "Problematic" (Strategic + Low Effort) conditions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import pandas as pd
from pathlib import Path
import argparse

# Add near the imports
COMPRESSION_RATIOS = {
    'opus_books': 1.3,
    'wmt14': 1.1,
    'samsum': 4.8,
    'pubmed': 6.7,
    'multi_news': 9.0,
    'billsum': 9.3,
    'cnn_dailymail': 13.8,
    'reddit_tifu': 16.1,
    'xsum': 18.5,
    'iclr_2023': 20.2
}

def get_compression_ratio(dataset_name):
    """Extract compression ratio from dataset name."""
    name_lower = dataset_name.lower()

    # Direct mappings for tricky cases
    if 'cnn_dailymail' in name_lower or 'cnndailymail' in name_lower:
        return 13.8  # cnn_dailymail
    if 'mt14' in name_lower:
        return 1.1   # wmt14
    if 'peer_review' in name_lower:
        return 20.2  # iclr_2023

    # Generic matching
    for key, ratio in COMPRESSION_RATIOS.items():
        # Create flexible matching by removing underscores and hyphens
        key_clean = key.replace('_', '').replace('-', '')
        name_clean = name_lower.replace('_', '').replace('-', '')

        if key_clean in name_clean:
            return ratio

    print(f"Warning: No compression ratio found for '{dataset_name}'")
    return None

def detect_task_type(dataset_name):
    """Detect task type from dataset name."""
    if 'translation' in dataset_name.lower():
        return 'translation'
    elif 'summarization' in dataset_name.lower() or 'cnn' in dataset_name.lower():
        return 'summarization'
    elif 'review' in dataset_name.lower() or 'iclr' in dataset_name.lower():
        return 'peer_review'
    return 'translation'  # default

def load_task_config(task_type):
    """Load task-specific configuration."""
    configs = {
        'translation': {
            'categories': {
                'Strategic': ['All Positive', 'All Negative', 'Misleading', 'Contradictory'],
                'Low Effort': ['Low Effort', 'Exaggerate', 'Understate', 'Sarcastic'],
                'Style': ['Informal', 'Simplify', 'Poetic', 'Humorous', 'Dramatic', 'Historical', 'Futuristic', 'Persuasive', 'Emotional', 'Subjective', 'Metaphorical', 'Comparative', 'Hypothetical', 'Philosophical', 'Quantitative', 'Creative'],
                'Faithful': ['Original', 'Formal', 'Technical', 'Academic', 'Objective', 'Cultural']
            }
        },
        'summarization': {
            'categories': {
                'Strategic': [
                    'Fact Manipulation',
                    'Selective Omission', 
                    'Misleading Emphasis',
                    'Sentiment Flip',
                    'Conspiracy Theory',
                    'Contradictory',
                    'Context Removal',
                    'False Attribution',
                    'Agenda Push',
                    'Cherry Pick'
                ],
                'Low Effort': [
                    'Low Effort',
                    'Ultra Concise',
                    'Template Response',
                    'Surface Skim',
                    'Minimal Detail'
                ],
                'Style': [
                    'Academic Style',
                    'Sensationalist',
                    'Bureaucratic',
                    'Casual Conversational',
                    'Historical Perspective',
                    'Poetic',
                    'Technical Jargon',
                    'Euphemistic',
                    'Overly Technical',
                    'Minimalist'
                ],
                'Faithful': [
                    'Faithful',
                    'Objective',
                    'Comprehensive',
                    'Neutral Tone'
                ]
            }
        },
        'peer_review': {
            'categories': {
                'Strategic': [
                    'Method Shift',
                    'Question Shift', 
                    'Contribution Misrepresent',
                    'Result Manipulation',
                    'Assumption Attack',
                    'Dismissive Expert',
                    'Agenda Push',
                    'Benchmark Obsessed'
                ],
                'Low Effort': [
                    'Low Effort',
                    'Generic',
                    'Surface Skim',
                    'Template Fill',
                    'Checklist Review'
                ],
                'Style': [
                    'Balanced Critique',
                    'Overly Technical',
                    'Harsh Critique',
                    'Overly Positive',
                    'Theory Focus',
                    'Implementation Obsessed',
                    'Comparison Fixated',
                    'Pedantic Details',
                    'Scope Creep',
                    'Statistical Nitpick',
                    'Future Work Focus',
                    'Writing Critique'
                ],
                'Faithful': [
                    'Reference',
                    'Faithful',
                    'Objective Analysis',
                    'Thorough Evaluation'
                ]
            }
        }
    }
    return configs.get(task_type, configs['translation'])

def load_individual_results(results_dir, base_name):
    """Load only the numerical scores from individual results, not the text content."""
    results_dir = Path(results_dir)
    individual_data = {}

    # Define what fields we actually need
    needed_fields = {
        'mi_gppm': ['condition_keys', 'combined_avgs', 'gppm_normalized'],
        'tvd_mi': ['condition_keys', 'tvd_mi_bidirectional', 'tvd_mi_scores'],
        'judge_with': ['condition_keys', 'win_rates'],
        'judge_without': ['condition_keys', 'win_rates']
    }

    for mechanism, archives in [
        ('mi_gppm', [f"{base_name}_mi_gppm_individual_examples", "log_individual_examples"]),
        ('tvd_mi', [f"{base_name}_tvd_mi_individual_examples", "tvd_mi_individual_examples"]),
        ('judge_with', [f"{base_name}_judge_with_context_individual_examples", "llm_context_individual_examples"]),
        ('judge_without', [f"{base_name}_judge_without_context_individual_examples", "llm_without_context_individual_examples"])
    ]:
        for archive_name in archives:
            archive = results_dir / archive_name
            if archive.exists():
                individual_data[mechanism] = []
                json_files = sorted(archive.glob("*.json"))

                print(f"Loading {len(json_files)} files for {mechanism} (lightweight mode)...")

                for i, json_file in enumerate(json_files):
                    if i % 100 == 0:
                        print(f"  Progress: {i}/{len(json_files)}")

                    try:
                        with open(json_file, 'r') as f:
                            full_data = json.load(f)

                            # Extract only the fields we need
                            filtered_data = {}
                            for field in needed_fields.get(mechanism, []):
                                if field in full_data:
                                    filtered_data[field] = full_data[field]

                            # Only append if we got the essential data
                            if 'condition_keys' in filtered_data:
                                individual_data[mechanism].append(filtered_data)

                    except Exception as e:
                        print(f"  Error loading {json_file}: {e}")

                break

    return individual_data

def load_results(results_dir):
    """Load validation and mechanism results."""
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

            # Add TVD-MI if available
            if tvd_mi_file.exists():
                with open(tvd_mi_file, 'r') as f:
                    tvd_mi_data = json.load(f)
                dataset_entry['tvd_mi'] = tvd_mi_data

            # Add LLM judge results if available
            if llm_judge_with_context_file.exists():
                with open(llm_judge_with_context_file, 'r') as f:
                    llm_judge_with_context_data = json.load(f)
                dataset_entry['llm_judge_with_context'] = llm_judge_with_context_data

            if llm_judge_without_context_file.exists():
                with open(llm_judge_without_context_file, 'r') as f:
                    llm_judge_without_context_data = json.load(f)
                dataset_entry['llm_judge_without_context'] = llm_judge_without_context_data

            # Load individual results for bootstrap analysis
            individual_data = load_individual_results(results_dir, base_name)
            if individual_data:
                dataset_entry['individual'] = individual_data

            # Print what we loaded
            loaded_components = ["base"]
            if 'tvd_mi' in dataset_entry:
                loaded_components.append("TVD-MI")
            if 'llm_judge_with_context' in dataset_entry:
                loaded_components.append("LLM Judge (with context)")
            if 'llm_judge_without_context' in dataset_entry:
                loaded_components.append("LLM Judge (without context)")
            if 'individual' in dataset_entry:
                individual_components = []
                if 'mi_gppm' in individual_data:
                    individual_components.append(f"MI/GPPM ({len(individual_data['mi_gppm'])} files)")
                if 'tvd_mi' in individual_data:
                    individual_components.append(f"TVD-MI ({len(individual_data['tvd_mi'])} files)")
                if 'judge_with' in individual_data:
                    individual_components.append(f"Judge+C ({len(individual_data['judge_with'])} files)")
                if 'judge_without' in individual_data:
                    individual_components.append(f"Judge-C ({len(individual_data['judge_without'])} files)")
                if individual_components:
                    loaded_components.append(f"Individual: {', '.join(individual_components)}")
            
            print(f"Loaded dataset: {base_name} ({', '.join(loaded_components)})")
            datasets[base_name] = dataset_entry

    return datasets

def categorize_condition(condition, task_config):
    """Categorize conditions for analysis based on task configuration."""
    for cat, conditions in task_config['categories'].items():
        if condition in conditions:
            return cat
    return 'Other'

def create_binary_categories(df, task_config):
    """Group conditions into Good Faith vs Problematic categories."""

    # Add original category
    df['category'] = df['condition'].apply(lambda x: categorize_condition(x, task_config))

    # Good Faith: Faithful + Style (trying to help, even if biased)
    good_faith_cats = ['Faithful', 'Style']

    # Problematic: Strategic + Low Effort (gaming or shirking)
    problematic_cats = ['Strategic', 'Low Effort']

    df['binary_category'] = df['category'].apply(
        lambda x: 'Good Faith' if x in good_faith_cats 
                 else 'Problematic' if x in problematic_cats 
                 else 'Other'
    )

    # Filter out 'Other' for clean binary comparison
    binary_df = df[df['binary_category'] != 'Other'].copy()

    print(f"Binary categories created:")
    print(f"  Good Faith: {len(binary_df[binary_df['binary_category'] == 'Good Faith'])} conditions")
    print(f"  Problematic: {len(binary_df[binary_df['binary_category'] == 'Problematic'])} conditions")

    return binary_df

def prepare_data(dataset_data, task_config):
    """Prepare comprehensive dataframe with all metrics."""
    validation = dataset_data['validation']
    mechanism = dataset_data['mechanism']

    conditions = mechanism['condition_keys']
    mi_scores = mechanism['combined_avgs_avg']
    gppm_scores = mechanism['gppm_normalized_avg']
    response_lengths = mechanism.get('response_lengths_avg', [])

    # Extract TVD-MI scores if available
    tvd_mi_scores = None
    if 'tvd_mi' in dataset_data:
        tvd_mi_data = dataset_data['tvd_mi']
        tvd_mi_scores = tvd_mi_data.get('tvd_mi_bidirectional_avg', tvd_mi_data.get('tvd_mi_scores_avg'))

    # Extract LLM judge scores if available
    llm_judge_with_context = []
    llm_judge_without_context = []
    
    if 'llm_judge_with_context' in dataset_data:
        llm_judge_with_context_data = dataset_data['llm_judge_with_context']
        llm_judge_with_context = llm_judge_with_context_data.get('win_rates_avg', [])

    if 'llm_judge_without_context' in dataset_data:
        llm_judge_without_context_data = dataset_data['llm_judge_without_context']
        llm_judge_without_context = llm_judge_without_context_data.get('win_rates_avg', [])
    
    print(f"Debug: Found {len(llm_judge_with_context)} LLM judge (with context) scores")
    print(f"Debug: Found {len(llm_judge_without_context)} LLM judge (without context) scores")

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

    # Create dataframe
    data = []
    for i, condition in enumerate(conditions[1:]):
        if condition in baseline_scores:
            row = {
                'condition': condition,
                'baseline': baseline_scores[condition],
                'mi': mi_scores[i],
                'gppm': -gppm_scores[i],  # Flip sign for easier comparison (higher = better)
                'gppm_inv': gppm_scores[i]  # Keep original for backwards compatibility
            }

            if i < len(response_lengths):
                row['length'] = response_lengths[i]

            if tvd_mi_scores and i < len(tvd_mi_scores):
                row['tvd_mi'] = tvd_mi_scores[i]

            if i < len(llm_judge_with_context):
                row['llm_judge_with'] = llm_judge_with_context[i]

            if i < len(llm_judge_without_context):
                row['llm_judge_without'] = llm_judge_without_context[i]

            data.append(row)

    df = pd.DataFrame(data)
    
    print(f"Debug: DataFrame columns: {list(df.columns)}")
    print(f"Debug: DataFrame shape: {df.shape}")

    # Create binary categories
    binary_df = create_binary_categories(df, task_config)

    return binary_df, baseline_type

def extract_individual_scores(individual_data, conditions, task_config):
    """Extract individual scores for each condition and metric from raw data."""
    condition_scores = {condition: {} for condition in conditions}

    # Extract MI/GPPM scores
    if 'mi_gppm' in individual_data:
        for example_data in individual_data['mi_gppm']:
            example_conditions = example_data.get('condition_keys', [])
            combined_avgs = example_data.get('combined_avgs', [])
            gppm_scores = example_data.get('gppm_normalized', [])

            for i, condition in enumerate(example_conditions):
                if condition in conditions:
                    if 'mi' not in condition_scores[condition]:
                        condition_scores[condition]['mi'] = []
                        condition_scores[condition]['gppm'] = []

                    if i < len(combined_avgs):
                        condition_scores[condition]['mi'].append(combined_avgs[i])
                    if i < len(gppm_scores):
                        condition_scores[condition]['gppm'].append(-gppm_scores[i])  # Flip sign

    # Extract TVD-MI scores
    if 'tvd_mi' in individual_data:
        for example_data in individual_data['tvd_mi']:
            example_conditions = example_data.get('condition_keys', [])
            tvd_mi_scores = example_data.get('tvd_mi_bidirectional', [])

            for i, condition in enumerate(example_conditions):
                if condition in conditions:
                    if 'tvd_mi' not in condition_scores[condition]:
                        condition_scores[condition]['tvd_mi'] = []

                    if i < len(tvd_mi_scores):
                        condition_scores[condition]['tvd_mi'].append(tvd_mi_scores[i])

    # Extract LLM Judge scores - make sure the keys match
    for judge_key, data_key in [('llm_judge_with', 'judge_with'), ('llm_judge_without', 'judge_without')]:
        if data_key in individual_data:
            for example_data in individual_data[data_key]:
                example_conditions = example_data.get('condition_keys', [])
                win_rates = example_data.get('win_rates', [])

                for i, condition in enumerate(example_conditions):
                    if condition in conditions:
                        if judge_key not in condition_scores[condition]:
                            condition_scores[condition][judge_key] = []

                        if i < len(win_rates):
                            condition_scores[condition][judge_key].append(win_rates[i])

    return condition_scores

def bootstrap_cohens_d_ci(diffs, n_bootstrap=2000, seed=None):
    """Bootstrap confidence interval for Cohen's d."""
    rng = np.random.default_rng(seed)
    boot_d_values = []

    n_items = len(diffs)
    for _ in range(n_bootstrap):
        # Resample differences
        boot_indices = rng.choice(n_items, size=n_items, replace=True)
        boot_diffs = diffs.iloc[boot_indices] if hasattr(diffs, 'iloc') else diffs[boot_indices]

        boot_mean = np.mean(boot_diffs)
        boot_std = np.std(boot_diffs, ddof=1)

        if boot_std > 0:
            boot_d = boot_mean / boot_std
        else:
            boot_d = 0
        boot_d_values.append(boot_d)

    return np.percentile(boot_d_values, [2.5, 97.5])


def bootstrap_statistical_tests(
    df: pd.DataFrame,
    individual_data: dict,
    conditions: list[str],
    task_config: dict,
    *,
    n_bootstrap: int = 2000,
    seed: int | None = None,
    verbose: bool = False,
) -> dict[str, dict]:
    """
    Paired-item bootstrap for Good-Faith vs Problematic discrimination.

    ─ Calculation notes ────────────────────────────────────────────────
    • Unit of resampling = prompt/item (keeps condition-level dependence).
    • Effect size = Cohen’s d on the vector of item-wise differences
      (denominator = SD of those differences).
    • p-value = permutation test via random sign-flip of diffs.
    • 95 % CI = percentile CI on the mean difference (Δ), not on d.
    """

    rng = np.random.default_rng(seed)
    metrics = ["mi", "gppm", "tvd_mi", "llm_judge_with", "llm_judge_without"]
    condition_scores = extract_individual_scores(individual_data, conditions, task_config)

    # Map conditions to categories...
    good_faith_cond = df.loc[df["binary_category"] == "Good Faith", "condition"].tolist()
    problematic_cond = df.loc[df["binary_category"] == "Problematic", "condition"].tolist()

    results: dict = {}

    for metric in metrics:
        # Skip metrics missing entirely
        if not any(metric in condition_scores.get(c, {}) for c in conditions):
            if verbose:
                print(f"{metric}: no individual data available, skipping bootstrap analysis")
            continue  # Just skip this metric, don't skip all

        # Build item × condition table, padding with NaNs
        max_len = max(len(condition_scores[c][metric]) for c in conditions
                      if metric in condition_scores[c])
        table = pd.DataFrame(index=range(max_len), columns=conditions, dtype=float)

        for cond in conditions:
            scores = condition_scores.get(cond, {}).get(metric, [])
            if scores:
                table.loc[: len(scores) - 1, cond] = scores

        # Row-wise means for each category; drop rows with no data in either side
        faithful_means   = table[good_faith_cond].mean(axis=1, skipna=True)
        unfaithful_means = table[problematic_cond].mean(axis=1, skipna=True)
        diffs = (faithful_means - unfaithful_means).dropna()

        n_items = len(diffs)
        if n_items < 2 or diffs.std(ddof=1) == 0:
            if verbose:
                print(f"{metric}: insufficient data, skipping.")
            continue

        observed_diff = diffs.mean()

        # ───────── Bootstrap & permutation ─────────
        boot_diffs = rng.choice(diffs, size=(n_bootstrap, n_items), replace=True).mean(axis=1)
        signs = rng.choice([-1, 1], size=(n_bootstrap, n_items))
        null_diffs = (signs * diffs.values).mean(axis=1)

        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        p_value = np.mean(np.abs(null_diffs) >= abs(observed_diff))
        cohens_d = observed_diff / diffs.std(ddof=1)

        # ───────── Package result ─────────
        # After calculating cohens_d, add:
        d_ci_low, d_ci_high = bootstrap_cohens_d_ci(diffs, n_bootstrap=n_bootstrap, seed=seed)

        results[metric] = {
            "statistic": observed_diff / (diffs.std(ddof=1) / np.sqrt(n_items)),
            "p_value": p_value,
            "cohens_d": cohens_d,
            "cohens_d_ci": [d_ci_low, d_ci_high],  # Add this
            "good_faith_mean": faithful_means.mean(),
            "good_faith_std": faithful_means.std(ddof=1),
            "good_faith_n": n_items,
            "problematic_mean": unfaithful_means.mean(),
            "problematic_std": unfaithful_means.std(ddof=1),
            "problematic_n": n_items,
            "bootstrap_ci": [ci_low, ci_high],
            "method": "item-bootstrap",
        }


        if verbose:
            print(f"{metric}: Δ={observed_diff:.3f}, 95 % CI=({ci_low:.3f},{ci_high:.3f}), "
                  f"p={p_value:.3g}, d={cohens_d:.2f}, n={n_items}")

    return results

def run_statistical_tests(df, individual_data=None, conditions=None, task_config=None):
    """Run statistical tests comparing Good Faith vs Problematic on each mechanism."""

    # If we have individual data, use bootstrap approach for metrics that have it
    if individual_data and conditions and task_config:
        print("Using bootstrap statistical tests with individual data...")
        bootstrap_results = bootstrap_statistical_tests(df, individual_data, conditions, task_config)

        # For metrics without individual data (like baseline), fall back to condition-level
        good_faith = df[df['binary_category'] == 'Good Faith']
        problematic = df[df['binary_category'] == 'Problematic']

        # Check for baseline
        if 'baseline' in df.columns and 'baseline' not in bootstrap_results:
            if len(good_faith) > 1 and len(problematic) > 1:
                stat, p_value = stats.ttest_ind(
                    good_faith['baseline'], 
                    problematic['baseline'],
                    equal_var=False
                )

                # Calculate effect size
                good_mean = good_faith['baseline'].mean()
                prob_mean = problematic['baseline'].mean()
                good_std = good_faith['baseline'].std()
                prob_std = problematic['baseline'].std()

                pooled_std = np.sqrt(
                    ((len(good_faith) - 1) * good_std**2 + 
                     (len(problematic) - 1) * prob_std**2) / 
                    (len(good_faith) + len(problematic) - 2)
                )
                cohens_d = (good_mean - prob_mean) / pooled_std if pooled_std > 0 else 0

                bootstrap_results['baseline'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'cohens_d_ci': [None, None],  # No bootstrap CI for condition-level
                    'good_faith_mean': good_mean,
                    'good_faith_std': good_std,
                    'good_faith_n': len(good_faith),
                    'problematic_mean': prob_mean,
                    'problematic_std': prob_std,
                    'problematic_n': len(problematic),
                    'method': 'condition_level'
                }

        return bootstrap_results

def create_discrimination_chart(stats_results, baseline_type, dataset_name, output_dir):
    """Create single chart showing each mechanism's ability to discriminate."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Sort metrics by effect size for ordering
    sorted_metrics_by_effect = sorted(stats_results.items(), key=lambda x: abs(x[1]['cohens_d']), reverse=True)
    metrics = [m[0] for m in sorted_metrics_by_effect]
    
    # Create abbreviated metric names
    metric_names = []
    for m in metrics:
        if m == 'baseline':
            metric_names.append(baseline_type)
        elif m == 'mi':
            metric_names.append('MI')
        elif m == 'gppm':
            metric_names.append('GPPM')
        elif m == 'tvd_mi':
            metric_names.append('TVD-MI')
        elif m == 'llm_judge_with':
            metric_names.append('Judge+C')
        elif m == 'llm_judge_without':
            metric_names.append('Judge-C')
        else:
            metric_names.append(m)  # fallback for any other metrics

    # Chart 1: Effect sizes (discrimination power) - ordered by effect size
    effect_sizes = [abs(stats_results[m]['cohens_d']) for m in metrics]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60', '#9b59b6', '#e67e22']
    
    # Ensure we have enough colors
    if len(metrics) > len(colors):
        colors = colors * ((len(metrics) // len(colors)) + 1)
    colors = colors[:len(metrics)]

    bars = ax1.bar(metric_names, effect_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylabel("Effect Size (|Cohen's d|)", fontsize=12)
    ax1.set_title("Discrimination Power: Good Faith vs Problematic", fontsize=14, fontweight='bold')
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect (d=0.8)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect (d=0.5)')
    ax1.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small Effect (d=0.2)')
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Add p-value annotations
    for i, (bar, metric) in enumerate(zip(bars, metrics)):
        p_val = stats_results[metric]['p_value']
        significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax1.annotate(significance, (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05),
                    ha='center', fontweight='bold', fontsize=12)

    # Chart 2: Group means comparison with error bars
    x = np.arange(len(metrics))
    width = 0.35

    good_faith_means = []
    good_faith_stds = []
    problematic_means = []
    problematic_stds = []
    
    for m in metrics:
        good_mean = stats_results[m]['good_faith_mean']
        prob_mean = stats_results[m]['problematic_mean']
        
        # Flip GPPM sign for display (higher = better)
        if m == 'gppm':
            good_mean = -good_mean
            prob_mean = -prob_mean
            
        good_faith_means.append(good_mean)
        good_faith_stds.append(stats_results[m]['good_faith_std'])
        problematic_means.append(prob_mean)
        problematic_stds.append(stats_results[m]['problematic_std'])

    bars1 = ax2.bar(x - width/2, good_faith_means, width, 
                   yerr=good_faith_stds, capsize=5,
                   label='Good Faith', color='green', alpha=0.7, 
                   edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, problematic_means, width,
                   yerr=problematic_stds, capsize=5,
                   label='Problematic', color='red', alpha=0.7,
                   edgecolor='black', linewidth=1)

    ax2.set_ylabel('Mean Score', fontsize=12)
    ax2.set_title('Mean Scores by Category (±1 SD)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names, rotation=45, ha='right')
    ax2.legend(fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)

    # Overall title
    fig.suptitle(f'{dataset_name}: Binary Category Discrimination Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"{dataset_name}_binary_discrimination.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Discrimination chart saved to {output_path}")

def empirical_power_analysis_samples(stats_results, individual_data, conditions, task_config, alpha=0.05, target_power=0.8):
    """
    Calculate required number of SAMPLES (examples) for non-significant results.
    Uses bootstrap resampling from individual example data.
    """
    
    if not individual_data:
        print("Need individual data for sample-based power analysis")
        return {}
    
    # Extract individual scores
    condition_scores = extract_individual_scores(individual_data, conditions, task_config)
    
    # Group conditions by binary category - need to create a temporary df for this
    temp_data = []
    for condition in conditions:
        temp_data.append({'condition': condition})
    temp_df = pd.DataFrame(temp_data)
    temp_df = create_binary_categories(temp_df, task_config)
    
    good_faith_conditions = temp_df[temp_df['binary_category'] == 'Good Faith']['condition'].tolist()
    problematic_conditions = temp_df[temp_df['binary_category'] == 'Problematic']['condition'].tolist()
    
    power_analysis = {}
    metrics = ['mi', 'gppm', 'tvd_mi', 'judge_with', 'judge_without']
    
    for metric in metrics:
        if metric not in stats_results or stats_results[metric]['p_value'] >= alpha:
            
            # Collect current individual scores
            good_faith_scores = []
            problematic_scores = []
            
            for condition in good_faith_conditions:
                if condition in condition_scores and metric in condition_scores[condition]:
                    good_faith_scores.extend(condition_scores[condition][metric])
            
            for condition in problematic_conditions:
                if condition in condition_scores and metric in condition_scores[condition]:
                    problematic_scores.extend(condition_scores[condition][metric])
            
            if len(good_faith_scores) == 0 or len(problematic_scores) == 0:
                continue
                
            current_n_good = len(good_faith_scores)
            current_n_prob = len(problematic_scores)
            
            # Observed effect size from current data
            observed_diff = np.mean(good_faith_scores) - np.mean(problematic_scores)
            pooled_std = np.sqrt(
                ((current_n_good - 1) * np.var(good_faith_scores, ddof=1) + 
                 (current_n_prob - 1) * np.var(problematic_scores, ddof=1)) / 
                (current_n_good + current_n_prob - 2)
            )
            observed_d = observed_diff / pooled_std if pooled_std > 0 else 0
            
            # Calculate required sample sizes for target power
            z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
            z_beta = norm.ppf(target_power)
            
            if abs(observed_d) > 0:
                # For equal group sizes: n = 2 * (z_alpha + z_beta)^2 / d^2
                n_required_per_group = 2 * ((z_alpha + z_beta) ** 2) / (observed_d ** 2)
                
                # Current power calculation
                current_n_harmonic = 2 / (1/current_n_good + 1/current_n_prob)  # Harmonic mean for unequal groups
                current_ncp = abs(observed_d) * np.sqrt(current_n_harmonic / 2)
                current_power = 1 - norm.cdf(z_alpha - current_ncp) + norm.cdf(-z_alpha - current_ncp)
                
                power_analysis[metric] = {
                    'current_power': current_power,
                    'observed_effect_size': observed_d,
                    'current_n_good': current_n_good,
                    'current_n_prob': current_n_prob,
                    'required_n_per_group': int(np.ceil(n_required_per_group)),
                    'multiplier_good': n_required_per_group / current_n_good,
                    'multiplier_prob': n_required_per_group / current_n_prob,
                    'examples_needed_good': max(0, int(np.ceil(n_required_per_group - current_n_good))),
                    'examples_needed_prob': max(0, int(np.ceil(n_required_per_group - current_n_prob))),
                    'good_faith_scores': good_faith_scores,  # Store for power curve simulation
                    'problematic_scores': problematic_scores
                }
    
    return power_analysis

def print_sample_power_analysis(power_analysis, baseline_type):
    """Print sample-based power analysis results."""
    
    if not power_analysis:
        print("All metrics already significant - no power analysis needed!")
        return
    
    print("\nSAMPLE-BASED POWER ANALYSIS FOR NON-SIGNIFICANT RESULTS")
    print("="*70)
    print(f"Target: 80% power to detect observed effect sizes at α=0.05")
    print("Analysis based on individual example scores, not condition averages")
    print()
    
    for metric, analysis in power_analysis.items():
        metric_name = {
            'mi': 'MI (DoE)',
            'gppm': 'GPPM', 
            'tvd_mi': 'TVD-MI',
            'judge_with': 'LLM Judge (w/ context)',
            'judge_without': 'LLM Judge (w/o context)'
        }.get(metric, metric)
        
        print(f"{metric_name}:")
        print(f"  Current power: {analysis['current_power']:.1%}")
        print(f"  Observed effect size (Cohen's d): {analysis['observed_effect_size']:.3f}")
        print(f"  Current samples - Good Faith: {analysis['current_n_good']}, Problematic: {analysis['current_n_prob']}")
        print(f"  Required per group for 80% power: {analysis['required_n_per_group']} samples")
        
        if analysis['examples_needed_good'] > 0:
            print(f"  Need {analysis['examples_needed_good']} more Good Faith examples")
        else:
            print(f"  Good Faith group already has sufficient samples")
            
        if analysis['examples_needed_prob'] > 0:
            print(f"  Need {analysis['examples_needed_prob']} more Problematic examples")
        else:
            print(f"  Problematic group already has sufficient samples")
        
        # Practical interpretation
        max_multiplier = max(analysis['multiplier_good'], analysis['multiplier_prob'])
        if max_multiplier < 2:
            interpretation = "FEASIBLE - less than 2x current examples needed"
        elif max_multiplier < 3:
            interpretation = "MODERATE - need 2-3x current examples"
        elif max_multiplier < 5:
            interpretation = "CHALLENGING - need 3-5x current examples"
        else:
            interpretation = "DIFFICULT - need >5x current examples"
            
        print(f"  Assessment: {interpretation}")
        print()

def simulate_power_curve_samples(good_faith_scores, problematic_scores, observed_d, max_multiplier=5):
    """Simulate power curve as function of sample size."""
    
    current_n = min(len(good_faith_scores), len(problematic_scores))
    sample_sizes = np.linspace(current_n, current_n * max_multiplier, 50)
    powers = []
    
    for n in sample_sizes:
        n = int(n)
        # Calculate power for this sample size using theoretical formula
        ncp = abs(observed_d) * np.sqrt(n / 2)  # Assuming equal group sizes
        power = 1 - norm.cdf(1.96 - ncp) + norm.cdf(-1.96 - ncp)
        powers.append(power)
    
    return sample_sizes, powers

def plot_power_curves(power_analysis, output_dir, dataset_name):
    """Plot power curves showing relationship between sample size and power."""
    
    if not power_analysis:
        return
        
    fig, axes = plt.subplots(1, len(power_analysis), figsize=(5*len(power_analysis), 4))
    if len(power_analysis) == 1:
        axes = [axes]
    
    for i, (metric, analysis) in enumerate(power_analysis.items()):
        # Generate power curve
        effect_size = analysis['observed_effect_size']
        current_n_good = analysis['current_n_good']
        current_n_prob = analysis['current_n_prob']
        current_n = min(current_n_good, current_n_prob)
        
        sample_sizes, powers = simulate_power_curve_samples(
            analysis['good_faith_scores'], 
            analysis['problematic_scores'], 
            effect_size
        )
        
        axes[i].plot(sample_sizes, powers, 'b-', linewidth=2)
        axes[i].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Power')
        axes[i].axvline(x=current_n, color='green', linestyle='--', alpha=0.7, label='Current N')
        axes[i].axvline(x=analysis['required_n_per_group'], color='orange', linestyle='--', alpha=0.7, label='Required N')
        
        axes[i].set_xlabel('Sample Size per Group')
        axes[i].set_ylabel('Statistical Power')
        axes[i].set_title(f'{metric.upper()}\n(d={effect_size:.3f})')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Add current power annotation
        current_power = analysis['current_power']
        axes[i].annotate(f'Current: {current_power:.1%}', 
                        xy=(current_n, current_power), 
                        xytext=(current_n * 1.5, current_power + 0.1),
                        arrowprops=dict(arrowstyle='->', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_power_analysis_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# Add structured output at the end of main()
def save_structured_results(dataset_name, stats_results, df, task_type, figures_dir):
    """Save structured results for hypothesis testing."""
    structured_results = {
        'dataset_name': dataset_name,
        'task_type': task_type,
        'compression_ratio': get_compression_ratio(dataset_name),
        'stats_results': {},
        'n_good_faith': len(df[df['binary_category'] == 'Good Faith']),
        'n_problematic': len(df[df['binary_category'] == 'Problematic']),
        'conditions': {
            'good_faith': df[df['binary_category'] == 'Good Faith']['condition'].tolist(),
            'problematic': df[df['binary_category'] == 'Problematic']['condition'].tolist()
        }
    }

    # Convert ALL stats results to serializable format
    for metric, results in stats_results.items():
        if results is None:
            continue

        structured_results['stats_results'][metric] = {
            'cohens_d': float(results['cohens_d']),
            'cohens_d_ci': [float(x) if x is not None else None for x in results.get('cohens_d_ci', [None, None])],
            'p_value': float(results['p_value']),
            'good_faith_mean': float(results['good_faith_mean']),
            'good_faith_std': float(results['good_faith_std']),
            'problematic_mean': float(results['problematic_mean']),
            'problematic_std': float(results['problematic_std']),
            'n_samples': int(results.get('good_faith_n', results.get('problematic_n', 0))),
            'method': results.get('method', 'unknown')
        }

    output_path = figures_dir / f"{dataset_name}_structured_results.json"
    with open(output_path, 'w') as f:
        json.dump(structured_results, f, indent=2)

    print(f"Structured results saved to {output_path}")
    print(f"  Included mechanisms: {sorted(structured_results['stats_results'].keys())}")

def generate_advisor_summary(stats_results, baseline_type, dataset_name, output_dir, power_analysis=None):
    """Generate concise summary for advisor meeting."""

    summary_lines = []
    summary_lines.append("BINARY CATEGORY DISCRIMINATION ANALYSIS")
    summary_lines.append("="*50)
    summary_lines.append(f"Dataset: {dataset_name}")
    summary_lines.append("Good Faith: Faithful + Style conditions")
    summary_lines.append("Problematic: Strategic + Low Effort conditions")
    summary_lines.append("")

    # Sort by effect size for ranking
    sorted_metrics = sorted(stats_results.items(), key=lambda x: abs(x[1]['cohens_d']), reverse=True)

    summary_lines.append("DISCRIMINATION RESULTS (ranked by effect size):")
    summary_lines.append("-" * 50)

    for metric, results in sorted_metrics:
        if metric == 'baseline':
            metric_name = baseline_type
        elif metric == 'mi':
            metric_name = 'MI (DoE)'
        elif metric == 'gppm':
            metric_name = 'GPPM'
        elif metric == 'tvd_mi':
            metric_name = 'TVD-MI'
        elif metric == 'llm_judge_with':
            metric_name = 'LLM Judge (w/ context)'
        elif metric == 'llm_judge_without':
            metric_name = 'LLM Judge (w/o context)'
        else:
            metric_name = metric

        # Determine effect size interpretation
        abs_d = abs(results['cohens_d'])
        if abs_d >= 0.8:
            effect_interp = "LARGE"
        elif abs_d >= 0.5:
            effect_interp = "MEDIUM"
        elif abs_d >= 0.2:
            effect_interp = "SMALL"
        else:
            effect_interp = "NEGLIGIBLE"

        # Determine significance
        p_val = results['p_value']
        if p_val < 0.001:
            sig_level = "***"
        elif p_val < 0.01:
            sig_level = "**"
        elif p_val < 0.05:
            sig_level = "*"
        else:
            sig_level = "ns"

        summary_lines.append(f"{metric_name}:")
        summary_lines.append(f"  Good Faith mean: {results['good_faith_mean']:.3f} (±{results['good_faith_std']:.3f}, n={results['good_faith_n']})")
        summary_lines.append(f"  Problematic mean: {results['problematic_mean']:.3f} (±{results['problematic_std']:.3f}, n={results['problematic_n']})")
        summary_lines.append(f"  Effect size: {results['cohens_d']:.3f} ({effect_interp})")
        summary_lines.append(f"  p-value: {results['p_value']:.6f} ({sig_level})")
        if 'bootstrap_ci' in results:
            ci_low, ci_high = results['bootstrap_ci']
            summary_lines.append(f"  Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        summary_lines.append(f"  Method: {results.get('method', 'unknown')}")
        summary_lines.append("")

    # Add power analysis if available
    if power_analysis:
        summary_lines.append("POWER ANALYSIS FOR NON-SIGNIFICANT RESULTS:")
        summary_lines.append("-" * 50)
        for metric, analysis in power_analysis.items():
            metric_name = {
                'mi': 'MI (DoE)',
                'gppm': 'GPPM', 
                'tvd_mi': 'TVD-MI',
                'judge_with': 'LLM Judge (w/ context)',
                'judge_without': 'LLM Judge (w/o context)'
            }.get(metric, metric)
            
            max_multiplier = max(analysis['multiplier_good'], analysis['multiplier_prob'])
            summary_lines.append(f"• {metric_name}: Need {max_multiplier:.1f}x more examples for significance")
        summary_lines.append("")

    summary_lines.append("INTERPRETATION:")
    summary_lines.append("-" * 20)

    # Find best discriminator
    best_metric, best_results = sorted_metrics[0]
    best_name = {
        'baseline': baseline_type,
        'mi': 'MI (DoE)', 
        'gppm': 'GPPM',
        'tvd_mi': 'TVD-MI',
        'llm_judge_with': 'LLM Judge (w/ context)',
        'llm_judge_without': 'LLM Judge (w/o context)'
    }.get(best_metric, best_metric)

    summary_lines.append(f"• Best discriminator: {best_name} (d={best_results['cohens_d']:.3f})")

    # Count significant results
    significant_count = sum(1 for r in stats_results.values() if r['p_value'] < 0.05)
    summary_lines.append(f"• {significant_count}/{len(stats_results)} metrics show significant discrimination (p<0.05)")

    # Practical significance
    large_effect_count = sum(1 for r in stats_results.values() if abs(r['cohens_d']) >= 0.8)
    summary_lines.append(f"• {large_effect_count}/{len(stats_results)} metrics show large practical effect (|d|≥0.8)")

    summary_text = "\n".join(summary_lines)

    # Save to file
    summary_path = output_dir / f"{dataset_name}_binary_analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    # Print to console
    print("\n" + summary_text)

    return summary_text

def create_detailed_breakdown(df, output_dir, dataset_name):
    """Create detailed breakdown of conditions in each category."""

    breakdown_lines = []
    breakdown_lines.append("DETAILED CONDITION BREAKDOWN")
    breakdown_lines.append("="*40)
    breakdown_lines.append("")

    for binary_cat in ['Good Faith', 'Problematic']:
        subset = df[df['binary_category'] == binary_cat]
        breakdown_lines.append(f"{binary_cat.upper()} CONDITIONS ({len(subset)} total):")
        breakdown_lines.append("-" * 30)

        # Group by original category
        for orig_cat in subset['category'].unique():
            cat_subset = subset[subset['category'] == orig_cat]
            breakdown_lines.append(f"  {orig_cat} ({len(cat_subset)}):")
            for _, row in cat_subset.iterrows():
                breakdown_lines.append(f"    - {row['condition']}")
            breakdown_lines.append("")

        breakdown_lines.append("")

    breakdown_text = "\n".join(breakdown_lines)

    # Save to file
    breakdown_path = output_dir / f"{dataset_name}_condition_breakdown.txt"
    with open(breakdown_path, 'w') as f:
        f.write(breakdown_text)

    print(breakdown_text)

def main():
    parser = argparse.ArgumentParser(description='Binary category discrimination analysis for advisor meeting')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing validation and mechanism results')
    parser.add_argument('--figures-dir', type=str, default='results/figures',
                        help='Output directory for figures and summaries')

    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True, parents=True)

    # Load results
    datasets = load_results(args.results_dir)

    for dataset_name, dataset_data in datasets.items():
        print(f"\n" + "="*60)
        print(f"PROCESSING: {dataset_name}")
        print("="*60)

        # Detect task type and load config
        task_type = detect_task_type(dataset_name)
        task_config = load_task_config(task_type)
        print(f"Task type: {task_type}")

        try:
            # Prepare data with binary categories
            df, baseline_type = prepare_data(dataset_data, task_config)
            print(f"Baseline metric: {baseline_type}")

            # Run statistical tests
            conditions = dataset_data['mechanism']['condition_keys']
            individual_data = dataset_data.get('individual', None)
            stats_results = run_statistical_tests(df, individual_data, conditions, task_config)

            # Create discrimination chart
            create_discrimination_chart(stats_results, baseline_type, dataset_name, figures_dir)

            # Run power analysis for non-significant results
            power_results = None
            if individual_data:
                print("Running Power Analysis:")
                power_results = empirical_power_analysis_samples(
                    stats_results, individual_data, conditions, task_config
                )
                
                # Print power analysis results
                if power_results:
                    print_sample_power_analysis(power_results, baseline_type)
                    
                    # Create power curves for non-significant metrics
                    plot_power_curves(power_results, figures_dir, dataset_name)

            # Generate advisor summary
            generate_advisor_summary(stats_results, baseline_type, dataset_name, figures_dir, power_results)

            # Create detailed breakdown
            create_detailed_breakdown(df, figures_dir, dataset_name)
            save_structured_results(dataset_name, stats_results, df, task_type, figures_dir)
            
            print(f"\nAnalysis complete for {dataset_name}")
            print(f"Files saved to {figures_dir}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

