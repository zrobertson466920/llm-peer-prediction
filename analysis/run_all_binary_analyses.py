#!/usr/bin/env python3
"""
Run binary category analysis on all results directories and aggregate outputs.
"""

import subprocess
import json
from pathlib import Path
import shutil
import argparse
from datetime import datetime
from scipy.stats import false_discovery_control 

# Directory to compression ratio mapping
DIRECTORY_TO_COMPRESSION = {
    'opus_translation_results': 1.3,          # Opus Books
    'MT14_config_results': 1.1,               # WMT14
    'samsum_200_config_results': 4.8,         # SamSum
    'pubmed_200_config_results': 6.7,         # PubMed
    'multi_news_200_config_results': 9.0,     # Multi-News
    'billsum_200_config_results': 9.3,        # BillSum
    'cnn_dailymail_config_results': 13.8,     # CNN/DailyMail
    'DailyCNN_config_results': 13.8,          # Alternative CNN/DailyMail naming
    'reddit_tifu_200_config_results': 16.1,   # Reddit TIFU
    'xsum_200_config_results': 18.5,          # XSum
    'peer_review_results_100': 20.2,          # ICLR 2023
}

# Display name mapping for cleaner output
DIRECTORY_TO_DISPLAY_NAME = {
    'opus_translation_results': 'Opus Books Translation',
    'MT14_config_results': 'WMT14 Translation',
    'samsum_200_config_results': 'SamSum',
    'pubmed_200_config_results': 'PubMed',
    'multi_news_200_config_results': 'Multi-News',
    'billsum_200_config_results': 'BillSum',
    'cnn_dailymail_config_results': 'CNN/DailyMail',
    'DailyCNN_config_results': 'CNN/DailyMail',
    'reddit_tifu_200_config_results': 'Reddit TIFU',
    'xsum_200_config_results': 'XSum',
    'peer_review_results_100': 'ICLR 2023 Peer Review',
}

def find_results_directories(base_dir):
    """Find all directories containing 'results' in their name, excluding transformed ablations."""
    base_path = Path(base_dir)
    results_dirs = []

    # Patterns to exclude (for ablation studies)
    exclude_patterns = [
        'case_flip_transformed',
        'format_transformed', 
        'padding_transformed',
        'pattern_transformed',
        'repetition_transformed',
        'transformed_results'  # catch-all for any transformed results
    ]

    # Find all directories with 'results' in the name
    for path in base_path.rglob('*results*'):
        if path.is_dir() and not path.name.startswith('.'):
            # Check if any exclude pattern matches
            if any(pattern in path.name for pattern in exclude_patterns):
                continue

            # Check if it contains validation files
            if any(path.glob('*_validation.json')):
                results_dirs.append(path)

    return sorted(results_dirs)

def run_binary_analysis(results_dir, output_base_dir):
    """Run binary_cat_analysis.py on a single results directory."""
    # Create output directory for this analysis
    dir_name = results_dir.name
    if results_dir.parent != Path('.'):
        dir_name = f"{results_dir.parent.name}_{dir_name}"

    output_dir = output_base_dir / dir_name / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the analysis
    cmd = [
        'python', 'analysis/binary_cat_analysis.py',
        '--results-dir', str(results_dir),
        '--figures-dir', str(output_dir)
    ]

    print(f"\nRunning analysis on: {results_dir}")
    print(f"Output directory: {output_dir}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Analysis failed for {results_dir}")
            print(f"STDERR: {result.stderr}")
            return False, output_dir
        else:
            print(f"SUCCESS: Analysis completed for {results_dir}")
            return True, output_dir
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False, output_dir

def get_compression_from_path(file_path):
    """Extract compression ratio from the file path."""
    path = Path(file_path)

    # Look for the parent directory that matches our mapping
    for parent in path.parents:
        # Check both with and without eval_results_ prefix
        clean_name = parent.name.replace('eval_results_', '')
        if clean_name in DIRECTORY_TO_COMPRESSION:
            return DIRECTORY_TO_COMPRESSION[clean_name]
        if parent.name in DIRECTORY_TO_COMPRESSION:
            return DIRECTORY_TO_COMPRESSION[parent.name]

    return None

def get_display_name_from_path(file_path):
    """Extract display name from the file path."""
    path = Path(file_path)

    for parent in path.parents:
        # Check both with and without eval_results_ prefix
        clean_name = parent.name.replace('eval_results_', '')
        if clean_name in DIRECTORY_TO_DISPLAY_NAME:
            return DIRECTORY_TO_DISPLAY_NAME[clean_name]
        if parent.name in DIRECTORY_TO_DISPLAY_NAME:
            return DIRECTORY_TO_DISPLAY_NAME[parent.name]

    return None

def aggregate_structured_results(output_base_dir):
    """Collect all structured results into a single file with compression ratios fixed."""
    all_results = []

    # Debug: Show what we're looking for
    print(f"\nSearching for structured results in: {output_base_dir}")

    # Look for all structured results files
    json_files = list(output_base_dir.rglob('*_structured_results.json'))
    print(f"Found {len(json_files)} structured result files")

    if len(json_files) == 0:
        # Try looking in the existing aggregated_results directory
        alt_dir = Path('aggregated_results')
        if alt_dir.exists():
            print(f"\nNo files in {output_base_dir}, checking {alt_dir}")
            json_files = list(alt_dir.rglob('*_structured_results.json'))
            print(f"Found {len(json_files)} structured result files in {alt_dir}")

    for json_file in json_files:
        try:
            print(f"  Loading: {json_file}")
            with open(json_file, 'r') as f:
                data = json.load(f)

                # Get compression ratio from path
                compression = get_compression_from_path(json_file)
                if compression is not None:
                    data['compression_ratio'] = compression

                # Get display name from path
                display_name = get_display_name_from_path(json_file)
                if display_name is not None:
                    data['display_name'] = display_name
                else:
                    data['display_name'] = data['dataset_name']

                data['source_file'] = str(json_file)
                data['parent_directory'] = json_file.parent.parent.name

                all_results.append(data)

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    # Sort by compression ratio (handle None values)
    all_results.sort(key=lambda x: x.get('compression_ratio') if x.get('compression_ratio') is not None else float('inf'))

    # Save aggregated results
    aggregate_file = output_base_dir / 'all_domains_results.json'
    with open(aggregate_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAggregated {len(all_results)} domain results to {aggregate_file}")
    return all_results

def apply_global_fdr_correction(all_results, alpha=0.05):
    """Apply global FDR correction across all mechanism-domain pairs."""
    from statsmodels.stats.multitest import multipletests

    # Collect all p-values with their locations
    p_values = []
    locations = []  # (result_idx, mechanism) tuples

    mechanisms = ['baseline', 'mi', 'gppm', 'tvd_mi', 'llm_judge_with', 'llm_judge_without']

    # Collect all p-values
    for i, result in enumerate(all_results):
        for mechanism in mechanisms:
            if mechanism in result['stats_results']:
                p_values.append(result['stats_results'][mechanism]['p_value'])
                locations.append((i, mechanism))

    if not p_values:
        print("Warning: No p-values found for FDR correction")
        return all_results

    print(f"\nApplying global FDR correction to {len(p_values)} tests...")

    # Apply Benjamini-Hochberg
    reject, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=alpha, method='fdr_bh', returnsorted=False
    )

    # Update results with FDR-corrected values
    for j, (result_idx, mechanism) in enumerate(locations):
        # Convert numpy types to Python native types for JSON serialization
        all_results[result_idx]['stats_results'][mechanism]['p_value_fdr'] = float(p_adjusted[j])
        all_results[result_idx]['stats_results'][mechanism]['significant_fdr'] = bool(reject[j])

    # Report summary
    n_sig_original = sum(p < 0.05 for p in p_values)
    n_sig_fdr = sum(reject)

    print(f"Original significant tests (p<0.05): {n_sig_original}/{len(p_values)}")
    print(f"FDR-corrected significant tests: {n_sig_fdr}/{len(p_values)}")
    print(f"Tests that lost significance: {n_sig_original - n_sig_fdr}")

    return all_results

def format_effect_size(stats, use_fdr=True):
    """Format effect size with CI-based labels and FDR-corrected significance."""
    if stats is None:
        return "--"

    d = stats['cohens_d']
    ci = stats.get('cohens_d_ci', [None, None])

    # Use FDR-corrected p-value if available and requested
    if use_fdr and 'p_value_fdr' in stats:
        p = stats['p_value_fdr']
    else:
        p = stats['p_value']

    # Format the effect size value with CI if available
    if ci[0] is not None and ci[1] is not None:
        ci_lower = ci[0]
        ci_upper = ci[1]

        # Calculate CI half-width using the maximum of the two sides
        left_width = d - ci_lower
        right_width = ci_upper - d
        ci_half_width = max(left_width, right_width)

        # Format as mean ± CI_width/2
        value_str = f"{d:.2f} ± {ci_half_width:.2f}"

        # Add qualitative label if CI overlaps zero
        if ci_lower <= 0 <= ci_upper:
            value_str += " (ns)"
    else:
        # No CI available, just show the value
        value_str = f"{d:.2f}"

    # Apply formatting based on significance
    if p < 0.001:
        return r"\textbf{" + value_str + "}"
    elif p < 0.05:
        return value_str
    else:
        return r"\textcolor{gray}{" + value_str + "}"

def create_latex_table(all_results, output_base_dir, use_fdr=True):
    """Create a LaTeX table of effect sizes with FDR correction."""
    # Start building the LaTeX table
    fdr_note = " P-values adjusted using Benjamini-Hochberg FDR correction." if use_fdr else ""

    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Effect sizes (Cohen's $d$) for discrimination between Good Faith and Problematic agents." +
        r" Cells show mean ± half-width of 95\% CI. (ns) = CI overlaps zero." +
        r" Bold = $p<0.001$, regular = $p<0.05$, gray = non-significant." + fdr_note + r"}",
        r"\label{tab:effect_sizes}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"\textbf{Domain (Compression)} & \textbf{Baseline} & \textbf{MI (DoE)} & \textbf{GPPM} & \textbf{TVD-MI} & \textbf{Judge (ctx)} & \textbf{Judge (no ctx)} \\",
        r"\midrule"
    ]
    
    # Group results by task type
    translation_results = []
    summarization_results = []
    peer_review_results = []
    
    for result in all_results:
        if result['task_type'] == 'translation':
            translation_results.append(result)
        elif result['task_type'] == 'summarization':
            summarization_results.append(result)
        elif result['task_type'] == 'peer_review':
            peer_review_results.append(result)
    
    # Process each task type
    for task_name, results in [
        ("Translation", translation_results),
        ("Summarization", summarization_results),
        ("Peer Review", peer_review_results)
    ]:
        if results:
            latex_lines.append(r"\multicolumn{7}{l}{\textit{" + task_name + r"}} \\")
            
            for result in results:
                # Get display name and compression
                display_name = result.get('display_name', result['dataset_name'])
                compression = result.get('compression_ratio')
                
                # Shorten display names for the table
                name_map = {
                    'WMT14 Translation': 'WMT14',
                    'Opus Books Translation': 'Opus Books',
                    'CNN/DailyMail': 'CNN/Daily',
                    'ICLR 2023 Peer Review': 'ICLR 2023'
                }
                short_name = name_map.get(display_name, display_name)
                
                if compression is not None:
                    row_name = f"{short_name} ({compression:.1f}:1)"
                else:
                    row_name = short_name
                
                # Get effect sizes for each mechanism
                baseline = format_effect_size(result['stats_results'].get('baseline'))
                mi = format_effect_size(result['stats_results'].get('mi'))
                gppm = format_effect_size(result['stats_results'].get('gppm'))
                tvd_mi = format_effect_size(result['stats_results'].get('tvd_mi'))
                judge_with = format_effect_size(result['stats_results'].get('llm_judge_with'))
                judge_without = format_effect_size(result['stats_results'].get('llm_judge_without'))
                
                latex_lines.append(f"{row_name} & {baseline} & {mi} & {gppm} & {tvd_mi} & {judge_with} & {judge_without} \\\\")
            
            # Add midrule after each section except the last
            if task_name != "Peer Review":
                latex_lines.append(r"\midrule")
    
    # Add success counts
    latex_lines.append(r"\midrule")
    
    # Count successes (d > 0.5) for each mechanism
    mechanisms = ['baseline', 'mi', 'gppm', 'tvd_mi', 'llm_judge_with', 'llm_judge_without']
    success_counts = []
    
    for mechanism in mechanisms:
        successes = 0
        total = 0
        for result in all_results:
            if mechanism in result['stats_results']:
                total += 1
                if result['stats_results'][mechanism]['cohens_d'] > 0.5:
                    successes += 1
        
        if total > 0:
            success_counts.append(f"{successes}/{total}")
        else:
            success_counts.append("--")
    
    latex_lines.append(r"\textbf{Success (d > 0.5)} & " + " & ".join(success_counts) + r" \\")
    
    # Close the table
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    
    # Save the LaTeX table
    latex_file = output_base_dir / 'effect_sizes_table.tex'
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"\nLaTeX table saved to {latex_file}")

def create_summary_report(all_results, output_base_dir):
    """Create a summary report of all analyses."""
    report_lines = [
        "BINARY CATEGORY ANALYSIS - ALL DOMAINS SUMMARY",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total domains analyzed: {len(all_results)}",
        "",
        "DOMAINS BY COMPRESSION RATIO:",
        "-" * 60
    ]

    # Group by task type
    translation_domains = []
    summarization_domains = []
    peer_review_domains = []

    for result in all_results:
        if result['task_type'] == 'translation':
            translation_domains.append(result)
        elif result['task_type'] == 'summarization':
            summarization_domains.append(result)
        elif result['task_type'] == 'peer_review':
            peer_review_domains.append(result)

    # Display by task type
    for task_type, domains in [
        ("TRANSLATION TASKS", translation_domains),
        ("SUMMARIZATION TASKS", summarization_domains),
        ("PEER REVIEW TASKS", peer_review_domains)
    ]:
        if domains:
            report_lines.append(f"\n{task_type}:")
            for result in domains:
                name = result.get('display_name', result['dataset_name'])
                compression = result.get('compression_ratio')
                compression_str = f"{compression:4.1f}:1" if compression is not None else "Unknown"
                n_good = result['n_good_faith']
                n_prob = result['n_problematic']

                report_lines.append(f"  {name:28} | Compression: {compression_str:8} | GF: {n_good:2} | Prob: {n_prob:2}")

    report_lines.extend([
        "",
        "EFFECT SIZES BY MECHANISM:",
        "-" * 60
    ])

    # Collect effect sizes by mechanism
    mechanisms = ['baseline', 'mi', 'gppm', 'tvd_mi', 'llm_judge_with', 'llm_judge_without']
    mechanism_names = {
        'baseline': 'BASELINE',
        'mi': 'MI (DoE)',
        'gppm': 'GPPM',
        'tvd_mi': 'TVD-MI',
        'llm_judge_with': 'LLM JUDGE (WITH CONTEXT)',
        'llm_judge_without': 'LLM JUDGE (WITHOUT CONTEXT)'
    }

    for mechanism in mechanisms:
        report_lines.append(f"\n{mechanism_names.get(mechanism, mechanism.upper())}:")
        found_any = False

        for result in all_results:
            if mechanism in result['stats_results']:
                found_any = True
                stats = result['stats_results'][mechanism]
                d = stats['cohens_d']
                p = stats['p_value']
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

                display_name = result.get('display_name', result['dataset_name'])
                if len(display_name) > 30:
                    display_name = display_name[:27] + '...'

                d_ci = stats.get('cohens_d_ci', [None, None])
                if d_ci[0] is not None and d_ci[1] is not None:
                    report_lines.append(f"  {display_name:30} | d={d:6.3f}, 95% CI: [{d_ci[0]:6.3f}, {d_ci[1]:6.3f}] {sig}")
                else:
                    report_lines.append(f"  {display_name:30} | d={d:6.3f} {sig}")

        if not found_any:
            report_lines.append("  No results found for this mechanism")

    # Add H1a preview
    report_lines.extend([
        "",
        "H1a TEST PREVIEW: All mechanisms achieve d > 0.5?",
        "-" * 60
    ])

    for mechanism in ['mi', 'gppm', 'tvd_mi', 'llm_judge_with', 'llm_judge_without']:
        effect_sizes = []
        for result in all_results:
            if mechanism in result['stats_results']:
                d = abs(result['stats_results'][mechanism]['cohens_d'])
                effect_sizes.append(d)

        if effect_sizes:
            all_above = all(d > 0.5 for d in effect_sizes)
            status = "✓ PASS" if all_above else "✗ FAIL"
            min_d = min(effect_sizes)
            report_lines.append(f"{mechanism_names[mechanism]:25} | {status} | Min d: {min_d:.3f}")

    report_text = "\n".join(report_lines)

    report_file = output_base_dir / 'all_domains_summary.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)

    print(f"\nSummary report saved to {report_file}")
    print("\n" + report_text)

def main():
    parser = argparse.ArgumentParser(description='Run binary analysis on all results directories')
    parser.add_argument('--base-dir', type=str, default='eval_results',
                        help='Base directory to search for results folders')
    parser.add_argument('--output-dir', type=str, default='aggregated_results',
                        help='Output directory for aggregated results')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip running analyses, just aggregate existing results')
    parser.add_argument('--no-fdr', action='store_true',
                        help='Do not apply FDR correction')
    args = parser.parse_args()

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(exist_ok=True)

    if not args.skip_analysis:
        # Find all results directories
        results_dirs = find_results_directories(args.base_dir)
        print(f"Found {len(results_dirs)} results directories:")
        for d in results_dirs:
            print(f"  - {d}")

        # Run analysis on each
        successful = 0
        failed = 0

        for results_dir in results_dirs:
            success, output_dir = run_binary_analysis(results_dir, output_base_dir)
            if success:
                successful += 1
            else:
                failed += 1

        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE: {successful} successful, {failed} failed")
        print(f"{'='*60}")

    # Aggregate all structured results
    all_results = aggregate_structured_results(output_base_dir)

    # Apply global FDR correction unless disabled
    if not args.no_fdr:
        all_results = apply_global_fdr_correction(all_results)

    # Save the FDR-corrected results
    aggregate_file = output_base_dir / 'all_domains_results_fdr.json'
    with open(aggregate_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create summary report
    create_summary_report(all_results, output_base_dir)

    # Create LaTeX table with FDR correction
    create_latex_table(all_results, output_base_dir, use_fdr=not args.no_fdr)

    print(f"\nAll outputs saved to: {output_base_dir}")

    print("Next step: Run hypothesis testing with test_h1_hypotheses.py")

if __name__ == "__main__":
    main()
