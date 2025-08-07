import json
import numpy as np
from pathlib import Path

def load_all_transformations():
    """Load results from all transformation experiments."""
    transformations = ['case_flip', 'format', 'padding', 'pattern']
    results = {}

    for transform in transformations:
        json_path = Path(f'ablation_results/ablation_analysis_{transform}/paired_analysis_results.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                results[transform] = json.load(f)

    return results

def generate_comprehensive_table(results):
    """Generate a comprehensive LaTeX table with averages."""

    mechanisms = ['tvd_mi', 'mi', 'gppm', 'judge_with_context', 'judge_without_context']
    mech_names = {
        'tvd_mi': 'TVD-MI',
        'mi': 'MI (DoE)', 
        'gppm': 'GPPM',
        'judge_with_context': 'Judge (w/ ctx)',
        'judge_without_context': 'Judge (w/o ctx)'
    }

    transforms = ['case_flip', 'format', 'padding', 'pattern']

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Effects of adversarial transformations on Reddit TIFU summarization. Score changes and discrimination")
    print("degradation across four transformation types, with averages showing overall robustness.}")
    print("\\label{tab:tampering_comprehensive}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Mechanism} & \\textbf{Case Flip} & \\textbf{Format} & \\textbf{Padding} & \\textbf{Pattern} & \\textbf{Average} \\\\")
    print("\\midrule")

    # Score changes section
    print("\\multicolumn{6}{l}{\\textit{Score Changes}} \\\\")

    for mech in mechanisms:
        row = [mech_names[mech]]
        score_changes = []

        for transform in transforms:
            if transform in results and 'paired_tests' in results[transform] and mech in results[transform]['paired_tests']:
                test_data = results[transform]['paired_tests'][mech]
                mean_diff = test_data['mean_difference']
                p_val = test_data['p_value']

                # Convert to percentage
                pct_change = mean_diff * 100
                score_changes.append(pct_change)

                # Format with significance and sign
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = ""

                # Format: always show sign, one decimal place
                if abs(pct_change) >= 10:
                    row.append(f"{pct_change:+.0f}\\%{sig}")
                else:
                    row.append(f"{pct_change:+.1f}\\%{sig}")
            else:
                row.append("--")

        # Calculate average
        if score_changes:
            avg = np.mean(score_changes)
            if abs(avg) >= 10:
                row.append(f"{avg:+.0f}\\%")
            else:
                row.append(f"{avg:+.1f}\\%")
        else:
            row.append("--")

        print(" & ".join(row) + " \\\\")

    print("\\midrule")

    # Discrimination degradation section
    print("\\multicolumn{6}{l}{\\textit{Discrimination Degradation (\\% change in Cohen's d)}} \\\\")

    for mech in mechanisms:
        row = [mech_names[mech]]
        disc_changes = []

        for transform in transforms:
            if transform in results and 'category_effect_sizes' in results[transform] and mech in results[transform]['category_effect_sizes']:
                effect_data = results[transform]['category_effect_sizes'][mech]
                original_d = effect_data['original_effect_size']
                transformed_d = effect_data['transformed_effect_size']
                pct_change = ((transformed_d - original_d) / abs(original_d)) * 100 if original_d != 0 else 0
                disc_changes.append(pct_change)

                # Format with consistent style
                if pct_change < -30:  # Severe degradation
                    if abs(pct_change) >= 100:
                        row.append(f"\\textcolor{{red}}{{{pct_change:.0f}\\%}}")
                    else:
                        row.append(f"\\textcolor{{red}}{{{pct_change:.0f}\\%}}")
                else:
                    if abs(pct_change) >= 100:
                        row.append(f"{pct_change:+.0f}\\%")
                    else:
                        row.append(f"{pct_change:+.0f}\\%")
            else:
                row.append("--")

        # Calculate average
        if disc_changes:
            avg = np.mean(disc_changes)
            # Special handling for extreme values
            if abs(avg) >= 100:
                if avg < -30:
                    row.append(f"\\textcolor{{red}}{{{avg:.0f}\\%}}")
                else:
                    row.append(f"{avg:+.0f}\\%")
            else:
                if avg < -30:
                    row.append(f"\\textcolor{{red}}{{{avg:.0f}\\%}}")
                else:
                    row.append(f"{avg:+.0f}\\%")
        else:
            row.append("--")

        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def main():
    # Load all transformation results
    results = load_all_transformations()

    print(f"% Loaded results for transformations: {list(results.keys())}\n")

    # Generate comprehensive table
    generate_comprehensive_table(results)

if __name__ == "__main__":
    main()
