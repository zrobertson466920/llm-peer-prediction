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

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Effects of adversarial transformations on Reddit TIFU summarization. Score changes and discrimination")
    lines.append("degradation across four transformation types, with averages showing overall robustness.}")
    lines.append("\\label{tab:tampering_comprehensive}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Mechanism} & \\textbf{Case Flip} & \\textbf{Format} & \\textbf{Padding} & \\textbf{Pattern} & \\textbf{Average} \\\\")
    lines.append("\\midrule")

    # Score changes section
    lines.append("\\multicolumn{6}{l}{\\textit{Score Changes}} \\\\")

    for mech in mechanisms:
        row = [mech_names[mech]]
        score_changes = []

        for transform in transforms:
            if transform in results and 'paired_tests' in results[transform] and mech in results[transform]['paired_tests']:
                test_data = results[transform]['paired_tests'][mech]
                mean_diff = test_data['mean_difference']
                p_val = test_data['p_value']

                # Keep as raw difference (delta)
                score_changes.append(mean_diff)

                # Format with significance and sign
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = ""

                # Format: always show sign, three decimal places for deltas
                row.append(f"{mean_diff:+.3f}{sig}")
            else:
                row.append("--")

        # Calculate average
        if score_changes:
            avg = np.mean(score_changes)
            row.append(f"{avg:+.3f}")
        else:
            row.append("--")

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\midrule")

    # Discrimination degradation section
    lines.append("\\multicolumn{6}{l}{\\textit{Discrimination Degradation (change in Cohen's d)}} \\\\")

    for mech in mechanisms:
        row = [mech_names[mech]]
        disc_changes = []

        for transform in transforms:
            if transform in results and 'category_effect_sizes' in results[transform] and mech in results[transform]['category_effect_sizes']:
                effect_data = results[transform]['category_effect_sizes'][mech]
                original_d = effect_data['original_effect_size']
                transformed_d = effect_data['transformed_effect_size']
                # Raw delta in Cohen's d
                delta_d = transformed_d - original_d
                disc_changes.append(delta_d)

                # Format with consistent style
                if delta_d < -0.3:  # Severe degradation (threshold adjusted for raw values)
                    row.append(f"\\textcolor{{red}}{{{delta_d:+.3f}}}")
                else:
                    row.append(f"{delta_d:+.3f}")
            else:
                row.append("--")

        # Calculate average
        if disc_changes:
            avg = np.mean(disc_changes)
            if avg < -0.3:
                row.append(f"\\textcolor{{red}}{{{avg:+.3f}}}")
            else:
                row.append(f"{avg:+.3f}")
        else:
            row.append("--")

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)

def main():
    # Load all transformation results
    results = load_all_transformations()

    # Generate comprehensive table
    table_tex = generate_comprehensive_table(results)

    # Create tables directory if it doesn't exist
    tables_dir = Path("tables")
    tables_dir.mkdir(exist_ok=True)

    # Save to file
    output_path = tables_dir / "tampering_table_reddit_tifu.tex"
    with open(output_path, 'w') as f:
        f.write(f"% Loaded results for transformations: {list(results.keys())}\n\n")
        f.write(table_tex)

    print(f"Table saved to: {output_path}")

if __name__ == "__main__":
    main()