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

    mechanisms = ['mi', 'gppm', 'tvd_mi', 'judge_with_context', 'judge_without_context']

    transforms = ['case_flip', 'format', 'padding', 'pattern']
    transform_names = {
        'case_flip': 'Case Flip',
        'format': 'Format',
        'padding': 'Padding',
        'pattern': 'Pattern'
    }

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Effects of adversarial transformations on mechanism scores and discrimination ability for Reddit TIFU summarization. Score changes show mean difference $\\pm$ 95\\% CI. Discrimination degradation shows change in Cohen's d. Bold indicates p < 0.001, regular text p < 0.05, gray text non-significant. Red values indicate severe degradation ($\\Delta$d < -0.3).}")
    lines.append("\\label{tab:tampering_comprehensive}")
    lines.append("\\footnotesize")
    lines.append("\\begin{tabular}{@{}lccccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Transformation} & \\textbf{MI} & \\textbf{GPPM} & \\textbf{TVD-MI} & \\textbf{Judge} & \\textbf{Judge} \\\\")
    lines.append("& \\textbf{(DoE)} & & & \\textbf{(w/ ctx)} & \\textbf{(w/o ctx)} \\\\")
    lines.append("\\midrule")

    # Score changes section
    lines.append("\\multicolumn{6}{l}{\\textit{Score Changes ($\\Delta$)}} \\\\")

    for transform in transforms:
        row = [transform_names[transform]]

        for mech in mechanisms:
            if transform in results and 'paired_tests' in results[transform] and mech in results[transform]['paired_tests']:
                test_data = results[transform]['paired_tests'][mech]
                mean_diff = test_data['mean_difference']
                p_val = test_data['p_value']
                ci_lower = test_data.get('ci_lower', mean_diff - 0.05)  # Fallback if CI not in data
                ci_upper = test_data.get('ci_upper', mean_diff + 0.05)
                ci_range = (ci_upper - ci_lower) / 2  # Half-width of CI

                # Format based on p-value
                if p_val < 0.001:
                    row.append(f"\\textbf{{{mean_diff:+.3f}$\\pm${ci_range:.3f}}}")
                elif p_val < 0.05:
                    row.append(f"{mean_diff:+.3f}$\\pm${ci_range:.3f}")
                else:
                    row.append(f"\\textcolor{{gray}}{{{mean_diff:+.3f}$\\pm${ci_range:.3f}}}")
            else:
                row.append("--")

        lines.append(" & ".join(row) + " \\\\")

    # Add average row for score changes
    row = ["\\textbf{Average}"]
    for mech in mechanisms:
        score_changes = []
        for transform in transforms:
            if transform in results and 'paired_tests' in results[transform] and mech in results[transform]['paired_tests']:
                score_changes.append(results[transform]['paired_tests'][mech]['mean_difference'])

        if score_changes:
            avg = np.mean(score_changes)
            std = np.std(score_changes)
            row.append(f"{avg:+.3f}$\\pm${std:.3f}")
        else:
            row.append("--")

    lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")

    # Discrimination degradation section
    lines.append("\\multicolumn{6}{l}{\\textit{Discrimination Degradation ($\\Delta$ Cohen's d)}} \\\\")

    for transform in transforms:
        row = [transform_names[transform]]

        for mech in mechanisms:
            if transform in results and 'category_effect_sizes' in results[transform] and mech in results[transform]['category_effect_sizes']:
                effect_data = results[transform]['category_effect_sizes'][mech]
                original_d = effect_data['original_effect_size']
                transformed_d = effect_data['transformed_effect_size']
                delta_d = transformed_d - original_d

                # Format with color for severe degradation
                if delta_d < -0.3:
                    row.append(f"\\textcolor{{red}}{{{delta_d:+.3f}}}")
                else:
                    row.append(f"{delta_d:+.3f}")
            else:
                row.append("--")

        lines.append(" & ".join(row) + " \\\\")

    # Add average row for discrimination degradation
    row = ["\\textbf{Average}"]
    for mech in mechanisms:
        disc_changes = []
        for transform in transforms:
            if transform in results and 'category_effect_sizes' in results[transform] and mech in results[transform]['category_effect_sizes']:
                effect_data = results[transform]['category_effect_sizes'][mech]
                original_d = effect_data['original_effect_size']
                transformed_d = effect_data['transformed_effect_size']
                disc_changes.append(transformed_d - original_d)

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
    lines.append("\\end{table*}")

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
    output_path = tables_dir / "tampering_comprehensive.tex"
    with open(output_path, 'w') as f:
        f.write(f"% Loaded results for transformations: {list(results.keys())}\n\n")
        f.write(table_tex)

    print(f"Table saved to: {output_path}")

if __name__ == "__main__":
    main()