import json
import numpy as np
from pathlib import Path
import pandas as pd

def load_all_transformations():
    """Load results from all transformation experiments."""
    transformations = ['case_flip', 'format', 'padding', 'pattern']
    results = {}

    for transform in transformations:
        json_path = Path(f'ablation_analysis_{transform}/paired_analysis_results.json')
        if json_path.exists():
            with open(json_path, 'r') as f:
                results[transform] = json.load(f)
        else:
            print(f"Warning: {json_path} not found")

    return results

def create_summary_table(results):
    """Create a summary table of all transformations' effects."""
    mechanisms = ['mi', 'gppm', 'tvd_mi', 'judge_with_context', 'judge_without_context']

    # Collect data for each mechanism across transformations
    summary_data = []

    for mech in mechanisms:
        mech_results = {
            'Mechanism': mech.replace('_', ' ').title().replace('Mi', 'MI').replace('Gppm', 'GPPM').replace('Tvd', 'TVD')
        }

        for transform in results:
            if 'paired_tests' in results[transform] and mech in results[transform]['paired_tests']:
                test_data = results[transform]['paired_tests'][mech]
                mech_results[f'{transform}_change'] = test_data['mean_difference']
                mech_results[f'{transform}_pvalue'] = test_data['p_value']

            if 'category_effect_sizes' in results[transform] and mech in results[transform]['category_effect_sizes']:
                effect_data = results[transform]['category_effect_sizes'][mech]
                original_d = effect_data['original_effect_size']
                transformed_d = effect_data['transformed_effect_size']
                pct_change = ((transformed_d - original_d) / abs(original_d)) * 100 if original_d != 0 else 0
                mech_results[f'{transform}_discrimination_change'] = pct_change

        summary_data.append(mech_results)

    return pd.DataFrame(summary_data)

def generate_latex_tables(results):
    """Generate LaTeX tables for the paper."""

    # Table 1: Score changes by transformation
    print("% Table 1: Mean score changes by transformation")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Score changes under adversarial transformations on Reddit TIFU}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Mechanism} & \\textbf{Case Flip} & \\textbf{Format} & \\textbf{Padding} & \\textbf{Pattern} \\\\")
    print("\\midrule")

    mechanisms = ['tvd_mi', 'mi', 'gppm', 'judge_with_context', 'judge_without_context']
    mech_names = {
        'tvd_mi': 'TVD-MI',
        'mi': 'MI (DoE)', 
        'gppm': 'GPPM',
        'judge_with_context': 'Judge (w/ ctx)',
        'judge_without_context': 'Judge (w/o ctx)'
    }

    for mech in mechanisms:
        row = [mech_names[mech]]
        for transform in ['case_flip', 'format', 'padding', 'pattern']:
            if transform in results and 'paired_tests' in results[transform] and mech in results[transform]['paired_tests']:
                test_data = results[transform]['paired_tests'][mech]
                mean_diff = test_data['mean_difference']
                p_val = test_data['p_value']

                # Format with significance stars
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = ""

                row.append(f"{mean_diff:+.3f}{sig}")
            else:
                row.append("--")

        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

    # Table 2: Discrimination changes
    print("% Table 2: Changes in discrimination ability (Good Faith vs Problematic)")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Discrimination degradation under transformations (\\% change in Cohen's d)}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Mechanism} & \\textbf{Case Flip} & \\textbf{Format} & \\textbf{Padding} & \\textbf{Pattern} \\\\")
    print("\\midrule")

    for mech in mechanisms:
        row = [mech_names[mech]]
        for transform in ['case_flip', 'format', 'padding', 'pattern']:
            if transform in results and 'category_effect_sizes' in results[transform] and mech in results[transform]['category_effect_sizes']:
                effect_data = results[transform]['category_effect_sizes'][mech]
                original_d = effect_data['original_effect_size']
                transformed_d = effect_data['transformed_effect_size']
                pct_change = ((transformed_d - original_d) / abs(original_d)) * 100 if original_d != 0 else 0

                # Color code severe degradations
                if pct_change < -30:
                    row.append(f"\\textcolor{{red}}{{{pct_change:.1f}\\%}}")
                else:
                    row.append(f"{pct_change:.1f}\\%")
            else:
                row.append("--")

        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

def generate_key_findings(results):
    """Generate key numerical findings for the text."""
    print("% Key findings for the measurement tampering section:")
    print()

    # Find the most paradoxical result (score increase + discrimination decrease)
    paradoxes = []
    for transform in results:
        if 'paired_tests' in results[transform] and 'category_effect_sizes' in results[transform]:
            for mech in ['tvd_mi', 'mi', 'gppm']:
                if mech in results[transform]['paired_tests'] and mech in results[transform]['category_effect_sizes']:
                    score_change = results[transform]['paired_tests'][mech]['mean_difference']
                    effect_data = results[transform]['category_effect_sizes'][mech]
                    disc_change = effect_data['transformed_effect_size'] - effect_data['original_effect_size']

                    if score_change > 0 and disc_change < 0:  # Paradoxical case
                        paradoxes.append({
                            'transform': transform,
                            'mechanism': mech,
                            'score_increase': score_change,
                            'discrimination_decrease': disc_change
                        })

    if paradoxes:
        best_paradox = max(paradoxes, key=lambda x: x['score_increase'])
        print(f"% Most paradoxical: {best_paradox['mechanism']} under {best_paradox['transform']}")
        print(f"% Score increased by {best_paradox['score_increase']:.3f} while discrimination dropped by {best_paradox['discrimination_decrease']:.3f}")

    # Average discrimination degradation across all transformations
    all_degradations = []
    for transform in results:
        if 'category_effect_sizes' in results[transform]:
            for mech, data in results[transform]['category_effect_sizes'].items():
                orig = data['original_effect_size']
                trans = data['transformed_effect_size']
                pct_change = ((trans - orig) / abs(orig)) * 100 if orig != 0 else 0
                all_degradations.append(pct_change)

    if all_degradations:
        print(f"\n% Average discrimination degradation: {np.mean(all_degradations):.1f}%")
        print(f"% Range: [{min(all_degradations):.1f}%, {max(all_degradations):.1f}%]")

def main():
    # Load all transformation results
    results = load_all_transformations()

    if not results:
        print("No results found!")
        return

    print(f"Loaded results for transformations: {list(results.keys())}")
    print()

    # Generate LaTeX tables
    generate_latex_tables(results)

    # Generate key findings
    generate_key_findings(results)

    # Note about repetition failure
    print("\n% Note: The repetition transformation caused parsing failures,")
    print("% likely functioning as a jailbreak. This is mentioned in the text.")

if __name__ == "__main__":
    main()