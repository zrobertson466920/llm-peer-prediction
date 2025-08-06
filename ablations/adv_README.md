# adv_analysis.py - Adversarial Transformation Analysis

## Overview

This script performs paired statistical analysis comparing peer prediction mechanism performance before and after adversarial transformations. It quantifies how transformations affect both raw scores and discrimination ability between response categories.

## Key Analyses

### 1. Category Effect Size Analysis
Measures discrimination between Good Faith (Faithful + Style) and Problematic (Strategic + Low Effort) response categories:

- **Good Faith conditions**: Academic Style, Bureaucratic, Casual Conversational, Comprehensive, Faithful, Historical Perspective, Minimalist, Neutral Tone, Objective, Poetic, Sensationalist, Technical Jargon, Euphemistic, Overly Technical
- **Problematic conditions**: Agenda Push, Cherry Pick, Conspiracy Theory, Context Removal, Contradictory, Fact Manipulation, False Attribution, Low Effort, Minimal Detail, Misleading Emphasis, Selective Omission, Sentiment Flip, Surface Skim, Template Response, Ultra Concise

For each mechanism, calculates:
- Cohen's d effect size using paired differences (same formula as `binary_cat_analysis.py`)
- Change in discrimination ability after transformation
- Warnings when discrimination is substantially reduced (>30%)

### 2. Paired T-Tests
For each mechanism, performs within-subjects comparison:
- Tests whether transformation significantly changes overall scores
- Uses all 200 examples as paired observations
- Reports mean difference, 95% CI, t-statistic, p-value, and effect size
- Uses standard deviation of paired differences for Cohen's d

### 3. Condition-Level Analysis
Examines transformation effects on individual conditions:
- Calculates absolute and relative changes for each condition
- Identifies conditions most affected by transformation
- Provides range of effects across all conditions

## Output Interpretation

### Category Effect Sizes Section
```
TVD-MI:
  Original effect size: 7.235      # Strong discrimination originally
  Transformed effect size: 4.975   # Reduced but still substantial
  Change: -2.259                   # Large reduction in discrimination
  Interpretation: substantial reduction
```

Effect size interpretation:
- d < 0.2: Negligible
- 0.2 ≤ d < 0.5: Small
- 0.5 ≤ d < 0.8: Medium
- d ≥ 0.8: Large

### Paired Test Results
```
TVD-MI:
  Mean change: 0.0698              # Average score increased
  p-value: 0.0000 (significant)    # Change is statistically significant
  Effect size: 1.036               # Large effect size for the change
```

This reveals paradoxes like TVD-MI having higher scores but worse discrimination.

### Visual Output
Creates `paired_analysis_visualization.png` with four panels:
1. **Before/After comparison**: Bar chart of mean scores by mechanism
2. **Effect sizes heatmap**: Shows effect sizes for all condition-mechanism pairs
3. **Changes by condition**: Grouped bars showing how each condition changed
4. **Category discrimination**: Compares Good Faith vs Problematic effect sizes before/after

## Usage

```bash
python ablations/adv_analysis.py \
  --original-dir path/to/original/results/ \
  --transformed-dir path/to/transformed/results/ \
  --output-dir output_directory/
```

Required inputs:
- Original and transformed results directories containing mechanism outputs
- Both must have evaluated the same examples for valid paired comparison

Output files:
- `paired_analysis_results.json`: Complete statistical results
- `paired_analysis_visualization.png`: Four-panel visualization
- Console output with formatted summary

## Key Insights

The script reveals that adversarial transformations can:
1. **Increase scores while reducing discrimination** (e.g., case-flipping improves TVD-MI scores but reduces category separation)
2. **Affect mechanisms differently** (GPPM most robust, judges most vulnerable)
3. **Create measurement paradoxes** where "better" scores indicate worse performance

This analysis is critical for understanding mechanism robustness and validates theoretical predictions about gaming vulnerabilities.