# Peer Prediction Analysis Tool for Machine Translation

This tool measures mutual information between translations using a peer prediction approach. It calculates how a given translation's probability changes when conditioned on other translations, revealing which translations have the highest mutual information with others.

## Overview

The tool performs two key operations:
1. It analyzes individual translation examples by calculating matrices of mutual influence
2. It aggregates results across multiple examples to find patterns in which translations have the strongest consensus alignment

This approach helps identify which machine translations are most "agreed upon" by an LLM's internal knowledge, working as a reference-free evaluation method for machine translation.

## Features

- Parallel processing of multiple examples using process pools
- Asynchronous API calls using thread pools for efficiency
- Progress tracking with tqdm progress bars
- Automatic skipping of already processed examples for resumable processing
- Minimal storage of essential data to avoid redundancy
- Comprehensive aggregation of results with rankings and normalized scores
- Two evaluation metrics:
  - MI Method (DoE): Measures mutual information using mean log probabilities
  - GPPM: Implements the Generative Peer Prediction Mechanism using sum of log probabilities

## Installation

```bash
pip install together numpy tqdm scipy
```

Ensure you have a `config.py` file with your Together API key:

```python
TOGETHER_API_KEY = "your-api-key-here"
```

## Usage

### Basic Usage

```bash
# Process a single example
python log_peer_prediction.py --data data/prompt_manipulation_experiment_MT_20240714_175417.json --single 0

# Process 10 examples with 5 conditions each
python log_peer_prediction.py --data data/prompt_manipulation_experiment_MT_20240714_175417.json --examples 10 --conditions 5 --aggregate
```

### Command Line Arguments

- `--data`: Path to the JSON data file containing translations (required)
- `--output`: Directory to save results (default: "results")
- `--conditions`: List of condition names to include or number of conditions (default: first 5 conditions)
- `--examples`: Number of examples to process (default: 1)
- `--workers`: Maximum number of parallel workers (default: 5)
- `--single`: Process only a single example with this index (default: -1, process multiple examples)
- `--aggregate`: Aggregate results after processing

### Input Data Format

The script expects a JSON file with the following structure:

```json
{
    "condition1": [
        {
            "id": 1,
            "reference": "Original source text",
            "completion": "Translated text"
        },
        ...
    ],
    "condition2": [
        {
            "id": 1,
            "reference": "Original source text",
            "completion": "Translated text"
        },
        ...
    ],
    ...
}
```

Where:
- Each key at the top level represents a different translation condition or prompting method
- Each condition contains an array of examples
- Each example contains:
  - `id`: A unique identifier for the example
  - `reference`: The original untranslated source text
  - `completion`: The translated text produced by this condition
- All conditions should contain the same examples with the same IDs and references
- The script will compare how different completions for the same source text predict each other

## Output

### Individual Example Output

For each example, the script generates a JSON file (`peer_prediction_example_{idx}.json`) with:

```json
{
  "example_idx": 0,
  "reference": "Original source text",
  "translations": ["Translation 1", "Translation 2", ...],
  "condition_keys": ["condition1", "condition2", ...],
  "logp_base": [0.123, -0.456, ...],
  "logp_cond": [[0, 0.234, ...], [0.567, 0, ...], ...],
  "difference_matrix": [[0, 0.111, ...], [1.023, 0, ...], ...],
  "row_avgs": [0.222, 0.789, ...],
  "col_avgs": [0.333, 0.456, ...],
  "combined_avgs": [0.278, 0.623, ...],
  "gppm": [-47.89, -52.34, ...],
  "token_counts": [15, 18, ...]
}
```

### Aggregated Output

The aggregated results file (`peer_prediction_aggregated.json`) contains:

```json
{
  "num_examples_processed": 10,
  "condition_keys": ["condition1", "condition2", ...],
  "difference_matrix_avg": [[0, 0.111, ...], [0.222, 0, ...], ...],
  "row_avgs_avg": [0.333, 0.444, ...],
  "col_avgs_avg": [0.555, 0.666, ...],
  "combined_avgs_avg": [0.444, 0.555, ...],
  "gppm_avg": [-45.67, -50.89, ...],
  "combined_avgs_ci": [[0.333, 0.555], [0.444, 0.666], ...],
  "gppm_ci": [[-50.12, -41.23], [-55.67, -46.12], ...],
  "condition_rankings": [1, 0, 2, ...],
  "ranked_condition_keys": ["condition2", "condition1", "condition3", ...],
  "normalized_scores": [100.0, 80.0, 65.0, ...],
  "gppm_rankings": [1, 0, 2, ...],
  "gppm_ranked_condition_keys": ["condition2", "condition1", "condition3", ...],
  "gppm_normalized_scores": [100.0, 90.5, 82.1, ...]
}
```

Where:
- `num_examples_processed`: Total number of examples included in the aggregation
- `condition_keys`: List of condition names evaluated
- `difference_matrix_avg`: Average difference matrix across all examples
- `row_avgs_avg`/`col_avgs_avg`: Average row/column values across all examples
- `combined_avgs_avg`: Average of row and column averages (main MI metric)
- `gppm_avg`: Average GPPM scores across all examples
- `combined_avgs_ci`/`gppm_ci`: Bootstrap confidence intervals for each metric
- `condition_rankings`: Indices of conditions sorted by combined_avgs score
- `ranked_condition_keys`: Condition names sorted by combined_avgs score
- `normalized_scores`: Combined_avgs scores normalized as percentages of highest score
- `gppm_rankings`: Indices of conditions sorted by GPPM score
- `gppm_ranked_condition_keys`: Condition names sorted by GPPM score
- `gppm_normalized_scores`: GPPM scores normalized as percentages of highest score

## How It Works

1. For each translation A, the script calculates P(A|task) - the base probability
2. For each pair of translations A and B, it calculates P(A|task,B) - the conditional probability
3. The difference P(A|task,B) - P(A|task) shows how much B boosts or reduces the probability of A
4. These boost values are averaged to determine which translations are most mutually reinforcing

The script implements two metrics:
- **MI Method (DoE)**: Uses the mean of log probabilities to measure mutual information
- **GPPM**: Uses the sum of log probabilities as described in peer prediction literature

## Technical Details

- Uses Together's API to access the LLM's token probabilities
- Caches API responses to avoid redundant calls
- Uses both token-level average and full sequence log probability sums
- Manages parallel processing to optimize throughput while avoiding API rate limits
- Bootstrap resampling provides confidence intervals for statistical validity
- Results are normalized to facilitate comparison across conditions

## Example Workflow

For a typical experiment comparing different translation prompting methods:

```bash
# First, run a small batch to test
python log_peer_prediction.py --data data/translations.json --examples 3 --conditions 5 --aggregate

# If everything looks good, run the full analysis
python log_peer_prediction.py --data data/translations.json --examples 100 --conditions 5 --workers 8 --aggregate

# To compare specific conditions of interest
python log_peer_prediction.py --data data/translations.json --examples 100 --conditions "condition1" "condition2" "condition3" --aggregate
```

## License

This project is open source and available under the MIT License.