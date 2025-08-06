# TVD-MI Analysis Tool for Machine Translation and Text Generation

This tool measures mutual information between agent responses using the Total Variation Distance - Mutual Information (TVD-MI) approach. It evaluates whether responses from different conditions share information about their common source, revealing which prompting strategies produce outputs with the highest task-specific information.

## Overview

The tool performs two key operations:
1. It analyzes responses by computing TVD-MI between all pairs of conditions using a binary classifier
2. It aggregates results across multiple examples to identify which conditions produce responses with the strongest shared information

This approach helps identify which prompting strategies lead to outputs that are most recognizably from the same source task, working as a reference-free evaluation method for controlled text generation.

## Features

- Parallel processing of multiple examples using process pools
- Thread-based parallelism for API calls within each example
- Proper P and Q distribution construction for valid TVD-MI estimation
- Progress tracking with nested progress bars
- Automatic skipping of already processed examples for resumable processing
- Direct saving to archive directory to keep workspace clean
- Comprehensive aggregation of results with bootstrap confidence intervals
- Caching of LLM critic responses to avoid redundant API calls
- Storage of raw LLM interactions for debugging and analysis

## Installation

```bash
pip install openai numpy tqdm
```

Ensure you have a `config.py` file with your OpenAI API key:

```python
OPENAI_API_KEY = "your-api-key-here"
```

## Usage

### Basic Usage

```bash
# Process agent-generated data file
python tvd_mi_peer_prediction.py --agent-data data/agents/translation_meta_llama_70B.json --examples 10

# Process with more workers for faster execution
python tvd_mi_peer_prediction.py --agent-data data/agents/translation_meta_llama_70B.json --examples 20 --workers 10

# Only aggregate existing results without new processing
python tvd_mi_peer_prediction.py --agent-data data/agents/translation_meta_llama_70B.json --examples 0 --aggregate
```

### Command Line Arguments

- `--agent-data`: Path to agent-generated data file containing responses (recommended usage)
- `--data`: Path to the JSON data file containing responses (legacy option)
- `--output`: Directory to save results (default: "results")
- `--conditions`: List of condition indices or number of conditions (default: all conditions)
- `--examples`: Number of examples to process (default: 1)
- `--workers`: Maximum number of parallel workers (default: 5)
- `--single`: Process only a single example with this index (default: -1, process multiple)
- `--aggregate`: Aggregate results after processing
- `--model`: Model to use for TVD-MI critic (default: "gpt-4o-mini")

### Input Data Format

The script expects an agent data file with the following structure:

```json
{
    "task_description": "German to English translation task",
    "agent_perspectives": [
        {
            "condition": "Reference",
            "strategy": "Translate the following German sentence to English..."
        },
        ...
    ],
    "tasks": [
        {
            "context": "Original source text",
            "responses": ["Response 1", "Response 2", ...],
            "reference": "Reference translation (optional)"
        },
        ...
    ]
}
```

Where:
- `task_description`: High-level description of the task for the critic
- `agent_perspectives`: Array of conditions with their prompting strategies
- `tasks`: Array of examples, each containing:
  - `context`: The input/source text
  - `responses`: Array of agent responses (one per condition)
  - `reference`: Optional reference output

## Output

### Individual Example Output

For each example, the script generates a JSON file (`tvd_mi_individual_examples/tvd_mi_example_{idx}.json`) with:

```json
{
  "example_idx": 0,
  "reference": "Original source text",
  "translations": ["Response 1", "Response 2", ...],
  "condition_keys": ["Reference", "Original", "Low Effort", ...],
  "tvd_mi_matrix": [[0, 0.25, ...], [0.5, 0, ...], ...],
  "tvd_mi_scores": [0.1875, 0.4375, ...],
  "tvd_mi_bidirectional": [0.21875, 0.40625, ...],
  "num_p_comparisons": 20,
  "num_q_comparisons": 20,
  "task_description": "German to English translation task",
  "response_lengths": [25, 32, ...],
  "llm_calls": [
    {
      "cached": false,
      "text_a": "Resumption of the session",
      "text_b": "Resumption of the session period",
      "prompt": "You are evaluating whether two responses...",
      "response": "[[Little Gain]]",
      "score": 0.25,
      "pair": [0, 1],
      "distribution": "p",
      "model": "gpt-4o-mini"
    },
    ...
  ]
}
```

### Aggregated Output

The aggregated results file (`{agent_file_name}_tvd_mi.json`) contains:

```json
{
  "num_examples_processed": 10,
  "condition_keys": ["Reference", "Original", "Low Effort", ...],
  "tvd_mi_matrix_avg": [[0, 0.625, ...], [0.75, 0, ...], ...],
  "tvd_mi_scores_avg": [0.5125, 0.5875, 0.65, ...],
  "tvd_mi_bidirectional_avg": [0.55625, 0.6125, ...],
  "response_lengths_avg": [116.1, 109.6, ...],
  "tvd_mi_scores_ci": [[0.369, 0.663], [0.438, 0.738], ...],
  "tvd_mi_bidirectional_ci": [[0.403, 0.719], [0.459, 0.763], ...],
  "tvd_mi_rankings": [2, 1, 0, 4, 3],
  "ranked_condition_keys": ["Low Effort", "Original", "Reference", ...],
  "normalized_scores": [100.0, 90.4, 78.8, ...],
  "tvd_mi_bidirectional_rankings": [1, 2, 0, 4, 3],
  "tvd_mi_bidirectional_ranked_condition_keys": ["Original", "Low Effort", ...],
  "tvd_mi_bidirectional_normalized_scores": [100.0, 97.4, ...]
}
```

Where:
- `tvd_mi_matrix_avg`: Average pairwise TVD-MI values across examples
- `tvd_mi_scores_avg`: Average TVD-MI score for each condition
- `tvd_mi_bidirectional_avg`: Symmetrized version using (i→j + j→i)/2
- `response_lengths_avg`: Average response length per condition
- `*_ci`: Bootstrap confidence intervals
- `tvd_mi_rankings`: Indices of conditions sorted by TVD-MI score
- `normalized_scores`: TVD-MI scores as percentages of highest score

## How It Works

1. **Distribution Construction**: 
   - P distribution: Pairs of responses (A,B) from the same source task
   - Q distribution: Pairs where response B comes from a different task

2. **Binary Critic**: An LLM evaluates each pair, classifying information gain as:
   - [[Significant Gain]]: Clear evidence of shared source (score: 1.0)
   - [[Little Gain]]: Some shared elements (score: 0.25)
   - [[No Gain]]: No evidence of shared source (score: 0.0)

3. **TVD-MI Calculation**: For each condition pair (i,j):
   - TVD-MI(i,j) = E[f(X_i,X_j)]_P - E[f(X_i,X_j)]_Q
   - This measures how much more likely the critic is to detect shared information when responses are actually from the same task

4. **Aggregation**: Results are averaged across examples with bootstrap confidence intervals

## Technical Details

- Uses OpenAI's chat API for the critic function
- Implements proper TVD-MI with cross-task Q distribution sampling
- Manages parallel processing at two levels:
  - Process pool for examples (avoiding GIL issues)
  - Thread pool for API calls within each example
- Handles API timeouts and failures gracefully
- Individual results saved directly to archive directory
- Caches critic responses to minimize API calls

## Example Workflow

For a typical experiment comparing different prompting strategies:

```bash
# First, test with a small batch
python tvd_mi_peer_prediction.py --agent-data data/agents/translation_test.json --examples 5

# Run full analysis with more workers
python tvd_mi_peer_prediction.py --agent-data data/agents/translation_full.json --examples 100 --workers 10

# Later, aggregate any additional results
python tvd_mi_peer_prediction.py --agent-data data/agents/translation_full.json --examples 0 --aggregate
```

## Interpreting Results

- **Higher TVD-MI scores** indicate responses that are more recognizably from the same source
- **Lower TVD-MI scores** suggest responses that could plausibly be from different sources
- Conditions with high TVD-MI preserve more task-specific information
- The bidirectional scores account for asymmetry in the critic function

## Common Issues

1. **Hanging at 99%**: Some API calls may timeout. The script will continue after 30 seconds.
2. **Negative TVD-MI**: Should not occur with proper implementation. Check that Q distribution uses different tasks.
3. **Rate limits**: Reduce `--workers` if hitting API rate limits.