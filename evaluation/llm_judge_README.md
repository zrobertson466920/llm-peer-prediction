# LLM Judge Peer Prediction Tool

This tool evaluates agent responses using LLM-based pairwise judging with and without query context. It performs head-to-head comparisons between all condition pairs to determine win rates and rankings, providing both reference-aware and reference-free evaluation of controlled text generation.

## Overview

The tool performs two key operations:
1. **With Context**: Judge responses knowing the original query/task, evaluating how well each response addresses the specific requirements
2. **Without Context**: Judge responses blindly based on general quality metrics like clarity, coherence, and informativeness

This dual approach helps distinguish between task-specific performance and general output quality, revealing which prompting strategies produce responses that are genuinely better versus those that simply appear higher quality without context.

## Features

- Parallel processing of multiple examples with thread-based API call management
- Dual evaluation modes (with/without context) that can be run separately or together
- Automatic skipping of already processed examples for resumable processing
- Archive directory organization to keep workspace clean
- Comprehensive aggregation with bootstrap confidence intervals
- Win matrix computation showing pairwise comparison results
- Caching of LLM judge responses to avoid redundant API calls
- Storage of complete judge explanations for analysis and debugging

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
# Judge with context (default)
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_dataset.json --examples 10

# Judge without context
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_dataset.json --examples 10 --no-context

# Run both analyses
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_dataset.json --examples 10 --both

# Use different model
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_dataset.json --examples 10 --model gpt-4o --both

# More workers for faster processing
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_dataset.json --examples 20 --workers 10 --both
```

### Aggregation

```bash
# Only aggregate existing results without new processing
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_dataset.json --examples 0 --aggregate --both

# Process more examples then aggregate
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_dataset.json --examples 50 --both
# Results are automatically aggregated after processing
```

### Command Line Arguments

- `--agent-data`: Path to agent-generated data file containing responses (required)
- `--output`: Directory to save results (default: "results")
- `--examples`: Number of examples to process (default: all available)
- `--workers`: Maximum number of parallel workers (default: 5)
- `--no-context`: Judge without query context
- `--both`: Run both with-context and without-context analyses
- `--aggregate`: Only aggregate existing results without new processing
- `--model`: Model to use for LLM judging (default: "gpt-4o-mini")

### Input Data Format

The script expects an agent data file with the following structure:

```json
{
    "task_description": "Summarize the following article",
    "agent_perspectives": [
        {
            "condition": "Reference",
            "strategy": "Provide a faithful summary of the key points..."
        },
        {
            "condition": "Low Effort", 
            "strategy": "Provide a brief, minimal summary..."
        },
        ...
    ],
    "tasks": [
        {
            "context": "Original article text to summarize",
            "responses": ["Summary 1", "Summary 2", ...],
            "reference": "Reference summary (optional)"
        },
        ...
    ]
}
```

Where:
- `task_description`: High-level description of the task
- `agent_perspectives`: Array of conditions with their prompting strategies
- `tasks`: Array of examples, each containing:
  - `context`: The input text/query that agents responded to
  - `responses`: Array of agent responses (one per condition)
  - `reference`: Optional reference output

## Output

### Individual Example Output

For each example, the script generates JSON files in archive directories:
- `llm_context_individual_examples/judge_with_context_example_{idx}.json`
- `llm_without_context_individual_examples/judge_without_context_example_{idx}.json`

Each file contains:

```json
{
  "example_idx": 0,
  "query": "Original article text to summarize",
  "responses": ["Summary 1", "Summary 2", ...],
  "condition_keys": ["Reference", "Low Effort", "Misleading", ...],
  "with_context": true,
  "win_matrix": [[0.5, 0.8, 0.9], [0.2, 0.5, 0.7], [0.1, 0.3, 0.5]],
  "win_rates": [0.85, 0.45, 0.2],
  "response_lengths": [156, 89, 203],
  "llm_calls": [
    {
      "cached": false,
      "text_a": "This article discusses...",
      "text_b": "The article talks about...",
      "query": "Original article text",
      "with_context": true,
      "prompt": "Please act as an impartial judge...",
      "response": "Response A provides a more comprehensive...\n\n[[A]]",
      "score": 1.0,
      "pair": [0, 1],
      "model": "gpt-4o-mini"
    },
    ...
  ]
}
```

### Aggregated Output

The aggregated results files (`{agent_file_name}_judge_with_context.json` and `{agent_file_name}_judge_without_context.json`) contain:

```json
{
  "num_examples_processed": 10,
  "with_context": true,
  "condition_keys": ["Reference", "Low Effort", "Misleading", ...],
  "win_matrix_avg": [[0.5, 0.75, 0.85], [0.25, 0.5, 0.65], [0.15, 0.35, 0.5]],
  "win_rates_avg": [0.8, 0.45, 0.25],
  "response_lengths_avg": [145.2, 92.7, 188.9],
  "win_rates_ci": [[0.72, 0.88], [0.38, 0.52], [0.18, 0.32]],
  "response_lengths_ci": [[138.1, 152.3], [87.2, 98.2], [180.4, 197.4]],
  "rankings": [0, 1, 2],
  "ranked_condition_keys": ["Reference", "Low Effort", "Misleading"],
  "normalized_scores": [100.0, 56.25, 31.25]
}
```

Where:
- `win_matrix_avg`: Average pairwise win probabilities across examples
- `win_rates_avg`: Average win rate for each condition against all others
- `win_rates_ci`: Bootstrap confidence intervals for win rates
- `rankings`: Indices of conditions sorted by win rate (best to worst)
- `normalized_scores`: Win rates as percentages of the highest win rate

## How It Works

1. **Pairwise Comparisons**: For each example, every pair of responses is compared using an LLM judge
2. **Dual Prompting**: 
   - **With Context**: Judge knows the original query and evaluates task-specific quality
   - **Without Context**: Judge evaluates general response quality without knowing the task
3. **Win Matrix Construction**: Results form a matrix where entry (i,j) = probability condition i beats condition j
4. **Win Rate Calculation**: Each condition's overall performance against all others
5. **Bootstrap Confidence Intervals**: Statistical robustness across multiple examples

## Judge Prompting

### With Context
The judge receives the original query and both responses, evaluating which better addresses the specific task requirements considering helpfulness, relevance, accuracy, depth, creativity, and detail.

### Without Context  
The judge receives only the two responses and evaluates general quality based on clarity, coherence, depth, and informativeness without knowing what task was being performed.

## Technical Details

- Uses OpenAI's chat API for judge evaluations
- Implements thread-based parallelism for API calls within each example
- Handles API timeouts and failures gracefully
- Individual results saved directly to archive directories
- Caches judge responses to minimize redundant API calls
- Bootstrap sampling (1000 samples) for confidence intervals

## Example Workflow

For a typical experiment comparing different summarization strategies:

```bash
# Test with small batch first
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_test.json --examples 5 --both

# Run full analysis
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_full.json --examples 100 --workers 10 --both

# Later, aggregate additional results if needed
python llm_judge_peer_prediction.py --agent-data data/agents/summarization_full.json --aggregate --both
```

## Interpreting Results

- **Higher win rates** indicate responses that are consistently preferred by the judge
- **With vs Without Context**: Comparing results reveals whether high performance is task-specific or due to general quality
- **Win Matrix**: Shows detailed pairwise relationships between conditions
- **Confidence Intervals**: Indicate statistical significance of observed differences

## Common Issues

1. **Rate Limits**: Reduce `--workers` if hitting OpenAI API rate limits
2. **Memory Usage**: Large numbers of conditions create O(n²) comparisons; monitor memory with many conditions
3. **Context Length**: Very long responses may hit token limits; consider truncation for extremely long outputs
4. **Model Consistency**: Different models may show different judging biases; stick to one model for comparisons

This tool provides comprehensive evaluation of agent responses through both reference-aware and reference-free judging, enabling detailed analysis of how different prompting strategies affect both task performance and general output quality.