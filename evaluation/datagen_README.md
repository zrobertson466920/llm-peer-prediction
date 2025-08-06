# Data Generation Pipeline for Translation, Summarization, and Peer Review Experiments

This package provides a robust, asynchronous pipeline for generating experimental data for translation, summarization, and peer review tasks using LLMs. It handles token estimation, efficient API usage, and structured output for consistent downstream evaluation.

## Overview

The pipeline takes input data (translation pairs, articles, or academic papers) and generates diverse responses across multiple agent perspectives. These perspectives can be configured to model different manipulation strategies, from faithful responses to various types of strategic manipulations.

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.9+
- API keys (set in environment or config.py):
  - Together API key (for open models)
  - OpenAI API key (for GPT models)
- Packages: asyncio, tqdm, openai, tiktoken, together

## Configuration

The pipeline uses JSON configuration files stored in the `configs/` directory:

```
configs/
├── translation.json     # German to English translation
├── summarization.json   # CNN/DailyMail summarization
└── peer_review.json     # ICLR paper reviews
```

### Configuration Structure

```json
{
  "task_type": "translation",
  "task_description": "German to English translation task",
  "model_config": {
    "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "max_tokens": 500,
    "temperature": 0.7
  },
  "agent_perspectives": [
    {
      "condition": "Original",
      "reading": null,
      "strategy": "Translate the following German sentence to English..."
    }
  ],
  "data_config": {
    "input_data_path": "data/wmt14_de_en_500.json",
    "sample_size": 500
  },
  "add_references": false,  // Optional: prepend reference as first agent
  "api_provider": "together"  // Optional: "together" or "openai" (auto-detected)
}
```

### Agent Perspectives

Agent perspectives define how responses are generated:

- **condition**: Label for this perspective (e.g., "Original", "Low Effort")
- **reading**: Optional pre-processing step (e.g., taking notes on a paper)
- **strategy**: Main instruction for generating the response

## Input Data Format

The pipeline expects input data in JSON format:

```json
[
  {
    "input": "Source text to process",
    "reference": "Optional reference output"
  }
]
```

## Usage

### Basic Usage

```bash
python data_generation.py --config configs/translation.json
```

### Test Mode

```bash
python data_generation.py --config configs/translation.json --test
```

Test mode reduces the dataset to 10 samples and uses only the first 4 agent perspectives for quick validation.

### Python API

```python
import asyncio
from data_generation import DataGenerationPipeline

async def run():
    with open('configs/translation.json', 'r') as f:
        config = json.load(f)
    
    pipeline = DataGenerationPipeline(config)
    output_path = await pipeline.generate_dataset()
    print(f"Dataset saved to: {output_path}")

asyncio.run(run())
```

## Output Format

The pipeline saves generated data to `data/agents/` with the naming convention:
```
data/agents/[task_type]_[model_name]_[timestamp].json
```

Output structure:
```json
{
  "task_type": "translation",
  "task_description": "German to English translation task",
  "agent_perspectives": [...],
  "tasks": [
    {
      "context": "Input text",
      "responses": ["Agent 1 response", "Agent 2 response", ...],
      "preparations": [null, "Agent 2 notes", ...]
    }
  ],
  "metadata": {
    "model_config": {...},
    "api_provider": "together",
    "data_config": {...},
    "generation_time": "2024-12-20T15:30:00",
    "token_usage": {
      "prompt_tokens": 10000,
      "completion_tokens": 50000,
      "total_tokens": 60000
    },
    "api_calls": {"total_calls": 200},
    "estimated_tokens": {...}
  }
}
```

## API Provider Support

The pipeline automatically detects the appropriate API provider based on model name:
- **OpenAI**: Models starting with "gpt-", "text-davinci-", "davinci-"
- **Together**: All other models (Llama, Mistral, etc.)

## Adding New Tasks

1. Create a new configuration file in `configs/`:
```json
{
  "task_type": "my_task",
  "task_description": "Description of my task",
  "model_config": {...},
  "agent_perspectives": [...],
  "data_config": {
    "input_data_path": "data/my_task_data.json",
    "sample_size": 500
  }
}
```

2. Prepare input data in the required format
3. Run: `python data_generation.py --config configs/my_task.json`

## Error Handling

The pipeline includes:
- Automatic retries with exponential backoff for API failures
- Token usage estimation before generation
- Progress bars showing generation status
- Detailed error messages for debugging

## License

MIT