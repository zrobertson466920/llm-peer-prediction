# LLM Assistant Guide to the Peer Prediction Codebase

## Overview
This codebase implements experiments for evaluating information-theoretic peer prediction mechanisms on text generation tasks. The system tests whether these mechanisms can detect strategic manipulation in LLM outputs across three task types: translation, summarization, and peer review.

## Pipeline Architecture
```
1. Data Generation (data_generation.py)
   ↓ [agent responses JSON with multiple perspectives]
2. Validation (validate_agent_data.py)
   ↓ [baseline ROUGE/BLEU scores + statistics]
3. Mechanism Evaluation (parallel)
   ├─ log_peer_prediction.py → [MI/GPPM scores]
   ├─ tvd_mi_peer_prediction.py → [TVD-MI scores]
   └─ llm_judge_peer_prediction.py → [Judge scores with/without context]
   ↓ [mechanism results JSONs]
4. Analysis
   ├─ test_primary_hyp.py → [test pre-registered experiments except H2c]
   └─ run_all_binary_analyses.py → [mechanism comparison + categories]
5. Ablations
   ├─ adversarial_mi.py → [generate tampered data for H2c]
   └─ adv_analysis.py → [exploratory analysis]
   └─ adv_results.py → [result tables]
```

## Critical Data Structures

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

### Agent Data Format (data_generation.py output)
```json
{
  "task_description": "Translate from German to English...",
  "agent_perspectives": [
    {
      "condition": "Faithful",
      "reading": "optional reading prompt",
      "strategy": "main generation prompt"
    }
  ],
  "tasks": [
    {
      "context": "input text",
      "responses": ["response1", "response2", ...],
      "preparations": ["reading output1", ...]
    }
  ],
  "add_references": true/false,
  "metadata": {...}
}
```

### Validation Results (validate_agent_data.py output)
```json
{
  "task_type": "translation|summarization|peer_review",
  "condition_stats": {
    "Faithful": {
      "count": N,
      "mean_length": X,
      "std_length": Y
    }
  },
  "baseline_scores": {
    "Faithful": {
      "bleu": {"corpus_score": 0.X, "confidence_interval": [...]},
      "rouge1_f1": {"mean": 0.X, "confidence_interval": [...]}
    }
  },
  "compression_analysis": {
    "mean_input_length": X,
    "compression_ratios": {"Faithful": Y}
  }
}
```

## Key Implementation Details

### 1. Data Generation (`data_generation.py`)
- Supports both Together API and OpenAI API (auto-detects based on model name)
- Two-stage generation: optional "reading" phase followed by "strategy" phase
- Can prepend reference responses when `add_references: true`
- Handles chat vs completion endpoints based on model type

### 2. Validation (`validate_agent_data.py`)
- **BLEU**: Corpus-level for translation (using sacrebleu)
- **ROUGE-1**: F1 score for summarization (simple unigram overlap)
- Bootstrap confidence intervals (default 200 samples)
- Compression ratio analysis reveals task characteristics:
  - Translation: ~0.89x (similar length)
  - Summarization: ~13.49x (high compression)
  - Peer Review: ~20.56x (extreme compression)

### 3. Mechanism Implementations

#### MI/GPPM (`log_peer_prediction.py`)
- Uses Together API with `echo=True` and `logprobs=True` for token probabilities
- **MI**: Mean log probability difference: `P(A|task,B) - P(A|task)`
- **GPPM**: Sum of log probabilities (also provides normalized version)
- Caches API calls to avoid redundancy
- Saves individual examples to archive, then aggregates

#### TVD-MI (`tvd_mi_peer_prediction.py`)
- Uses OpenAI chat models as LLM critic
- Computes proper P and Q distributions:
  - **P**: Pairs from same task
  - **Q**: Pairs from different tasks
- TVD-MI = E[f(X,Y)]_P - E[f(X',Y)]_Q
- Critic scores: [[Significant Gain]]=1.0, [[Little Gain]]=0.25, [[No Gain]]=0.0
- Provides both directional and bidirectional scores
- Reads directly from archive directory during aggregation (no temp files)
- Supports verbosity levels to control output size

#### LLM Judge (`llm_judge_peer_prediction.py`)
- Head-to-head comparisons using OpenAI models
- Two modes:
  - **With context**: Judge sees the original query
  - **Without context**: Judge evaluates responses blindly
- Win matrix → win rates for each condition
- Returns: [[A]]=1.0, [[B]]=0.0, [[C]]=0.5 (tie)
- Already uses direct aggregation from archive
- Supports verbosity levels to control output size

#### Verbosity Levels (TVD-MI and Judge)
Both scripts support three verbosity levels to manage memory usage:
- **VERBOSITY_MINIMAL (0)**: Default. Only scores, matrices, and metadata. Dramatically reduces file sizes.
- **VERBOSITY_COMPACT (1)**: Includes full responses/translations but excludes queries, prompts, and contexts that could contain entire papers.
- **VERBOSITY_FULL (2)**: Complete output including all prompts and LLM calls (original behavior).

Memory optimization is critical for peer review tasks where prompts can contain multiple full papers.

### 4. Analysis Scripts

#### `test_primary_hyp.py` - Pre-registered hypothesis testing
- Loads all mechanism results + validation scores
- Tests mechanisms discrimination effect-size (H1a)
- Compressions effects (H1b)
- Robustness tests (H1c, H2a, H2b)

#### `run_all_binary_analyses.py` - Mechanism Comparison
- Categorizes conditions: Strategic, Low Effort, Style, Faithful
- Creates comprehensive results with:
  - Discrimination effect-sizes by mechanism
  - Confidence-intervals

#### `u_hyp.py` - Exploratory 
- Allows quadratic model for compression effects (H1b)

## File Organization

### Input/Output Structure
```
data/
  agents/           # Generated agent responses
  reference/        # Original datasets (WMT, CNN/DailyMail, ICLR)
results/
  *_validation.json
  *_mi_gppm.json
  *_tvd_mi.json
  *_judge_with_context.json
  *_judge_without_context.json
  */individual_examples/  # Archives of per-example results
```

### Naming Conventions
- Agent data: `{task_type}_{model}_{timestamp}.json`
- Results: `{agent_filename}_{mechanism}.json`
- Archives: `{mechanism}_individual_examples/`

## Critical Context for AI Assistants

### 1. Task Delimiters
The code uses task-specific delimiters:
- Translation: `"Translations:\n"`
- Summarization: `"Summaries:\n"`
- Peer Review: `"Reviews:\n"`

### 2. API Management
- Together API: Used for models requiring logprobs (MI/GPPM)
- OpenAI API: Used for chat-based evaluation (TVD-MI, Judge)
- Both APIs implement caching to reduce costs
- Parallel processing with configurable workers

### 3. Bootstrap Confidence Intervals
All mechanisms compute 95% CIs using bootstrap resampling:
- Sample with replacement from example-level scores
- Default 1000 bootstrap samples
- Reports [lower, upper] bounds for each condition

### 4. Response Length Considerations
- Length is tracked throughout pipeline
- Some mechanisms (especially GPPM) may correlate with length
- Analysis scripts visualize length effects
- Normalized GPPM attempts to control for length bias

### 5. Condition Categories (from `binary_cat_analysis.py`)
- **Strategic**: Misleading, Selective
- **Low Effort**: Lazy
- **Style**: Formal Style, Style_Transfer
- **Faithful**: Faithful, Pedantic, Roll playing

## Common Workflows

### Running Full Pipeline on New Data

We have two bash scripts to run the pipeline: `run_pipeline.sh` and `eval_pipeline.sh`. 

```bash
# 1. Generate agent responses (only used for the full pipeline)
python data_generation.py --config config_file.json

# 2. Validate and get baseline scores
python validate_agent_data.py --filepath data/agents/output.json

# 3. Run mechanisms (can parallelize)
python log_peer_prediction.py --agent-data data/agents/output.json
python tvd_mi_peer_prediction.py --agent-data data/agents/output.json --examples 100 --verbosity 0
python llm_judge_peer_prediction.py --agent-data data/agents/output.json --both --verbosity 0

# 4. Analyze results
python binary_cat_analysis.py --results-dir results --agent-file output
```

### Memory-Efficient Usage
```bash
# Use minimal verbosity (default) for large-scale experiments
python tvd_mi_peer_prediction.py --agent-data peer_review_data.json --examples 100

# Use compact verbosity to debug responses without full prompts
python llm_judge_peer_prediction.py --agent-data data.json --examples 10 --verbosity 1

# Use full verbosity only when debugging specific issues
python tvd_mi_peer_prediction.py --agent-data data.json --examples 5 --verbosity 2
```

### Debugging Tips
1. Check archive directories for individual example results
2. MI/GPPM requires models with logprobs support
3. TVD-MI needs multiple examples to compute Q distribution properly
4. Judge can timeout - check for partial results in archives
5. Validation script detects task type from task_description field
6. Use `--verbosity 1` to see responses without full prompts when debugging
7. For peer review tasks, always use `--verbosity 0` or `1` to avoid memory issues

## Key Insights from Implementation

1. **Compression ratios** directly impact how much room agents have for strategic behavior
2. **Baseline mechanisms** (MI/GPPM) largely track surface quality rather than detecting manipulation
3. **TVD-MI** attempts to capture deeper semantic relationships by comparing same-task vs different-task pairs
4. **Response length** is a potential confound that needs careful consideration
5. The **two-stage prompting** (reading + strategy) allows for more sophisticated agent behaviors
6. **Memory management** is critical - peer review tasks can create files with 180+ copies of papers without verbosity controls
7. **Direct aggregation** from archive directories avoids unnecessary file copying and memory usage