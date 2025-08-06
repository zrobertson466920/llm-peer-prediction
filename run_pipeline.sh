#!/bin/bash

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file.json>"
    exit 1
fi

CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Extract config name without extension and path
CONFIG_NAME=$(basename "$CONFIG_FILE" .json)
RESULTS_DIR="${CONFIG_NAME}_results"

# Get the number of conditions and sample size from config file
NUM_CONDITIONS=$(python -c "import json; config = json.load(open('$CONFIG_FILE')); print(len(config['agent_perspectives']))")
SAMPLE_SIZE=$(python -c "import json; config = json.load(open('$CONFIG_FILE')); print(config.get('data_config', {}).get('sample_size', 100))")

echo "Configuration details:"
echo "  - Number of conditions: $NUM_CONDITIONS"
echo "  - Sample size: $SAMPLE_SIZE"

# Calculate workers for different mechanisms
JUDGE_WORKERS=$((NUM_CONDITIONS * (NUM_CONDITIONS - 1)))
TVD_WORKERS=$((JUDGE_WORKERS * 2))
LOG_WORKERS=5

echo "Worker configuration:"
echo "  - Judge workers: $JUDGE_WORKERS"
echo "  - TVD-MI workers: $TVD_WORKERS"
echo "  - Log prob workers: $LOG_WORKERS"

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/figures"

echo "=== Running pipeline for config: $CONFIG_NAME ==="
echo "Results will be saved in: $RESULTS_DIR/"

# Step 1: Generate agent responses
echo "Step 1: Generating agent responses..."
python evaluation/data_generation.py --config "$CONFIG_FILE"

# The data_generation script saves to data/agents/ directory
# We need to find the most recently created file
AGENT_FILE=$(ls -t data/agents/*.json | head -1)

if [ ! -f "$AGENT_FILE" ]; then
    echo "Error: Agent data generation failed - no output file found"
    exit 1
fi

# Copy the agent file to our results directory
cp "$AGENT_FILE" "$RESULTS_DIR/"
AGENT_FILE_BASENAME=$(basename "$AGENT_FILE")
LOCAL_AGENT_FILE="$RESULTS_DIR/$AGENT_FILE_BASENAME"

echo "Generated agent data: $AGENT_FILE_BASENAME"

# Step 2: Validate agent data
echo "Step 2: Validating agent data..."
python evaluation/validate_agent_data.py --filepath "$LOCAL_AGENT_FILE"

# Move validation file from default location to results directory
VALIDATION_FILE="results/${AGENT_FILE_BASENAME%.json}_validation.json"
if [ -f "$VALIDATION_FILE" ]; then
    mv "$VALIDATION_FILE" "$RESULTS_DIR/"
fi

# Step 3: Run mechanism evaluations sequentially
echo "Step 3: Running mechanism evaluations..."

# MI/GPPM
echo "  - Running MI/GPPM with $LOG_WORKERS workers..."
python evaluation/log_peer_prediction.py --agent-data "$LOCAL_AGENT_FILE" \
    --output "$RESULTS_DIR" \
    --workers $LOG_WORKERS

# Check if MI/GPPM succeeded
if [ $? -ne 0 ]; then
    echo "Error: MI/GPPM evaluation failed"
    exit 1
fi

# TVD-MI (with minimal verbosity for memory efficiency)
echo "  - Running TVD-MI with $TVD_WORKERS workers, $SAMPLE_SIZE examples..."
python evaluation/tvd_mi_peer_prediction.py --agent-data "$LOCAL_AGENT_FILE" \
    --output "$RESULTS_DIR/" \
    --examples $SAMPLE_SIZE \
    --verbosity 0 \
    --workers $TVD_WORKERS

# Check if TVD-MI succeeded
if [ $? -ne 0 ]; then
    echo "Error: TVD-MI evaluation failed"
    exit 1
fi

# LLM Judge (both modes with minimal verbosity)
echo "  - Running LLM Judge with $JUDGE_WORKERS workers, $SAMPLE_SIZE examples..."
python evaluation/llm_judge_peer_prediction.py --agent-data "$LOCAL_AGENT_FILE" \
    --output "$RESULTS_DIR" \
    --both \
    --verbosity 0 \
    --workers $JUDGE_WORKERS \
    --examples $SAMPLE_SIZE

# Check if Judge succeeded
if [ $? -ne 0 ]; then
    echo "Error: LLM Judge evaluation failed"
    exit 1
fi

echo "All mechanisms completed!"

# Move individual_examples directories to results directory with correct names
echo "Moving individual example directories..."
AGENT_NAME="${AGENT_FILE_BASENAME%.json}"

# Move and rename individual example directories
if [ -d "${AGENT_NAME}_mi_gppm_individual_examples" ]; then
    mv "${AGENT_NAME}_mi_gppm_individual_examples" "$RESULTS_DIR/log_individual_examples"
fi

if [ -d "${AGENT_NAME}_tvd_mi_individual_examples" ]; then
    mv "${AGENT_NAME}_tvd_mi_individual_examples" "$RESULTS_DIR/tvd_mi_individual_examples"
fi

if [ -d "${AGENT_NAME}_judge_with_context_individual_examples" ]; then
    mv "${AGENT_NAME}_judge_with_context_individual_examples" "$RESULTS_DIR/llm_context_individual_examples"
fi

if [ -d "${AGENT_NAME}_judge_without_context_individual_examples" ]; then
    mv "${AGENT_NAME}_judge_without_context_individual_examples" "$RESULTS_DIR/llm_without_context_individual_examples"
fi

# Step 4: Run binary category analysis
echo "Step 4: Running binary category analysis..."
python analysis/binary_cat_analysis.py \
    --results-dir "$RESULTS_DIR" \
    --figures-dir "$RESULTS_DIR/figures"

# Check if analysis succeeded
if [ $? -ne 0 ]; then
    echo "Error: Binary category analysis failed"
    exit 1
fi

echo "=== Pipeline completed successfully! ==="
echo "Results are in: $RESULTS_DIR/"
echo "Figures are in: $RESULTS_DIR/figures/"

# Optional: List generated files
echo ""
echo "Generated files:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null
echo ""
echo "Individual example directories:"
ls -d "$RESULTS_DIR"/*_individual_examples/ 2>/dev/null
echo ""
echo "Figures:"
ls -la "$RESULTS_DIR/figures"/*.png 2>/dev/null