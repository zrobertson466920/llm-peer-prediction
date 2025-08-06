#!/bin/bash

# Check if agent data file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <agent_data_file.json>"
    exit 1
fi

AGENT_FILE="$1"

# Check if agent data file exists
if [ ! -f "$AGENT_FILE" ]; then
    echo "Error: Agent data file '$AGENT_FILE' not found"
    exit 1
fi

# Extract filename without extension and path
AGENT_FILE_BASENAME=$(basename "$AGENT_FILE")
AGENT_NAME="${AGENT_FILE_BASENAME%.json}"
RESULTS_DIR="${AGENT_NAME}_results"

# Get the number of conditions from agent data file
NUM_CONDITIONS=$(python -c "import json; data = json.load(open('$AGENT_FILE')); print(len(data['agent_perspectives']))")

# Get the number of tasks (for sample size)
SAMPLE_SIZE=$(python -c "import json; data = json.load(open('$AGENT_FILE')); print(len(data['tasks']))")

# Get task type for display
TASK_TYPE=$(python -c "import json; data = json.load(open('$AGENT_FILE')); desc = data['task_description'].lower(); print('translation' if 'translat' in desc else 'summarization' if 'summar' in desc else 'peer_review' if 'review' in desc else 'unknown')")

echo "Agent data details:"
echo "  - File: $AGENT_FILE_BASENAME"
echo "  - Task type: $TASK_TYPE"
echo "  - Number of conditions: $NUM_CONDITIONS"
echo "  - Number of tasks: $SAMPLE_SIZE"

# Check for adversarial transformation metadata
HAS_TRANSFORM=$(python -c "import json; data = json.load(open('$AGENT_FILE')); print('adversarial_transform' in data.get('metadata', {}))" 2>/dev/null)
if [ "$HAS_TRANSFORM" = "True" ]; then
    TRANSFORM_TYPE=$(python -c "import json; data = json.load(open('$AGENT_FILE')); print(data['metadata']['adversarial_transform']['type'])")
    echo "  - Adversarial transform: $TRANSFORM_TYPE"
fi

# Calculate workers for different mechanisms
JUDGE_WORKERS=$((NUM_CONDITIONS * (NUM_CONDITIONS - 1)))
TVD_WORKERS=$((JUDGE_WORKERS * 2))
LOG_WORKERS=5

echo ""
echo "Worker configuration:"
echo "  - Judge workers: $JUDGE_WORKERS"
echo "  - TVD-MI workers: $TVD_WORKERS"
echo "  - Log prob workers: $LOG_WORKERS"

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/figures"

echo ""
echo "=== Running evaluation pipeline for: $AGENT_NAME ==="
echo "Results will be saved in: $RESULTS_DIR/"

# Copy the agent file to our results directory
cp "$AGENT_FILE" "$RESULTS_DIR/"
LOCAL_AGENT_FILE="$RESULTS_DIR/$AGENT_FILE_BASENAME"

# Step 1: Validate agent data
echo ""
echo "Step 1: Validating agent data..."
python evaluation/validate_agent_data.py --filepath "$LOCAL_AGENT_FILE"

# Move validation file from default location to results directory
VALIDATION_FILE="results/${AGENT_FILE_BASENAME%.json}_validation.json"
if [ -f "$VALIDATION_FILE" ]; then
    mv "$VALIDATION_FILE" "$RESULTS_DIR/"
else
    echo "Warning: Validation file not found at expected location"
fi

# Step 2: Run mechanism evaluations sequentially
echo ""
echo "Step 2: Running mechanism evaluations..."

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

echo ""
echo "All mechanisms completed!"

# Move individual_examples directories to results directory with correct names
echo "Moving individual example directories..."

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

# Step 3: Run binary category analysis
echo ""
echo "Step 3: Running binary category analysis..."
python analysis/binary_cat_analysis.py \
    --results-dir "$RESULTS_DIR" \
    --figures-dir "$RESULTS_DIR/figures"

# Check if analysis succeeded
if [ $? -ne 0 ]; then
    echo "Error: Binary category analysis failed"
    exit 1
fi

echo ""
echo "=== Pipeline completed successfully! ==="
echo "Results are in: $RESULTS_DIR/"
echo "Figures are in: $RESULTS_DIR/figures/"

# Optional: List generated files
echo ""
echo "Generated files:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null | grep -v "$AGENT_FILE_BASENAME"
echo ""
echo "Individual example directories:"
ls -d "$RESULTS_DIR"/*_individual_examples/ 2>/dev/null
echo ""
echo "Figures:"
ls -la "$RESULTS_DIR/figures"/*.png 2>/dev/null

# If this was a transformed file, provide comparison hint
if [ "$HAS_TRANSFORM" = "True" ]; then
    echo ""
    echo "TIP: To compare with original results, look for results from the non-transformed version"
fi