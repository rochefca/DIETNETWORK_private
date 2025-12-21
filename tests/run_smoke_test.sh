#!/bin/bash
# DietNetwork Smoke Test - Validates installation and model accuracy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/smoke_test_config.sh"

echo "=========================================="
echo "DietNetwork Smoke Test"
echo "=========================================="
echo ""

# Activate venv
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "WARNING: No virtual environment found at $PROJECT_ROOT/.venv"
fi

# Step 1: Download test data
echo "Step 1/3: Ensuring test data is available..."
bash "$SCRIPT_DIR/download_test_data.sh"

# Step 2: Run inference with model preset
echo ""
echo "Step 2/3: Running inference with 1kgp_default preset..."

PLINK_BASE="1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1"
PLINK_PREFIX="$TEST_DATA_DIR/$PLINK_BASE"
LABELS="$TEST_DATA_DIR/labels_pop_subsampleV1.tsv"
OUTPUT="$SCRIPT_DIR/smoke_test_predictions.tsv"

# Get model path (downloads if needed)
MODEL_PATH=$(python3 "$SCRIPT_DIR/get_model_path.py" "1kgp_default")

# Run inference with all models (ensemble)
# Note: temp-dir must be absolute path since we cd into PROJECT_ROOT
TEMP_DIR="$SCRIPT_DIR/preprocessed_plink"
cd "$PROJECT_ROOT" && python3 Dietnet/predict_with_plink.py --model-dir "$MODEL_PATH" --plink-prefix "$PLINK_PREFIX" --output "$OUTPUT" --batch-size 256 --device cpu --num-workers 4 --temp-dir "$TEMP_DIR"

# Step 3: Check accuracy
echo ""
echo "Step 3/3: Validating accuracy..."
python3 "$SCRIPT_DIR/check_predictions.py" "$OUTPUT" "$LABELS"

# Extract accuracy percentage
ACCURACY=$(python3 - "$OUTPUT" "$LABELS" << 'PYTHON_EOF'
import sys
import pandas as pd

pred = pd.read_csv(sys.argv[1], sep='\t')
labels = pd.read_csv(sys.argv[2], sep='\t')

merged = pred.merge(labels, left_on='sample_id', right_on='sample_id')
accuracy = (merged['predicted_class'] == merged.iloc[:, 1]).mean() * 100
print(f"{accuracy:.2f}")
PYTHON_EOF
)

echo ""
echo "=========================================="
echo "Smoke Test Results"
echo "=========================================="
echo "Accuracy: ${ACCURACY}%"
echo "Expected: ${EXPECTED_ACCURACY_MIN}-${EXPECTED_ACCURACY_MAX}%"

# Validate accuracy is in expected range
if command -v bc &> /dev/null; then
    # Use bc for decimal comparison if available
    if (( $(echo "$ACCURACY < $EXPECTED_ACCURACY_MIN" | bc -l) )); then
        echo "❌ FAILED: Accuracy below minimum threshold"
        exit 1
    elif (( $(echo "$ACCURACY > $EXPECTED_ACCURACY_MAX" | bc -l) )); then
        echo "❌ FAILED: Accuracy above maximum (suspicious)"
        exit 1
    else
        echo "✓ PASSED"
    fi
else
    # Fallback to integer comparison (truncates decimals)
    ACC_INT=${ACCURACY%.*}
    if (( ACC_INT < EXPECTED_ACCURACY_MIN )); then
        echo "❌ FAILED: Accuracy below minimum threshold"
        exit 1
    elif (( ACC_INT > EXPECTED_ACCURACY_MAX )); then
        echo "❌ FAILED: Accuracy above maximum (suspicious)"
        exit 1
    else
        echo "✓ PASSED"
    fi
fi

echo "=========================================="
