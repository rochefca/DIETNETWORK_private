#!/bin/bash
# DietNetwork Smoke Test V2 - Uses CLI only
#
# This version uses only the dietnet CLI commands:
#   - dietnet preprocess-plink: Align test data to model SNPs
#   - dietnet predict: Run inference
#   - dietnet check: Validate predictions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/smoke_test_config.sh"

echo "=========================================="
echo "DietNetwork Smoke Test (CLI Version)"
echo "=========================================="
echo ""

# Activate venv
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "WARNING: No virtual environment found at $PROJECT_ROOT/.venv"
fi

# Step 1: Download test data
echo "Step 1/4: Ensuring test data is available..."
bash "$SCRIPT_DIR/download_test_data.sh"

# Step 2: Preprocess PLINK data (once for all models)
echo ""
echo "Step 2/4: Preprocessing PLINK data with model alignment..."

PLINK_BASE="1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1"
PLINK_PREFIX="$TEST_DATA_DIR/$PLINK_BASE"

# Use absolute paths for preprocessed files
PREPROCESSED_DIR="$SCRIPT_DIR/preprocessed_plink"
PREPROCESSED_PREFIX="$PREPROCESSED_DIR/test_preprocessed"

mkdir -p "$PREPROCESSED_DIR"

# Convert to absolute path
PREPROCESSED_PREFIX="$(cd "$SCRIPT_DIR" && pwd)/preprocessed_plink/test_preprocessed"

dietnet preprocess-plink \
    --model 1kgp_default \
    --plink-prefix "$PLINK_PREFIX" \
    --output-prefix "$PREPROCESSED_PREFIX"

# Step 3: Run inference with all models (ensemble)
echo ""
echo "Step 3/4: Running inference with 1kgp_default preset (ensemble)..."

LABELS="$TEST_DATA_DIR/labels_pop_subsampleV1.tsv"
OUTPUT="$SCRIPT_DIR/smoke_test_predictions.tsv"

dietnet predict \
    --model 1kgp_default \
    --plink-prefix "$PREPROCESSED_PREFIX" \
    --output "$OUTPUT" \
    --batch-size 256 \
    --device cpu \
    --num-workers 4 \
    --skip-preprocess

# Step 4: Validate predictions
echo ""
echo "Step 4/4: Validating accuracy..."

dietnet check \
    --predictions "$OUTPUT" \
    --labels "$LABELS" \
    --min-accuracy 0.85 \
    --max-accuracy 1.0

echo ""
echo "=========================================="
echo "✓ Smoke Test Complete (CLI Version)"
echo "=========================================="
