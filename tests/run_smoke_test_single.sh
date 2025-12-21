#!/bin/bash
# DietNetwork Single Model Test V2 - Uses CLI only
#
# Usage: bash tests/run_smoke_test_single_v2.sh [seed] [fold]
# Example: bash tests/run_smoke_test_single_v2.sh 78 0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/smoke_test_config.sh"

# Get seed and fold from arguments (default to 78 and 0)
SEED=${1:-78}
FOLD=${2:-0}

echo "=========================================="
echo "DietNetwork Single Model Test (CLI Version)"
echo "Seed: $SEED, Fold: $FOLD"
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

# Step 2: Preprocess PLINK data
echo ""
echo "Step 2/4: Preprocessing PLINK data with model alignment..."

PLINK_BASE="1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1"
PLINK_PREFIX="$TEST_DATA_DIR/$PLINK_BASE"

OUTPUT_DIR="$SCRIPT_DIR/outputs"
PREPROCESSED_DIR="$OUTPUT_DIR/preprocessed_plink"
PREPROCESSED_PREFIX="$PREPROCESSED_DIR/test_preprocessed"

mkdir -p "$PREPROCESSED_DIR"

dietnet preprocess-plink \
    --model 1kgp_default \
    --plink-prefix "$PLINK_PREFIX" \
    --output-prefix "$PREPROCESSED_PREFIX"

# Step 3: Run inference with single model
echo ""
echo "Step 3/4: Running inference with seed $SEED fold $FOLD..."

LABELS="$TEST_DATA_DIR/labels_pop_subsampleV1.tsv"
OUTPUT="$OUTPUT_DIR/single_model_predictions.tsv"
LOGITS="$OUTPUT_DIR/single_model_predictions_logits.npz"
HIDDEN="$OUTPUT_DIR/single_model_predictions_hidden.npz"
mkdir -p "$OUTPUT_DIR"

dietnet predict \
    --model 1kgp_default \
    --plink-prefix "$PREPROCESSED_PREFIX" \
    --output "$OUTPUT" \
    --seeds $SEED \
    --folds $FOLD \
    --batch-size 256 \
    --device cpu \
    --num-workers 4 \
    --skip-preprocess \
    --save-logits "$LOGITS" \
    --save-hidden "$HIDDEN"

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
echo "✓ Single Model Test Complete (CLI Version)"
echo "Model: seed $SEED, fold $FOLD"
echo "=========================================="
