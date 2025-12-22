#!/usr/bin/env bash

# Purpose: Simple wrapper to run DietNetwork inference from the CLI.
# Usage: edit the configuration block, then run:
#   bash scripts/predict_external_data.sh
#
# Supports either a pretrained preset (e.g., 1kgp_default) or a local model
# package directory produced by dietnet train.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

############################################
# User configuration (edit these)
############################################

# Use either a preset name (e.g., "1kgp_default") OR a local package dir.
# If MODEL_DIR is non-empty, it takes precedence over MODEL_PRESET.
MODEL_PRESET="1kgp_default"          # leave empty to use MODEL_DIR instead
MODEL_DIR=""                         # e.g., "/abs/path/to/my_packages"

# PLINK test data (prefix without extension)
TEST_PLINK_PREFIX="/abs/path/to/test_prefix"

# Output predictions file
PREDICTIONS_OUT="/abs/path/to/predictions.tsv"

# Optional seeds/folds filters (comma-separated). Leave empty to use all.
SEEDS=""
FOLDS=""

# Temp directory for PLINK preprocessing (will be created if missing)
TEMP_DIR="./preprocessed_plink"

############################################
# Build model argument
############################################
MODEL_ARG=""
if [[ -n "${MODEL_DIR}" ]]; then
  MODEL_ARG="${MODEL_DIR}"
else
  MODEL_ARG="${MODEL_PRESET}"
fi

if [[ -z "${MODEL_ARG}" ]]; then
  echo "ERROR: Set either MODEL_PRESET or MODEL_DIR." >&2
  exit 1
fi

mkdir -p "$(dirname "${PREDICTIONS_OUT}")" "${TEMP_DIR}"

echo "[Predict] Using model: ${MODEL_ARG}"
echo "[Predict] Test data: ${TEST_PLINK_PREFIX}"
echo "[Predict] Output: ${PREDICTIONS_OUT}"

CMD=(dietnet predict
  --model "${MODEL_ARG}"
  --plink-prefix "${TEST_PLINK_PREFIX}"
  --output "${PREDICTIONS_OUT}"
  --temp-dir "${TEMP_DIR}"
)

if [[ -n "${SEEDS}" ]]; then
  CMD+=(--seeds "${SEEDS}")
fi
if [[ -n "${FOLDS}" ]]; then
  CMD+=(--folds "${FOLDS}")
fi

echo "[Predict] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[Predict] Done. Predictions saved to ${PREDICTIONS_OUT}"
