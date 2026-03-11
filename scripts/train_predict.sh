#!/usr/bin/env bash

# Purpose: Train a DietNetwork ensemble on PLINK data and run inference on another PLINK dataset.
# Usage: edit the paths in the "User configuration" section, then run:
#   bash scripts/train_predict_ensemble.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

############################################
# User configuration (edit these)
############################################

# Training data (PLINK prefix without extension)
TRAIN_PLINK_PREFIX="/abs/path/to/train_prefix"

# Labels TSV (sample_id<TAB>label) for training and stratification
LABEL_TSV="/abs/path/to/labels.tsv"

# Experiment tracking/output
EXP_PATH="/abs/path/to/experiment_dir"
EXP_NAME="my_experiment"
CONFIG="/abs/path/to/config.yaml"

# Number of folds for cross-validation
N_FOLDS=5

# Optional seeds/folds filters (comma-separated). Leave empty to use all.
SEEDS=""
FOLDS=""

# Test data for inference (PLINK prefix)
TEST_PLINK_PREFIX="/abs/path/to/test_prefix"
PREDICTIONS_OUT="/abs/path/to/predictions.tsv"

# Batch size for inference
BATCH_SIZE=256

# Optional temp dir for PLINK preprocessing during predict (defaults to outputs/ inside EXP_PATH)
TEMP_DIR=""

############################################
# Derived paths (usually no edits needed)
############################################
TRAIN_DATASET_BED="${TRAIN_PLINK_PREFIX}.bed"
PARTITION_NAME="partitioned_idx.npz"
EMBED_NAME="embedding.npz"
INPUT_STATS_NAME="input_features_means.npz"
PACKAGE_DIR="${EXP_PATH}/${EXP_NAME}_packages"

if [[ -z "${TEMP_DIR}" ]]; then
  TEMP_DIR="${EXP_PATH}/outputs"
fi

mkdir -p "${EXP_PATH}" "${TEMP_DIR}"

echo "[1/6] Partitioning (stratified, ${N_FOLDS} folds)..."
dietnet partition \
  --exp-path "${EXP_PATH}" \
  --dataset "${TRAIN_DATASET_BED}" \
  --label-file "${LABEL_TSV}" \
  --nb-folds "${N_FOLDS}" \
  --stratify \
  --output-name "${PARTITION_NAME}"

echo "[2/6] Computing embeddings..."
dietnet generate-embedding \
  --exp-path "${EXP_PATH}" \
  --dataset "${TRAIN_DATASET_BED}" \
  --partition "${PARTITION_NAME}" \
  --label-file "${LABEL_TSV}" \
  --output-name "${EMBED_NAME}"

echo "[3/6] Computing input stats..."
dietnet compute-input-stats \
  --exp-path "${EXP_PATH}" \
  --dataset "${TRAIN_DATASET_BED}" \
  --partition "${PARTITION_NAME}" \
  --output-name "${INPUT_STATS_NAME}"

echo "[4/6] Training ensemble and packaging..."
TRAIN_CMD=(dietnet train
  --exp-path "${EXP_PATH}"
  --exp-name "${EXP_NAME}"
  --config "${CONFIG}"
  --plink-prefix "${TRAIN_PLINK_PREFIX}"
  --label-file "${LABEL_TSV}"
  --partition "${PARTITION_NAME}"
  --embedding "${EMBED_NAME}"
  --input-features-means "${INPUT_STATS_NAME}"
  --output-dir "${PACKAGE_DIR}"
)
[[ -n "${SEEDS}" ]] && TRAIN_CMD+=(--seeds "${SEEDS}")
[[ -n "${FOLDS}" ]] && TRAIN_CMD+=(--folds "${FOLDS}")
"${TRAIN_CMD[@]}"

echo "[5/6] Running inference..."
PREDICT_CMD=(dietnet predict
  --model "${PACKAGE_DIR}"
  --plink-prefix "${TEST_PLINK_PREFIX}"
  --output "${PREDICTIONS_OUT}"
  --temp-dir "${TEMP_DIR}"
  --batch-size "${BATCH_SIZE}"
)
[[ -n "${SEEDS}" ]] && PREDICT_CMD+=(--seeds "${SEEDS}")
[[ -n "${FOLDS}" ]] && PREDICT_CMD+=(--folds "${FOLDS}")
"${PREDICT_CMD[@]}"

echo "[6/6] Done. Predictions saved to ${PREDICTIONS_OUT}"
