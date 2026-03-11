#!/usr/bin/env bash
#SBATCH --job-name=dietnet_train_predict
#SBATCH --output=logs/dietnet_train_predict.%j.out
#SBATCH --error=logs/dietnet_train_predict.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --account=ctb-hussinju
#SBATCH --time=24:00:00

# Purpose: train a DietNetwork ensemble on a PLINK dataset and run inference on another PLINK dataset.
# Usage: edit the absolute paths below, then sbatch this script.

set -euo pipefail

############################################
# User configuration (absolute paths)
############################################

# Training data (PLINK prefix without extension)
TRAIN_PLINK_PREFIX="/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/tests/data/ukbb_test_data/hgdp_1kgp"

# Label TSV (sample_id<TAB>label) for training/partition stratification
LABEL_TSV="/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/tests/data/ukbb_test_data/hgdp_1kgp_labels.tsv"

# Where to write intermediate artifacts (partitions, embeddings, stats, packages)
EXP_PATH="/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/tests/data/ukbb_test_data/outputs"
EXP_NAME="EXP1"

# Config YAML used for training
CONFIG="/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/tests/data/ukbb_test_data/config.yaml"

# Ensemble seeds/folds
SEEDS=(42 43 44)
FOLDS=(0 1 2 3 4)

# Test data (PLINK prefix) for external inference (e.g. UKBB)
TEST_PLINK_PREFIX="/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/tests/data/ukbb_test_data/ukbb"

# Prediction output file for external data
PREDICTIONS_OUT="/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/tests/data/ukbb_test_data/outputs/predictions.tsv"

############################################
# Derived paths (no edits usually needed)
############################################
TRAIN_DATASET_BED="${TRAIN_PLINK_PREFIX}.bed"
PARTITION_FILE="${EXP_PATH}/partitioned_idx.npz"
EMBED_FILE="${EXP_PATH}/embedding.npz"
INPUT_STATS_FILE="${EXP_PATH}/input_features_means.npz"
PACKAGE_DIR="${EXP_PATH}/${EXP_NAME}_packages"

# Click multiple=True requires one flag per value: --seeds 42 --seeds 43 ...
SEED_FLAGS=(); for s in "${SEEDS[@]}"; do SEED_FLAGS+=(--seeds "$s"); done
FOLD_FLAGS=(); for f in "${FOLDS[@]}"; do FOLD_FLAGS+=(--folds "$f"); done

mkdir -p "${EXP_PATH}" logs
source "$(dirname "$(dirname "$(realpath "$0")")")/.venv/bin/activate"

echo "[1/7] Partitioning (stratified)..."
dietnet partition \
  --exp-path "${EXP_PATH}" \
  --dataset "${TRAIN_DATASET_BED}" \
  --label-file "${LABEL_TSV}" \
  --nb-folds ${#FOLDS[@]} \
  --stratify \
  --output-name "$(basename "${PARTITION_FILE}")"

echo "[2/7] Computing embeddings..."
dietnet generate-embedding \
  --exp-path "${EXP_PATH}" \
  --dataset "${TRAIN_DATASET_BED}" \
  --partition "$(basename "${PARTITION_FILE}")" \
  --label-file "${LABEL_TSV}" \
  --output-name "$(basename "${EMBED_FILE}")"

echo "[3/7] Computing input stats..."
dietnet compute-stats \
  --exp-path "${EXP_PATH}" \
  --dataset "${TRAIN_DATASET_BED}" \
  --partition "$(basename "${PARTITION_FILE}")" \
  --output-name "$(basename "${INPUT_STATS_FILE}")"

echo "[4/7] Training ensemble and packaging..."
dietnet train \
  --exp-path "${EXP_PATH}" \
  --exp-name "${EXP_NAME}" \
  --config "${CONFIG}" \
  --plink-prefix "${TRAIN_PLINK_PREFIX}" \
  --label-file "${LABEL_TSV}" \
  --partition "$(basename "${PARTITION_FILE}")" \
  --embedding "$(basename "${EMBED_FILE}")" \
  --input-features-means "$(basename "${INPUT_STATS_FILE}")" \
  "${SEED_FLAGS[@]}" \
  "${FOLD_FLAGS[@]}" \
  --output-dir "${PACKAGE_DIR}"

echo "[5/7] Checking hold-out test accuracy per fold..."
for s in "${SEEDS[@]}"; do
  for f in "${FOLDS[@]}"; do
    PRED="${EXP_PATH}/${EXP_NAME}/${EXP_NAME}_seed${s}_fold${f}/predictions.tsv"
    echo "  seed=${s} fold=${f}:"
    dietnet check --predictions "${PRED}" --labels "${LABEL_TSV}"
  done
done

echo "[6/7] Running inference on external dataset..."
dietnet predict \
  --model "${PACKAGE_DIR}" \
  --plink-prefix "${TEST_PLINK_PREFIX}" \
  --output "${PREDICTIONS_OUT}" \
  "${SEED_FLAGS[@]}" \
  "${FOLD_FLAGS[@]}"

# echo "[7/7] Checking external dataset accuracy (fill in UKBB_LABEL_TSV if available)..."
# UKBB_LABEL_TSV="/abs/path/to/ukbb_labels.tsv"
# dietnet check \
#   --predictions "${PREDICTIONS_OUT}" \
#   --labels "${UKBB_LABEL_TSV}"

echo "[7/7] Done. External predictions saved to ${PREDICTIONS_OUT}"
echo ""
echo "Results written to: ${EXP_PATH}"
