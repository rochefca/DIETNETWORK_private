#!/usr/bin/env bash
#SBATCH --job-name=dietnet_train_predict
#SBATCH --output=logs/dietnet_train_predict.%j.out
#SBATCH --error=logs/dietnet_train_predict.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Purpose: train a DietNetwork ensemble on a PLINK dataset and run inference on another PLINK dataset.
# Usage: edit the absolute paths below, then sbatch this script.

set -euo pipefail

############################################
# User configuration (absolute paths)
############################################

# Training data (PLINK prefix without extension)
TRAIN_PLINK_PREFIX="/abs/path/to/train"

# Label TSV (sample_id<TAB>label) for training/partition stratification
LABEL_TSV="/abs/path/to/labels.tsv"

# Where to write intermediate artifacts (partitions, embeddings, stats, packages)
EXP_PATH="/abs/path/to/experiment_dir"
EXP_NAME="my_experiment"

# Config YAML used for training
CONFIG="/abs/path/to/config.yaml"

# Ensemble seeds/folds
SEEDS=(42 43 44)
FOLDS=(0 1 2 3 4)

# Test data (PLINK prefix) for inference
TEST_PLINK_PREFIX="/abs/path/to/test"

# Prediction output file
PREDICTIONS_OUT="/abs/path/to/predictions.tsv"

############################################
# Derived paths (no edits usually needed)
############################################
TRAIN_DATASET_BED="${TRAIN_PLINK_PREFIX}.bed"
PARTITION_FILE="${EXP_PATH}/partitioned_idx.npz"
EMBED_FILE="${EXP_PATH}/embedding.npz"
INPUT_STATS_FILE="${EXP_PATH}/input_features_means.npz"
PACKAGE_DIR="${EXP_PATH}/${EXP_NAME}_packages"

mkdir -p "${EXP_PATH}" logs

echo "[1/6] Partitioning (stratified)..."
dietnet partition \
  --exp-path "${EXP_PATH}" \
  --dataset "${TRAIN_DATASET_BED}" \
  --label-file "${LABEL_TSV}" \
  --nb-folds ${#FOLDS[@]} \
  --stratify \
  --output-name "$(basename "${PARTITION_FILE}")"

echo "[2/6] Computing embeddings..."
dietnet generate-embedding \
  --exp-path "${EXP_PATH}" \
  --dataset "${TRAIN_DATASET_BED}" \
  --partition "$(basename "${PARTITION_FILE}")" \
  --label-file "${LABEL_TSV}" \
  --output-name "$(basename "${EMBED_FILE}")"

echo "[3/6] Computing input stats..."
python3 Dietnet/compute_input_features_mean.py \
  --exp-path "${EXP_PATH}" \
  --dataset "${TRAIN_DATASET_BED}" \
  --partition "$(basename "${PARTITION_FILE}")" \
  --out "$(basename "${INPUT_STATS_FILE}")"

echo "[4/6] Training ensemble and packaging..."
dietnet train \
  --exp-path "${EXP_PATH}" \
  --exp-name "${EXP_NAME}" \
  --config "${CONFIG}" \
  --plink-prefix "${TRAIN_PLINK_PREFIX}" \
  --label-file "${LABEL_TSV}" \
  --partition "$(basename "${PARTITION_FILE}")" \
  --embedding "$(basename "${EMBED_FILE}")" \
  --input-features-means "$(basename "${INPUT_STATS_FILE}")" \
  --seeds "${SEEDS[@]}" \
  --folds "${FOLDS[@]}" \
  --output-dir "${PACKAGE_DIR}"

echo "[5/6] Running inference..."
dietnet predict \
  --model "${PACKAGE_DIR}" \
  --plink-prefix "${TEST_PLINK_PREFIX}" \
  --output "${PREDICTIONS_OUT}" \
  --seeds "${SEEDS[@]}" \
  --folds "${FOLDS[@]}"

echo "[6/6] Done. Predictions saved to ${PREDICTIONS_OUT}"
