#!/usr/bin/env bash
#SBATCH --job-name=dietnet_1kgp_train_eval
#SBATCH --output=logs/dietnet_1kgp_train_eval.%j.out
#SBATCH --error=logs/dietnet_1kgp_train_eval.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --account=ctb-hussinju
#SBATCH --time=24:00:00

# Purpose: train DietNetwork on 1KGP (1 seed, 1 fold) and evaluate on the held-out test fold.
# Usage: sbatch scripts/sbatch_1kgp_train_eval.sh

set -euo pipefail

############################################
# Paths
############################################

REPO_DIR="/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK"

PLINK_BASENAME="1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1"
PLINK_PREFIX="${REPO_DIR}/tests/data/1kgp_test_data/${PLINK_BASENAME}"
LABEL_TSV="${REPO_DIR}/tests/data/1kgp_test_data/labels_pop_subsampleV1.tsv"
CONFIG="${REPO_DIR}/tests/data/ukbb_test_data/config.yaml"

EXP_PATH="${REPO_DIR}/logs/1kgp_single_run"
EXP_NAME="1kgp_run"

SEED=78
FOLD=0
NB_FOLDS=5

############################################
# Derived paths
############################################

PLINK_BED="${PLINK_PREFIX}.bed"
PARTITION_FILE="${EXP_PATH}/partitioned_idx.npz"
EMBED_FILE="${EXP_PATH}/embedding.npz"
INPUT_STATS_FILE="${EXP_PATH}/input_features_means.npz"
PACKAGE_DIR="${EXP_PATH}/seed78_fold0_packages"

############################################
# Setup
############################################

mkdir -p "${EXP_PATH}" logs
source "${REPO_DIR}/.venv/bin/activate"

echo "================================================"
echo " DietNetwork 1KGP train+eval (seed=${SEED}, fold=${FOLD})"
echo "================================================"

############################################
# Pipeline
############################################

# echo "[1/5] Partitioning (stratified, ${NB_FOLDS} folds)..."
# dietnet partition \
#   --exp-path "${EXP_PATH}" \
#   --dataset "${PLINK_BED}" \
#   --label-file "${LABEL_TSV}" \
#   --nb-folds ${NB_FOLDS} \
#   --stratify \
#   --output-name "$(basename "${PARTITION_FILE}")"

# echo "[2/5] Computing embeddings (fold ${FOLD})..."
# dietnet generate-embedding \
#   --exp-path "${EXP_PATH}" \
#   --dataset "${PLINK_BED}" \
#   --partition "$(basename "${PARTITION_FILE}")" \
#   --label-file "${LABEL_TSV}" \
#   --output-name "$(basename "${EMBED_FILE}")"

# echo "[3/5] Computing input stats..."
# dietnet compute-stats \
#   --exp-path "${EXP_PATH}" \
#   --dataset "${PLINK_BED}" \
#   --partition "$(basename "${PARTITION_FILE}")" \
#   --output-name "$(basename "${INPUT_STATS_FILE}")"

# echo "[4/5] Training (seed=${SEED}, fold=${FOLD})..."
# dietnet train \
#   --exp-path "${EXP_PATH}" \
#   --exp-name "${EXP_NAME}" \
#   --config "${CONFIG}" \
#   --plink-prefix "${PLINK_PREFIX}" \
#   --label-file "${LABEL_TSV}" \
#   --partition "$(basename "${PARTITION_FILE}")" \
#   --embedding "$(basename "${EMBED_FILE}")" \
#   --input-features-means "$(basename "${INPUT_STATS_FILE}")" \
#   --seeds ${SEED} \
#   --folds ${FOLD} \
#   --output-dir "${PACKAGE_DIR}"

echo "[5/5] Checking hold-out test accuracy (seed=${SEED}, fold=${FOLD})..."
PRED="${EXP_PATH}/${EXP_NAME}/${EXP_NAME}_seed${SEED}_fold${FOLD}/predictions.tsv"
dietnet check \
  --predictions "${PRED}" \
  --labels "${LABEL_TSV}"

echo ""
echo "Results written to: ${EXP_PATH}"
