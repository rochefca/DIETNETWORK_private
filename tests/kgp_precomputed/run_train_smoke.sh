#!/usr/bin/env bash
# Smoke test: train DietNetwork on the bundled 1KGP test PLINK data and evaluate on its held-out fold.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."

if [ -d "$PROJECT_ROOT/.venv" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

DATA_DIR="$SCRIPT_DIR/data/1kgp_test_data"
PLINK_BASE="1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1"
PLINK_PREFIX="$DATA_DIR/$PLINK_BASE"
LABELS="$DATA_DIR/labels_pop_subsampleV1.tsv"
CONFIG="$SCRIPT_DIR/data/config.yaml"

EXP_PATH="$SCRIPT_DIR/output_train"
EXP_NAME="train_smoke"
PARTITION_FILE="$EXP_PATH/partitioned_idx.npz"
EMBED_FILE="$EXP_PATH/embedding.npz"
INPUT_STATS_FILE="$EXP_PATH/input_features_means.npz"
PACKAGE_DIR="$EXP_PATH/${EXP_NAME}_packages"
PREPROCESSED_DIR="$EXP_PATH/preprocessed_plink"

SEED=78
FOLD=0
N_FOLDS=5

mkdir -p "$EXP_PATH" "$PROJECT_ROOT/logs"

echo "[1/6] Partitioning (${N_FOLDS} folds)..."
dietnet partition \
  --exp-path "$EXP_PATH" \
  --dataset "${PLINK_PREFIX}.bed" \
  --label-file "$LABELS" \
  --nb-folds "$N_FOLDS" \
  --stratify \
  --output-name "$(basename "$PARTITION_FILE")"

echo "[2/6] Embeddings..."
dietnet generate-embedding \
  --exp-path "$EXP_PATH" \
  --dataset "${PLINK_PREFIX}.bed" \
  --partition "$(basename "$PARTITION_FILE")" \
  --label-file "$LABELS" \
  --output-name "$(basename "$EMBED_FILE")"

echo "[3/6] Input stats..."
python3 "$PROJECT_ROOT/Dietnet/compute_input_features_mean.py" \
  --exp-path "$EXP_PATH" \
  --dataset "${PLINK_PREFIX}.bed" \
  --partition "$(basename "$PARTITION_FILE")" \
  --out "$(basename "$INPUT_STATS_FILE")"

echo "[4/6] Training (seed ${SEED}, fold ${FOLD})..."
dietnet train \
  --exp-path "$EXP_PATH" \
  --exp-name "$EXP_NAME" \
  --config "$CONFIG" \
  --plink-prefix "$PLINK_PREFIX" \
  --label-file "$LABELS" \
  --partition "$(basename "$PARTITION_FILE")" \
  --embedding "$(basename "$EMBED_FILE")" \
  --input-features-means "$(basename "$INPUT_STATS_FILE")" \
  --seeds "$SEED" \
  --folds "$FOLD" \
  --output-dir "$PACKAGE_DIR"

echo "[5/6] Predict on the same fold (sanity check)..."
dietnet predict \
  --model "$PACKAGE_DIR" \
  --plink-prefix "$PLINK_PREFIX" \
  --output "$EXP_PATH/train_smoke_predictions.tsv" \
  --seeds "$SEED" \
  --folds "$FOLD" \
  --temp-dir "$PREPROCESSED_DIR"

echo "[6/6] Done. Check outputs in $EXP_PATH"
