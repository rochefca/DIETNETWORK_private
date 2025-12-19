#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --account=ctb-hussinju
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=dietnet_plink_test_train
#SBATCH --output=test_output_plink/train_%j.out
#SBATCH --error=test_output_plink/train_%j.err

set -euo pipefail

echo "=========================================="
echo "DietNetwork PLINK Training Test Script"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Configuration
WORK_DIR="$(pwd)/test_output_plink"
PLINK_SOURCE="/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_V1/1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1"
LABEL_FILE="/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_V1/labels_pop_subsampleV1.tsv"
DATASET_NAME="1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1.bed"
FOLD=0
NB_FOLDS=5
SEED=42

# Create working directory
echo "Creating working directory: $WORK_DIR"
mkdir -p "$WORK_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Symlink the PLINK files (.bed, .bim, .fam)
echo ""
echo "Linking PLINK files..."
ln -sf "${PLINK_SOURCE}.bed" "$WORK_DIR/$DATASET_NAME"
ln -sf "${PLINK_SOURCE}.bim" "$WORK_DIR/${DATASET_NAME%.bed}.bim"
ln -sf "${PLINK_SOURCE}.fam" "$WORK_DIR/${DATASET_NAME%.bed}.fam"
echo "✓ PLINK files linked"

# Symlink label file
echo "Linking label file..."
ln -sf "$LABEL_FILE" "$WORK_DIR/labels.tsv"
echo "✓ Label file linked"

# Create config.yaml
echo ""
echo "Creating config.yaml..."
cat > "$WORK_DIR/config.yaml" << 'CONFIG_EOF'
# Training hyperparameters for DietNetwork
batch_size: 138
epochs: 8000
input_dropout: 0.995
dropout_main: 0.1
lr_aux: 0.0012
lr_main: 0.0006
learning_rate_annealing: 0.995
nb_hidden_u_aux:
  - 100
  - 100
nb_hidden_u_main:
  - 100
patience: 2000
seed: 78
uniform_init_limit: 0.02
CONFIG_EOF
echo "✓ Config created"

# Step 1: Partition data into folds
echo ""
echo "=========================================="
echo "Step 1: Partitioning data into $NB_FOLDS folds"
echo "=========================================="
python -m Dietnet.partition_data \
    --exp-path "$WORK_DIR" \
    --dataset "$DATASET_NAME" \
    --out "partitioned_idx.npz" \
    --nb-folds $NB_FOLDS \
    --train-valid-ratio 0.8 \
    --seed $SEED

echo "✓ Partitioning complete"

# Step 2: Compute input features means (for missing value imputation)
echo ""
echo "=========================================="
echo "Step 2: Computing input feature means"
echo "=========================================="
python -m Dietnet.compute_input_features_mean \
    --exp-path "$WORK_DIR" \
    --dataset "$DATASET_NAME" \
    --partition "partitioned_idx.npz" \
    --out "input_features_means.npz"

echo "✓ Input features means computed"

# Step 3: Generate embeddings
echo ""
echo "=========================================="
echo "Step 3: Generating genotype frequency embeddings"
echo "=========================================="
python -m Dietnet.generate_embedding \
    --exp-path "$WORK_DIR" \
    --dataset "$DATASET_NAME" \
    --partition "partitioned_idx.npz" \
    --label-file "labels.tsv" \
    --out "embedding.npz" \
    --task classification

echo "✓ Embeddings generated"

# Step 4: Train model on fold 0
echo ""
echo "=========================================="
echo "Step 4: Training DietNetwork on fold $FOLD"
echo "=========================================="
mkdir -p "$WORK_DIR/experiment1"
cp "$WORK_DIR/config.yaml" "$WORK_DIR/experiment1/config.yaml"

python -m Dietnet.train \
    --exp-path "$WORK_DIR" \
    --exp-name "experiment1" \
    --config "config.yaml" \
    --dataset "$DATASET_NAME" \
    --partition "partitioned_idx.npz" \
    --embedding "embedding.npz" \
    --input-features-means "input_features_means.npz" \
    --label-file "labels.tsv" \
    --which-fold $FOLD \
    --task classification \
    --normalize

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: $WORK_DIR/experiment1/"
echo ""
echo "To view results:"
echo "  ls -lh $WORK_DIR/experiment1/"
echo ""
echo "To clean up:"
echo "  rm -rf $WORK_DIR"
