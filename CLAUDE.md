# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch implementation of DietNetwork (https://arxiv.org/abs/1611.09340), a neural network architecture for genetic ancestry classification. The network uses:
- **Auxiliary Network** (feature embedding network): Learns genotype frequency embeddings
- **Discriminative Network** (main network): Performs classification using the learned embeddings

## Current State

The project is a packaged Python tool (`pyproject.toml`, `hatchling` build backend) with a `dietnet` CLI entry point. PLINK files are read directly — no HDF5 intermediate step required for the main pipeline.

**Remaining limitations:**
1. Unit tests are minimal (smoke tests only)
2. `other_code/` contains old scripts that diverged from current `Dietnet/helpers/`

## CLI — `dietnet`

Installed via `pip install -e .` (or `uv pip install -e .`). Entry point: `Dietnet/cli.py`.

### Pipeline commands (canonical order)

```bash
# [1] Partition data into k folds
dietnet partition --exp-path DIR --dataset PLINK.bed --label-file LABELS.tsv \
  --nb-folds 5 --stratify --output-name partitioned_idx.npz

# [2] Compute genotype frequency embeddings per fold
dietnet generate-embedding --exp-path DIR --dataset PLINK.bed \
  --partition partitioned_idx.npz --label-file LABELS.tsv \
  --output-name embedding.npz

# [3] Compute per-fold feature means/stds (for imputation & normalization)
dietnet compute-input-stats --exp-path DIR --dataset PLINK.bed \
  --partition partitioned_idx.npz --output-name input_features_means.npz

# [4] Train
dietnet train --exp-path DIR --exp-name NAME --config config.yaml \
  --plink-prefix PLINK_PREFIX --label-file LABELS.tsv \
  --partition partitioned_idx.npz --embedding embedding.npz \
  --input-features-means input_features_means.npz \
  --seeds 78 --folds 0 --output-dir PACKAGE_DIR

# [5] Predict
dietnet predict --model PACKAGE_DIR --plink-prefix PLINK_PREFIX \
  --output predictions.tsv --seeds 78 --folds 0
```

### Other commands
- `dietnet create-dataset` — convert text genotypes → HDF5 (legacy use)
- `dietnet preprocess-plink` — pre-convert PLINK to npz for faster loading
- `dietnet check` — evaluate predictions vs ground-truth labels
- `dietnet analyze-population` — population-level accuracy plots
- `dietnet info` — show package/environment info

**Removed command:** `dietnet compute-stats` was a broken duplicate of `compute-input-stats` and has been deleted.

## Architecture

### Core Components

**Main Scripts (in `Dietnet/`):**
- `cli.py`: Click-based CLI entry point for all commands
- `train.py`: Training loop with early stopping
- `generate_embedding.py`: Computes genotype frequency embeddings per fold
- `compute_input_features_mean.py`: Computes per-fold feature means/stds; key function: `get_preprocessing_params(args=None)`
- `predict_with_plink.py`: Inference on PLINK datasets
- `partition_data.py`: Creates k-fold cross-validation splits

**Helper Modules (in `Dietnet/helpers/`):**
- `dataset_utils.py`: Data loading, preprocessing (missing value imputation, normalization)
- `model.py`: Neural network definitions (`Feat_emb_net`, `Discrim_net`, `Discrim_net2`)
- `model_package.py`: Model packaging/loading for inference
- `mainloop_utils.py`: Training utilities (accuracy computation, evaluation step)
- `log_utils.py`: Model and experiment tracking utilities
- `model_handlers.py`: Handlers for DietNetwork and MLP architectures
- `task_handlers.py`: Classification and regression task handlers

### Network Architecture

**Auxiliary Network (Feature Embedding):**
- Input: Genotype frequency embedding (computed from training data)
- Learns compressed representation of genetic features
- Uses tanh activation
- First layer weights become "fat layer" for discriminative network

**Discriminative Network (Main):**
- First layer uses weights from auxiliary network (parameter sharing)
- 2+ hidden layers with ReLU, BatchNorm, Dropout
- Output layer for classification/regression

## Data Formats

### Input Files
- **Genotypes**: PLINK binary format (.bed/.bim/.fam) — read directly via `pyplink`
  - Missing values encoded as -1 internally
- **Labels**: Tab-separated file with sample IDs and labels

### Key Intermediate Files
- **partitioned_idx.npz**: Array of [train_idx, valid_idx, test_idx] for each fold
- **embedding.npz**: Genotype frequency embeddings per fold
- **input_features_means.npz**: Per-fold feature means and stds for imputation/normalization

### Training Configuration
Training uses YAML config files (see `tests/kgp_precomputed/data/config.yaml`):
```yaml
batch_size: 138
epochs: 8000
input_dropout: 0.995  # High dropout on input layer
dropout_main: 0.1
lr_aux: 0.000600
lr_main: 0.0003
learning_rate_annealing: 0.99
nb_hidden_u_aux: [100, 100]
nb_hidden_u_main: [100]
patience: 2000  # Early stopping patience
seed: 78
uniform_init_limit: 0.02
```

## Training Reference Data

- **PLINK files**: `/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_V1/1000G.2504_WGS30x.GSA17k_MHI.intersectGSA.miss10perc.maf0.05.pruned.autosomes.noHLA.phased_imputed_V1.{bed,bim,fam}`
- **Labels**: `/lustre06/project/6065672/shared/DietNet/1KGB_POP24/1KGP/WGS30X_V1/labels_pop_subsampleV1.tsv`
- **Pre-trained models**: `other_code/neededfiles/best_model_seed_{78,79,80}_fold{0-4}/best_model.pt`

## Data Preprocessing

**Missing Values:**
- Encoded as -1 in input data
- Replaced by feature mean (computed on training set only)
- Not included in genotype frequency embedding computation

**Normalization:**
- Auxiliary network: Square Euclidean distance normalization
- Discriminative network: Standardization (z-score) using training set statistics

## Key Design Patterns

**Two-Stage Training:**
1. Auxiliary network learns from genotype frequency embeddings
2. Discriminative network uses auxiliary weights as first layer (frozen or fine-tuned)

**Cross-Validation:**
- K-fold CV with stratified splits
- Embeddings computed per fold (avoiding data leakage)
- Early stopping on validation set

## Scripts

- `scripts/sbatch_1kgp_train_eval.sh` — full train+eval pipeline on 1KGP data (SLURM)
- `scripts/sbatch_train_predict_ensemble.sh` — ensemble train+predict (SLURM)
- `scripts/sbatch_train_hgdp_ukbb.sh` — train on HGDP/UKBB data (SLURM)
- `scripts/train_predict.sh` — local train+predict convenience script
- `scripts/predict_external.sh` — predict on an external dataset

## Tests

- `tests/kgp_precomputed/run_smoke_test.sh` — smoke test with pre-trained model
- `tests/kgp_precomputed/run_train_smoke.sh` — smoke test full train pipeline
- `tests/kgp_precomputed/download_test_data.sh` — download 1KGP test data
- `tests/kgp_precomputed/download_model.sh` — download pre-trained model

## Dependencies

Managed via `pyproject.toml`. Key packages:
```
python >= 3.10
torch >= 2.0.0
numpy, pandas, h5py, pyyaml, click, pyplink, scikit-learn, tqdm, matplotlib
captum (optional, for interpretability)
comet-ml (optional, for experiment tracking)
```

Install: `uv pip install -e .` or `pip install -e .`

## Interpretability

The `Dietnet/Interpretability/` module provides attribution analysis (Captum library), SNP importance experiments, and graph-based attribution management.
