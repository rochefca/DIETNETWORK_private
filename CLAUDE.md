# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch implementation of DietNetwork (https://arxiv.org/abs/1611.09340), a neural network architecture for genetic ancestry classification. The network uses:
- **Auxiliary Network** (feature embedding network): Learns genotype frequency embeddings
- **Discriminative Network** (main network): Performs classification using the learned embeddings

## Current State and Known Issues

**CRITICAL LIMITATIONS:**
1. No dependency management (no setup.py, pyproject.toml, or requirements.txt)
2. No unit tests
3. No CLI or proper entry points
4. Pipeline converts PLINK files → HDF5 → DietNetwork (inefficient, should read PLINK directly)

The codebase has working pre-trained models in `other_code/neededfiles/` but needs significant refactoring.

## Architecture

### Data Flow Pipeline

```
PLINK files (.bed/.bim/.fam)
  → create_dataset.py → dataset.hdf5
  → partition_data.py → folds_indexes.npz
  → generate_embedding.py → embedding.npz
  → train.py → model checkpoints (.pt)
  → test_external_dataset.py → predictions
```

### Core Components

**Main Scripts (in `Dietnet/`):**
- `create_dataset.py`: Parses PLINK/text genotype files into HDF5 format
- `partition_data.py`: Creates k-fold cross-validation splits
- `generate_embedding.py`: Computes genotype frequency embeddings per fold
- `train.py`: Main training loop with early stopping
- `test_external_dataset.py`: Inference on external datasets (SNP alignment required)

**Helper Modules (in `Dietnet/helpers/`):**
- `dataset_utils.py`: Data loading, shuffling, partitioning, preprocessing (missing value imputation, normalization)
- `model.py`: Neural network definitions (`Feat_emb_net`, `Discrim_net`, `Discrim_net2`)
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
- **Genotypes**: Tab-separated text file with SNPs in additive encoding (0/1/2) or PLINK binary format (.bed/.bim/.fam)
  - Missing values encoded as -1
  - Header row: sample IDs
  - First column: SNP names
- **Labels**: Tab-separated file with sample IDs and labels
  - For regression: Include separate class labels file for embedding computation

### Intermediate Files
- **dataset.hdf5**: Contains `inputs` (genotypes), `labels`, `samples`, `snp_names`, `label_names`
- **folds_indexes.npz**: Array of [train_idx, valid_idx, test_idx] for each fold
- **embedding.npz**: Genotype frequency embeddings per fold
- **input_stats.npz** or **input_features_means.npz**: Mean values for missing value imputation

### Training Configuration
Training uses YAML config files (see `other_code/neededfiles/config.yaml`):
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

Pre-trained models exist for 1000 Genomes Project data:
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

**SNP Alignment (for external datasets):**
- `test_external_dataset.py` aligns test SNPs to training SNPs by chromosome:position
- Missing SNPs in test data filled with -1 (then imputed)

## Running Inference (Example from other_code)

The bash script `other_code/generalisation_v4.sh` shows production inference workflow:
1. Load environment and modules
2. Loop over seeds × folds × data chunks
3. Run `test_independent_dataset_v4.py` for each chunk
4. Merge chunk predictions
5. Aggregate predictions across models (voting)

## Key Design Patterns

**Two-Stage Training:**
1. Auxiliary network learns from genotype frequency embeddings
2. Discriminative network uses auxiliary weights as first layer (frozen or fine-tuned)

**Cross-Validation:**
- K-fold CV with stratified splits
- Embeddings computed per fold (avoiding data leakage)
- Early stopping on validation set

**HDF5 Usage:**
- Lazy loading via torch.utils.data.Dataset
- Files kept open during training (`FoldDataset.f`)
- External test data uses separate `IndepTestDataset` class

## Interpretability

The `Dietnet/Interpretability/` module provides:
- Attribution analysis (using Captum library)
- SNP importance experiments
- Graph-based attribution management

## Dependencies (Inferred)

```
python >= 3.6
torch >= 1.5.0
numpy
pandas
h5py
yaml
comet_ml (optional, for experiment tracking)
captum (for interpretability)
pyplink (for reading PLINK files directly)
```

## Future Refactoring Priorities

1. **Direct PLINK reading**: See `999_recompute_pca.ipynb` for pyplink example
2. **Dependency management**: Create pyproject.toml with uv/pip
3. **CLI**: Use argparse/click for unified train/inference commands
4. **Unit tests**: Test data loading, preprocessing, model forward passes
5. **Consolidate duplicated code**: `other_code/` has copies of helpers that diverged from `Dietnet/helpers/`
