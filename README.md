# DIETNETWORK

PyTorch implementation of DietNetwork (https://arxiv.org/abs/1611.09340) for genetic ancestry classification.

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/rochefca/DIETNETWORK.git
cd DIETNETWORK

# Create virtual environment and install package
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Verify installation
dietnet info
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .
```

### Optional dependencies

```bash
# For interpretability features
uv pip install -e ".[interpretability]"

# For experiment tracking with Comet.ML
uv pip install -e ".[tracking]"

# For development
uv pip install -e ".[dev]"
```

## Quick Start - Smoke Tests

Run the bundled smoke tests to verify the code works:

- **Run immediately after cloning** (everything downloads on first run):  
  ```bash
  
  # Downloads data and model and performs inference on data using a single model
  bash tests/kgp_precomputed/run_smoke_test_single.sh

  # Performs inference using ensemble model
  bash tests/kgp_precomputed/run_smoke_test.sh

  # Train + predict on the small bundled 1KGP subset (much slower, GPU is recommended)
  bash tests/kgp_precomputed/run_train_smoke.sh

  ```
  These commands stay within `tests/kgp_precomputed/` and manage their own cached data/model downloads.


All smoke tests download required assets on first run, then reuse cached data/models.

## Using Model Presets

Run inference with pretrained models:

```bash
# Using a preset model (downloads automatically)
dietnet predict --model 1kgp_default \
                --plink-prefix /path/to/your/data \
                --output predictions.tsv

# Using a local model package
dietnet predict --model /path/to/pretrained/model \
                --plink-prefix /path/to/your/data \
                --output predictions.tsv
```

**Available presets:**
- `1kgp_default`: 1000 Genomes Phase 3 (24 populations, single model)
- `hgdp_ukbb`: HGDP+1KGP for UKBB inference (coming soon)

## Train on PLINK Data (recommended)

Train directly from PLINK files and produce packaged models (`seed_X/fold_Y` with `model.pt`, `metadata.json`, `snps.txt`, `input_stats.npz`, `embedding.npz`, `label_mapping.json`, `allpos.bim`):

```bash
# 1) Partition your PLINK dataset (optionally stratify by population)
dietnet partition \
    --exp-path /path/to/experiment_dir \
    --dataset /path/to/train_prefix.bed \
    --nb-folds 5 \
    --stratify \
    --label-file /path/to/labels.tsv   # required for stratified PLINK

# 2) Compute embeddings per fold
dietnet generate-embedding \
    --exp-path /path/to/experiment_dir \
    --dataset /path/to/train_prefix.bed \
    --label-file /path/to/labels.tsv \
    --output-name embedding.npz

# 3) Compute input stats (means/stds) per fold
python Dietnet/compute_input_features_mean.py \
    --exp-path /path/to/experiment_dir \
    --dataset /path/to/train_prefix.bed \
    --partition partitioned_idx.npz \
    --out input_features_means.npz

# 4) Train and package a single model (one fold)
dietnet train \
    --exp-path /path/to/experiment_dir \
    --exp-name my_experiment \
    --config config.yaml \
    --plink-prefix /path/to/train_prefix \
    --label-file /path/to/labels.tsv \
    --folds 0   # pick one fold if you only want a single model

# 5) Train an ensemble across seeds/folds
dietnet train \
    --exp-path /path/to/experiment_dir \
    --exp-name my_experiment \
    --config config.yaml \
    --plink-prefix /path/to/train_prefix \
    --label-file /path/to/labels.tsv \
    --seeds 42 43 44 \
    --folds 0 1 2 3 4 \
    --output-dir /path/to/my_packages   # optional override
```

Packages land in `<exp-path>/<exp-name>_packages/seed_*/fold_*/` by default and are ready for `dietnet predict`.
`--seeds` and `--folds` are Click “multiple” options: repeat the flag (`--seeds 42 --seeds 43`) or provide comma-separated values in one flag (`--seeds 42,43`). The same applies to `--folds`.

## Inference (presets or your own packages)

### Preset models (downloaded automatically)
```bash
dietnet predict --model 1kgp_default \
                --plink-prefix /path/to/test_data \
                --output predictions.tsv \
                --temp-dir ./outputs/preprocessed_plink
```
`--temp-dir` controls where intermediate PLINK preprocessing files are written (default: `./preprocessed_plink`).

### Your own packaged models
Point `--model` to the directory that contains `seed_*` folders (the parent of the packages):
```bash
dietnet predict --model ./my_packages \
                --plink-prefix /path/to/test_data \
                --output predictions.tsv \
                --temp-dir ./outputs/preprocessed_plink

# Use a subset of seeds/folds from your ensemble
dietnet predict --model ./my_packages \
                --plink-prefix /path/to/test_data \
                --output predictions.tsv \
                --seeds 42 43 --folds 0 1
```

## Training from Scratch (legacy HDF5 path)

Note: this will become deprecated soon.

### Complete workflow

```bash
# 1. Create dataset from genotype and label files
dietnet create-dataset \
    --genotypes data/snps.txt \
    --labels data/labels.txt \
    --output-dir ./processed

# 2. Partition into cross-validation folds
dietnet partition \
    --exp-path ./processed \
    --nb-folds 5

# 3. Generate genotype frequency embeddings
dietnet generate-embedding \
    --exp-path ./processed

# 4. Train model on a specific fold
dietnet train \
    --exp-path ./processed \
    --exp-name experiment1 \
    --which-fold 0 \
    --config config.yaml

# 5. Run predictions on external data
dietnet predict \
    --test-dataset test.hdf5 \
    --train-dataset ./processed/dataset.hdf5 \
    --config ./processed/experiment1/config.yaml \
    --embedding ./processed/embedding.npz \
    --input-features-stats ./processed/input_features_means.npz \
    --model-params ./processed/experiment1/best_model.pt \
    --output-dir ./results \
    --which-fold 0
```

### Get help

```bash
# General help
dietnet --help

# Command-specific help
dietnet train --help
dietnet predict --help
```

## FAQ

### I can't download the data or model since my compute node has no access to the internet

You can run this prior to running the smoke tests:
```bash
bash tests/kgp_precomputed/download_test_data.sh
```

Likewise the models can be downloaded using the following:
```bash
bash tests/kgp_precomputed/download_model.sh
```
