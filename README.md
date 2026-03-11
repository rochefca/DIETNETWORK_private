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
bash setup.sh # installs virtual environment dependencies and external dependencies

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

# 3) Compute input stats (means/stds) per fold via CLI
dietnet compute-input-stats \
    --exp-path /path/to/experiment_dir \
    --dataset /path/to/train_prefix.bed \
    --partition partitioned_idx.npz \
    --output-name input_features_means.npz

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
    --seeds 42,43,44 \
    --folds 0,1,2,3,4 \
    --output-dir /path/to/my_packages   # optional override
```

Packages land in `<exp-path>/<exp-name>_packages/seed_*/fold_*/` by default and are ready for `dietnet predict`.
`--seeds` and `--folds` are Click “multiple” options: repeat the flag (`--seeds 42 --seeds 43`) or provide comma-separated values in one flag (`--seeds 42,43`). The same applies to `--folds`.

## Evaluate Accuracy

```bash
# Check accuracy on all predicted samples
dietnet check \
    --predictions predictions.tsv \
    --labels /path/to/labels.tsv

# Check accuracy on test-fold only (avoids train/val inflation)
dietnet check \
    --predictions predictions.tsv \
    --labels /path/to/labels.tsv \
    --partition-file /path/to/partitioned_idx.npz \
    --fold 0

# Assert accuracy is within expected range (exits non-zero if violated)
dietnet check \
    --predictions predictions.tsv \
    --labels /path/to/labels.tsv \
    --min-accuracy 0.85
```

## Using Pre-trained Model

### Preset models

Specify an existing model to use (will download automatically if not present in `~/.cache/dietnet/`).

```bash
# Using a preset model (downloads automatically)
dietnet predict --model 1kgp_default \
                --plink-prefix /path/to/test_data \
                --output predictions.tsv \
                --temp-dir /path/to/test_data
```
`--temp-dir` controls where preprocessed PLINK files **and** the genotype cache are written (default: alongside `--plink-prefix`, e.g., `/path/to/test_data`).

**Available presets:**
- `1kgp_default`: 1000 Genomes Phase 3 (24 populations, single model)
- `hgdp_ukbb`: HGDP+1KGP for UKBB inference (coming soon)

### Your own packaged models
Point `--model` to the directory that contains `seed_*` folders (the parent of the packages):

```bash
# Using a local model package
dietnet predict --model /path/to/pretrained/model \
                --plink-prefix /path/to/test_data \
                --output predictions.tsv \
                --temp-dir /path/to/test_data
```

As with training, adding `--seeds` and `--folds` specifies which models to train on.

### Get help

```bash
# General help
dietnet --help

# Command-specific help
dietnet train --help
dietnet predict --help
```

## FAQ

### How do I run this on my data?

We provided 2 scripts that you can use as a template for your research:
- `scripts/predict_external.sh` for inference on your dataset
- `scripts/train_predict.sh` to train on a reference dataset and do inference on another

### I can't download the data or model since my compute node has no access to the internet

You can run this prior to running the smoke tests:
```bash
bash tests/kgp_precomputed/download_test_data.sh
```

Likewise the models can be downloaded using the following:
```bash
bash tests/kgp_precomputed/download_model.sh
```
