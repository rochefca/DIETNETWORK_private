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

## Quick Start

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

## Training pipeline

![code_wf](Images/dn_workflow.png)
## Scripts
### Main scripts
1. **create_dataset.py** : Create dataset and partition data into folds. The script takes snps.txt and labels.txt files as input to create dataset.npz and folds_indexes.npz
1. **generate_embedding.py** : Takes dataset.npz and folds_indexes.npz files created in the previous step and computes the embedding (genotypic frequency) of every fold. Embedding of each fold is saved in embedding.npz
    1. Missing values are -1 and are not included in the computation of genotypic frequencies embedding
    1. Embedding values are computed on train and valid sets
1. **train.py** : Whole training process. The data is divided in train/valid and test sets. Performance is reported on the test set.
    1. Data preprocessing of auxiliary net : Square Euclidean distance normalization
    1. Data preprocessing of discrim net: Missing values are replaced by the mean of the feature computed on training set. Data normalization (standardization) using mean and sd computed on training set.
1. **test_external_dataset.py** : Test model on an external set, ie on individuals that are not part of dataset.npz
1. **evaluate.py** : Utilities to visualize the model performance such as confusion matrix
  
### Helper scripts
- **dataset_utils.py** : Data related functions (shuffle, partition, split, get_fold_data, replace_missing_values, normalize, ...)
- **model.py** : Model definition of feature embedding (auxiliary) and discriminative (main) networks.
- **mainloop_utils.py** : Function used in the training loop (get_predictions, compute_accuracy, eval_step, ...)
- **log_utils.py** : Utilities to save data (model summary and parameters, experiment parameters, predictions, etc.)
- **test_utils.py** : Utilities related to testing a trained model on an external set

## Files
### Raw files provided by user
- **snps.txt** : File of genotypes in additive encoding format and tab-separated.
- **labels.txt** : File of samples and their label.
### Files created before training
- **dataset.npz** : Dataset created from the parsed snps.txt and labels.txt files.
- **folds_indexes.npz** : Array index (arrays are in dataset.npz) for each fold. The indexes are those of the data points to use as test.
- **embedding.npz** : Computed embeddings of each fold.
### Files returned after training
- **exp_params.log** : Experiment parameters (fixed seed, learning rate, number of epochs, etc.)
- **model_summary.log** : Model information (number of hidden layers, number of neurons in each layers, activation functions, etc.)
- **model_params.pt** : Model parameters of final trained model
- **model_predictions.npz**: Scores and predictions returned by the trained model for test samples
- **additional_data.npz** : Some more information used at training time (mus and sigmas values used for normalization, feature names, label names, training samples ids, validation samples ids, etc.) 

## To do
- [x] Embedding
- [x] Data preprocessing : Missing values
- [x] Data preprocessing : Data normalization
- [x] Dataset class (for dataloader)
- [x] Auxiliary and Main networks models
- [x] Training loop
- [x] Loss/Accuracy monitoring of train and valid
- [x] Early stopping
- [x] Test for in-sample data
- [ ] Test in-sample with missing values rates
- [x] Test for out-of-sample data
- [x] Save model params, results

## Requirements
- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- pandas >= 2.0.0
- h5py >= 3.8.0
- PyYAML >= 6.0
- click >= 8.1.0
- pyplink >= 1.3.0
- scikit-learn >= 1.3.0

### Optional
- captum >= 0.6.0 (for interpretability)
- matplotlib >= 3.7.0 (for visualization)
- comet-ml >= 3.33.0 (for experiment tracking)
