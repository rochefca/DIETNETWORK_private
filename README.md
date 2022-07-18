# DIETNETWORK

Pytorch implementation of DietNetwork (https://arxiv.org/abs/1611.09340)

## Files
- **Genotype**: File of samples and their genotypes (genotypes = {0,1,2}, additive encoding) in tab-separated format. One sample per line, first column are sample ids and other columns are samples genotypes for every SNPs. Missing genotypes can be encoded with *NA*, *./.* or *-1*.
- **Labels**: Samples and their labels in tab-separated format. One sample per line, first column are sample ids followed by their label.
    - **Class labels**: used to compute the embedding (in classification and regression tasks) and used as prediction in classification task
    - **Regression labels**: used as prediction in regression task 
- **Config**

## Usage

### Create Dataset
Create a hdf5 dataset from genotype and label files.
```
python create_dataset --help

usage: create_dataset.py [-h] --exp-path EXP_PATH --genotypes GENOTYPES
                         --class-labels CLASS_LABELS
                         [--regression-labels REGRESSION_LABELS]
                         [--ncpus NCPUS] [--out OUT]

Create hdf5 dataset from genotype and label files.

optional arguments:
  -h, --help            show this help message and exit
  --exp-path EXP_PATH   Path to directory where dataset will be written
  --genotypes GENOTYPES
                        File of samples and their genotypes (tab-separated
                        format, one sample per line). Missing genotypes can be
                        encoded with NA, .\. or -1. Provide full path
  --class-labels CLASS_LABELS
                        File of samples and their class labels (tab-separated
                        format, one sample per line). Provide full path
  --regression-labels REGRESSION_LABELS
                        File of samples and their regression labels (tab-
                        separated format, one sample per line). Provide full
                        path
  --ncpus NCPUS         Number of cpus available to parse genotypes in
                        parallel. Default: 1
  --out OUT             Optional filename for the returned dataset. If not
                        provided the file will be named dataset_date.hdf5
```

#### Create dataset for classification task
```
python create_dataset.py \
    --exp-path <experiment_dir_path> \
    --genotypes <genotype_file> \
    --class-labels <class_labels_file>
```

#### Create dataset for regression task
```
python create_dataset.py \
    --exp-path <experiment_dir_path> \
    --genotypes <genotype_file> \
    --class-labels <class_labels_file> \
    --regression-labels <regression_labels_file>
```

### Partition data
Partition samples into folds and split each fold into train and validation sets. This is done before training, because embeddings are computed on each fold.

```
python partition_data.py --help

usage: partition_data.py [-h] --exp-path EXP_PATH --dataset DATASET
                         [--seed SEED] [--nb-folds NB_FOLDS]
                         [--train-valid-ratio TRAIN_VALID_RATIO] [--out OUT]

Partition data into folds. This script creates an array containing samples'
indexes of every partition

optional arguments:
  -h, --help            show this help message and exit
  --exp-path EXP_PATH   Path to directory where partition will be written
  --dataset DATASET     Hdf5 dataset created with create_dataset.py Provide
                        full path
  --seed SEED           Seed for fixing random shuffle of samples before
                        partitioning. Default: 23
  --nb-folds NB_FOLDS   Number of folds. Use 1 for no folds. Default: 5
  --train-valid-ratio TRAIN_VALID_RATIO
                        Ratio (between 0-1) for split of train and valid sets.
                        For example, 0.75 will use 75% of data for training
                        and 25% of data for validation. Default: 0.75
  --out OUT             Optional filename for dataset partition. If not
                        provided the file will be named
                        partition_datasetFilename_date
```

### Compute input features statistics
Computes mean and standard deviation of every input feature (SNP). The means and standard deviations are computed by fold, on samples of the training set. The mean is used to replace missing values (missing genotypes) and mean+standard deviation are used to normalize input features at training time.

```
python compute_input_features_statistics.py --help

usage: compute_input_features_statistics.py [-h] --exp-path EXP_PATH --dataset
                                            DATASET --partition PARTITION
                                            [--mean-only] [--ncpus NCPUS]
                                            [--out OUT]

Compute features means and standard deviations for missing values filing and
input normalization at training time

optional arguments:
  -h, --help            show this help message and exit
  --exp-path EXP_PATH   Path to directory where input features statistics will
                        be written
  --dataset DATASET     Hdf5 dataset created with create_dataset.py. Provide
                        full path
  --partition PARTITION
                        Npz dataset partition returned by partition_data.py
                        Provide full path
  --mean-only           Use this flag to compute only input features means and
                        not the standard deviations
  --ncpus NCPUS         Number of cpus for parallel computation of means and
                        of standard deviations. Default: 1
  --out OUT             Optional filename for input features statistics file.
                        If not provided the file will be named
                        input_features_stats_datasetFilename_date
```

### Generate embedding
Compute classes genotype frequencies embedding by using samples of the training set.

```
python generate_embedding.py --help

usage: generate_embedding.py [-h] --exp-path EXP_PATH --dataset DATASET
                             --partition PARTITION [--ncpus NCPUS]
                             [--include-valid] [--only-valid] [--out OUT]

Generate embedding

optional arguments:
  -h, --help            show this help message and exit
  --exp-path EXP_PATH   Path to directory where embedding will be saved.
  --dataset DATASET     Hdf5 dataset created with create_dataset.py Provide
                        full path
  --partition PARTITION
                        Npz dataset partition returned by partition_data.py
                        Provide full path
  --ncpus NCPUS         Number of cpus available to compute folds embedding in
                        parallel. Default:1
  --include-valid       Use this flag to include samples from the validation
                        set in the embedding computation.
  --only-valid          Use this flag to compute embedding only on samples of
                        the validation set
  --out OUT             Optional filename for embedding file. If not provided
                        the file will be named
                        genotype_class_freq_embedding_datasetFilename_date
```

## OLD README
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

## Packages
- Python 3.6
- torch 1.5.0+cu101
- numpy 1.19.1
- pandas 1.1.0
- matplotlib 3.2.1
- captum 0.2.0 (https://captum.ai/)
- h5py 2.10.0
