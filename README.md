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
