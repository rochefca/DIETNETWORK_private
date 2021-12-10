import math

import numpy as np

import h5py

import torch


class FoldDataset(torch.utils.data.Dataset):
    # These variables are set in train.py
    dataset_file = None #path to h5py file
    label_type = None # Int if classification, float if regression

    def __init__(self, set_indexes):
        self.set_indexes = set_indexes

    def __len__(self):
        return len(self.set_indexes)

    def __getitem__(self, index):
        # Data of all sets (train, valid, test) is in one file
        # so we convert the index to match file index
        file_index = self.set_indexes[index]
        """
        with h5py.File(FoldDataset.dataset_file, 'r') as f:
            x = np.array(f['inputs'][file_index], dtype=np.int8)
            y = (f['labels'][file_index]).astype(np.int)
            sample = (f['samples'][file_index]).astype(np.str_)
        """
        x = np.array(self.f['inputs'][file_index], dtype=np.int8)
        y = (self.f['labels'][file_index]).astype(self.label_type)
        sample = (self.f['samples'][file_index]).astype(np.str_)
        return x, y, sample
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_file, 'r')

        return np.array(self.dataset['inputs'][file_index], dtype=np.int8), \
               np.array(self.dataset['labels'][file_index]), \
               np.array(self.dataset['samples'][file_index], dtype=np.int)
        """

    def get_samples(self):
        indexes = np.sort(self.set_indexes)
        samples = (self.f['samples'][indexes]).astype(np.str_)

        return samples


class ExternalTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file):
        self.dataset = h5py.File(dataset_file, 'r')

    def __len__(self):
        return len(self.dataset['samples'])

    def __getitem__(self, index):
        x = np.array(self.dataset['inputs'][index], dtype=np.int8)
        sample = (self.dataset['samples'][index]).astype(np.str_)

        return x, sample


def shuffle(indices, seed=None):
    # Fix seed so shuffle is always the same
    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(indices)


def partition(indices, nb_folds, train_valid_ratio, seed=None):
    """
    The partitions contains indices of train. valid and test sets
    for each fold.
    If folds test sets with equal nb of samples is not possible:
    test set of last fold will have more samples
    The number of extra samples will always be < nb_folds
    """
    # Shuffle data
    if seed is not None:
        np.random.seed(seed)
    shuffle(indices, seed=seed)

    # Get indices of examples in test set for each fold
    step = math.floor(len(indices)/nb_folds)
    split_pos = [i for i in range(0, len(indices), step)]

    test_indices_byfold = []
    start = split_pos[0] # same as start=0
    for i in range(nb_folds-1):
        test_indices_byfold.append(indices[start:(start+step)])
        start = split_pos[i+1]

    test_indices_byfold.append(indices[start:]) # append last fold

    # Get indices of train+valid sets for each fold
    train_indices_byfold = []
    valid_indices_byfold = []
    for i in range(nb_folds):
        other_folds = [f for f in range(nb_folds) if f!=i]
        # Concat test indices of other folds: this is train+valid indices
        train_valid_indices = np.concatenate(
                [test_indices_byfold[f] for f in other_folds]
                )
        # Split into train and valid sets
        train_indices, valid_indices = split(train_valid_indices,
                train_valid_ratio, seed)
        train_indices_byfold.append(train_indices)
        valid_indices_byfold.append(valid_indices)

    # Train, valid and test indices of examples for each fold
    indices_byfold = []
    for train_indices, valid_indices, test_indices in zip(
            train_indices_byfold, valid_indices_byfold, test_indices_byfold):
        indices_byfold.append([train_indices, valid_indices, test_indices])

    return indices_byfold


def split(indices, split_ratio, seed):
    # Fix seed so shuffle is always the same
    if seed is not None:
        np.random.seed(seed)

    # Shuffle so that validation set is different between folds
    #np.random.shuffle(indices)

    split_pos = int(len(indices)*split_ratio)

    train_indexes = indices[0:split_pos]
    valid_indexes = indices[split_pos:]

    return train_indexes, valid_indexes


def load_genotypes(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # SNP ids
    snps = np.array([i.strip() for i in lines[0].split('\t')[1:]])

    # Sample ids
    samples = np.array([i.split('\t')[0] for i in lines[1:]])

    # Genotypes
    genotypes = np.empty((len(samples), len(snps)), dtype="int8")
    for i,line in enumerate(lines[1:]):
        for j,genotype in enumerate(line.split('\t')[1:]):
            if genotype.strip() == './.' or genotype.strip() == 'NA':
                genotype = -1
            else:
                genotype = int(genotype.strip())
            genotypes[i,j] = genotype

        # Log number of parsed samples
        if i % 100 == 0 and i != 0:
            print('Loaded', i, 'out of', len(samples), 'samples')

    print('Loaded', str(genotypes.shape[1]), 'genotypes of', str(genotypes.shape[0]), 'samples')

    return samples, snps, genotypes


def load_genotypes_parallel(line):
    # Line : Sample id and genotype values across all SNPs
    sample = (line.split('\t')[0]).strip()

    # Fill with genotypes of all SNPs for the individual
    genotypes = []
    for i in line.split('\t')[1:]:
        # Replace missing values with -1
        if i.strip() == './.' or i.strip() == 'NA':
            genotype = -1
        else:
            genotype = int(i.strip())

        genotypes.append(genotype)

    genotypes = np.array(genotypes, dtype='int8')

    return sample, genotypes


def load_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    mat = np.array([l.strip('\n').split('\t') for l in lines])

    samples = mat[1:,0]
    labels = mat[1:,1]

    print('Loaded', str(len(labels)),'labels of', str(len(samples)),'samples')

    return samples, labels


def order_labels(samples, samples_in_labels, labels):
    idx = [np.where(samples_in_labels == s)[0][0] for s in samples]

    return np.array([labels[i] for i in idx])


def load_data(filename):
    data = np.load(filename)

    return data

# Not sure if this will be used
def load_data_(filename):
    data = np.load(filename)

    return data['inputs'], data['labels'], data['samples'],\
           data['label_names'], data['snp_names']


def load_folds_indexes(filename):
    data = np.load(filename, allow_pickle=True)

    return data['folds_indexes']


def load_embedding(filename, which_fold):
    data = np.load(filename)
    embs = data['emb']
    emb = torch.from_numpy(embs[which_fold])

    return emb


def get_fold_data(which_fold, folds_indexes, data, label='labels'):
    # Indices of each set for the fold (0:train, 1:valid, 2:test)
    fold_indexes = folds_indexes[which_fold]
    train_indexes = np.sort(fold_indexes[0]) # sort is a hdf5 requirement
    valid_indexes = np.sort(fold_indexes[1])
    test_indexes = np.sort(fold_indexes[2])

    # Get data (x,y,samples) of each set (train, valid, test)
    x_train = data['inputs'][train_indexes]
    y_train = data[label][train_indexes]
    samples_train = data['samples'][train_indexes]

    x_valid = data['inputs'][valid_indexes]
    y_valid = data[label][valid_indexes]
    samples_valid = data['samples'][valid_indexes]

    x_test = data['inputs'][test_indexes]
    y_test = data[label][test_indexes]
    samples_test = data['samples'][test_indexes]

    return train_indexes, valid_indexes, test_indexes,\
           x_train, y_train, samples_train,\
           x_valid, y_valid, samples_valid,\
           x_test, y_test, samples_test


# !!This is the old function that has to be removed eventually!!
def _get_fold_data(which_fold, folds_indexes, data, split_ratio=None, seed=None):
    # Set aside fold nb of which_fold for test
    test_indexes = folds_indexes[which_fold]

    # Other folds are used for train and valid sets
    other_folds = [i for i in range(len(folds_indexes)) if i!=which_fold]

    # Concat indices of other folds
    other_indexes = np.concatenate([folds_indexes[i] for i in other_folds])

    # If we are generating embeddings, we don't need train/valid sets
    if split_ratio is None:
        x = data['inputs'][other_indexes]
        y = data['labels'][other_indexes]
        samples = data['samples'][other_indexes]

        return x, y, samples

    # Split indexes into train and valid set
    train_indexes, valid_indexes = split(other_indexes, split_ratio, seed)

    # Get data (x,y,samples) of each set (train, valid, test)
    x_train = torch.from_numpy(data['inputs'][train_indexes])
    y_train = torch.from_numpy(data['labels'][train_indexes])
    samples_train = data['samples'][train_indexes]

    x_valid = torch.from_numpy(data['inputs'][valid_indexes])
    y_valid = torch.from_numpy(data['labels'][valid_indexes])
    samples_valid = data['samples'][valid_indexes]

    x_test = torch.from_numpy(data['inputs'][test_indexes])
    y_test = torch.from_numpy(data['labels'][test_indexes])
    samples_test = data['samples'][test_indexes]

    return train_indexes, valid_indexes, test_indexes,\
           x_train, y_train, samples_train,\
           x_valid, y_valid, samples_valid,\
           x_test, y_test, samples_test


def compute_norm_values(x):
    """
    x is a tensor
    """
    # Non missing values
    mask = (x >= 0)

    # Compute mean of every column (feature)
    per_feature_mean = torch.sum(x*mask, dim=0) / torch.sum(mask, dim=0)

    print('Computed per feature mean')

    # S.d. of every column (feature)
    per_feature_sd = torch.sqrt(
            torch.sum((x*mask-mask*per_feature_mean)**2, dim=0) / \
                    (torch.sum(mask, dim=0) - 1)
                    )
    per_feature_sd += 1e-6

    print('Computed per feature sd')

    return per_feature_mean, per_feature_sd


def replace_missing_values(x, per_feature_mean):
    """
    x and per_feature_mean are tensors
    """
    mask = (x >= 0)

    for i in range(x.shape[0]):
        x[i] =  mask[i]*x[i] + (~mask[i])*per_feature_mean


def normalize(x, per_feature_mean, per_feature_sd):
    """
    x, per_feature_mean and per_feature_sd are tensors
    """
    x_norm = (x - per_feature_mean) / per_feature_sd

    return x_norm
