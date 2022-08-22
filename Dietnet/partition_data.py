import argparse
import os
import math
from datetime import datetime

import numpy as np

import h5py


"""
The partitions contains indices of train. valid and test sets
for each fold.
If folds test sets with equal nb of samples is not possible:
test set of last fold will have more samples
The number of extra samples will always be < nb_folds
"""
def main():
    args = parse_args()

    # Nb of folds to create
    nb_folds = args.nb_folds
    # Train/valid ratio
    train_valid_ratio = args.train_valid_ratio

    # Get samples
    dataset_file = os.path.join(args.exp_path, args.dataset)
    f = h5py.File(dataset_file, 'r')
    samples = np.array(f['samples'])
    indices = np.arange(len(samples))
    f.close()

    print('\n---\nPartitioning indices of {} samples into {} folds'.format(
          len(indices), nb_folds))

    # Shuffle data
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    indices = list(indices)

    # Step and split positions for test sets in each fold
    step = math.floor(len(indices)/nb_folds)
    split_pos = [i for i in range(0, len(indices), step)]

    test_indices_byfold = []
    valid_indices_byfold = []
    train_indices_byfold = []
    test_start = split_pos[0] # this is start=0

    # Get indices by set
    for i in range(nb_folds-1):
        # Test set
        test_indices = indices[test_start:(test_start+step)]
        test_indices_byfold.append(test_indices)

        # Nb of samples in train and valid sets
        nb_train_and_valid = len(indices) - len(test_indices)
        nb_train = math.floor(nb_train_and_valid*train_valid_ratio)
        nb_valid = nb_train_and_valid - nb_train

        # Valid set
        valid_start = (test_start + len(test_indices)) % len(indices)
        valid_end = (valid_start + nb_valid) % len(indices)

        if (valid_end > valid_start):
            valid_indices = indices[valid_start:valid_end]
        else:
            valid_indices = indices[valid_start:] + indices[:valid_end]

        valid_indices_byfold.append(valid_indices)

        # Train set
        train_start = valid_end
        train_end = (valid_end + nb_train) % len(indices)

        if (train_end > train_start):
            train_indices = indices[train_start:train_end]
        else:
            train_indices = indices[train_start:] + indices[:train_end]

        train_indices_byfold.append(train_indices)

        # Update test start for next loop iteration
        test_start = split_pos[i+1]

    # Get indices : Last fold
    test_indices = indices[test_start:]
    test_indices_byfold.append(test_indices) # append last fold

    # Nb of samples in train and valid sets
    nb_train_and_valid = len(indices) - len(test_indices)
    nb_train = math.floor(nb_train_and_valid*train_valid_ratio)
    nb_valid = nb_train_and_valid - nb_train

    # Valid set
    valid_start = 0
    valid_end = (valid_start + nb_valid) % len(indices)

    if (valid_end > valid_start):
        valid_indices = indices[valid_start:valid_end]
    else:
        valid_indices = indices[valid_start:] + indices[:valid_end]

    valid_indices_byfold.append(valid_indices)

    # Train set
    train_start = valid_end
    train_end = (valid_end + nb_train) % len(indices)

    if (train_end > train_start):
        train_indices = indices[train_start:train_end]
    else:
        train_indices = indices[train_start:] + indices[:train_end]

    train_indices_byfold.append(train_indices)

    # Grouping train, valid and test indices by fold
    indices_byfold = []
    for train_indices, valid_indices, test_indices in zip(
            train_indices_byfold, valid_indices_byfold, test_indices_byfold):
        indices_byfold.append([train_indices, valid_indices, test_indices])

    # Get samples by fold
    samples_byfold = []
    for fold in range(nb_folds):
        train_samples = samples[indices_byfold[fold][0]]
        valid_samples = samples[indices_byfold[fold][1]]
        test_samples = samples[indices_byfold[fold][2]]

        samples_byfold.append([train_samples, valid_samples, test_samples])

    print('Created {} folds with the following number of samples'.format(
          nb_folds))
    for i,fold_indices in enumerate(indices_byfold):
        print('FOLD:',i)
        print('Train: {} samples'.format(len(fold_indices[0])))
        print('Valid: {} samples'.format(len(fold_indices[1])))
        print('Test: {} samples'.format(len(fold_indices[2])))
        print('Fold number of samples:',
              len(fold_indices[0])+len(fold_indices[1])+len(fold_indices[2]))
        print('***')

    # Saving results
    if args.out is not None:
        partition_filename = args.out
    else:
        partition_filename = 'partition_' \
                + args.dataset.split('/')[-1][0:-5] \
                + '_' \
                + datetime.now().strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    partition_fullpath = os.path.join(args.exp_path, partition_filename)

    # dtype is object because in a fold, train/valid/test lists of
    # indices don't have the same length
    np.savez(partition_fullpath,
             folds_indexes=np.array(indices_byfold, dtype=object),
             folds_samples=np.array(samples_byfold, dtype=object))

    print('Dataset partition was saved to {}'.format(partition_fullpath+'.npz'))
    print('---\n')


def parse_args():
    parser = argparse.ArgumentParser(
            description=('Partition data into folds. This script creates an array '
                         'containing samples\' indexes of every partition')
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory where partition will be written'
            )

    parser.add_argument(
            '--dataset',
            type=str,
            required=True,
            help=('Hdf5 dataset created with create_dataset.py '
                  'Provide full path')
            )

    parser.add_argument(
            '--seed',
            type=int,
            default=23,
            help=('Seed for fixing random shuffle of samples before '
                  'partitioning. Default:  %(default)i')
            )

    parser.add_argument(
            '--nb-folds',
            type=int,
            default=5,
            help='Number of folds. Use 1 for no folds. Default: %(default)i',
            )

    parser.add_argument(
            '--train-valid-ratio',
            type=float,
            default=0.75,
            help=('Ratio (between 0-1) for split of train and valid sets. '
                  'For example, 0.75 will use 75%% of data for training '
                  'and 25%% of data for validation. Default: %(default).2f')
            )

    parser.add_argument(
            '--out',
            help=('Optional filename for dataset partition. '
                  'If not provided the file will be named '
                  'partition_datasetFilename_date')
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
