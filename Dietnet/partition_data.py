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

    # Get indices of test set samples for each fold
    step = math.floor(len(indices)/nb_folds)
    split_pos = [i for i in range(0, len(indices), step)]

    test_indices_byfold = []
    start = split_pos[0] # this is start=0
    for i in range(nb_folds-1):
        test_indices_byfold.append(indices[start:(start+step)])
        start = split_pos[i+1]

    test_indices_byfold.append(indices[start:]) # append last fold

    print('Partitioned data into {} folds of length {}'.format(
          nb_folds, [len(i) for i in test_indices_byfold]))

    # Get indices of train+valid sets samples for each fold
    train_indices_byfold = []
    valid_indices_byfold = []
    for i in range(nb_folds):
        other_folds = [f for f in range(nb_folds) if f!=i]
        # Concat test indices of other folds: this is train+valid indices
        train_valid_indices = np.concatenate(
                [test_indices_byfold[f] for f in other_folds])
        # Split into train and valid sets
        split_pos = int(len(train_valid_indices)*train_valid_ratio)
        train_indices = train_valid_indices[0:split_pos]
        valid_indices = train_valid_indices[split_pos:]
        # Append train and valid indices
        train_indices_byfold.append(train_indices)
        valid_indices_byfold.append(valid_indices)
    print('Each fold was split into train/valid sets with ratio {} '.format(
          train_valid_ratio))

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
