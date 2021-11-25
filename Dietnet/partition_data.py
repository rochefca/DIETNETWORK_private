import argparse
import os

import numpy as np

import h5py

import helpers.dataset_utils as du


def partition_data():
    args = parse_args()

    # Load samples
    dataset_file = os.path.join(args.exp_path, args.dataset)
    f = h5py.File(dataset_file, 'r')
    indices = np.arange(len(f['samples']))
    f.close()

    print('Partitioning indices of', len(indices), 'samples')

    # Partition
    #indices = np.arange(len(samples))
    partition = du.partition(indices, args.nb_folds,
                             args.train_valid_ratio, args.seed)

    print('Saving partition to', os.path.join(args.exp_path,args.out))
    np.savez(os.path.join(args.exp_path,args.out),
             folds_indexes=np.array(partition,dtype=object),
             seed=np.array([args.seed]))


def parse_args():
    parser = argparse.ArgumentParser(
            description=('Partition data into folds. This script creates an array '
                         'containing samples\' indexes of every partition')
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory where dataset is saved'
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.hdf5',
            help=('Filename of dataset created with create_dataset.py '
                  'Default %(default)s')
            )

    parser.add_argument(
            '--seed',
            type=int,
            default=23,
            help=('Seed for fixing random shuffle of samples before '
                  'partitioning samples. Default:  %(default)i')
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
            default='partitioned_idx.npz',
            help=('Filename for returned samples indexes of each fold. '
                  'Default: %(default)s')
            )

    return parser.parse_args()


if __name__ == '__main__':
    partition_data()
