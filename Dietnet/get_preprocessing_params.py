import argparse
import os
import time

import numpy as np

import h5py

import torch

import helpers.dataset_utils as du


def get_preprocessing_params():
    start_time = time.time()
    args = parse_args()

    # Load data
    data = h5py.File(os.path.join(args.exp_path,args.dataset))
    folds_indexes = du.load_folds_indexes(
            os.path.join(args.exp_path,args.partition)
            )

    means_by_fold = []
    sd_by_fold = []
    for fold in range(len(folds_indexes)):
        print('Computing preprocessing parameters of fold', str(fold))
        # Get fold inputs (x)
        fold_indexes = folds_indexes[fold]
        train_indexes = np.sort(fold_indexes[0]) # sort is a hdf5 requirement
        x_train = data['inputs'][train_indexes]

        x_train = torch.from_numpy(x_train)

        # Compute features means and sd
        means, sd = du.compute_norm_values(x_train)

        means_by_fold.append(means.numpy())
        sd_by_fold.append(sd.numpy())


    data.close()
    """
    # nb of folds and nb of snps
    nb_folds = len(folds_indexes)
    nb_snps = len(data['snp_names'])

    means_by_fold = torch.empty(size=(nb_folds,nb_snps))
    sd_by_fold = torch.empty(size=(nb_folds,nb_snps))

    for fold in range(len(folds_indexes)):
        print('Computing preprocessing parameters of fold', str(fold))
        # Get fold inputs (x)
        fold_indexes = folds_indexes[fold]
        train_indexes = np.sort(fold_indexes[0]) # sort is a hdf5 requirement
        x_train = data['inputs'][train_indexes]

        # Compute means and sd
        means, sd = du.compute_norm_values(x_train)

        means_by_fold[fold] = means
        sd_by_fold[fold] = sd
    """

    # Save
    print('Saving preprocessing params to', os.path.join(args.exp_path,args.out))
    np.savez(os.path.join(args.exp_path,args.out),
             means_by_fold=means_by_fold,
             sd_by_fold=sd_by_fold)

    end_time=time.time()
    print('End of execution. Execution time:', end_time-start_time, 'seconds')


def parse_args():
    parser = argparse.ArgumentParser(
            description=('Compute features means and standard deviations '
                         'for input normalization at training time')
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to experiment directory where to save embedding. '
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.hdf5',
            help=('Filename of dataset returned by create_dataset.py '
                  'The file must be in directory specidifed with exp-path. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--partition',
            type=str,
            default='partitioned_idx.npz',
            help=('Filename of folds indexes returned by partition_data.py '
                  'The file must be in directory specified with exp-path. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--out',
            type=str,
            default='preprocessing_params.npz',
            help='Output filename. Default: %(default)s'
            )

    return parser.parse_args()


if __name__ == '__main__':
    get_preprocessing_params()
