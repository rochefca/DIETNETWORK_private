import argparse
import os

import numpy as np

import h5py

import torch

import helpers.dataset_utils as du


def get_preprocessing_params():
    args = parse_args()

    # Load data
    data = h5py.File(os.path.join(args.exp_path,args.dataset), 'r')
    folds_indexes = du.load_folds_indexes(
            os.path.join(args.exp_path,args.folds_indexes)
            )

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

        print('loaded x_train:')
        print(x_train)
        print(x_train.dtype)

        # Order data to fit train_indexes in folds_indexes file
        positions = []
        for i in fold_indexes[0]:
            positions.append(np.where(train_indexes == i)[0])

        x_train_ordered = x_train[positions]

        print('x train ordered:')
        print(x_train_ordered)
        print(x_train_ordered.dtype)

        # Convert to tensor
        x_train_ordered = torch.from_numpy(x_train_ordered)

        print('X_train converted to tensor:')
        print(x_train_ordered)
        print(x_train_ordered.dtype)

        # Set GPU
        print('Cuda available:', torch.cuda.is_available())
        print('Current cuda device ', torch.cuda.current_device())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device:', device)

        x_train_ordered = x_train_ordered.to(device)
        print('On device:')
        print(x_train_ordered)
        print(x_train_ordered.dtype)

        x_train_ordered = x_train_ordered.float()
        print('converted to float:')
        print(x_train_ordered)
        print(x_train_ordered.dtype)

        # Compute features means and sd
        means, sd = du.compute_norm_values(x_train_ordered)
        print('means:')
        print(means)
        print(means.dtype)
        print('sd:')
        print(sd)
        print(sd.dtype)

        means_by_fold[fold] = means
        sd_by_fold[fold] = sd

    data.close()

    # Save
    print('Saving preprocessing params to', os.path.join(args.exp_path,args.out))
    np.savez(os.path.join(args.exp_path,args.out),
             means_by_fold=means_by_fold,
             sd_by_fold=sd_by_fold)


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
            '--folds-indexes',
            type=str,
            default='folds_indexes.npz',
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
