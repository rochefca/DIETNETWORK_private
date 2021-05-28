import argparse
import os

import numpy as np

import h5py

import torch

import helpers.dataset_utils as du


def get_preprocessing_params():
    args = parse_args()

    # Load data
    data = h5py.File(os.path.join(args.exp_path,args.dataset))
    folds_indexes = du.load_folds_indexes(
            os.path.join(args.exp_path,args.folds_indexes)
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

        break

    data.close()

    # Save
    print('Saving preprocessing params to', os.path.join(args.exp_path,args.out))
    np.savez(os.path.join(args.exp_path,args.out),
             means_by_fold=np.array(means_by_fold),
             sd_by_fold=np.array(sd_by_fold))




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
