import argparse
import os
import time
from pathlib import PurePath

import numpy as np
import h5py
import torch

from Dietnet.helpers import dataset_utils as du


def compute_stats_with_args(args):
    start_time = time.time()

    # Load partition indexes
    folds_indexes = du.load_folds_indexes(
            os.path.join(args.exp_path, args.partition)
            )

    # Detect dataset type
    if args.dataset.endswith('.hdf5') or args.dataset.endswith('.h5'):
        print('Using HDF5 dataset')
        data = h5py.File(os.path.join(args.exp_path, args.dataset))

        means_by_fold = []
        sd_by_fold = []
        for fold in range(len(folds_indexes)):
            print('Computing input features statistics of fold', fold)
            train_indexes = np.sort(folds_indexes[fold][0])  # sort is an h5py requirement
            x_train = torch.from_numpy(data['inputs'][train_indexes].astype(np.float32))
            mean, sd = du.compute_norm_values(x_train)
            means_by_fold.append(mean.numpy())
            sd_by_fold.append(sd.numpy())

        data.close()

    else:
        # PLINK mode
        print('Using PLINK dataset')
        from Dietnet.helpers.dataset_utils import load_plink_genotypes
        from tqdm import tqdm

        plink_prefix = os.path.join(args.exp_path, args.dataset).replace('.bed', '')
        cache_file = PurePath(args.exp_path, args.dataset.replace('.bed', '_genotypes.npy'))
        genotypes, fam_data, bim_data = load_plink_genotypes(plink_prefix, cache_file)

        means_by_fold = []
        sd_by_fold = []
        for fold in tqdm(range(len(folds_indexes)), desc='Computing fold statistics', unit='fold'):
            train_indices = folds_indexes[fold][0]
            x_train = torch.from_numpy(genotypes[train_indices].astype(np.float32))
            mean, sd = du.compute_norm_values(x_train)
            means_by_fold.append(mean.numpy())
            sd_by_fold.append(sd.numpy())

    out_path = os.path.join(args.exp_path, args.out)
    print('Saving input features stats to', out_path)
    np.savez(out_path, means_by_fold=means_by_fold, sd_by_fold=sd_by_fold)
    print('Execution time: {} seconds'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(
            description='Compute per-fold feature means and standard deviations '
                        'for missing value imputation and input normalization.'
            )

    parser.add_argument('--exp-path', type=str, required=True,
                        help='Path to experiment directory.')
    parser.add_argument('--dataset', type=str, default='dataset.hdf5',
                        help='Dataset filename (default: %(default)s).')
    parser.add_argument('--partition', type=str, default='partitioned_idx.npz',
                        help='Partition filename (default: %(default)s).')
    parser.add_argument('--out', type=str, default='input_features_means.npz',
                        help='Output filename (default: %(default)s).')

    return parser.parse_args()


def get_preprocessing_params():
    compute_stats_with_args(parse_args())


if __name__ == '__main__':
    get_preprocessing_params()
