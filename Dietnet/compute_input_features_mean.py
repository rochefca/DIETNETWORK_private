import argparse
import os
import multiprocessing as mp
import time
from pathlib import PurePath

import numpy as np

import h5py

import torch

from Dietnet.helpers import dataset_utils as du


def _resolve_path(base_dir: str, path: str) -> str:
    """Return absolute path, respecting already-absolute inputs."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def get_preprocessing_params(args=None):
    """
    Compute per-fold feature statistics for normalization.

    Args:
        args: argparse.Namespace-like object with exp_path, dataset, partition,
              parallel_loading, ncpus, and out attributes. If None, CLI args
              are parsed.
    """
    if args is None:
        args = parse_args()

    start_time = time.time()

    # Load partition indexes
    partition_path = _resolve_path(args.exp_path, args.partition)
    folds_indexes = du.load_folds_indexes(partition_path)

    # Detect dataset type
    dataset_path = _resolve_path(args.exp_path, args.dataset)

    if dataset_path.endswith('.hdf5') or dataset_path.endswith('.h5'):
        # HDF5 mode
        print('Using HDF5 dataset')
        data = h5py.File(dataset_path)

        means_by_fold = []
        for fold in range(len(folds_indexes)):
            print('Computing preprocessing parameters of fold', str(fold))
            # Get fold inputs (x)
            print('Loading fold {} indexes'.format(fold))
            fold_indexes = folds_indexes[fold]

            # Genotypes of training samples
            print('Getting genotypes of train samples')
            train_indexes = np.sort(fold_indexes[0]) # sort is a hdf5 requirement
            x_train = data['inputs'][train_indexes]

            # Convert to torch tensor
            x_train = torch.from_numpy(x_train.astype(np.float32))

            if args.parallel_loading:
                # Compute mean of every features in parallel
                ncpus = args.ncpus
                print('Computing input features means in parallel using '
                      '{} processes'.format(ncpus))
                pool = mp.Pool(processes=ncpus) #this starts ncpus nb of processes

                # Results that will be returned by processes
                results = []

                # The mean of every snp feature is computed by a different process
                nb_snps = x_train.shape[1]
                for i in range(nb_snps):
                    results.append(
                            pool.apply_async(
                                compute_feature_mean, args=(i, x_train[:,i])))

                # Get order (snp nb) and mean of features from multiprocess results
                print('Computed mean of every feature, getting the results')
                results_values = np.array([np.array(p.get()) for p in results])

                print(results_values)
                print(results_values.shape)

                # Order means based on snp nb
                print('Ordering means returned by multiprocessing')
                idx = results_values[:,0].astype(int)
                ordered_idx = idx.argsort()

                # Ordered means
                means = results_values[:,1][ordered_idx]

                print(means)


            else:
                # Compute features means and sd
                means = du.compute_norm_values(x_train)

            means_by_fold.append(means)

        data.close()

    else:
        # PLINK mode - load genotypes and compute stats
        print('Using PLINK dataset')
        from Dietnet.helpers.dataset_utils import load_plink_genotypes
        from tqdm import tqdm

        plink_prefix = dataset_path.replace('.bed', '')
        cache_file = PurePath(args.exp_path, PurePath(dataset_path).name.replace('.bed', '_genotypes.npy'))

        # Load all genotypes
        genotypes, fam_data, bim_data = load_plink_genotypes(plink_prefix, cache_file)

        # Compute per-fold means and stds
        means_by_fold = []
        nb_folds = len(folds_indexes)
        for fold in tqdm(range(nb_folds), desc='Computing fold statistics', unit='fold'):
            train_indices = folds_indexes[fold][0]
            train_genotypes = genotypes[train_indices]  # Subset to train samples

            # Convert to torch tensor
            x_train = torch.from_numpy(train_genotypes.astype(np.float32))

            # Compute features means and sd
            means = du.compute_norm_values(x_train)

            means_by_fold.append(means)

    # Save
    out_path = _resolve_path(args.exp_path, args.out)
    print('Saving input features mean to', out_path)
    np.savez(out_path, means_by_fold=means_by_fold)

    end_time=time.time()
    print('End of execution. Execution time:', end_time-start_time, 'seconds')


def compute_all_features_mean():
    pass

def compute_feature_mean(i, x):
    """
    i is the snp nb (from 0 to nb_snps-1)
    x is the vector of genotypes for the snp (dim : nb_samples x 1)
    """
    # Set to false missing value (missing values are encoded with -1)
    mask = (x >= 0)

    # Compute mean :
    # in numerator, missing values are 0
    # in the denominator we divise by the nb of non missing values
    mean = np.sum(x*mask) / np.sum(mask)

    return (i,mean)


def parse_args():
    parser = argparse.ArgumentParser(
            description=('Compute features means and standard deviations '
                         'for input normalization at training time')
            )

    # Path
    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to experiment directory where to save embedding. '
            )

    # Files
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

    # Parallel computation
    parser.add_argument(
            '--parallel-loading',
            action='store_true',
            help='Use this flag to load samples in parallel.'
            )

    parser.add_argument(
            '--ncpus',
            type=int,
            help='Number of cpus for parallel loading'
            )

    # Output
    parser.add_argument(
            '--out',
            type=str,
            default='input_features_means.npz',
            help='Output filename. Default: %(default)s'
            )

    return parser.parse_args()


if __name__ == '__main__':
    get_preprocessing_params()
