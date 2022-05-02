import argparse
import os
import multiprocessing as mp
import time

import numpy as np

import h5py

import torch

import helpers.dataset_utils as du


def get_preprocessing_params():
    start_time = time.time()
    args = parse_args()

    mean_only = args.mean_only

    # Load data
    data = h5py.File(os.path.join(args.exp_path,args.dataset))
    folds_indexes = du.load_folds_indexes(
            os.path.join(args.exp_path,args.partition)
            )

    means_by_fold = []
    sds_by_fold = []
    for fold in range(len(folds_indexes)):
        print('Computing preprocessing parameters of fold', str(fold))
        # Get fold inputs (x)
        print('Loading fold {} indexes'.format(fold))
        fold_indexes = folds_indexes[fold]

        # Genotypes of training samples
        print('Getting genotypes of train samples')
        train_indexes = np.sort(fold_indexes[0]) # sort is a hdf5 requirement
        x_train = data['inputs'][train_indexes]

        #x_train = torch.from_numpy(x_train)

        if args.parallel_loading:
            # Compute statistics of every features in parallel
            ncpus = args.ncpus
            print('Computing input features statistics in parallel using '
                  '{} processes'.format(ncpus))
            pool = mp.Pool(processes=ncpus) #this starts ncpus nb of processes

            # Results that will be returned by processes
            results = []

            # The statistics of every snp feature is computed by a different process
            nb_snps = x_train.shape[1]
            for i in range(nb_snps):
                # Compute snps mean
                if mean_only :
                    results.append(
                            pool.apply_async(
                                compute_feature_mean, args=(i, x_train[:,i])))

                # Compute mean and standard deviation (sd)
                else:
                    results.append(
                            pool.apply_async(
                                compute_feature_mean_and_sd, args=(i, x_train[:,i])))

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

            # Ordered sd
            if not mean_only:
                sds = results_values[:,2][ordered_idx]

            print(means)
            print(sds)


        else:
            # Compute features means and sd
            means = du.compute_norm_values(x_train)

        means_by_fold.append(means)
        sds_by_fold.append(sds)

    data.close()

    # Save
    print('Saving input features statictics to', os.path.join(args.exp_path,args.out))
    if mean_only:
        np.savez(os.path.join(args.exp_path,args.out),
                means_by_fold=means_by_fold)

    else:
        np.savez(os.path.join(args.exp_path,args.out),
                means_by_fold=means_by_fold,
                sd_by_fold=sds_by_fold)

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


def compute_feature_mean_and_sd(i, x):
    """
    i is the snp nb (from 0 to nb_snps-1)
    x is the vector of genotypes for the snp (dim : nb_samples x 1)
    """
    # Compute mean
    _,mean = compute_feature_mean(i,x)

    # --- Compute sd ---
    mask = (x >= 0) #set missing values (-1) to False

    numerator = np.sum((x*mask - mean*mask)**2)

    denominator = np.sum(mask) - 1

    sd = np.sqrt(numerator/denominator)

    sd += 1e-6

    return (i,mean,sd)


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

    # Which statistic tom compute (mean or mean+sd)
    parser.add_argument(
            '--mean-only',
            action='store_true',
            help=('Use this flag to compute only input features means '
                  'and not the standard deviations')
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
            default='input_features_statistics.npz',
            help='Output filename. Default: %(default)s'
            )

    return parser.parse_args()


if __name__ == '__main__':
    get_preprocessing_params()
