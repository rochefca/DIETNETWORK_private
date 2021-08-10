"""
Script to parse data into a hdf5 format
Creates dataset.hdf5 (default filename)
"""
import argparse
import os
import multiprocessing as mp

import numpy as np

import h5py

import helpers.dataset_utils as du


def create_dataset():
    args = parse_args()

    # Load data
    if args.parallel_loading:
        # Multiprocessing to load lines in parallel
        #ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
        ncpus = args.ncpus
        print('Loading data in parallel using', ncpus, 'processes')
        pool = mp.Pool(processes=ncpus) #this starts 4 worker processes

        # Results returned by processes
        # (samples will be shuffled  because of parallel processes)
        results = []

        # Every line is read by a different process
        with open(args.genotypes, 'r') as f:
            header_line = next(f)
            for line in f:
                # Add sample, sample's genotype accros all SNPs
                results.append(pool.apply_async(du.load_genotypes_parallel, args=(line,)))

        # Get samples and genotypes from multiprocess results
        print('Parsed genotypes, getting the results')
        results_values = [p.get() for p in results]
        samples = np.array([i[0] for i in results_values])
        genotypes = np.array([i[1] for i in results_values])
        print('Loaded', genotypes.shape[0], 'samples with', genotypes.shape[1], 'genotypes')

        # Get snps names from header line
        print('Loading snps names')
        snps = np.array([i.strip() for i in header_line.split('\t')[1:]], dtype='S')
        print('Loaded', len(snps), 'snps')

    else:
        print('Loading data')
        # Load samples, snp names and genotype values
        samples, snps, genotypes = du.load_genotypes(args.genotypes)

    # Load samples with their labels
    print('Loading labels')
    samples_in_labels, labels = du.load_labels(args.labels)

    # Order labels to match samples order obtained from genotypes file
    print('Ordering labels to match genotypes order, based on samples ids')
    ordered_labels = du.order_labels(samples, samples_in_labels, labels)


    # If labels are categories, encode labels as numbers
    if args.prediction == 'classification' :
        label_names, encoded_labels = numeric_encode_labels(ordered_labels)

        # Save dataset to file
        print('Saving dataset to', os.path.join(args.exp_path,args.out))
        f = h5py.File(os.path.join(args.exp_path,args.out), 'w')
        f.create_dataset('inputs', data=genotypes)
        snps = snps.astype('S') # hdf5 doesn't support np UTF-8 encoding
        f.create_dataset('snp_names', data=snps)
        f.create_dataset('labels', data=encoded_labels)
        label_names = label_names.astype('S')
        f.create_dataset('label_names', data=label_names)
        samples = samples.astype('S')
        f.create_dataset('samples', data=samples)
        f.close()

        """
        np.savez(os.path.join(args.exp_path,args.data_out),
                 inputs=genotypes,
                 snp_names=snps,
                 labels=encoded_labels,
                 label_names=label_names,
                 samples=samples)
        """

    # If labels are not categories
    """
    else:
        print('Saving dataset and fold indexes to', args.exp_path)
        np.savez(os.path.join(args.exp_path, args.data_out),
                 inputs=genotypes,
                 snp_names=snp_names,
                 labels=ordered_labels,
                 samples=samples)
    """

    # Partition data into fold (using indexes of the numpy arrays)
    """
    indices = np.arange(len(samples))
    partition = du.partition(indices,
                             args.nb_folds,
                             args.train_valid_ratio,
                             seed=args.seed)
    np.savez(os.path.join(args.exp_path,args.fold_out),
             folds_indexes=np.array(partition,dtype=object),
             seed=np.array([args.seed]))
    """

def _load_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    mat = np.array([l.strip('\n').split('\t') for l in lines])

    samples = mat[1:,0]
    labels = mat[1:,1]

    print('Loaded', str(len(labels)),'labels of', str(len(samples)),'samples')

    return samples, labels


def _order_labels(samples, samples_in_labels, labels):
    idx = [np.where(samples_in_labels == s)[0][0] for s in samples]

    return np.array([labels[i] for i in idx])


def onehot_encode_labels(labels):
    label_names = np.sort(np.unique(labels))

    encoded_labels = np.zeros((len(labels), len(label_names)))
    for i,label in enumerate(labels):
        encoded_labels[i,np.where(label_names==label)[0][0]] = 1.0

    return label_names, encoded_labels


def numeric_encode_labels(labels):
    label_names = np.sort(np.unique(labels))

    encoded_labels = [np.where(label_names==i)[0][0] for i in labels]

    return label_names, encoded_labels


def parse_args():
    parser = argparse.ArgumentParser(
            description='Create dataset and partition data into folds.'
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help=('Path to directory where returned results (parsed dataset '
                  ' and fold indexes) will be saved.')
            )

    parser.add_argument(
            '--genotypes',
            type=str,
            required=True,
            help=('File of genotypes (additive-encoding) in tab-separated '
                  'format. Each line contains a sample id followed '
                  'by its genotypes for every SNP. '
                  'Missing genotypes can be encoded NA, ./. or -1 ')
            )

    parser.add_argument(
            '--parallel-loading',
            action='store_true',
            help='Use this flag to load samples in parallel'
            )

    parser.add_argument(
            '--ncpus',
            type=int,
            help='Number of cpus for parallel loading'
            )

    parser.add_argument(
            '--labels',
            type=str,
            required=True,
            help=('File of samples labels. Each line contains a sample '
                  'id followed by its label in tab-separated format.')
            )

    parser.add_argument(
            '--prediction',
            choices=['classification', 'regression'],
            default='classification',
            help=('Type of prediction (for labels encoding) '
                  'Classification: Labels are numerically encoded '
                  '(one number per category). '
                  'Regression: Labels are kept the same. '
                  'Default: %(default)s')
            )

    """
    parser.add_argument(
            '--nb-folds',
            type=int,
            default=5,
            help='Number of folds. Use 1 for no folds. Default: %(default)i'
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
            '--seed',
            type=int,
            default=23,
            help=('Seed to use for fixing the shuffle of samples '
                  'before partitioning into folds. '
                  'Default: %(default)i')
            )
    """

    parser.add_argument(
            '--out',
            default='dataset.hdf5',
            help='Filename for the returned dataset. Default: %(default)s'
            )

    """
    parser.add_argument(
            '--fold-out',
            default='folds_indexes.npz',
            help=('Filename for returned samples indexes of each fold. '
                  'Default: %(default)s')
            )
    """
    return parser.parse_args()


if __name__ == '__main__':
    create_dataset()
