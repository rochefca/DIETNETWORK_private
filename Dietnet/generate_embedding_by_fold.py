import argparse
import os

import numpy as np

import helpers.dataset_utils as du

NB_POSSIBLE_GENOTYPES = 3


def generate_embedding():
    args = parse_args()

    # Load data
    data = np.load(os.path.join(args.exp_path,args.dataset))
    folds_indexes = du.load_folds_indexes(
            os.path.join(args.exp_path,args.folds_indexes)
            )

    fold = args.fold
    print('Computing embedding of fold', str(fold))

    (_,_,_,
    x_train, y_train, _,
    x_valid, y_valid, _,
    _,_,_) = du.get_fold_data(fold, folds_indexes, data)

    # Embedding on train+valid sets
    if args.include_valid:
        x = np.concatenate((x_train, x_valid))
        y = np.concatenate((y_train, y_valid))

    # Embedding on valid set
    elif args.only_valid:
        print('Emb on valid')
        x = x_valid
        y = y_valid

    # Embedding on training set
    else:
        x = x_train
        y = y_train

    # Compute embedding for the fold
    emb = compute_fold_embedding(x, y)

    # Save
    np.savez(os.path.join(args.exp_path,args.out), emb=emb)


def compute_fold_embedding(xs, ys):
    # Total number of classes
    nb_class = ys.max() + 1 #class 0

    # Compute sum of genotypes (0-1-2) per class
    xs = xs.transpose() # rows are snps, col are inds
    embedding = np.zeros((xs.shape[0],nb_class*NB_POSSIBLE_GENOTYPES))
    for c in range(nb_class):
        # Select genotypes for samples of same class
        class_genotypes = xs[:,ys==c]
        nb = class_genotypes.shape[1] #nb of samples in that class
        print('Class:', c, 'NB:', nb)
        for genotype in range(NB_POSSIBLE_GENOTYPES):
            col = NB_POSSIBLE_GENOTYPES*c+genotype
            embedding[:,col] = (class_genotypes == genotype).sum(axis=1)/nb

    return embedding



def parse_args():
    parser = argparse.ArgumentParser(
            description='Generate embedding'
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
            default='dataset.npz',
            help=('Filename of dataset returned by create_dataset.py '
                  'The file must be in directory specidifed with exp-path. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--folds-indexes',
            type=str,
            default='folds_indexes.npz',
            help=('Filename of folds indexes returned by create_dataset.py '
                  'The file must be in directory specified with exp-path. '
                  'Default: %(default)s')
            )

    parser.add_argument('--fold', type=int)

    parser.add_argument(
            '--include-valid',
            action='store_true',
            help=('Use this flag if to include samples from validation set '
                  'in the embedding computation. Otherwise embedding is '
                  'computed using only samples from training set.')
            )

    parser.add_argument(
            '--only-valid',
            action='store_true',
            help='Compute embedding on validation set'
            )

    parser.add_argument(
            '--out',
            type=str,
            default='embedding.npz',
            help='Filename for returned embedding. Default: %(default)s'
            )

    return parser.parse_args()


if __name__ == '__main__':
    generate_embedding()
