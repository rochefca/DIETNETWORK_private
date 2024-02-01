"""
Script to parse data into a npz format and partition data into folds
Creates dataset.npz and folds_indexes.npz (default filenames)
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
import argparse

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/Dietnet')
import helpers.dataset_utils as du
import helpers.mainloop_utils as mlu
import helpers.model as model
import helpers.log_utils as lu
from Interpretability import graph_attribution_manager as gam


def create_dataset():
    args = parse_args()

    exp_path = Path(args.exp_path)
    full_path = exp_path / args.exp_name
    fold_folders = os.listdir(full_path) #[full_path / '*_fold{}/'.format(args.exp_name, fold) for fold in args.which_fold]

    print('Loading data')
    # Load samples, snp names and genotype values

    # Dataset
    du.FoldDataset.dataset_file = os.path.join(exp_path, args.dataset)
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')
    #data = du.load_data(os.path.join(exp_path, args.dataset))
    genotypes = du.FoldDataset.f['inputs'][:]
    class_label_names = du.FoldDataset.f['class_label_names'][:]
    snp_names = du.FoldDataset.f['snp_names'][:]
    class_labels = du.FoldDataset.f['class_labels'][:]
    samples = du.FoldDataset.f['samples'][:]

    du.FoldDataset.f.close()

    # Load embedding
    emb = np.load(os.path.join(exp_path, args.embedding))['emb']

    # load feature statistics
    inp_feat_stats = np.load(args.input_features_stats)
    means_by_fold = inp_feat_stats.f.means_by_fold[:]
    sd_by_fold = inp_feat_stats.f.sd_by_fold[:]

    attr_manager = gam.GraphAttributionManager()
    attr_manager.set_device(torch.device('cpu'))

    abs_attr_all_pops = []

    for fold_folder in fold_folders:
        hf = h5py.File(os.path.join(full_path, fold_folder, 'attrs_avg.h5'), 'r')
        attr_manager.set_agg_attributions(np.nan_to_num(hf['avg_attr'][:, :, :]))

        #abs_attr_all_pops.append(np.abs(attr_manager.agg_attributions).sum(1).sum(1))        # rank attributions based on sum of absolute values across variant and across populations
        abs_attr_all_pops.append(np.amax(np.abs(attr_manager.agg_attributions), axis=(1,2)))  # rank attributions based on max of absolute values across variant and across populations
    hf.close()

    abs_attr_all_pops = np.stack(abs_attr_all_pops).mean(0) # take mean across all folds

    #attr_manager.set_feat_names(np.load(os.path.join(full_paths[0], 'additional_data.npz'))['feature_names']) # pick one arbitrarily
    attr_manager.set_feat_names(snp_names) # pick one arbitrarily
    #attr_manager.set_label_names(np.load(os.path.join(full_paths[0], 'additional_data.npz'))['label_names'])
    attr_manager.set_label_names(class_label_names)
    #attr_manager.set_labels(torch.from_numpy(np.load(os.path.join(full_paths[0], 'additional_data.npz'))['test_labels']))
    attr_manager.set_labels(torch.from_numpy(class_labels))

    if args.percentile_to_remove is not None:
        remove_threshold = np.percentile(abs_attr_all_pops, args.percentile_to_remove)
        to_keep = abs_attr_all_pops < remove_threshold
        print('Datset size reduced to: {} (top {}% removed)'.format(to_keep.sum(), 
                                                                    args.percentile_to_remove))
    elif args.num_to_remove is not None:
        to_keep = np.full(abs_attr_all_pops.shape, False)
        to_keep[np.argsort(abs_attr_all_pops)[::-1][args.num_to_remove:]] = True
        print('Datset size reduced to: {} (top {} removed)'.format(to_keep.sum(), 
                                                                   args.num_to_remove))
    
    if args.random_snp_removal:
        # randomly shuffle which entries to keep
        # this is to compute baseline for SNP attribution removal experiment
        to_keep = np.random.permutation(to_keep)

    # Create dataset
    dataset_fullpath = os.path.join(exp_path, args.dataset_out)
    f_out = h5py.File(dataset_fullpath, 'w')

    # Input features
    genotypes = genotypes[:,np.arange(len(to_keep))[to_keep]] # subset genotypes
    f_out.create_dataset('inputs', data=genotypes)
    # SNP names (Hdf5 doesn't support np UTF-8 encoding: snps.astype('S'))
    f_out.create_dataset('snp_names', data=snp_names.astype('S'))
    # Samples
    f_out.create_dataset('samples', data=samples.astype('S'))
    # Class labels
    f_out.create_dataset('class_labels', data=class_labels)
    f_out.create_dataset('class_label_names', data=class_label_names.astype('S'))
    
    # Regression labels
    #if args.regression_labels is not None:
    #    f_out.create_dataset('regression_labels', data=regression_labels)

    f_out.close()

    #genotypes = data['inputs'][:, to_keep]
    #snps = data['snp_names'][to_keep]

    #  Save resulting data and embeddings
    #np.savez(os.path.join(exp_path, args.dataset_out),
    #         inputs=genotypes,
    #         snp_names=snps,
    #         labels=data['labels'],
    #         label_names=data['label_names'],
    #         samples=data['samples'])
    print('saved dataset to: {}'.format(os.path.join(exp_path, args.dataset_out)))

    np.savez(os.path.join(exp_path, args.embedding_out),
             emb=emb[:, to_keep, :])

    #np.savez(os.path.join(exp_path, args.embedding_out),
    #         emb=emb[:, to_keep, :])
    print('saved embedding to: {}'.format(os.path.join(exp_path, args.embedding_out)))

    np.savez(os.path.join(exp_path, args.input_features_stats_out),
             means_by_fold=means_by_fold[:, np.arange(len(to_keep))[to_keep]],
             sd_by_fold=sd_by_fold[:, np.arange(len(to_keep))[to_keep]])
    print('saved input feature stats to to: {}'.format(os.path.join(exp_path, args.input_features_stats_out)))



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
            '--exp-name',
            type=str,
            required=True,
            help=('Name of directory where to the attribution are. '
                  'The attributions are used to determine which SNPs to remove'
                  'This directory must be in the directory specified with '
                  'exp-path. ')
            )

    parser.add_argument(
            '--which-fold', 
            type=int,
            nargs="*", 
            help='Which fold(s) to train (1st fold is 0).'
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.npz',
            help=('Filename of dataset returned by create_dataset.py '
                  'The file must be in direcotry specified with exp-path '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--embedding',
            type=str,
            default='embedding.npz',
            help=('Filename of embedding returned by generate_embedding.py '
                  'The file must be in directory specified with exp-path. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--percentile-to-remove',
            type=int,
            default=None,
            help=('remove attributions above this percentile (None ignores this)')
            )
    parser.add_argument(
            '--num-to-remove',
            type=int,
            default=None,
            help=('remove attributions above this percentile (None ignores this)')
            )

    parser.add_argument(
            '--dataset-out',
            default='dataset-new.npz',
            help='Filename for the returned dataset from this script. Default: %(default)s'
            )

    parser.add_argument(
            '--embedding-out',
            default='embedding-new.npz',
            help=('Filename for returned embedding from this script. Default: %(default)s')
            )
    
    parser.add_argument(
            '--random-snp-removal',
            action='store_true',
            help=('Randomly remove SNPs (for baseline comparison)')
            )

    parser.add_argument(
            '--input-features-stats',
            type=str,
            required=True,
            help=('Input features mean and sd in npz format returned by '
                  'compute_input_features_statistics.py '
                  'Provide full path')
            )
    parser.add_argument(
            '--input-features-stats-out',
            type=str,
            required=True,
            help=('Input features mean and sd in npz format returned by '
                  'compute_input_features_statistics.py '
                  'Provide full path')
            )

    parser.add_argument(
            '--partition',
            type=str,
            required=True,
            help=('Npz dataset partition returned by partition_data.py '
                  'Provide full path')
            )


    return parser.parse_args()


if __name__ == '__main__':
    create_dataset()
