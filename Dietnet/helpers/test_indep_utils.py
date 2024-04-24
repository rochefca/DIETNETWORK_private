from pathlib import Path, PurePath
import sys

import numpy as np

import torch

import helpers.dataset_utils as du


def get_train_dir(exp_path, exp_name, fold):
    train_dir_name = exp_name + '_fold' + str(fold)
    train_dir_path = PurePath(exp_path, exp_name, train_dir_name)

    if Path(train_dir_path).exists():
        return train_dir_path
    else:
        print('Could not find training information. Path',
                train_dir_path, 'does not exist.')
        sys.exit(1)


def match_input_features(genotypes, test_snps, train_snps):
    """
    Check if input features in test and training sets are the same.
    If not:
        1. Input features that are in test set but not in training set
           will be ignored.
           
        2. Input features that are in training set but not in test set
           are added to the matrix of genotypes as missing values (-1).
           
        3. Non missing genotypes in test set will be scaled according to the 
           proportion of SNPs from train set that are missing in test set. We
           return the scale to be used for scaling.
           (We don't do the scaling here, we do it on the gpu device right
           before feeding the data into the network for faster transfer from 
           cpu to gpu) 
    """

    # Train and test snps are the same, return test genotypes and scale=1.0
    if (test_snps.shape==train_snps.shape) and (test_snps==train_snps).all():
        return genotypes, 1.0

    # Test and train snps are not the same :
    # set genotypes of missing snps in test set to -1
    # scale non-missing genotypes by a factor = nb_train_snps / nb_test_snps
    nb_ignored_features = 0
    nb_matching_features = 0
    matched_genotypes = -np.ones((genotypes.shape[0], train_snps.shape[0]),
            dtype='int8')

    for i,snp in enumerate(test_snps):
        if snp in train_snps:
            matched_genotypes[:,(train_snps==snp).argmax()] = genotypes[:,i]
            nb_matching_features +=1
        else:
            nb_ignored_features +=1

        if i % 1000 == 0 and i != 0:
            print('Matched', str(i), 'of', str(test_snps.shape[0]), 'test input features',nb_matching_features/(nb_matching_features+nb_ignored_features)*100,'% match')
    
    print('Matched', str(test_snps.shape[0]), 'of',
            str(test_snps.shape[0]), 'input features')

    # SNPs in test set but not in training set
    if nb_ignored_features > 0:
        print(str(nb_ignored_features), 'test features ignored')

    nb_missing_features = train_snps.shape[0] - nb_matching_features
   
    # Scale
    if nb_missing_features > 0:
        scale = float(train_snps.shape[0]) / nb_matching_features
        print('\nScale:', str(scale))
    
    else:
        scale = 1.0

    return matched_genotypes, scale


def save_test_results(out_dir, test_name, samples, score, pred, label_names):
    filename = test_name + '_eval_results.npz'

    print('Saving eval results to %s' % PurePath(out_dir, filename))

    np.savez(PurePath(out_dir, filename),
             samples=samples,
             score=score.cpu(),
             pred=pred.cpu(),
             label_names=label_names)