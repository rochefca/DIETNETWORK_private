import argparse
import math
import os
import pprint
import yaml

import h5py

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu

from helpers.model_handlers import DietNetworkHandler, MlpHandler
from helpers.task_handlers import ClassificationHandler, RegressionHandler

def main():
    args = parse_args()

    # ---------------
    # Loading config
    # ---------------
    # The config file used to train the model
    f = open(args.config, 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    print('\n---\nSpecifications from config file used in training:')
    pprint.pprint(config)
    print('---\n')

    # Set device
    print('\n---\nSetting device')
    print('Cuda available:', torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    print('---\n')
    
    # Fix seed
    seed = config['seed']
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # ----------------------------------------
    #               TASK HANDLER
    # ----------------------------------------
    # Task : clasification or regression
    task = args.task

    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
        task_handler = ClassificationHandler(args.dataset, criterion)

    elif args.task == 'regression':
        criterion = nn.MSELoss()
        task_handler = RegressionHandler(args.dataset, criterion)
    
    # ---------------
    #      DATA
    # ---------------
    print('\n---\nLoading data')
    
    # Fold number
    fold = args.which_fold
    
    # Fold indices
    indices_byfold = np.load(args.partition, allow_pickle=True)
    fold_indices = indices_byfold['folds_indexes'][fold]
    
    # Input features statistics
    inp_feat_stats = np.load(args.input_features_stats)
    
    mus = inp_feat_stats['means_by_fold'][fold]
    # Send to GPU
    mus = torch.from_numpy(mus).float().to(device)

    if 'sd_by_fold' in inp_feat_stats.files:
        sigmas = inp_feat_stats['sd_by_fold'][fold]
        print('Loaded {} means and {} standard deviations of input features'.format(
              len(mus), len(sigmas)))

        # Send to GPU
        sigmas = torch.from_numpy(sigmas).float().to(device)
    else:
        sigmas = None
        print('Loaded {} means of input features'.format(len(mus)))
    
    # Dataset
    du.FoldDataset.dataset_file = args.dataset
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')
    # This tells what label to load in getitem
    du.FoldDataset.task_handler = task_handler
    
    # Load data
    # xs
    du.FoldDataset.data_x_original = np.array(
        du.FoldDataset.f['inputs'],
        dtype=np.int8)
    # ys
    if du.FoldDataset.task_handler.name == 'regression':
        du.FoldDataset.data_y = np.array(
            du.FoldDataset.f['regression_labels'],
            dtype=np.float32)
    
    elif du.FoldDataset.task_handler.name == 'classification':
        du.FoldDataset.data_y = np.array(
            du.FoldDataset.f['class_labels'],
            dtype=np.int64)
    
    # samples : the fold indexes
    du.FoldDataset.data_samples = np.array(fold_indices[0]+fold_indices[1]+fold_indices[2])
    
    # We only need data from test set
    test_set = du.FoldDataset(fold_indices[2])
    
    print('Loaded {} test samples with {} snps'.format(
        len(test_set), du.FoldDataset.data_x_original.shape[1]))
    
    print('---\n')
    
    
    # ----------------------------------------
    #                 MODEL
    # ----------------------------------------
    param_init = None

    # Model architecture (Dietnet or Mlp)
    if args.model == 'Dietnet':
        model_handler = DietNetworkHandler(task_handler, fold,
                args.embedding, device, args.dataset, config, param_init)
    elif args.model == 'Mlp':
        model_handler = MlpHandler(task_handler, args.dataset, config)
    else:
        raise Exception('{} is not a recognized model'.format(
            args.model))

    # Send mmodel to GPU
    model_handler.model.to(device)

    print('\nModel:', model_handler.model)
    
    # Loading trained model parameters
    checkpoint = torch.load(args.model_params)
    model_handler.model.load_state_dict(checkpoint['model_state_dict'])
    print('\nLoaded model parameters from {} at epoch {}'.format(
          args.model_params, checkpoint['epoch']))
    print('---\n')
    

    #------------------------------
    # Missing data simulation loop
    #------------------------------
    print('\n---\nMissing data simulation\n')
    
    miss_percent_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    nb_feats = du.FoldDataset.data_x_original.shape[1]
    
    # Total nb of exepriments : we want same nb of exeperiments for all
    # miss_percent in miss_percent_list. First batch of experiments is based
    # on the nb of snp_groups (created so that every snp is removed or kept
    # once) and we add experiments to match the total nb of experiments
    min_percent = min(miss_percent_list)
    max_percent = max(miss_percent_list)
    
    # We divise the number of SNPs by the nb of snp_groups and ceiling
    # this number because we will do an extra group with the remaining 
    # never use SNPs and random reselected SNPs.
    print('Nb snps in group min percent:', int(math.floor(nb_feats)*min_percent))
    print('Nb snps in group max percent:', int(math.floor(nb_feats)*(1-max_percent)))
    total_exp = max(
        math.ceil(nb_feats/(int(math.floor(nb_feats)*min_percent))),
        math.ceil(nb_feats/(int(math.floor(nb_feats)*(1-max_percent))))
    )
    
    print('Total nb of experiments:', total_exp, '\n')
    
    # For saving data
    if task == 'classification':
        f_str = ""
        for i in miss_percent_list:
            f_str += '_'+str(i)
        results_filename = args.out_prefix+'_missing_data_simulations'+f_str+'.txt'
        results_file = os.path.join(args.results_path, results_filename)
        f = open(results_file, 'w')
        f.write('missing\taccuracy\n')
    
    
    for miss_percent in miss_percent_list:
            print('\n***')
            print('% of missing:', miss_percent)
            # Nb of snps to remove or to keep
            # If the % of missing is <=50% we make groups of SNPs to remove
            # If the % of missing is > 50% we make groups of SNPs to keep
            # This ensure that every (or most) SNP is kept (or remove) at least once
            if miss_percent > 0.5:
                reverse_miss_percent = 1 - miss_percent
                # How many SNPs to keep in a experiment
                group_size = int(math.floor(nb_feats)*reverse_miss_percent)
                # Scale is used to increase info of non-missing genotypes by a 
                # factor = nb_snps/nb_non_missing_snps
                scale = nb_feats / group_size
                # If true, remove the SNPs, if False keep the SNPs
                to_remove = False
                
            else:
                # How many SNPs to remove in an experiment
                group_size = int(math.floor(nb_feats)*miss_percent)
                # Scale is used to increase info of non-missing genotypes by a 
                # factor = nb_snps/nb_non_missing_snps
                scale = nb_feats / (nb_feats-group_size)
                # If true, remove the SNPs, if False keep the SNPs
                to_remove = True
            
            print('Scale:', scale)
            
            # Make the groups of SNPs to remove or to keep
            snp_indices = np.arange(nb_feats)
            np.random.shuffle(snp_indices)
            
            # Drop last group because it will be of smaller size
            all_snp_groups = [snp_indices[i:i + group_size] for i in range(0, len(snp_indices), group_size)]
            last_snp_group = all_snp_groups[-1]
            snp_groups = [snp_indices[i:i + group_size] for i in range(0, len(snp_indices), group_size)][:-1]
            
            print('\nNb of groups:', len(snp_groups))
            print('First group size:', len(snp_groups[0]))
            print('Last (complete) group size:', len(snp_groups[-1]))
            
            # Complete snp_groups to make equal nb of experiments for
            # the different miss_percent
            nb_exp_to_do = total_exp - len(snp_groups)
            print('Last group size:', len(last_snp_group))
            for i in range(0,nb_exp_to_do):
                # Complete the last snp_group
                if i == 0:
                    # nb of snps to add to last_snp_group to make a complete group
                    gr_len = group_size - len(last_snp_group)
                    print('gr len:', gr_len)
                    
                    # Make sure we don't pick a snp that is already in last_snp_group
                    snps_pick = list(set(snp_indices) - set(last_snp_group))
                    gr = np.hstack(
                        (last_snp_group,
                        np.random.choice(snps_pick, size=gr_len, replace=False)))
                
                # Make a new snp_group
                else:
                    gr = np.random.choice(snp_indices, size=group_size, replace=False)
            
                snp_groups.append(gr)
            
            print('\n')
            
            # Iterate over snp_groups to remove SNPs and make test step in model
            for snp_group in snp_groups:
                du.FoldDataset.data_x = du.FoldDataset.data_x_original.copy()
                # Remove SNPs in snp_group
                if to_remove:
                    du.FoldDataset.data_x[:,snp_group] = -1
                
                # Keep SNPs in snp_group (remove the rest)
                else:
                    snps_to_remove = list(set(snp_indices) - set(snp_group))
                    du.FoldDataset.data_x[:,snps_to_remove] = -1
                
                # Data loader
                test_generator = DataLoader(test_set,
                                            batch_size = config['batch_size'],
                                            shuffle=False,
                                            num_workers=0)
                
                # Test step
                model_handler.model.eval()
                test_results = mlu.eval_step(model_handler,
                                             device,
                                             test_set,
                                             test_generator,
                                             mus, sigmas, args.normalize,
                                             args.results_path, 'test_step',
                                             scale=scale)
                
                model_handler.task_handler.print_test_results(test_results)
                
                # Save results
                if task == 'classification':
                    acc = test_results['n_right'].sum()/len(test_results['ys'])
                    f.write(str(miss_percent)+'\t'+str(acc)+'\n')

    print('---\n')
            
    # Where results are saved
    if task == 'classification':
        print('Results saved to', results_file)
        f.close()
                
            
    
    

def parse_args():
    parser = argparse.ArgumentParser(
            description=('Plot loss or accuracy curves for proportion '
                          'of missing SNPs')
            )

    # Input files
    parser.add_argument(
            '--dataset',
            type=str,
            required=True,
            help='Hdf5 dataset. Provide full path'
            )
    
    parser.add_argument(
            '--partition',
            type=str,
            required=True,
            help=('Npz dataset partition returned by partition_data.py '
                  'Provide full path')
            )
    
    parser.add_argument(
            '--embedding',
            type=str,
            required=True,
            help=('Embedding per fold in npz format (ex: class genotype '
                  'frequencies returned by generate_embedding.py) '
                  'Provide full path')
            )
    
    parser.add_argument(
            '--input-features-stats',
            type=str,
            required=True,
            help=('Input features mean and sd in npz format returned by '
                  'compute_input_features_statistics.py '
                  'Provide full path')
            )
    
    # Model specifications
    parser.add_argument(
            '--model',
            type=str,
            choices=['Dietnet', 'Mlp'],
            default='Dietnet',
            help='Model architecture. Default: %(default)s'
            )
    
    parser.add_argument(
            '--config',
            type=str,
            required=True,
            help='Yaml file of hyperparameters. Provide full path'
            )
    
    parser.add_argument(
            '--model-params',
            type=str,
            required=True,
            help='Pt file of params of the trained model.'
            )

    # Input features normalization
    parser.add_argument(
            '--normalize',
            action='store_true',
            help='Use this flag to normalize input features.'
            )
    
    # Task
    parser.add_argument(
            '--task',
            choices = ['classification', 'regression'],
            required=True,
            help='Type of prediction : classification or regression'
            )
    
    # Fold
    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help=('Trained model of which fold to test (1st fold is 0). '
                  'Default: %(default)i')
            )
    
    # Results path
    parser.add_argument(
            '--results-path',
            required=True,
            help='Where to save the results'
            )
    
    parser.add_argument(
            '--out-prefix',
            default='',
            help='Optional prefix for the output file of results'
            )
    
    return parser.parse_args()


if __name__ == '__main__':
    main()