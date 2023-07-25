import argparse
import math
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
    miss_percent_list = [0.25]
    nb_feats = du.FoldDataset.data_x_original.shape[1]
    
    for miss_percent in miss_percent_list:
            # Nb of snps to remove
            group_size = int(math.floor(nb_feats)*miss_percent)
            reverse_mask = False
            
            print('Nb of SNPs to remove:', group_size)
            
            # Make the groups of SNPs to remove
            snp_indices = np.arange(nb_feats)
            np.random.shuffle(snp_indices)
            
            # Drop last group because it will be of smaller size
            snp_groups = [snp_indices[i:i + group_size] for i in range(0, len(snp_indices), group_size)][:-1]
            print('Nb of groups:', len(snp_groups))
            print('First group size:', len(snp_groups[0]))
            print('Last group size:', len(snp_groups[-1]))
            
            # Iterate over the groups to remove
            for snp_group in snp_groups:
                du.FoldDataset.data_x = du.FoldDataset.data_x_original.copy()
                du.FoldDataset.data_x[:,snp_group] = -1
                
                print('Snp group:', snp_group)
                
                print('Original data:')
                print(du.FoldDataset.data_x_original)
                print('Data with missing injected:')
                print(du.FoldDataset.data_x)
                
                # Data loader
                test_generator = DataLoader(test_set,
                                            batch_size = config['batch_size'],
                                            shuffle=False,
                                            num_workers=0)
                
                # Test step
                model_handler.model.eval()
                results_fullpath = '/lustre06/project/6065672/cam27/DIETNET_EXP/HEIGHT/SANDBOX_DEBUG/MISS_TEST'
                test_results = mlu.eval_step(model_handler,
                                             device,
                                             test_set,
                                             test_generator,
                                             mus, sigmas, args.normalize,
                                             results_fullpath, 'test_step')
                
                model_handler.task_handler.print_test_results(test_results)
                
                
        
            
            
            
            
            
            
            
    
    

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
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )
    
    return parser.parse_args()


if __name__ == '__main__':
    main()