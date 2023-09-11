
import argparse
import os
import pprint
import time
import yaml

import h5py

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import helpers.test_indep_utils as tu
import helpers.dataset_utils as du
import helpers.log_utils as lu
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
    
    # ---------------------------------------------------------------
    #                 ---- INFO FROM TRAINING PHASE ----
    # ---------------------------------------------------------------
    print('\n---\nLoading info from training phase')
    
    # Which training fold
    fold = args.which_fold
    print('Training fold:', fold)
    
    # ----------------------------------------
    #               TASK HANDLER
    # ----------------------------------------
    # Task : clasification or regression
    # Note: dataset param is used to get 'class_label_names'
    # and set the nb of classes in the classification task.
    # This is why we give train instead of test dataset
    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
        task_handler = ClassificationHandler(args.train_dataset, criterion)

    elif args.task == 'regression':
        criterion = nn.MSELoss()
        task_handler = RegressionHandler(args.train_dataset, criterion)

    print('Task:', task_handler.name)
    
    # ----------------------------------------
    #               INP FEAT STATS
    # ----------------------------------------
    # Input features statistics
    inp_feat_stats = np.load(args.input_features_stats)

    mus = inp_feat_stats['means_by_fold'][fold]
    # Send to GPU
    mus = torch.from_numpy(mus).float().to(device)

    if 'sd_by_fold' in inp_feat_stats.files:
        sigmas = inp_feat_stats['sd_by_fold'][fold]
        print('Input features stats: {} means and {} standard deviations'.format(
              len(mus), len(sigmas)))

        # Send to GPU
        sigmas = torch.from_numpy(sigmas).float().to(device)
    else:
        sigmas = None
        print('Input feautres stats: {} means'.format(len(mus)))
    
    # ----------------------------------------
    #                 DATA
    # ----------------------------------------
    print('\n---\nLoading data')
    
    # Fold indices
    indices_byfold = np.load(args.partition, allow_pickle=True)
    fold_indices = indices_byfold['folds_indexes'][fold]
    
    # Dataset info
    du.FoldDataset.dataset_file = args.dataset
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')
    
    # This tells what label to load in getitem
    du.FoldDataset.task_handler = task_handler
    
    # OPTION TO LOAD ALL DATA TO CPU
    data_start_time = time.time()
    print('Loading all data to cpu')
    print('Loading input features')
    du.FoldDataset.data_x = np.array(du.FoldDataset.f['inputs'], dtype=np.int8)
    print('Loading labels')
    if du.FoldDataset.task_handler.name == 'regression':
        du.FoldDataset.data_y = np.array(du.FoldDataset.f['regression_labels'], dtype=np.float32)
    elif du.FoldDataset.task_handler.name == 'classification':
        du.FoldDataset.data_y = np.array(du.FoldDataset.f['class_labels'], dtype=np.int64)
    print('Loading samples')
    du.FoldDataset.data_samples = np.array(fold_indices[0]+fold_indices[1]+fold_indices[2])
    print('Loaded data in {} seconds'.format(time.time()-data_start_time))
    
    test_set = du.FoldDataset(fold_indices[2])
    
    print('Loaded {} test samples'.format(len(test_set)))
    print('---\n')
    
    # ----------------------------------------
    #                 MODEL
    # ----------------------------------------
    param_init = None

    # Model architecture (Dietnet or Mlp)
    if args.model == 'Dietnet':
        model_handler = DietNetworkHandler(task_handler, fold,
                args.embedding, device, args.train_dataset, config, param_init)
    elif args.model == 'Mlp':
        model_handler = MlpHandler(task_handler, args.train_dataset, config)
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
    
def parse_args():
    parser = argparse.ArgumentParser(
            description='Test a trained model in another dataset'
            )

    parser.add_argument(
            '--model-params',
            type=str,
            required=True,
            help='Pt file of params of the trained model.'
            )

    parser.add_argument(
            '--train-dataset',
            type=str,
            required=True,
            help=('Dataset used to train the model. '
                  'Samples from test set will be used to '
                  'compute the attributions'
                  'Provide full path')
            )
    
    parser.add_argument(
            '--normalize',
            action='store_true'
            )

    parser.add_argument(
            '--config',
            type=str,
            required=True,
            help=('The config file used for training the model. '
                  'Provide full path')
            )
    
    parser.add_argument(
            '--task',
            choices = ['classification', 'regression'],
            required=True,
            help='Type of prediction : classification or regression'
            )

    parser.add_argument(
            '--model',
            type=str,
            choices=['Dietnet', 'Mlp'],
            default='Dietnet',
            help='Model architecture. Default: %(default)s'
            )

    parser.add_argument(
            '--input-features-stats',
            type=str,
            help = ('a')
            )

    parser.add_argument(
            '--embedding',
            required = True,
            help=('Filename of embedding returned by generate_embedding.py '
                  'and used at training time. The file must be in directory '
                  'specified with exp-path. Default: %(default)s')
            )

    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help=('Trained model of which fold to test (1st fold is 0). '
                  'Default: %(default)i')
            )

    return parser.parse_args()
if __name__=='__main__':
    main()