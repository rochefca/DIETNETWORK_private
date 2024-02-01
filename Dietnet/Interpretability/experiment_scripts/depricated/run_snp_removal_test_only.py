import os
import sys
import time
import yaml
import pprint
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
from Interpretability import utils
#from Interpretability.utils import load_data_snps_removed
from helpers.model_handlers import DietNetworkHandler, MlpHandler
from helpers.task_handlers import ClassificationHandler, RegressionHandler


def main():
    args = parse_args()


    # ---------------
    # Loading config
    # ---------------
    # Load hyperparameters from config file
    # Info in the config file:
    #   - batch_size
    #   - epochs
    #   - input_dropout
    #   - dropout_main
    #   - learning_rate
    #   - learning_rate_annealing
    #   - nb_hidden_u_aux
    #   - nb_hidden_u_main
    #   - patience
    #   - seed
    #   - uniform_init_limit
    f = open(args.config, 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    print('\n---\nExperiment specifications:')
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

    # ----------------------------------------
    #                   DATA
    # ----------------------------------------
    # Fold number
    fold = args.which_fold

    print('\n---\nLoading fold {} data'.format(fold))

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

    # TO DO
    param_init=None

    # Dataset
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

    # ----------------------------------------
    #                 MODEL
    # ----------------------------------------
    print('\n---\nInitializing model')

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

    print(model_handler.model)

    # Where to save fold results
    exp_identifier = model_handler.get_exp_identifier(config, fold)

    results_dirname = 'RESULTS_' + exp_identifier
    results_fullpath = os.path.join(args.exp_path,
            args.exp_name, results_dirname)

    #lu.create_dir(results_fullpath)

    # Monitoring best and last models
    bestmodel_fullpath = os.path.join(results_fullpath, 'best_model.pt')
    lastmodel_fullpath = os.path.join(results_fullpath, 'last_model.pt')

    # Batch generators
    batch_size = config['batch_size']

    print('\n---\nStarting test')

    # Load best model to do the test
    if not torch.cuda.is_available():
        checkpoint = torch.load(bestmodel_fullpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(bestmodel_fullpath)
    print('Loading best model from epoch {}'.format(checkpoint['epoch']))

    model_handler.model.load_state_dict(checkpoint['model_state_dict'])

    accs = []

    # Load attributions
    with h5py.File(os.path.join(results_fullpath, 'attrs_avg.h5'), 'r') as hf:
        agg_attr = hf['avg_attr'][:]
        print('loaded attribution averages from {}'.format(results_fullpath, 'attrs_avg.h5'))
    # we take the absolute value of the (26) class/target averages per SNP
    # we then take the max of these. This is our attribution based "score" per SNP
    abs_attr = np.amax(np.abs(np.nan_to_num(agg_attr)), axis=(1,2))

    # Test step
    model_handler.model.eval()
    model_handler.get_attribution_model()
    model_handler.model_attr.eval()

    for percentile_to_remove in [100*(0.85**i) for i in range(1, 4)]: #range(0, 21)

        remove_threshold = np.percentile(abs_attr, percentile_to_remove)
        to_keep = abs_attr <= remove_threshold

        if args.baseline:
            amt_to_keep = (~to_keep).sum()
            to_keep = np.array([True]*len(to_keep))
            to_keep_indx = np.random.choice(to_keep.shape[0], amt_to_keep, replace=False)
            to_keep[np.sort(to_keep_indx)] = False
        
        formatted_genotypes, feature_scaling = utils.match_features(du.FoldDataset.data_x, to_keep)
        #formatted_genotypes2, feature_scaling2 = utils.match_input_features(du.FoldDataset.data_x, np.arange(len(to_keep))[to_keep], np.arange(len(to_keep)))
        #(formatted_genotypes[:, to_keep] == du.FoldDataset.data_x[:, to_keep]).mean()
        #(formatted_genotypes2[:, to_keep] == du.FoldDataset.data_x[:, to_keep]).mean()

        du.FoldDataset.data_x = formatted_genotypes

        test_set = du.FoldDataset(fold_indices[2])

        test_generator = DataLoader(test_set,
                batch_size=batch_size, shuffle=False, num_workers=0)

        print('---\n')

        # ----------------------------------------
        #                   TEST
        # ----------------------------------------

        acc, total, preds = utils.test_net_quickly(model_handler.model_attr, 
                                                   test_generator, 
                                                   mus, 
                                                   sigmas, 
                                                   args.normalize, 
                                                   device=torch.device('cpu'), 
                                                   scale_out=feature_scaling)
        print('Accuracy: {:.2f}'.format(acc/total))

        # Reset test dataset
        du.FoldDataset.data_x = np.array(du.FoldDataset.f['inputs'], dtype=np.int8)

        #  check that model gets correct performance
        #for i, data in enumerate(test_generator):

        #    # Replace missing values
        #    du.replace_missing_values(data[0], mus)

        #    # Normalize
        #    if args.normalize:
        #        data[0] = du.normalize(data[0], mus, sigmas)

        #test_results = mlu.eval_step(model_handler,
        #                             device,
        #                             test_set,
        #                             test_generator,
        #                             mus, sigmas, args.normalize,
        #                             results_fullpath, 
        #                             'test_step')
        #acc = (test_results['preds'] == test_results['ys']).mean(0)
        #accs.append( [percentile_to_remove, acc] )
        #model_handler.task_handler.print_test_results(test_results)

    #np.savez(os.path.join(args.exp_path, 'accs_fold_{}'), accs=accs)


def parse_args():
    parser = argparse.ArgumentParser(
            description=('Test model on given dataset')
            )

    # Paths
    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory where results will be saved.'
            )

    parser.add_argument(
            '--exp-name',
            type=str,
            required=True,
            help=('Name of directory where to write results in exp-path '
                  'The results will be written to exp-path/exp-name')
            )

    # Files
    parser.add_argument(
            '--config',
            type=str,
            required=True,
            help='Yaml file of hyperparameters. Provide full path'
            )

    parser.add_argument(
            '--model',
            type=str,
            choices=['Dietnet', 'Mlp'],
            default='Dietnet',
            help='Model architecture. Default: %(default)s'
            )

    parser.add_argument(
            '--dataset',
            type=str,
            required=True,
            help=('Hdf5 dataset created with create_dataset.py '
                  'Provide full path')
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

    # Do random baseline
    parser.add_argument(
            '--baseline',
            action='store_true',
            help='Use this flag to do random feature removal (versus attribution based).'
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
