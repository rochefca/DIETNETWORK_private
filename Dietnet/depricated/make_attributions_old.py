import os
import time
import numpy as np
from pathlib import Path
import argparse
import yaml
import pprint

from captum.attr import IntegratedGradients
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import helpers.dataset_utils as du
from helpers import mainloop_utils as mlu
#from helpers.mainloop_utils import load_model
import helpers.model as model
from Interpretability import attribution_manager as am
import helpers.log_utils as lu


def load_data(exp_path, dataset, folds_indexes, which_fold, seed, train_valid_ratio, device, task, batch_size=12):
    # Get fold data (indexes and samples are np arrays, x,y are tensors)
    data = du.load_data(os.path.join(exp_path, dataset))
    folds_indexes = du.load_folds_indexes(
            os.path.join(exp_path, folds_indexes))
    (train_indexes, valid_indexes, test_indexes,
     x_train, y_train, samples_train,
     x_valid, y_valid, samples_valid,
     x_test, y_test, samples_test) = du.get_fold_data(which_fold,
                                        folds_indexes,
                                        data,
                                        split_ratio=train_valid_ratio,
                                        seed=seed)
    # Put data on GPU
    x_train, x_valid, x_test = x_train.to(device), x_valid.to(device), \
            x_test.to(device)
    x_train, x_valid, x_test = x_train.float(), x_valid.float(), \
            x_test.float()

    y_train, y_valid, y_test = y_train.to(device), y_valid.to(device), \
            y_test.to(device)

    # Compute mean and sd of training set for normalization
    mus, sigmas = du.compute_norm_values(x_train)

    # Replace missing values
    du.replace_missing_values(x_train, mus)
    du.replace_missing_values(x_valid, mus)
    du.replace_missing_values(x_test, mus)

    # Normalize
    x_train_normed = du.normalize(x_train, mus, sigmas)
    x_valid_normed = du.normalize(x_valid, mus, sigmas)
    x_test_normed = du.normalize(x_test, mus, sigmas)

    # Make fold final dataset
    train_set = du.FoldDataset(x_train_normed, y_train, samples_train)
    valid_set = du.FoldDataset(x_valid_normed, y_valid, samples_valid)
    test_set = du.FoldDataset(x_test_normed, y_test, samples_test)

    test_generator = DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=False)

    del train_set, valid_set, x_train_normed, x_valid_normed, x_train, x_valid, samples_train, samples_valid, samples_test, folds_indexes, mus, sigmas
    torch.cuda.empty_cache()

    return test_generator, x_test

def main(args):
    # Directory to save experiment info
    out_dir = lu.create_out_dir(args.exp_path, args.exp_name, args.which_fold)

    # Save experiment parameters
    #lu.save_exp_params(out_dir, args)

    # Create the full config
    """
    The full config contains 2 level info
        - hyperparams : provided in the config file
        - specifics : paths and files used in the training process
                      (specified with command line arguments)
    """
    config = {}

    # Hyperparameters
    f = open(os.path.join(args.exp_path, args.exp_name, args.config), 'r')
    config_hyperparameters = yaml.load(f, Loader=yaml.FullLoader)

    # Overwrite batch size
    """
    In attribution computation mode we want a smaller batch size
    for memory capacity
    """
    config['params'] = config_hyperparams

    # Add fold to config hyperparams
    config['params']['fold'] = args.which_fold

    # Specifics
    specifics = {}
    specifics['exp_path'] = args.exp_path
    specifics['exp_name'] = args.exp_name
    specifics['out_dir'] = out_dir
    specifics['partition'] = args.partition
    specifics['dataset'] = args.dataset
    specifics['embedding'] = args.embedding
    specifics['normalize'] = args.normalize
    specifics['preprocess_params'] = args.preprocess_params
    specifics['task'] = args.task
    specifics['param_init'] = None # Trained model will be loaded
    specifics['model_weights'] = args.model_name

    config['specifics'] = specifics

    print('Attribution confiduration:')
    pprint.pprint(config)

    # ----------------------------------------
    #               SET DEVICE
    # ----------------------------------------
    print('Cuda available:', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # ----------------------------------------
    #               FIX SEED
    # ----------------------------------------
    seed = config['seed']
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print('Seed:', str(seed))

    """
    test_generator, x_test = load_data(args.exp_path,
                                       args.dataset,
                                       args.folds_indexes,
                                       args.which_fold,
                                       args.seed,
                                       args.train_valid_ratio,
                                       device,
                                       args.batch_size)
    """
    # ----------------------------------------
    #           LOAD MEAN and SD
    # ----------------------------------------
    print('loading preprocessing parameters')

    # Mean and sd per feature computed on training set
    preprocess_params = np.load(os.path.join(
        config['specifics']['exp_path'],
        config['specifics']['preprocess_params'])
        )
    mus = preprocess_params['means_by_fold'][config['params']['fold']]
    sigmas = preprocess_params['sd_by_fold'][config['params']['fold']]

    # ----------------------------------------
    #               TEST SET
    # ----------------------------------------
    print('Loading fold indexes to get test set indexes')
    all_folds_idx = np.load(os.path.join(args.exp_path, args.folds_indexes),
                            allow_pickle=True)

    fold_idx = all_folds_idx['folds_indexes'][args.which_fold]

    # Instantiate FoldDataset class
    du.FoldDataset.dataset_file = os.path.join(args.exp_path, args.dataset)
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')

    # Label conversion depending on task
    if args.task == 'classification':
        du.FoldDataset.label_type = np.int64
    elif args.task == 'regression':
        du.FoldDataset.label_type = np.float32

    # Test set
    test_set = du.FoldDataset(fold_idx[2])
    print('test set:', len(test_set))

    # Test batch generator
    test_generator = DataLoader(test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=0)

    # ----------------------------------------
    #             LOAD EMBEDDING
    # ----------------------------------------
    print('Loading embedding')
    emb = du.load_embedding(os.path.join(args.exp_path,args.embedding),
                            args.which_fold)
    # Send to device
    emb = emb.to(device)
    emb = emb.float()

    # Normalize embedding
    emb_norm = (emb ** 2).sum(0) ** 0.5
    emb = emb/emb_norm

    # ----------------------------------------
    #               MAKE MODEL
    # ----------------------------------------
    # Aux net input size (nb of emb features)
    if len(emb.size()) == 1:
        n_feats_emb = 1 # input of aux net, 1 value per SNP
        emb = torch.unsqueeze(emb, dim=1) # match size in Linear fct (nb_snpsx1)
    else:
        n_feats_emb = emb.size()[1] # input of aux net

    # Main net input size (nb of features)
    n_feats = emb.size()[0] # input of main net

    # Main net output size (nb targets)
    if args.task == 'classification':
        with h5py.File(dataset_file, 'r') as f:
            n_targets = len(f['label_names'])
    elif args.task == 'regression':
        n_targets = 1

    # Model init
    comb_model = model.CombinedModel(
        n_feats=n_feats_emb,
        n_hidden_u_aux=config['params']['nb_hidden_u_aux'],
        n_hidden_u_main=config['params']['nb_hidden_u_aux'][-1:] \
                +config['params']['nb_hidden_u_main'],
        n_targets=n_targets,
        param_init=None)

    """
    n_feats_emb = emb.size()[1] # input of aux net
    n_feats = emb.size()[0] # input of main net
    # Hidden layers size
    emb_n_hidden_u = 100
    discrim_n_hidden1_u = 100
    discrim_n_hidden2_u = 100
    # Output layer
    n_targets = test_generator.dataset.ys.max().item()+1 # zero-based encoding
    """

    print('\n***Nb features in models***')
    print('n_feats_emb:', n_feats_emb)
    print('n_feats:', n_feats)
    print('n_targets:', n_targets)

    """
    model_path = os.path.join(out_dir, args.model_name)

    discrim_model = load_model(model_path, emb, device,
                               n_feats=n_feats_emb,
                               n_hidden_u=emb_n_hidden_u,
                               n_hidden1_u=discrim_n_hidden1_u,
                               n_hidden2_u=discrim_n_hidden2_u,
                               n_targets=n_targets,
                               input_dropout=0.,
                               incl_bias=True)
    """
    #del data, folds_indexes, train_indexes, valid_indexes, samples_train, samples_valid, x_train, x_valid, y_train, y_valid, mus, sigmas, x_train_normed, x_valid_normed, train_set, valid_set
    #torch.cuda.empty_cache()
    #print('Cleared out unneeded memory. Ready for inference')

    #baseline = torch.zeros(1, x_test[0].shape[0]).to(device)                # this is doing ordinary 0-baseline
    baseline = test_generator.dataset.xs.min(0).values.view(1,-1).to(device) # this is doing "encoded" 0-baseline

    attr_manager = am.AttributionManager()

    attr_manager.set_model(discrim_model)
    attr_manager.init_attribution_function(attr_type='int_grad', backend='captum')
    # attr_manager.init_attribution_function(attr_type='int_grad', backend='custom')
    attr_manager.set_data_generator(test_generator)
    attr_manager.set_genotypes_data(x_test)
    attr_manager.set_raw_attributions_file(os.path.join(out_dir, 'attrs.h5'))
    attr_manager.set_device(device)

    attr_manager.create_raw_attributions(False,
                                         only_true_labels=False,
                                         baselines=baseline,
                                         n_steps=100,
                                         method='riemann_left')

    out = attr_manager.get_attribution_average()
    with h5py.File(os.path.join(out_dir, 'attrs_avg.h5'), 'w') as hf:
        hf['avg_attr'] = out.cpu().numpy()
        print('Saved attribution averages to {}'.format(out_dir, 'attrs_avg.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=('Preprocess features for main network '
                         'and train model for a given fold')
            )

    # Paths
    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory of dataset, folds indexes and embedding.'
            )

    parser.add_argument(
            '--exp-name',
            type=str,
            required=True,
            help=('Name of directory where to save the results. '
                  'This direcotry must be in the directory specified with '
                  'exp-path. ')
            )

    # Files
    parser.add_argument(
            '--config',
            type=str,
            default='config.yaml',
            help='Yaml file of hyperparameter. Default: %(default)s'
            )

    parser.add_argument(
            '--model-name',
            type=str,
            default='model_params.pt',
            help='Filename of model saved in main script '
                  'The file must be in direcotry specified with exp-path '
                  'Default: %(default)s'
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
            '--partition',
            type=str,
            default='partitioned_idx.npz',
            help=('Filename of folds indexes returned by create_dataset.py '
                  'The file must be in directory specified with exp-path. '
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

    # Fold
    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    # Batch size
    parser.add_argument(
            '--batch_size',
            type=int,
            default=12,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    # Task
    parser.add_argument(
            '--task',
            choices = ['classification', 'regression'],
            required=True,
            help='Type of prediction : classification or regression'
            )

    # Input features normalization
    parser.add_argument(
            '--normalize',
            action='store_true',
            help='Use this flag to normalize input features.'
            )

    parser.add_argument(
            '--input-features-stats',
            type=str,
            required=True,
            help=('Input features mean and sd in npz format returned by '
                  'compute_input_features_statistics.py '
                  'Provide full path')
            )

    args = parser.parse_args()

    main(args)
