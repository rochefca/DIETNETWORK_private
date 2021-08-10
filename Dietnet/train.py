import argparse
import os
import sys
import time
import yaml
import pprint

import h5py

import numpy as np

#from comet_ml import Experiment, Optimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchinfo import summary

import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu
import helpers.log_utils as lu


def main():
    args = parse_args()

    # Import comet-ml to log experiment
    if args.comet_ml:
        from comet_ml import Experiment, Optimizer

    # Create dir where training info will be saved
    """
    The directory will be created in exp_path/exp_name with the name
    exp_name_foldi where i is the number of the fold
    """
    out_dir = lu.create_out_dir(args.exp_path, args.exp_name, args.which_fold)

    # Create the full config
    """
    The full config contains 2 level info
        - hyperparams : provided in the config file
        - specifics : paths and files used in the training process
                      (specified with command line arguments)
    """
    config = {}

    # Hyperparameters
    f = open(os.path.join(args.exp_path, args.exp_name, args.config))
    config_hyperparams = yaml.load(f, Loader=yaml.FullLoader)

    # Project name (will be added to specifics item in config dict)
    project_name = config_hyperparams['project_name']
    config_hyperparams.pop('project_name')

    config['params'] = config_hyperparams

    # Add fold to config hyperparams
    config['params']['fold'] = args.which_fold

    # Specifics
    specifics = {}
    specifics['project_name'] = project_name
    specifics['exp_path'] = args.exp_path
    specifics['exp_name'] = args.exp_name
    specifics['out_dir'] = out_dir
    specifics['folds_indexes'] = args.folds_indexes
    specifics['dataset'] = args.dataset
    specifics['embedding'] = args.embedding
    specifics['preprocess_params'] = args.preprocess_params
    specifics['param_init'] = args.param_init

    config['specifics'] = specifics

    # This is the full configurations for the training
    pprint.pprint(config)

    # Save experiment configurations (out_dir/full_config.log)
    lu.save_exp_params(config['specifics']['out_dir'],'full_config.log', config)

    # Training
    train(config, args.comet_ml)


def train(config, comet_log):
    # ----------------------------------------
    #               COMET PROJECT
    # ----------------------------------------
    if comet_log:
        # Init experiment (will be sent to the project project_name)
        experiment = Experiment(
                project_name=config['specifics']['project_name'],
                auto_histogram_weight_logging=True
                )

        # Log hyperparams
        experiment.log_parameters(config['params'])

        # Log specifics
        experiment.log_others(config['specifics'])

    # ----------------------------------------
    #               SET GPU
    # ----------------------------------------
    print('Cuda available:', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # ----------------------------------------
    #               FIX SEED
    # ----------------------------------------
    seed = config['params']['seed']
    #torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

    # Send mus and sigmans to GPU
    mus = torch.from_numpy(mus).float().to(device)
    sigmas = torch.from_numpy(sigmas).float().to(device)

    # ----------------------------------------
    #           LOAD FOLD INDEXES
    # ----------------------------------------
    print('Loading fold indexes split into train, valid, test sets')
    all_folds_idx = np.load(os.path.join(
        config['specifics']['exp_path'],
        config['specifics']['folds_indexes']),
        allow_pickle=True)

    fold_idx = all_folds_idx['folds_indexes'][config['params']['fold']]

    # ----------------------------------------
    #       LOAD TRAIN, VALID, TEST SETS
    # ----------------------------------------
    print('Making train, valid, test sets classes')

    # Dataset hdf5 file
    dataset_file = os.path.join(
            config['specifics']['exp_path'],
            config['specifics']['dataset'])

    du.FoldDataset.dataset_file = dataset_file
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')

    train_set = du.FoldDataset(fold_idx[0])
    print('training set:', len(train_set))
    valid_set = du.FoldDataset(fold_idx[1])
    print('valid set:', len(valid_set))
    test_set = du.FoldDataset(fold_idx[2])
    print('test set:', len(test_set))

    # ----------------------------------------
    #             LOAD EMBEDDING
    # ----------------------------------------
    print('Loading embedding')
    emb = du.load_embedding(os.path.join(
        config['specifics']['exp_path'],
        config['specifics']['embedding']),
        config['params']['fold'])

    # Send to GPU
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
    with h5py.File(dataset_file, 'r') as f:
        n_targets = len(f['label_names'])

    print('\n***Nb features in models***')
    print('n_feats_emb:', n_feats_emb)
    print('n_feats:', n_feats)
    print('n_targets:', n_targets)

    # Model init
    comb_model = model.CombinedModel(
            n_feats=n_feats_emb,
            n_hidden_u_aux=config['params']['nb_hidden_u_aux'],
            n_hidden_u_main=config['params']['nb_hidden_u_aux'][-1:] \
                            +config['params']['nb_hidden_u_main'],
            n_targets=n_targets,
            param_init=config['specifics']['param_init'],
            input_dropout=config['params']['input_dropout'])

    # Note: runs script in single GPU mode only!
    comb_model.to(device)
    #print(summary(comb_model.feat_emb, input_size=(294427,1,1,78)))
    #print(summary(comb_model.disc_net, input_size=[(138,1,1,294427),(100,294427)]))

    # ----------------------------------------
    #               OPTIMIZATION
    # ----------------------------------------
    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    lr = config['params']['learning_rate']
    optimizer = torch.optim.Adam(comb_model.parameters(), lr=lr)

    # Max nb of epochs
    n_epochs = config['params']['epochs']

    # ----------------------------------------
    #             BATCH GENERATORS
    # ----------------------------------------
    batch_size = config['params']['batch_size']

    train_generator = DataLoader(train_set,
                                 batch_size=batch_size, num_workers=0)
    valid_generator = DataLoader(valid_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0)
    test_generator = DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)

    # ----------------------------------------
    #           TRAINING LOOP
    # ----------------------------------------
    # Save model summary
    lu.save_model_summary(config['specifics']['out_dir'],
                          comb_model, criterion, optimizer)

    # Monitoring: Epoch loss and accuracy setup
    train_losses = []
    train_acc = []
    valid_losses = []
    valid_acc = []

    # Baseline
    comb_model.eval()
    min_loss, best_acc = mlu.eval_step(comb_model, device,
            valid_generator, len(valid_set), criterion, mus, sigmas, emb)

    print('baseline loss:',min_loss, 'baseline acc:', best_acc)

    # Patience: Nb epoch without improvement after which to stop training
    patience = 0
    max_patience = config['params']['patience']
    has_early_stoped = False

    total_time = 0
    for epoch in range(n_epochs):
        print('Epoch {} of {}'.format(epoch+1, n_epochs), flush=True)
        start_time = time.time()

        # ---Training---
        comb_model.train()

        epoch_loss, epoch_acc = mlu.train_step(comb_model, device, optimizer,
                train_generator, len(train_set), criterion, mus, sigmas, emb)

        print('train loss:', epoch_loss, 'train acc:', epoch_acc, flush=True)

        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Comet
        if comet_log:
            experiment.log_metric("train_accuracy", epoch_acc, epoch=epoch)


        # ---Validation---
        comb_model.eval()

        epoch_loss, epoch_acc = mlu.eval_step(comb_model, device,
                valid_generator, len(valid_set), criterion, mus, sigmas, emb)

        print('valid loss:', epoch_loss, 'valid acc:', epoch_acc,flush=True)

        valid_losses.append(epoch_loss)
        valid_acc.append(epoch_acc)

        # Comet
        if comet_log:
            experiment.log_metric("train_accuracy", epoch_acc, epoch=epoch)
            experiment.log_metric("train_loss", epoch_loss, epoch=epoch)

        # ---Baseline: check  improvement---
        if mlu.has_improved(best_acc, epoch_acc, min_loss, epoch_loss):
            patience = 0
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            if epoch_loss < min_loss:
                min_loss = epoch_loss
            # Save model parameters (for later inference)
            print('best validation acc achieved: {} (loss {}) at epoch {} saving model ...'.format(best_acc, epoch_loss, epoch))
            lu.save_model_params(config['specifics']['out_dir'], comb_model)
        else:
            patience += 1

        # ---Early stopping---
        if patience >= max_patience:
            has_early_stoped = True
            n_epochs = epoch - patience
            break # exit training loop

        # ---Anneal laerning rate---
        for param_group in optimizer.param_groups:
            param_group['lr'] = \
                    param_group['lr'] * config['params']['learning_rate_annealing']

        # ---Time---
        end_time = time.time()
        total_time += end_time-start_time
        print('time:', end_time-start_time, flush=True)

    # End of training phase
    print('Early stoping:', has_early_stoped)

    # ----------------------------------------
    #                 TEST
    # ----------------------------------------
    # Reload weights from early stoping
    model_weights_path = os.path.join(config['specifics']['out_dir'], 'model_params.pt')
    comb_model.load_state_dict(torch.load(model_weights_path))

    # Put model in eval mode
    comb_model.eval()

    # Test step
    test_samples, test_ys, score, pred, acc = mlu.test_step(comb_model, device,
            test_generator, len(test_set), mus, sigmas, emb)

    print('Final accuracy:', str(acc))
    print('total running time:', str(total_time))

    # Comet
    if comet_log:
        experiment.log_metric("test accuracy", acc)

    # Save test results (model_predictions.npz)
    with h5py.File(dataset_file, 'r') as f:
        label_names = np.array(f['label_names']).astype(np.str_)

    lu.save_results(config['specifics']['out_dir'], test_samples, test_ys,
                    label_names, score.cpu(), pred.cpu())

    # Save additional data (additional_data.npz)
    train_samples = train_set.get_samples()
    valid_samples = valid_set.get_samples()
    with h5py.File(dataset_file, 'r') as f:
        snp_names = np.array(f['snp_names']).astype(np.str_)

    lu.save_additional_data(config['specifics']['out_dir'],
                            train_samples, valid_samples, test_samples,
                            test_ys, pred.cpu(), score.cpu(),
                            label_names, snp_names, mus.cpu(), sigmas.cpu())


def parse_args():
    parser = argparse.ArgumentParser(
            description=('Train, eval and test model of a given fold')
            )

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

    parser.add_argument(
            '--config',
            type=str,
            default='config.yaml',
            help='Yaml file of hyperparameter. Default: %(default)s'
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.hdf5',
            help=('Filename of dataset returned by create_dataset.py '
                  'The file must be in direcotry specified with exp-path '
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

    parser.add_argument(
        '--embedding',
        type=str,
        default='embedding.npz',
        help=('Filename of embedding returned by generate_embedding.py '
              'The file must be in directory specified with exp-path. '
              'Default: %(default)s')
        )

    parser.add_argument(
        '--preprocess-params',
        type=str,
        default='preprocessing_params.npz',
        help='Normalization parameters obtained with get_preprocessing_params.py'
        )

    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    parser.add_argument(
            '--param-init',
            type=str,
            help='File of parameters initialization values'
            )

    parser.add_argument(
            '--comet-ml',
            action='store_true',
            help='Use this flag to run experiment with comet ml'
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
