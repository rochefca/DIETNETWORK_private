import argparse
import os
import sys
import time
import yaml
import pprint

import h5py

import numpy as np

try:
    from comet_ml import Experiment, Optimizer
except:
    # cannot load comet. Proceed...
    Experiment, Optimizer = None, None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.profiler import profiler

import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu
import helpers.log_utils as lu
from helpers.model_handlers import DietNetworkHandler, MlpHandler
from helpers.task_handlers import ClassificationHandler, RegressionHandler


def main():
    print('in main', flush=True)
    args = parse_args()

    # Monitoring time to execute the whole experiment
    exp_start_time = time.time()

    # ----------------------------------------
    #        EXPERIMENT CONFIGURATION
    # ----------------------------------------
    # Experiment's hyperparameters
    f = open(args.config, 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    print('\n---\nExperiment specifications:')
    pprint.pprint(config)
    print('---\n')

    # Task : clasification or regression
    #task = args.task

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

    train_set = du.FoldDataset(fold_indices[0])
    valid_set = du.FoldDataset(fold_indices[1])
    test_set = du.FoldDataset(fold_indices[2])

    print('Loaded train ({} samples), valid ({} samples) and '
          'test ({} samples) sets'.format(
              len(train_set), len(valid_set), len(test_set)))

    print('---\n')


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

    # Optimizer
    lr = config['learning_rate']
    optimizer = torch.optim.Adam(model_handler.get_parameters(), lr=lr)


    # ----------------------------------------
    #          TRAINING LOOP SET UP
    # ----------------------------------------
    print('\n---\nTraining loop set up')
    # Where to save fold results
    exp_identifier = model_handler.get_exp_identifier(config, fold)

    results_dirname = 'RESULTS_' + exp_identifier
    results_fullpath = os.path.join(args.exp_path,
            args.exp_name, results_dirname)

    lu.create_dir(results_fullpath)
    print('Results will be saved to:', results_fullpath)

    # Monitoring best and last models
    bestmodel_fullpath = os.path.join(results_fullpath, 'best_model.pt')
    lastmodel_fullpath = os.path.join(results_fullpath, 'last_model.pt')

    # Batch generators
    batch_size = config['batch_size']
    train_generator = DataLoader(train_set,
            batch_size=batch_size, num_workers=0)

    valid_generator = DataLoader(valid_set,
            batch_size=batch_size, shuffle=False, num_workers=0)

    test_generator = DataLoader(test_set,
            batch_size=batch_size, shuffle=False, num_workers=0)

    print('---\n')

    # ----------------------------------------
    #             TRAINING LOOP
    # ----------------------------------------
    training_start_time = time.time()

    # Max nb of epochs
    n_epochs = config['epochs']
    start_epoch = 0 # first epoch

    # Patience for early stopping
    patience = 0
    max_patience = config['patience']
    has_early_stoped = False

    # Baseline
    if not args.resume_training:
        print('\n---\nComputing baseline (forward pass in model with valid set)')
        baseline_start_time = time.time()

        model_handler.model.eval()

        baseline = mlu.eval_step(model_handler, device, valid_generator,
                                 mus, sigmas, args.normalize)

        # Print baseline results
        model_handler.task_handler.print_baseline_results(baseline)
        print('Computed baseline in {} seconds'.format(
            time.time() - baseline_start_time))

        # Init best results
        model_handler.task_handler.init_best_results(baseline)

    # Resume training
    else:
        print('\n---\nResuming training')
        # Load last model
        checkpoint = torch.load(lastmodel_fullpath)

        # Set model weights with weights from last model
        model_handler.model.load_state_dict(checkpoint['model_state_dict'])

        # Set results from best model
        best_checkpoint = torch.load(bestmodel_fullpath)
        model_handler.task_handler.resume_best_results(best_checkpoint['best_results'])

        # Set start epoch
        start_epoch = checkpoint['epoch']

        # Set patience
        patience = checkpoint['patience']

        print('Loaded last model from epoch {}'.format(start_epoch))
        print('Patience is {}'.format(patience))
        model_handler.task_handler.print_resumed_best_results(best_checkpoint['best_results'])

    # Training loop
    print('\nTraining:')
    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = time.time()
        print('Epoch {} of {}'.format(epoch+1, n_epochs), flush=True)

        # Train step
        model_handler.model.train()
        train_results = mlu.train_step(model_handler, device, train_generator,
                                       mus, sigmas, args.normalize, optimizer)

        # Eval step
        model_handler.model.eval()
        evaluated_train_results = mlu.eval_step(model_handler, device, valid_generator,
                                                mus, sigmas, args.normalize)

        valid_results = mlu.eval_step(model_handler, device, train_generator,
                                      mus, sigmas, args.normalize)

        # Print epoch results
        model_handler.task_handler.print_epoch_results(
                train_results, valid_results)
        print('New train results:')
        model_handler.task_handler.print_epoch_results(
                evaluated_train_results, valid_results)

        # Write epoch results
        train_filename = 'train_results_epoch'+str(epoch+1)
        valid_filename = 'valid_results_epoch'+str(epoch+1)

        train_fullpath = os.path.join(results_fullpath, train_filename)
        valid_fullpath = os.path.join(results_fullpath, valid_filename)

        model_handler.task_handler.save_predictions(
                evaluated_train_results, train_fullpath)

        model_handler.task_handler.save_predictions(
                valid_results, valid_fullpath)

        # Anneal learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = \
                param_group['lr']*config['learning_rate_annealing']


        # Check model improvement and update best results
        has_improved = model_handler.task_handler.update_best_results(
                       valid_results)
        if has_improved:
            # Reset patience
            patience = 0
            # Save best model
            torch.save({'epoch': epoch+1,
                'model_state_dict': model_handler.model.state_dict(),
                'best_results': model_handler.task_handler.best_results},
                bestmodel_fullpath)
            print('Saving best model')

        else:
            patience += 1

        # Save last model
        torch.save({'epoch': epoch+1,
            'model_state_dict': model_handler.model.state_dict(),
            'results': model_handler.task_handler.best_results,
            'patience':patience},
            lastmodel_fullpath)

        print('Epoch execution time: {} seconds'.format(
              time.time() - epoch_start_time))

        # Check early stopping
        if patience >= max_patience:
            has_early_stoped = True
            print('\nEarly stoping, exiting training loop', flush=True)
            break

    print('Executed training in {} seconds'.format(
          time.time() - training_start_time))
    print('---\n')

    # ----------------------------------------
    #                   TEST
    # ----------------------------------------
    test_start_time = time.time()
    print('\n---\nStarting test')

    # Load best model to do the test
    checkpoint = torch.load(bestmodel_fullpath)
    print('Loading best model from epoch {}'.format(checkpoint['epoch']))

    model_handler.model.load_state_dict(checkpoint['model_state_dict'])

    # Test step
    model_handler.model.eval()
    test_results = mlu.eval_step(model_handler, device, test_generator,
                                 mus, sigmas, args.normalize)

    model_handler.task_handler.print_test_results(test_results)

    # Save test results
    test_filename = 'test_results_epoch'+str(checkpoint['epoch'])
    test_fullpath = os.path.join(results_fullpath, test_filename)
    model_handler.task_handler.save_predictions(
            test_results, test_fullpath)


    # Training
    #train(config, args.comet_ml, args.comet_ml_project_name, args.optimization)


def train(config, comet_log, comet_project_name, optimization_exp):

    # ----------------------------------------
    #       EXPERIMENT IDENTIFIER
    # ----------------------------------------
    """
    # Experiment identifier for naming files
    elif config['specifics']['model'] == 'Mlp':
        exp_identifier = 'mlp_' \
                + str(config['params']['n_hidden_u'])[1:-1].replace(', ','_') \
                + '_lr_' + str(config['params']['learning_rate']) \
                + '_lra_' + str(config['params']['learning_rate_annealing']) \
                + '_epochs_' + str(config['params']['epochs']) \
                + '_patience_' + str(config['params']['patience']) \
                + '_inpdrop_' + str(config['params']['input_dropout']) \
                + '_seed_' + str(config['params']['seed'])
    """

    # ----------------------------------------
    #               COMET PROJECT
    # ----------------------------------------
    if comet_log:
        # Init experiment
        if comet_project_name is None:
            experiment = Experiment(auto_histogram_weight_logging=True)

        else:
            experiment = Experiment(
                project_name=comet_project_name,
                auto_metric_logging=False,
                parse_args=False
                )

        # Set experiment name
        experiment.set_name(exp_identifier)

        # Log hyperparams
        experiment.log_parameters(config['params'])

        # Log specifics
        experiment.log_others(config['specifics'])


    # ----------------------------------------
    #             BATCH GENERATORS
    # ----------------------------------------
    print('\n --- Making batch generators ---')
    batch_gen_start_time = time.time()

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

    print('Batch generators initiated in:', time.time()-batch_gen_start_time, 'seconds')

    print('TRAIN SET SAMPLE 1')
    print(train_set.__getitem__(1))

    # ----------------------------------------
    #          TRAINING LOOP SET UP
    # ----------------------------------------

    # Monitoring set up: Epoch
    train_results_by_epoch = []
    valid_results_by_epoch = []

    # File where to save model params
    model_params_filename = 'model_params_' + exp_identifier + '.pt'

    # Baseline (and best result at this point)
    print('\n --- Computing baseline (forward pass in model with valid set) ---')
    baseline_start_time = time.time()

    baseline = mlu.eval_step(mod_handler, device,
            valid_generator, len(valid_set), criterion, mus, sigmas,
            config['specifics']['task'], config['specifics']['normalize'])

    if config['specifics']['task'] == 'classification':
        print('baseline loss:', baseline[0], 'baseline acc:', baseline[1])

    elif config['specifics']['task'] == 'regression':
        print('baseline loss:', baseline[0])

    best_result = baseline
    print('Baseline computed in:', time.time()-baseline_start_time, 'seconds')

    # Save the baseline model
    mod_handler.save(config['specifics']['out_dir'], filename=model_params_filename)

    # Log weights initialisation values to comet-ml
    if comet_log:
        mod_handler.log_weight_initialization(experiment)

    # Patience: Nb epoch without improvement after which to stop training
    patience = 0
    max_patience = config['params']['patience']
    has_early_stoped = False

    # ----------------------------------------
    #           TRAINING LOOP
    # ----------------------------------------
    total_time = 0
    for epoch in range(n_epochs):
        print('Epoch {} of {}'.format(epoch+1, n_epochs), flush=True)
        start_time = time.time()

        # ---Training---
        mod_handler.train_mode()


        epoch_train_result = mlu.train_step(mod_handler, device, optimizer,
                train_generator, len(train_set), criterion, mus, sigmas,
                config['specifics']['task'], config['specifics']['normalize'])

        train_results_by_epoch.append(epoch_train_result)

        # Print result (and optional save to comet-ml)
        if config['specifics']['task'] == 'classification':
            print('train loss:', epoch_train_result[0],
                  'train acc:', epoch_train_result[1], flush=True)

            if comet_log:
                experiment.log_metric("train_loss", epoch_train_result[0], epoch=epoch, step=epoch)
                experiment.log_metric("train_accuracy", epoch_train_result[1], epoch=epoch, step=epoch)

        elif config['specifics']['task'] == 'regression':
            print('train loss:', epoch_train_result[0], flush=True)

            if comet_log:
                experiment.log_metric("train_loss", epoch_train_result[0], epoch=epoch, step=epoch)

        # ---Validation---
        mod_handler.eval_mode()

        epoch_valid_result = mlu.eval_step(mod_handler, device,
                valid_generator, len(valid_set), criterion, mus, sigmas,
                config['specifics']['task'], config['specifics']['normalize'])

        valid_results_by_epoch.append(epoch_valid_result)

        # Print result (and optional save to comet-ml)
        if config['specifics']['task'] == 'classification':
            print('valid loss:', epoch_valid_result[0],
                  'valid acc:', epoch_valid_result[1], flush=True)

            if comet_log:
                experiment.log_metric("valid_loss", epoch_valid_result[0], epoch=epoch, step=epoch)
                experiment.log_metric("valid_accuracy", epoch_valid_result[1], epoch=epoch, step=epoch)

        elif config['specifics']['task'] == 'regression':
            print('valid loss:', epoch_valid_result[0], flush=True)

            if comet_log:
                experiment.log_metric("valid_loss", epoch_valid_result[0], epoch=epoch, step=epoch)

        # Save the predictions for regression
        if config['specifics']['task'] == 'regression':
            filename = 'model_predictions_epoch{}.npz'.format(epoch+1)
            print('Saving predictions to %s' % os.path.join(config['specifics']['out_dir'], filename))

            np.savez(os.path.join(config['specifics']['out_dir'], filename),
                     test_samples=epoch_valid_result[3],
                     test_labels=epoch_valid_result[2],
                     test_preds=epoch_valid_result[1].detach().squeeze().cpu())


        # ---Baseline: check  improvement---

        if mlu.has_improved(best_result, epoch_valid_result):
            # Reset patience
            patience = 0
            # Update best results
            best_result = mlu.update_best_result(best_result, epoch_valid_result)

            # Save model parameters (for later inference)
            print('best validation achieved at epoch {} saving model'.format(epoch+1))
            lu.save_model_params(config['specifics']['out_dir'], comb_model)

        else:
            patience += 1

        # ---Early stopping---
        if patience >= max_patience:
            has_early_stoped = True
            n_epochs = epoch - patience

            # log best validation results to comet
            if comet_log:
                if config['specifics']['task'] == 'classification':
                    experiment.log_metric("best_valid_loss", best_result[0])
                    experiment.log_metric("best_valid_acc", best_result[1])

                if config['specifics']['task'] == 'regression':
                    experiment.log_metric("best_valid_loss", best_result[0])
            break # exit training loop

        # ---Anneal learning rate---
        for param_group in optimizer.param_groups:
            param_group['lr'] = \
                    param_group['lr'] * config['params']['learning_rate_annealing']

        # ---Time---
        #end_time = time.time()
        epoch_time = time.time() - start_time
        total_time += epoch_time
        print('time:', epoch_time, flush=True)

        # Pytorch profiler
        #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

        if comet_log:
            experiment.log_metric("epoch_time", epoch_time, epoch=epoch, step=epoch)

    # End of training phase
    print('Early stoping:', has_early_stoped, flush=True)

    # ----------------------------------------
    #                 TEST
    # ----------------------------------------
    #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
    print(prof.key_averages().table())

    # Monitoring time
    start_time = time.time()

    # Reload weights from early stoping
    model_weights_path = os.path.join(config['specifics']['out_dir'], model_params_filename)

    mod_handler.load(torch.load(model_weights_path))

    # Put model in eval mode
    mod_handler.eval_mode()

    # Test step
    print('Testing model', flush=True)
    test_samples, test_ys, test_results = mlu.test_step(mod_handler, device,
            test_generator, len(test_set), criterion, mus, sigmas,
            config['specifics']['task'], config['specifics']['normalize'])

    """
    print('REAL TEST:')
    print('Test loss:', str(test_results[0]), flush=True)

    # Fake tests
    results = mlu.eval_step(comb_model, device,
            valid_generator, len(valid_set), criterion, mus, sigmas, emb,
            config['specifics']['task'], config['specifics']['normalize'])
    print('eval results:', results)

    print('FAKE TEST:')
    test_samples, test_ys, test_results = mlu.test_step(comb_model, device,
            valid_generator, len(valid_set), criterion, mus, sigmas, emb,
            config['specifics']['task'], config['specifics']['normalize'])
    """
    # Monitoring time
    print('Test time:', time.time()-start_time, flush=True)

    # Print result (and optional save to comet-ml)
    if config['specifics']['task'] == 'classification':
        print('Final accuracy:', str(test_results[2]), flush=True)

        if comet_log:
            experiment.log_metric("test accuracy", test_results[2])

    elif config['specifics']['task'] == 'regression':
        print('Test loss:', str(test_results[0]), flush=True)
        print('Pearson correlation between outputs and targets:',
              str(test_results[2]), flush=True)

        if comet_log:
            experiment.log_metric("Test loss", test_results[0])
            experiment.log_metric("Pearson_r", test_results[1])

    # Save test results (model_predictions.npz)
    if not optimization_exp:
        print('Saving results', flush=True)
        if config['specifics']['task'] == 'classification':
            with h5py.File(dataset_file, 'r') as f:
                label_names = np.array(f['label_names']).astype(np.str_)

            lu.save_results(config['specifics']['out_dir'],
                    test_samples, test_ys, label_names, test_results[0].cpu(), test_results[1].cpu())

        elif config['specifics']['task'] == 'regression':
            lu.save_results_regression(config['specifics']['out_dir'],
                    test_samples, test_ys, test_results[1].detach().squeeze().cpu())

        # Save additional data (additional_data.npz)
        #print('saving additional results', flush=True)
        """
        train_samples = train_set.get_samples()
        valid_samples = valid_set.get_samples()
        with h5py.File(dataset_file, 'r') as f:
            snp_names = np.array(f['snp_names']).astype(np.str_)

        lu.save_additional_data(config['specifics']['out_dir'],
                                train_samples, valid_samples, test_samples,
                                test_ys, pred.cpu(), score.cpu(),
                                label_names, snp_names, mus.cpu(), sigmas.cpu())
        """

        print('\n--- End of execution ---')
        print('Executed training process in {} seconds'.format(
            time.time() - whole_exp_start_time))

def parse_args():
    parser = argparse.ArgumentParser(
            description=('Train, eval and test model of a given fold')
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

    # Resume training
    parser.add_argument(
            '--resume-training',
            action='store_true',
            help='Use this flag to resume training'
            )

    # Fold
    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    # Optional param init
    parser.add_argument(
            '--param-init',
            type=str,
            help='File of parameters initialization values'
            )

    # Comet-ml
    parser.add_argument(
            '--comet-ml',
            action='store_true',
            help='Use this flag to run experiment with comet ml'
            )

    parser.add_argument(
            '--comet-ml-project-name',
            type=str,
            help='Specific project were to send comet Experiment'
            )

    # Optimization process (do not save results)
    parser.add_argument(
            '--optimization',
            action='store_true',
            help=('Use this flag when in optimization process. '
                  '(Config files created with optimize.py).')
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
