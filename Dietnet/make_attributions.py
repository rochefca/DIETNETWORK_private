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
from captum.attr import IntegratedGradients
from Interpretability import attribution_manager as am

def make_baseline(mod_handler, dset, data_generator, mus, sigmas, device, normalize):
    #baseline = torch.zeros(1, x_test[0].shape[0]).to(device)                # this is doing ordinary 0-baseline
    #baseline = test_generator.dataset.xs.min(0).values.view(1,-1).to(device) # this is doing "encoded" 0-baseline
    #baseline = test_generator.dataset.data_x.min(0).values.view(1,-1).to(device) # this is doing "encoded" 0-baseline

    task_handler = mod_handler.task_handler

    # Reset to 0 batches results from previous epoch
    task_handler.init_batches_results(dset, data_generator)

    full_dset = []
    bstart = 0
    for batch, (x_batch, y_batch, samples) in enumerate(data_generator):
        # Compile batch samples and labels
        bend = bstart + len(samples) # batch end pos
        task_handler.batches_results['samples'][bstart:bend] = samples

        # Send data to device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch = x_batch.float()

        # Replace missing values
        du.replace_missing_values(x_batch, mus)

        # Normalize
        if normalize:
            x_batch = du.normalize(x_batch, mus, sigmas)
        full_dset.append(x_batch)
    x_dset = torch.cat(full_dset)
    baseline = x_dset.min(0).values.view(1,-1)
    return x_dset, baseline


def main():
    # Monitoring execution time
    exp_start_time = time.time()

    # ----------------------------------------
    #        EXPERIMENT CONFIGURATION
    # ----------------------------------------

    # -------------
    # Loading args
    # -------------
    # Where to save the results:
    # --exp-path: Path to directory where results will be saved
    # --exp-name: Dir where to save results (dir will be created in exp-path)

    # The files to provide:
    # --config: Yaml file of hyperparameters
    # --dataset: Hdf5 file
    # --partition: npz file (samples partitioning of folds)
    # --embedding: npz file, one embedding per fold
    # --input-features-stats: npz file, stats per fold

    # Specifications
    # --model: {Dietnet, Mlp} : which model to use
    # --normalize: Flag used to normalize or not input features
    # --task: {classification, regression}
    # --which-fold

    # Other args
    # --resume-training: continue training from last saved epoch
    # --param-init: PAS FONCTIONNEL DANS CETTE IMPLEM
    # --comet-ml et --comet-ml-project-name : PAS SURE QUE ÇA FONCTIONNE ENCORE CETTE CHOSE
    # --optimization : JE PENSE QUE C'ÉTAIT AVEC COMET
    args = parse_args()


    # ---------------
    # Loading config
    # ---------------
    # Load hyperparameters from config file
    # Info in the config file:
    #   - batch_size (ignoring here)
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
    batch_size = args.batch_size #config['batch_size'] is too big!
    train_generator = DataLoader(train_set, shuffle=True,
            batch_size=batch_size, num_workers=0, drop_last=True)

    valid_generator = DataLoader(valid_set,
            batch_size=batch_size, shuffle=False, num_workers=0)

    test_generator = DataLoader(test_set,
            batch_size=batch_size, shuffle=False, num_workers=0)

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
    test_results = mlu.eval_step(model_handler,
                                 device,
                                 test_set,
                                 test_generator,
                                 mus, sigmas, args.normalize,
                                 results_fullpath, 'test_step')

    model_handler.task_handler.print_test_results(test_results)

    # Save test results
    test_filename = 'test_results_epoch'+str(checkpoint['epoch'])
    test_fullpath = os.path.join(results_fullpath, test_filename)
    model_handler.task_handler.save_predictions(
            test_results, test_fullpath)

    # ----------------------------------------
    #               Attributions
    # ----------------------------------------
    out_dir = results_fullpath # to be consistent with the old code!

    x_test, baseline = make_baseline(model_handler, test_set, test_generator, mus, sigmas, device, args.normalize)

    attr_manager = am.AttributionManager()

    attr_manager.set_model(model_handler.model.main_net) # hack for now!
    attr_manager.init_attribution_function(attr_type='int_grad', backend='captum')
    # attr_manager.init_attribution_function(attr_type='int_grad', backend='custom')
    attr_manager.set_data_generator(test_generator)
    attr_manager.set_genotypes_data(x_test)
    attr_manager.set_raw_attributions_file(os.path.join(out_dir, 'attrs.h5'))
    attr_manager.set_device(device)
    
    import pdb
    pdb.set_trace()
    #model_handler.model.main_net(x_test[:10,:])
    #*** TypeError: forward() missing 5 required positional arguments: 'fatLayer_weights', 'results_fullpath', 'epoch', 'batch', and 'step'
    #model_handler.model.main_net(x_test[:10,:], , '', 1, 1, 1)

    attr_manager.create_raw_attributions(False,
                                         only_true_labels=False,
                                         baselines=baseline,
                                         n_steps=100,
                                         method='riemann_left')
    # TypeError: forward() missing 5 required positional arguments: 'fatLayer_weights', 'results_fullpath', 'epoch', 'batch', and 'step'

    out = attr_manager.get_attribution_average()
    with h5py.File(os.path.join(out_dir, 'attrs_avg.h5'), 'w') as hf:
        hf['avg_attr'] = out.cpu().numpy()
        print('Saved attribution averages to {}'.format(out_dir, 'attrs_avg.h5'))


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
    
    #parser.add_argument(
    #        '--model-name',
    #        type=str,
    #        default='model_params.pt',
    #        help='Filename of model saved in main script '
    #              'The file must be in direcotry specified with exp-path '
    #              'Default: %(default)s'
    #        )

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

    # Batch size
    parser.add_argument(
            '--batch_size',
            type=int,
            default=12,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
