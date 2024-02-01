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

import copy
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


# These generate multiple baselines, so have to be handled differently
MULTI_BASELINE_LIST = ['random_sample', 'random_gen_sample', 
                       'random_gen_weighted_sample', 'random_gen_pop_weighted_sample',
                       'random_sample_YRI', 'random_sample_CEU', 'random_sample_JPT']

def name_attr_corr_title(attr_method,
                          corruption_style,
                          attribution_score,
                          baseline_style,
                          fold,
                          baseline_sample_size=104,
                          random_baseline=False, 
                          reverse_baseline=False):
    #title = '{} corruption={} {} bl={}'.format(attr_method,
    #                                             corruption_style, 
    #                                             attribution_score, 
    #                                             baseline_style)
    #if reverse_baseline:
    #    title += ' -- rev bl'
    #if random_baseline:
    #    title += ' -- random bl'
    
    results_filename_pre = 'snp_corr_exp_fold_'+str(fold) + \
    '_'+str(attr_method) + \
    '_'+str(corruption_style) + \
    '_'+str(attribution_score)+ \
    '_'+str(baseline_style)

    if random_baseline:
        results_filename_pre += '_random'
    if reverse_baseline:
        results_filename_pre += '_reverse'
    if (not random_baseline) and (not reverse_baseline):
        results_filename_pre += '_attrbased'

    results_filename = results_filename_pre + '_'+str(baseline_sample_size) + '.txt' # dont include info about thresholds anymore!
    return results_filename

def get_task_handler(task, dataset):
    # ----------------------------------------
    #               TASK HANDLER
    # ----------------------------------------
    # Task : clasification or regression

    if task == 'classification':
        criterion = nn.CrossEntropyLoss()
        task_handler = ClassificationHandler(dataset, 
                                             criterion)

    elif task == 'regression':
        criterion = nn.MSELoss()
        task_handler = RegressionHandler(dataset, 
                                         criterion)
    return task_handler


def load_data(fold,
              partition,
              input_features_stats,
              dataset,
              task_handler=None,
              device='cpu',
              mode='test'):
    # ---------------
    #      DATA
    # ---------------
    print('\n---\nLoading data')

    # Fold indices
    indices_byfold = np.load(partition, allow_pickle=True)
    fold_indices = indices_byfold['folds_indexes'][fold]

    # Input features statistics
    inp_feat_stats = np.load(input_features_stats)

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
    du.FoldDataset.dataset_file = dataset
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')
    # This tells what label to load in getitem
    if task_handler is None:
        # ignore this step if not loading model
        pass
    else:
        du.FoldDataset.task_handler = task_handler

    # Load data
    # xs
    du.FoldDataset.data_x = np.array(
        du.FoldDataset.f['inputs'],
        dtype=np.int8)

    # keep a (deep) copy of original for 
    # attribution validation experiments
    du.FoldDataset.data_x_original = np.array(
        du.FoldDataset.f['inputs'],
        dtype=np.int8)

    # ys
    if task_handler is None:
        # assume classification if not loading model!
        du.FoldDataset.data_y = np.array(
            du.FoldDataset.f['class_labels'],
            dtype=np.int64)
    else:
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

    # What subset of data will we use
    if mode == 'test':
        subset = fold_indices[2]
    elif mode == 'train':
        subset = fold_indices[0]
    elif mode == 'valid':
        subset = fold_indices[1]
    elif mode == 'train+valid':
        subset = fold_indices[0] + fold_indices[1]
    else:
        subset = fold_indices[0] + fold_indices[1] + fold_indices[2]
    test_set = du.FoldDataset(subset)

    #print('Loaded train ({} samples), valid ({} samples) and '
    #      'test ({} samples) sets'.format(
    #          len(train_set), len(valid_set), len(test_set)))

    print('---\n')

    return du, test_set, fold_indices, mus, sigmas


def load_model(model,
               task_handler,
               embedding,
               device,
               dataset,
               config,
               fold,
               exp_path,
               exp_name,
               model_params=None,
               param_init=None):
     

    # Model architecture (Dietnet or Mlp)
    if model == 'Dietnet':
        model_handler = DietNetworkHandler(task_handler, 
                                           fold,
                                           embedding, 
                                           device, 
                                           dataset, 
                                           config, 
                                           param_init)
    elif model == 'Mlp':
        model_handler = MlpHandler(task_handler, 
                                   dataset, 
                                   config)
    else:
        raise Exception('{} is not a recognized model'.format(model))

    # Send model to GPU
    model_handler.model.to(device)

    print('\nModel:', model_handler.model)

    # Where to load model
    if model_params:
        bestmodel_fullpath = model_params
        exp_identifier = None
        results_fullpath = None
    else:
        exp_identifier = model_handler.get_exp_identifier(config, 
                                                          fold)

        results_dirname = 'RESULTS_' + exp_identifier
        results_fullpath = os.path.join(exp_path,
                                        exp_name, 
                                        results_dirname)

        #lu.create_dir(results_fullpath)

        # Monitoring best and last models
        bestmodel_fullpath = os.path.join(results_fullpath, 
                                          'best_model.pt')

    if not torch.cuda.is_available():
        checkpoint = torch.load(bestmodel_fullpath, 
                                map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(bestmodel_fullpath)

    model_handler.model.load_state_dict(checkpoint['model_state_dict'])
    print('\nLoaded model parameters from {} at epoch {}'.format(
          bestmodel_fullpath, checkpoint['epoch']))
    return model_handler, results_fullpath, checkpoint


def get_highest_entropy_datapoint(fold, 
                                  partition, 
                                  input_features_stats, 
                                  dataset, 
                                  task_handler,
                                  config,
                                  device,
                                  mus,
                                  sigmas,
                                  model_handler):
    du, test_set, fold_indices, mus, sigmas = load_data(fold,
                                                        partition,
                                                        input_features_stats,
                                                        dataset,
                                                        task_handler,
                                                        mode='train+valid')

    test_generator = DataLoader(test_set,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=0)
    du.FoldDataset.data_x = du.FoldDataset.data_x_original.copy()
    model_handler.model.eval()
    model_handler.model.to(device) # send to device

    test_results = mlu.eval_step(model_handler,
                                 device,
                                 test_set,
                                 test_generator,
                                 mus.to(device), sigmas.to(device), True,
                                 '', 'test_step_all_reference',
                                 scale=1.)
    entropies = ((-np.log(test_results['scores']))*test_results['scores']).sum(1)
    return test_set[entropies.argmax()][0]


def compute_allele_freqs(data_x):
    return np.array([(data_x==0).mean(0), 
                     (data_x==1).mean(0), 
                     (data_x==2).mean(0)])


def make_baseline(mus, 
                  sigmas, 
                  normalize, 
                  train_valid_set, 
                  train_valid_set_labels=None,
                  kind='reference',
                  fold=None, 
                  partition=None, 
                  input_features_stats=None, 
                  dataset=None, 
                  task_handler=None,
                  config=None,
                  device=None,
                  model_handler=None,
                  process_baseline=True,
                  sample_size=104):

    # create baseline
    if kind == 'reference':
        #baseline = torch.zeros_like(x_dset.min(0).values.view(1,-1))
        baseline = np.zeros_like(mus.reshape(1,-1))
    elif kind == 'missing':
        #baseline = torch.zeros_like(x_dset.min(0).values.view(1,-1)) - 1
        baseline = np.zeros_like(mus.reshape(1,-1)) - 1.
    elif kind == 'hetero':
        #baseline = torch.zeros_like(x_dset.min(0).values.view(1,-1)) - 1
        baseline = np.zeros_like(mus.reshape(1,-1)) + 1.
    elif kind == 'uniform':
        baseline = get_highest_entropy_datapoint(fold, 
                                                 partition, 
                                                 input_features_stats, 
                                                 dataset, 
                                                 task_handler,
                                                 config,
                                                 device,
                                                 mus,
                                                 sigmas,
                                                 model_handler).reshape(1,-1)
    elif kind == 'random_gen':
        # generates a random genotype to use as baseline
        # 0,1,2 has equal probability        
        baseline = np.random.choice([0,1,2], size=(1, len(mus)))
    elif kind == 'random_gen_weighted':
        # generates a random genotype to use as baseline
        # weights 0,1,2 according to empirical distribution
        ps = compute_allele_freqs(du.FoldDataset.data_x)
        baseline = np.array([np.random.choice([0,1,2], 
                                              size=1, 
                                              p=ps[:,i]) for i in range(ps.shape[1])]).T
    elif kind == 'random_gen_sample':
        # this one generates multiple random baselines (unweighted)
        baseline = np.random.choice([0,1,2], 
                                    size=(sample_size, 
                                          len(mus)))
    elif kind == 'random_gen_weighted_sample':
        # this one generates multiple random baselines (weighted)
        ps = compute_allele_freqs(du.FoldDataset.data_x)
        baseline = np.array([np.random.choice([0,1,2], 
                                              size=sample_size, 
                                              p=ps[:,i]) for i in range(ps.shape[1])]).T
    elif kind == 'random_gen_pop_weighted_sample':
        baseline = []
        # this one generates multiple random baselines (weighted)
        categories = np.unique(du.FoldDataset.data_y)
        assert sample_size <= (len(du.FoldDataset.data_x) - len(train_valid_set_labels)), 'Sample size requested is > then number of test points!'
        assert sample_size % categories  == 0, 'sample size not a multiple of  number of categories'
        sample_size_per_pop = sample_size // categories
        for label in categories:
            subset = du.FoldDataset.data_x[du.FoldDataset.data_y==label]
            ps = compute_allele_freqs(subset)
            baseline_pop = np.array([np.random.choice([0,1,2], 
                                                      size=sample_size_per_pop, 
                                                      p=ps[:,i]) for i in range(ps.shape[1])]).T
            baseline.append(baseline_pop)
        baseline = np.vstack(baseline)
        
    elif kind == 'random_sample':
        # sample 100 random points from training+validation set
        baseline = train_valid_set[np.random.choice(len(train_valid_set), 
                                                    sample_size, 
                                                    replace=False)]
    elif kind == 'random_sample_YRI':
        indices = np.arange(len(train_valid_set_labels))[train_valid_set_labels == 25]
        baseline = train_valid_set[np.random.choice(indices, 
                                                    sample_size, 
                                                    replace=False)]
    elif kind == 'random_sample_CEU':
        indices = np.arange(len(train_valid_set_labels))[train_valid_set_labels == 4]
        baseline = train_valid_set[np.random.choice(indices, 
                                                    sample_size, 
                                                    replace=False)]
    elif kind == 'random_sample_JPT':
        indices = np.arange(len(train_valid_set_labels))[train_valid_set_labels == 15]
        baseline = train_valid_set[np.random.choice(indices, 
                                                    sample_size, 
                                                    replace=True)] # need to bootstrap since < 100 samples!
    else:
        raise Exception("kind {} not valid!".format(kind))

    if process_baseline:
        # process baseline
        baseline = baseline.astype(np.float32)

        # Replace missing values
        du.replace_missing_values(baseline, mus)

        # Normalize
        if normalize:
            baseline = du.normalize(baseline, mus, sigmas)

    return baseline
    #return x_dset, baseline
    #return full_dset_raw, baseline


def make_feature_mask(test_set, how='equal_space'):
    if how == 'equal_space':
        # this feature mask makes nearby SNPs have the same attribution values
        nb_feats = int(np.ceil(test_set.data_x.shape[1]/10000)) # captum suggests < 10k features
        feature_mask = torch.arange(10000, dtype=torch.long)
        feature_mask = feature_mask.repeat_interleave(nb_feats)
        feature_mask = feature_mask[:test_set.data_x.shape[1]]
    elif how == 'exactly_10k':
        # or can get 10000 equally sized features and assign the rest randomly...
        nb_feats = int(np.floor(test_set.data_x.shape[1]/10000)) # captum suggests < 10k features
        feature_mask = torch.arange(10000, dtype=torch.long)
        feature_mask = feature_mask.repeat_interleave(nb_feats)
        rest = np.random.choice(np.arange(10000), test_set.data_x.shape[1]-feature_mask.shape[0], replace=False)
        feature_mask = torch.cat([feature_mask, torch.from_numpy(rest)])
    elif how =='random':
        # or can just pick totally randomly
        feature_mask = torch.from_numpy(np.random.choice(10000, test_set.data_x.shape))
    return feature_mask


def name_attr_file(attr_method, baseline_style, baseline_sample_size, agg):
    if agg:
        prefix = 'attrs_avg'
    else:
        prefix = 'attrs'

    if baseline_style in MULTI_BASELINE_LIST:
        return '{}_{}_{}_{}.h5'.format(prefix, attr_method, baseline_style, baseline_sample_size)
    else:
        return '{}_{}_{}.h5'.format(prefix, attr_method, baseline_style)
        

def main():
    args = parse_args()


    # ---------------
    # Loading config
    # ---------------
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
    task_handler = get_task_handler(task, args.dataset)

    # ----------------------------------------
    #                   DATA
    # ----------------------------------------
    du, test_set, fold_indices, mus, sigmas = load_data(args.which_fold,
                                                        args.partition,
                                                        args.input_features_stats,
                                                        args.dataset,
                                                        task_handler,
                                                        device)

    # ----------------------------------------
    #                 MODEL
    # ----------------------------------------
    print('\n---\nInitializing model')

    # Model architecture (Dietnet or Mlp)
    model_handler, results_fullpath, checkpoint = load_model(args.model,
                                                           task_handler,
                                                           args.embedding,
                                                           device,
                                                           args.dataset,
                                                           config,
                                                           args.which_fold,
                                                           args.exp_path,
                                                           args.exp_name)

    # ----------------------------------------
    #          TRAINING LOOP SET UP
    # ----------------------------------------

    # make copy of unmodified test set
    genotypes_data = copy.deepcopy(test_set.data_x)
    genotypes_data_tv = copy.deepcopy(test_set.data_x)
    genotypes_data = genotypes_data[fold_indices[2]]

    # Convert to float32 and normalize (so Saliency will work!)
    # process x_dset
    test_set.data_x = test_set.data_x.astype(np.float32)

    # Replace missing values
    du.replace_missing_values(test_set.data_x, 
                              mus.cpu().numpy())

    # Normalize
    if args.normalize:
        test_set.data_x = du.normalize(test_set.data_x, 
                                       mus.cpu().numpy(), 
                                       sigmas.cpu().numpy())

    # Batch generators
    batch_size = args.batch_size #config['batch_size'] is too big!
    if (args.attr_method == 'Lime'):
        batch_size = 1 # this should be 1 for Lime!
        print('Changing batch size to 1 for Lime')
    
    #train_generator = DataLoader(train_set, shuffle=True,
    #        batch_size=batch_size, num_workers=0, drop_last=True)

    #valid_generator = DataLoader(valid_set,
    #        batch_size=batch_size, shuffle=False, num_workers=0)

    test_generator = DataLoader(test_set,
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=0)

    print('---\n')

    # ----------------------------------------
    #                   TEST
    # ----------------------------------------

    # change targets from ground truth to model predictions (for attribution computation)
    test_results = np.load(os.path.join(results_fullpath, 
                                        'test_results_epoch{}.npz'.format(checkpoint['epoch'])
                                       )
                          )
    pred_idxs = test_results['preds'].astype('int')
    # super hacky. Just replace targets with predictions for test set. 
    # This will cause the attribution to be w.r.t. predicted class (instead of ground truth class).
    test_set.data_y[fold_indices[2]] = pred_idxs

    # ----------------------------------------
    #               Attributions
    # ----------------------------------------
    out_dir = results_fullpath # to be consistent with the old code!

    model_handler.get_attribution_model()
    model_handler.model_attr.to(device) # send to device
    model_handler.model_attr = model_handler.model_attr.eval()

    baseline = make_baseline(mus.cpu().numpy(), 
                             sigmas.cpu().numpy(), 
                             args.normalize,
                             genotypes_data_tv[fold_indices[0]+fold_indices[1]],
                             test_set.data_y[fold_indices[0]+fold_indices[1]],
                             args.baseline_style,
                             args.which_fold, 
                             args.partition,
                             args.input_features_stats,
                             args.dataset,
                             task_handler,
                             config,
                             device,
                             model_handler,
                             True,
                             args.baseline_sample_size)

    baseline = torch.from_numpy(baseline).to(device)

    attr_manager = am.AttributionManager()

    attr_manager.set_model(model_handler.model_attr)
    attr_manager.init_attribution_function(attr_type=args.attr_method, 
                                           backend='captum')

    attr_manager.set_data_generator(test_generator)

    attr_manager.set_genotypes_data(torch.from_numpy(genotypes_data))
    
    fname_file = name_attr_file(args.attr_method, args.baseline_style, args.baseline_sample_size, agg=False)
    attr_manager.set_raw_attributions_file(os.path.join(out_dir, fname_file))

    attr_manager.set_device(device)

    if args.attr_method == 'int_grad':
        attrs_args = {'baselines': baseline,
                      'n_steps': 75, # 100 too slow?
                      'method': 'riemann_left',
                      'return_convergence_delta': True}
    elif args.attr_method == 'saliency':
        attrs_args = {}
    elif args.attr_method == 'feat_ablation':
        # make feature mask for permutation based methods
        feature_mask = make_feature_mask(test_set, how='equal_space')
        attrs_args = {'show_progress': True,
                      'baselines': baseline,
                      'feature_mask': feature_mask}
    elif args.attr_method == 'DeepLift':
        attrs_args = {}
    elif args.attr_method == 'Lime':
        # make feature mask for permutation based methods
        feature_mask = make_feature_mask(test_set, how='equal_space')
        attrs_args = {'show_progress': True,
                      'baselines': baseline,
                      'feature_mask': feature_mask}

    attr_manager.create_raw_attributions(False,
                                         only_true_labels=False, # remove attrs w.r.t. incorrect labels next step!
                                         **attrs_args)

    out = attr_manager.get_attribution_average(use_true_class_only=args.only_true_labels)
    fname_file = name_attr_file(args.attr_method, args.baseline_style, args.baseline_sample_size, agg=True)
    fname = os.path.join(out_dir, fname_file)
    attr_manager.set_raw_attributions_file(fname)
    with h5py.File(fname, 'w') as hf:
        hf['avg_attr'] = out.cpu().numpy()
        print('Saved attribution averages to {}'.format(fname))


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
            '--attr-method',
            type=str,
            choices=['int_grad', 'saliency', 'feat_ablation', 'DeepLift', 'Lime'],
            default='int_grad',
            help='Attribution Method to use. Default: %(default)s'
            )

    parser.add_argument(
            '--baseline-style',
            type=str,
            choices=['random_sample', 'reference', 'missing', 
                     'uniform', 'random_gen', 'random_gen_weighted',
                     'random_gen_sample', 'random_gen_weighted_sample', 
                     'random_gen_pop_weighted_sample',
                     'random_sample_YRI', 
                     'random_sample_CEU', 'random_sample_JPT'],
            default='reference',
            help='Should baseline be all reference or all missing. Default: %(default)s'
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
            required=True,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    # Batch size
    parser.add_argument(
            '--batch_size',
            type=int,
            default=12,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    # Number of Baselines to Use (When using Multiple)
    parser.add_argument(
            '--baseline_sample_size',
            type=int,
            default=104,
            help='Number of baseline to use (when using multiple baselines). Default: %(default)i'
            )

    parser.add_argument(
            '--only_true_labels',
            action='store_true',
            help='Compute attribution averages with respect to labels given to AttributionManager'
            )


    return parser.parse_args()


if __name__ == '__main__':
    main()
