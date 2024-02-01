import argparse
import math
import os
import pprint
import yaml
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/Dietnet')
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

    # Send model to GPU
    model_handler.model.to(device)

    print('\nModel:', model_handler.model)

    # Where to load model
    if args.model_params:
        bestmodel_fullpath = args.model_params
    else:
        exp_identifier = model_handler.get_exp_identifier(config, fold)

        results_dirname = 'RESULTS_' + exp_identifier
        results_fullpath = os.path.join(args.exp_path,
                args.exp_name, results_dirname)

        #lu.create_dir(results_fullpath)

        # Monitoring best and last models
        bestmodel_fullpath = os.path.join(results_fullpath, 'best_model.pt')
    
    if not torch.cuda.is_available():
        checkpoint = torch.load(bestmodel_fullpath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(bestmodel_fullpath)

    #model_handler.model.load_state_dict(checkpoint['model_state_dict'])
    #print('\nLoaded model parameters from {} at epoch {}'.format(
    #      args.model_params, checkpoint['epoch']))
    
    bestmodel_fullpath2 = '/lustre07/scratch/sciclun4/results/dietnet_attr_exp/kepler_exps/model_params.pt'
    checkpoint2 = torch.load(bestmodel_fullpath2, map_location=torch.device('cpu'))

    assert checkpoint['model_state_dict']['aux_net.hidden_layers.0.weight'].shape == checkpoint2['feat_emb.hidden_1.weight'].shape
    assert checkpoint['model_state_dict']['aux_net.hidden_layers.1.weight'].shape == checkpoint2['feat_emb.hidden_2.weight'].shape
    assert checkpoint['model_state_dict']['main_net.fat_bias'].shape == checkpoint2['disc_net.fat_bias'].shape
    assert checkpoint['model_state_dict']['main_net.bn_fatLayer.weight'].shape == checkpoint2['disc_net.bn1.weight'].shape    
    assert checkpoint['model_state_dict']['main_net.bn_fatLayer.bias'].shape == checkpoint2['disc_net.bn1.bias'].shape
    assert checkpoint['model_state_dict']['main_net.bn_fatLayer.running_mean'].shape == checkpoint2['disc_net.bn1.running_mean'].shape
    assert checkpoint['model_state_dict']['main_net.bn_fatLayer.running_var'].shape == checkpoint2['disc_net.bn1.running_var'].shape
    assert checkpoint['model_state_dict']['main_net.bn_fatLayer.num_batches_tracked'].shape == checkpoint2['disc_net.bn1.num_batches_tracked'].shape
    assert checkpoint['model_state_dict']['main_net.out.weight'].shape == checkpoint2['disc_net.out.weight'].shape
    assert checkpoint['model_state_dict']['main_net.out.bias'].shape == checkpoint2['disc_net.out.bias'].shape
    assert checkpoint['model_state_dict']['main_net.bn.0.weight'].shape == checkpoint2['disc_net.bn2.weight'].shape
    assert checkpoint['model_state_dict']['main_net.bn.0.bias'].shape == checkpoint2['disc_net.bn2.bias'].shape
    assert checkpoint['model_state_dict']['main_net.bn.0.running_mean'].shape == checkpoint2['disc_net.bn2.running_mean'].shape
    assert checkpoint['model_state_dict']['main_net.bn.0.running_var'].shape == checkpoint2['disc_net.bn2.running_var'].shape
    assert checkpoint['model_state_dict']['main_net.bn.0.num_batches_tracked'].shape == checkpoint2['disc_net.bn2.num_batches_tracked'].shape
    assert checkpoint['model_state_dict']['main_net.hidden_layers.0.weight'].shape == checkpoint2['disc_net.hidden_2.weight'].shape
    assert checkpoint['model_state_dict']['main_net.hidden_layers.0.bias'].shape == checkpoint2['disc_net.hidden_2.bias'].shape 

    checkpoint2['aux_net.hidden_layers.0.weight'] = checkpoint2.pop('feat_emb.hidden_1.weight')
    checkpoint2['aux_net.hidden_layers.1.weight'] = checkpoint2.pop('feat_emb.hidden_2.weight')
    checkpoint2['main_net.fat_bias'] = checkpoint2.pop('disc_net.fat_bias')
    checkpoint2['main_net.bn_fatLayer.weight'] = checkpoint2.pop('disc_net.bn1.weight')
    checkpoint2['main_net.bn_fatLayer.bias'] = checkpoint2.pop('disc_net.bn1.bias')
    checkpoint2['main_net.bn_fatLayer.running_mean'] = checkpoint2.pop('disc_net.bn1.running_mean')
    checkpoint2['main_net.bn_fatLayer.running_var'] = checkpoint2.pop('disc_net.bn1.running_var')
    checkpoint2['main_net.bn_fatLayer.num_batches_tracked'] = checkpoint2.pop('disc_net.bn1.num_batches_tracked')
    checkpoint2['main_net.out.weight'] = checkpoint2.pop('disc_net.out.weight')
    checkpoint2['main_net.out.bias'] = checkpoint2.pop('disc_net.out.bias')
    checkpoint2['main_net.bn.0.weight'] = checkpoint2.pop('disc_net.bn2.weight')
    checkpoint2['main_net.bn.0.bias'] = checkpoint2.pop('disc_net.bn2.bias')
    checkpoint2['main_net.bn.0.running_mean'] = checkpoint2.pop('disc_net.bn2.running_mean')
    checkpoint2['main_net.bn.0.running_var'] = checkpoint2.pop('disc_net.bn2.running_var')
    checkpoint2['main_net.bn.0.num_batches_tracked'] = checkpoint2.pop('disc_net.bn2.num_batches_tracked')
    checkpoint2['main_net.hidden_layers.0.weight'] = checkpoint2.pop('disc_net.hidden_2.weight')
    checkpoint2['main_net.hidden_layers.0.bias'] = checkpoint2.pop('disc_net.hidden_2.bias')

    checkpoint3 = {}
    checkpoint3['model_state_dict'] = checkpoint2
    checkpoint3['epoch'] = -1
    checkpoint3['best_results'] = -1

    model_handler.model.load_state_dict(checkpoint3['model_state_dict'])
    
    print('---\n')
    
    # Load attributions
    #with h5py.File(os.path.join(results_fullpath, 'attrs_avg_{}.h5'.format(args.attr_method)), 'r') as hf:
    #    agg_attr = hf['avg_attr'][:]
    #    print('loaded attribution averages from {}'.format(results_fullpath, 'attrs_avg_{}.h5'.format(args.attr_method)))
    # we take the absolute value of the (26) class/target averages per SNP
    # we then take the max of these. This is our attribution based "score" per SNP
    with h5py.File(os.path.join('/lustre07/scratch/sciclun4/results/dietnet_attr_exp/kepler_exps', 'attrs_avg.h5'), 'r') as hf:
        agg_attr = hf['avg_attr'][:]

    abs_attr = np.amax(np.abs(np.nan_to_num(agg_attr)), axis=(1,2))

    # Test step
    model_handler.model.eval()
    #model_handler.get_attribution_model()
    #model_handler.model_attr.eval()
    

    #------------------------------
    # Missing data simulation loop
    #------------------------------
    print('\n---\nMissing data simulation\n')
    
    miss_percent_list = [0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    nb_feats = du.FoldDataset.data_x_original.shape[1]
    
    # Total nb of exepriments : we want same nb of exeperiments for all
    # miss_percent in miss_percent_list. First batch of experiments is based
    # on the nb of snp_groups (created so that every snp is removed or kept
    # once) and we add experiments to match the total nb of experiments
    #min_percent = min(miss_percent_list)
    #max_percent = max(miss_percent_list)
    
    # We divide the number of SNPs by the nb of snp_groups and ceiling
    # this number because we will do an extra group with the remaining 
    # never use SNPs and random reselected SNPs.
    #print('Nb snps in group min percent:', int(math.floor(nb_feats)*min_percent))
    #print('Nb snps in group max percent:', int(math.floor(nb_feats)*(1-max_percent)))
    total_exp = len(miss_percent_list)
    #total_exp = max(
    #    math.ceil(nb_feats/(int(math.floor(nb_feats)*min_percent))),
    #    math.ceil(nb_feats/(int(math.floor(nb_feats)*(1-max_percent))))
    #)

    print('Total nb of experiments:', total_exp, '\n')

    # For saving data
    if task == 'classification':
        f_str = ""
        for i in miss_percent_list:
            f_str += '_'+str(i)
        results_filename = 'attr_based_missing_data_simulations_fold_'+str(fold)+f_str+'.txt'
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

            # Make the groups of SNPs to keep
            snp_indices = np.argsort(abs_attr)

        else:
            # How many SNPs to remove in an experiment
            group_size = int(math.floor(nb_feats)*miss_percent)
            # Scale is used to increase info of non-missing genotypes by a 
            # factor = nb_snps/nb_non_missing_snps
            scale = nb_feats / (nb_feats-group_size)
            # If true, remove the SNPs, if False keep the SNPs
            to_remove = True

            # Make the groups of SNPs to keep
            snp_indices = np.argsort(abs_attr)[::-1]

        #snp_indices = snp_indices[::-1]
        # Sanity check
        #print('shuffling indices')
        #snp_indices = np.random.permutation(np.arange(abs_attr.shape[0]))

        print('Scale:', scale)

        all_snp_groups = [snp_indices[0: 0 + group_size], snp_indices[group_size: ]]
        #last_snp_group = all_snp_groups[-1]
        snp_groups = [snp_indices[0: 0 + group_size], snp_indices[group_size: ]][:-1]

        print('\nNb of groups:', len(snp_groups))
        print('First group size:', len(snp_groups[0]))
        print('Last (complete) group size:', len(snp_groups[-1]))

        # Complete snp_groups to make equal nb of experiments for
        # the different miss_percent
        #nb_exp_to_do = total_exp - len(snp_groups)
        nb_exp_to_do = 1

        # Iterate over snp_groups to remove SNPs and make test step in model
        for snp_group in snp_groups:
            du.FoldDataset.data_x = du.FoldDataset.data_x_original.copy()
            # Remove SNPs in snp_group
            if to_remove:
                print(abs_attr[snp_group].min() >= abs_attr[all_snp_groups[1]].max(), snp_group.shape[0])
                du.FoldDataset.data_x[:,snp_group] = -1
            # Keep SNPs in snp_group (remove the rest)
            else:
                snps_to_remove = list(set(snp_indices) - set(snp_group))
                print(abs_attr[np.array(snps_to_remove)].min() >= abs_attr[snp_group].max(), len(snps_to_remove))        
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

    parser.add_argument(
            '--exp-path',
            type=str,
            help='Path to directory where results were saved. Used to load model'
            )

    parser.add_argument(
            '--exp-name',
            type=str,
            help=('Name of directory of exp-path where results were written '
                  'Used to load model')
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
    
    # Results path
    parser.add_argument(
            '--results-path',
            required=True,
            help='Where to save the results'
            )

    parser.add_argument(
            '--attr-method',
            type=str,
            choices=['int_grad'],
            default='IntGrad',
            help='Attribution Method to use. Default: %(default)s'
            )
    
    return parser.parse_args()


if __name__ == '__main__':
    main()