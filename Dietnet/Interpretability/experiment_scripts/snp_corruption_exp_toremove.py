import argparse
import math
import os
import pprint
import yaml
import sys
import copy

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append('/lustre06/project/6065672/sciclun4/ActiveProjects/DIETNETWORK/Dietnet')
import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu
from make_attributions import get_task_handler, load_data, load_model


# helper functions
def load_attributions(attribution_score, 
                      results_fullpath, 
                      checkpoint, 
                      attr_method, 
                      baseline_style,
                      load_all_attrs=False):
    # Load averaged attributions
    if (attribution_score != 'none') and (attribution_score != 'none_raw'):
        with h5py.File(os.path.join(results_fullpath, 
                                    'attrs_avg_{}_{}.h5'.format(attr_method, 
                                                                baseline_style)), 'r') as hf:
            attrs = hf['avg_attr'][:]
            print('loaded attribution averages from {}'.format(results_fullpath,
                                                               'attrs_avg_{}_{}.h5'.format(attr_method,
                                                                                           baseline_style)))
    # load attributions
    else:
        with h5py.File(os.path.join(results_fullpath, 
                                    'attrs_{}_{}.h5'.format(attr_method, 
                                                            baseline_style)), 'r') as hf:
            attrs = hf[attr_method][:]
        
        if not load_all_attrs:
            test_results = np.load(os.path.join(
                results_fullpath, 
                'test_results_epoch{}.npz'.format(checkpoint['epoch'])
            )
                                  )
            pred_idxs = test_results['preds'].astype('int')

            # Use advanced indexing to select values from attrs using pred_idxs
            attrs = attrs[np.arange(attrs.shape[0])[:, np.newaxis], :, pred_idxs[:, np.newaxis]].squeeze()

        print('loaded attributions from {}'.format(results_fullpath, 
                                                   'attr_{}_{}.h5'.format(attr_method, 
                                                                          baseline_style)))
    return attrs


def compute_score(score_style, 
                  agg_attr):

    if score_style == 'old':
        # we take the absolute value of the (26) class/target averages per SNP
        # we then take the max of these. This is our attribution based "score" per SNP
        score = np.amax(np.abs(np.nan_to_num(agg_attr)), axis=(1,2))
    elif score_style == 'new':
        score = np.amax(np.nan_to_num(agg_attr), axis=(1,2))
    elif score_style == 'none':
        # takes absolute values
        score = np.abs(agg_attr)
    elif score_style == 'none_raw':
        # including sign of attribution
        score = agg_attr
    else:
        raise Exception
    
    return score


def compute_score_indices(score, to_remove):
    # for feature scores
    # i.e. all samples have the same indices
    if len(score.shape) == 1:
        if to_remove:
            snp_indices = np.argsort(score)[::-1]
        else:
            snp_indices = np.argsort(score)
    # for attributions.
    # i.e. each sample gets its own indices
    elif len(score.shape) == 2:
        if to_remove:
            snp_indices = np.argsort(score, axis=1)[:, ::-1]
        else:
            snp_indices = np.argsort(score, axis=1)
    return snp_indices


def create_random_indices(scores, 
                          to_remove, 
                          random_style='random_incl_baseline'):
    print('shuffling indices (sanity check)')
    if len(scores.shape) == 1:
        snp_indices = np.random.permutation(np.arange(scores.shape[0]))
    elif len(scores.shape) == 2:
        if random_style == 'random_incl_baseline':
            # keep 0 attributions the same.
            # make pseudo data (with baseline attrs set to 0)
            # use this if using (local) attributions without sign (abs value)
            snp_indices = np.stack([np.concatenate([np.random.permutation(np.arange(len(score))[score==0]),
                                                    np.random.permutation(np.arange(len(score))[score!=0])
                                                   ]) for score in scores])
        elif random_style == 'random_incl_baseline_and_sign':
            # use this if using (local) attributions with sign
            snp_indices = np.stack([np.concatenate([np.random.permutation(np.arange(len(score))[score<0]),
                                                    np.random.permutation(np.arange(len(score))[score==0]),
                                                    np.random.permutation(np.arange(len(score))[score>0])
                                                   ]) for score in scores])
        else:
            # completely random order of SNPs
            snp_indices = np.stack([np.random.permutation(np.arange(scores.shape[1])) 
                                    for _ in range(scores.shape[0])])

        # if removing the selected features, we reverse the indices order!
        if to_remove:
            snp_indices = snp_indices[:, ::-1]

    return snp_indices

def make_corrupt_dataset(du, 
                         snp_group, 
                         snp_indices, 
                         fold_indices, 
                         to_remove, 
                         corruption_style):

    du.FoldDataset.data_x = du.FoldDataset.data_x_original.copy()

    # Remove SNPs in snp_group
    if to_remove:
        snps_to_remove = snp_group
    else:
        if len(snp_group.shape) == 1:
            snps_to_remove = np.array(list(set(snp_indices) - set(snp_group)))
        else:
            snps_to_remove = np.array(
                [list(set(snp_indices[i]) - set(snp_group[i])) 
                 for i in range(snp_indices.shape[0])])

    # generate corrupted values!
    corrupted_vals = generate_corrupted_values(corruption_style,
                                               snps_to_remove,
                                               du.FoldDataset.data_x,
                                               fold_indices)

    # set values of test data to corrupted values!
    du.FoldDataset.data_x[np.array(fold_indices[2])[:, np.newaxis], 
                          snps_to_remove] = corrupted_vals


def generate_corrupted_values(corruption_style, snps_to_remove, data, fold_indices):
    if corruption_style == 'missing':
        return -1 # -1 = missing
    elif corruption_style == 'permute':
        vals_to_substitute = copy.deepcopy(data)
        for col in range(vals_to_substitute.shape[1]):
            # shuffle values for each column (using the entire dataset!)
            vals_to_substitute[:, col] = np.random.permutation(vals_to_substitute[:, col])
        return vals_to_substitute[np.array(fold_indices[2])[:, np.newaxis], snps_to_remove]


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
    task_handler = get_task_handler(task, args.dataset)
    
    # ---------------
    #      DATA
    # ---------------
    du, test_set, fold_indices, mus, sigmas = load_data(args.which_fold,
                                                        args.partition,
                                                        args.input_features_stats,
                                                        args.dataset,
                                                        task_handler,
                                                        device)

    # ----------------------------------------
    #                 MODEL
    # ----------------------------------------
    param_init = None
    
    model_handler, results_fullpath, checkpoint = load_model(args.model,
                                                           task_handler,
                                                           args.embedding,
                                                           device,
                                                           args.dataset,
                                                           config,
                                                           args.which_fold,
                                                           args.exp_path,
                                                           args.exp_name,
                                                           args.model_params,
                                                           param_init)
    print('---\n')
    
    # Load attributions
    attrs = load_attributions(args.attribution_score, 
                              results_fullpath, 
                              checkpoint, 
                              args.attr_method, 
                              args.baseline_style)
    if args.random_baseline:
        # indices are randomly generated. Just pass attrs so it knows where the 0 vals are!
        scores = attrs
    else:
        scores = compute_score(args.attribution_score, 
                               attrs)

    # Test step
    model_handler.model.eval()
    #model_handler.get_attribution_model()
    #model_handler.model_attr.eval()
    
    # get label names
    with h5py.File(os.path.join(args.exp_path, args.dataset), "r") as f:
        label_names = f['class_label_names'][:]
    #------------------------------
    # Missing data simulation loop
    #------------------------------
    print('\n---\nMissing data simulation\n')
    
    corr_percent_list = args.corr_percent_list
    nb_feats = du.FoldDataset.data_x_original.shape[1]
    
    # Total nb of exepriments : we want same nb of exeperiments for all
    # corruption_percent in corr_percent_list. First batch of experiments is based
    # on the nb of snp_groups (created so that every snp is removed or kept
    # once) and we add experiments to match the total nb of experiments
    #min_percent = min(corr_percent_list)
    #max_percent = max(corr_percent_list)
    
    # We divide the number of SNPs by the nb of snp_groups and ceiling
    # this number because we will do an extra group with the remaining 
    # never use SNPs and random reselected SNPs.
    #print('Nb snps in group min percent:', int(math.floor(nb_feats)*min_percent))
    #print('Nb snps in group max percent:', int(math.floor(nb_feats)*(1-max_percent)))
    total_exp = len(corr_percent_list)
    #total_exp = max(
    #    math.ceil(nb_feats/(int(math.floor(nb_feats)*min_percent))),
    #    math.ceil(nb_feats/(int(math.floor(nb_feats)*(1-max_percent))))
    #)

    print('Total nb of experiments:', total_exp, '\n')

    # For saving data
    if task == 'classification':
        f_str = ""
        for i in corr_percent_list:
            f_str += '_'+str(i)
        #results_filename = 'attr_based_missing_data_simulations_fold_'+str(fold)+f_str+'.txt'
        results_filename_pre = 'snp_corr_exp_fold_'+str(args.which_fold) + \
        '_'+str(args.attr_method) + \
        '_'+str(args.corruption_style) + \
        '_'+str(args.attribution_score)+ \
        '_'+str(args.baseline_style)
        
        if args.random_baseline:
            results_filename_pre += '_random'
        if args.reverse_baseline:
            results_filename_pre += '_reverse'
        if (not args.random_baseline) and (not args.reverse_baseline):
            results_filename_pre += '_attrbased'

        results_filename = results_filename_pre + '.txt' # dont include info about thresholds anymore!

        results_file = os.path.join(args.results_path, results_filename)
        f = open(results_file, 'w')
        f.write('missing\taccuracy\n')
        counts_df = [] # more detailed output

    for corruption_percent in corr_percent_list:
        print('\n***')
        print('% of missing:', corruption_percent)
        # Nb of snps to remove or to keep
        # If the % of missing is <=50% we make groups of SNPs to remove
        # If the % of missing is > 50% we make groups of SNPs to keep
        # This ensure that every (or most) SNP is kept (or remove) at least once
        if corruption_percent > 0.5:
            reverse_corruption_percent = 1 - corruption_percent
            # How many SNPs to keep in a experiment
            group_size = int(math.floor(nb_feats)*reverse_corruption_percent)
            # Scale is used to increase info of non-missing genotypes by a 
            # factor = nb_snps/nb_non_missing_snps
            scale = nb_feats / group_size
            # If true, remove the SNPs, if False keep the SNPs
            to_remove = False
        else:
            group_size = int(math.floor(nb_feats)*corruption_percent)
            scale = nb_feats / (nb_feats-group_size)
            to_remove = True
    
        if args.random_baseline:
            if args.attribution_score == 'none_raw':
                # keep sign info
                #random_style = 'random_incl_baseline'
                random_style = 'completely_random'
                #random_style = 'random_incl_baseline_and_sign' # don't include sign anymore!
            else:
                #random_style = 'random_incl_baseline'
                random_style = 'completely_random'
            snp_indices = create_random_indices(scores, 
                                                to_remove, 
                                                random_style)
        else:
            snp_indices = compute_score_indices(scores, 
                                                to_remove)

        if args.reverse_baseline:
            print('reversing indices (sanity check)')
            if len(snp_indices.shape) == 1:
                snp_indices = snp_indices[::-1]
            if len(snp_indices.shape) == 2:
                snp_indices = snp_indices[:, ::-1]
        
        if args.corruption_style == 'missing':
            pass
        elif args.corruption_style == 'permute':
            # don't use scaling here!
            scale = 1.
        print('Scale:', scale)

        if len(snp_indices.shape) == 1:
            all_snp_groups = [snp_indices[0: 0 + group_size], 
                              snp_indices[group_size: ]]
            #last_snp_group = all_snp_groups[-1]
            snp_groups = [snp_indices[0: 0 + group_size], 
                          snp_indices[group_size: ]][:-1]

            print('\nNb of groups:', len(snp_groups))
            print('First group size:', len(snp_groups[0]))
            print('Last (complete) group size:', len(snp_groups[-1]))

            # Complete snp_groups to make equal nb of experiments for
            # the different corruption_percent
            #nb_exp_to_do = total_exp - len(snp_groups)
            nb_exp_to_do = 1
        else:
            all_snp_groups = [snp_indices[:, 0: 0 + group_size], snp_indices[:, group_size: ]]
            snp_groups = [snp_indices[:, 0: 0 + group_size], snp_indices[:, group_size: ]][:-1]

        # Iterate over snp_groups to remove SNPs and make test step in model
        for snp_group in snp_groups:
            make_corrupt_dataset(du,
                                 snp_group,
                                 snp_indices,
                                 fold_indices,
                                 to_remove,
                                 args.corruption_style) 

            #print('{:.3f}'.format((test_set.data_x[fold_indices[2]] == -1).mean()))

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

            # Store prediction freq info
            _idxs = np.unique(test_results['preds'], return_counts=True)
            counts = np.zeros(26)
            counts[_idxs[0].astype(int)] = _idxs[1]
            counts_df.append(pd.DataFrame({'populations': label_names, 
                                           'frequencies': counts, 
                                           'missing': np.ones(26)*corruption_percent}))

            # Save results
            if task == 'classification':
                acc = test_results['n_right'].sum()/len(test_results['ys'])
                f.write(str(corruption_percent)+'\t'+str(acc)+'\n')

                counts_df2 = pd.concat(counts_df)
                counts_df2.to_csv(os.path.join(args.results_path, 
                                               'detailed_' + results_filename))

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

    parser.add_argument(
            '--random-baseline',
            action='store_true',
            help='Use this flag to ignore attributions and use random order of SNPs'
            )

    parser.add_argument(
            '--reverse-baseline',
            action='store_true',
            help='Use this flag to ignore attributions and use random order of SNPs'
            )

    parser.add_argument(
            '--baseline-style',
            type=str,
            choices=['random_sample', 'reference', 'missing', 
                     'uniform', 'random_gen', 'random_gen_weighted',
                     'random_gen_sample', 'random_gen_weighted_sample', 'random_sample_YRI', 
                     'random_sample_CEU', 'random_sample_JPT'],
            default='reference',
            help='Should baseline be all reference or all missing. Default: %(default)s'
            )
    
    parser.add_argument(
            '--corr-percent-list',
            type=float,
            nargs='+',
            help='Thresholds of missingness to use',
            default=[0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
            )

    parser.add_argument(
            '--attribution-score',
            choices = ['none', 'none_raw', 'old'],
            default='none',
            help='Should we compute a corruption score and remove SNPs based on it. none = no score computed: instead we use remove different SNPs per sample.'
            )   

    parser.add_argument(
            '--corruption-style',
            choices = ['missing', 'permute'],
            default='median',
            help='How to corrupt SNPs. median = replace values with feature median. permute = shuffle values of features.'
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()