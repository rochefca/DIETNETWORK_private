import argparse
import os
import time
import yaml
import pprint

import h5py

import numpy as np

from comet_ml import Experiment, Optimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu
import helpers.log_utils as lu


def main():
    args = parse_args()

    # ----------------------------------------
    #               OPTIMIZER
    # ----------------------------------------
    # Parse optimization file
    f = open(os.path.join(args.exp_path, args.exp_name, args.config_opt), 'r')
    opt_config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    print("Hypersearch config:")
    pprint.pprint(opt_config)

    # Create optimizer with values from config-opt file
    opt = Optimizer(opt_config)

    # ----------------------------------------
    #           EXPERIMENTS LOOP
    # ----------------------------------------
    exp_count = 0
    for experiment in opt.get_experiments(project_name=args.comet_ml_project_name):
        # ----------------------------------------
        #           EXP CONFIG FILE
        # ----------------------------------------
        config = {}
        config['batch_size'] = experiment.get_parameter('batch_size')
        config['epochs'] = experiment.get_parameter('epochs')
        config['input_dropout'] = experiment.get_parameter('input_dropout')
        config['learning_rate'] = experiment.get_parameter('learning_rate')
        config['learning_rate_annealing'] = experiment.get_parameter('learning_rate_annealing')
        config['uniform_init_limit'] = experiment.get_parameter('uniform_init_limit')

        print(experiment.get_parameter('nb_hidden_u_aux'))
        print(type(experiment.get_parameter('nb_hidden_u_aux')))

        # Nb of layers and hidden units in aux net in list format
        """
        Note: exeriment.get returns a string in the form '[nb1,nb2,...,nbN]'
        We convert this to list of intergers
        """
        nb_hidden_u_aux = \
                experiment.get_parameter('nb_hidden_u_aux')[1:-1].split(',')
        config['nb_hidden_u_aux'] = [int(i) for i in nb_hidden_u_aux]

        # Nb of layers and hidden units in main net in list format
        nb_hidden_u_main = \
                experiment.get_parameter('nb_hidden_u_main')[1:-1].split(',')
        config['nb_hidden_u_main'] = [int(i) for i in nb_hidden_u_main]

        config['patience'] = experiment.get_parameter('patience')
        config['seed'] = experiment.get_parameter('seed')

        pprint.pprint(config)

        # Filename details
        filename_details = '_bs_' + str(config['batch_size']) \
                + '_epochs_' + str(config['epochs']) \
                + '_inpdrop_' + str(config['input_dropout']) \
                + '_lr_' + str(config['learning_rate']) \
                + '_lra_' + str(config['learning_rate_annealing']) \
                + '_auxu_' \
                    + str(config['nb_hidden_u_aux'])[1:-1].replace(', ','_') \
                + '_mainu_' \
                    + str(config['nb_hidden_u_aux'][-1]) + '_' \
                    + str(config['nb_hidden_u_main'])[1:-1].replace(', ','_') \
                + '_uniform_init_limit_' + str(config['uniform_init_limit']) \
                + '_patience_' + str(config['patience']) \
                + '_seed_' + str(config['seed'])

        # Experiment config filename
        config_filename = 'config_' + args.exp_name \
                + filename_details + '.yaml'

        # Write experiment config file
        config_filename_fullpath = os.path.join(
                args.exp_path, args.exp_name, config_filename
                )
        config_handle = open(config_filename_fullpath, 'w')
        yaml.dump(config, config_handle)
        config_handle.close()

        # ----------------------------------------
        #           BASH TRAIN SCRIPT
        # ----------------------------------------
        script = '#!/bin/bash\n\n'

        # Compute Canada specifications
        script += '#SBATCH --account=' + args.cc_account + '\n'
        script += '#SBATCH --' + args.cc_gpu + '\n'
        script += '#SBATCH --time=' + args.cc_time + '\n'
        script += '#SBATCH --mem=' + args.cc_mem + '\n'
        script += '#SBATCH --job-name=' + args.cc_job + '\n'
        script += '#SBATCH --output=' + args.cc_job + '_%j.out\n'
        script += '#SBATCH --error=' + args.cc_job + '_%j.err\n\n'

        # Paths
        script += 'DIETNET=\"' + args.dietnet_path + '\"\n'
        script += 'EXP_PATH=\"' + args.exp_path + '\"\n'
        script += 'EXP_NAME=\"' + args.exp_name + '\"\n\n'

        # Activation of virtual env
        script += 'source ' + args.cc_python_env + '/bin/activate\n\n'

        # Internet access
        script += 'module load httpproxy' + '\n\n'

        # Call to dietnet train.py
        script += 'python $DIETNET/train.py \\\n'
        script += '--exp-path $EXP_PATH \\\n'
        script += '--exp-name $EXP_NAME \\\n'
        script += '--dataset ' + args.dataset + ' \\\n'
        script += '--partition ' + args.partition + ' \\\n'
        script += '--embedding ' + args.embedding + ' \\\n'
        script += '--input-features-means ' + args.input_features_means + ' \\\n'
        script += '--task ' + args.task + ' \\\n'
        script += '--config ' + config_filename + ' \\\n'
        script += '--comet-ml ' + ' \\\n'
        script += '--comet-ml-project-name ' + args.comet_ml_project_name + ' \\\n'
        script += '--optimization'


        # Write script to file
        script_filename = 'train_' + args.exp_name + filename_details + '.sh'

        script_filename_full_path = os.path.join(
                args.exp_path, args.exp_name, script_filename
                )

        script_file_handle = open(script_filename_full_path, 'w')
        script_file_handle.write(script)
        script_file_handle.close()

        # Tracking number of generated experiments
        exp_count += 1

    print('Number of experiments:', exp_count)


def parse_args():
    parser = argparse.ArgumentParser(
            description=('Train, eval and test model of a given fold')
            )

    # Paths
    parser.add_argument(
            '--dietnet-path',
            type=str,
            required=True,
            help='Path to dietnet code'
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

    # Files
    parser.add_argument(
            '--config-opt',
            type=str,
            default='config_opt.yaml',
            help=('Yaml file of hyperparameters optimization. '
                  'Default: %(default)s')
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
            '--partition',
            type=str,
            default='partition_idx.npz',
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
            '--input-features-means',
            type=str,
            default='input_features_means.npz',
            help=('Filename of computed input features means. The means are '
                  'used to replace missing genotypes. Default: %(default)s')
            )

    # Input features normalization
    parser.add_argument(
            '--normalize',
            action='store_true',
            help='Use this flag to normalize input features.'
            )

    """
    parser.add_argument(
            '--preprocess-params',
            type=str,
            default='preprocessing_params.npz',
            help='Normalization parameters obtained with get_preprocessing_params.py'
            )
    """
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

    # Optional param init
    parser.add_argument(
            '--param-init',
            type=str,
            help='File of parameters initialization values'
            )

    # Comet-ml
    parser.add_argument(
            '--comet-ml-project-name',
            type=str,
            required=True,
            help='Specific project were to send comet Experiment'
            )

    # Compute Canada specifications
    parser.add_argument(
            '--cc-account',
            type=str,
            required=True,
            help='Compute Canada --account argument value'
            )

    parser.add_argument(
            '--cc-gpu',
            type=str,
            default='gres=gpu:1',
            help=('Compute Canada GPU specification. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--cc-time',
            type=str,
            required=True,
            help='Compute Canada --time argument value (hh:mm:ss)'
            )

    parser.add_argument(
            '--cc-mem',
            type=str,
            required=True,
            help='Compute Canada --mem argument value (ex:46G)'
            )

    parser.add_argument(
            '--cc-job',
            type=str,
            default='%j',
            help='Compute Canada --job argument value. Default: %(default)s'
            )

    parser.add_argument(
            '--cc-python-env',
            type=str,
            required=True,
            help='Path to python virtual env to activate'
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
