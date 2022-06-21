import unittest
from unittest import mock
import os
import argparse
import os
from shutil import copyfile, rmtree

import numpy as np

from train import main

EXP_PATH = '/lustre06/project/6065672/sciclun4/Experiments/Dietnet_exp'
CONFIG_DIETNET = 'config_dietnet.yaml'
CONFIG_MLP = 'config_mlp.yaml'
EXP_NAME1 = 'TMP1'
EXP_NAME2 = 'TMP2'
EXP_NAME3 = 'TMP3'
EXP_NAME4 = 'TMP4'


class TestDietNetwork(unittest.TestCase):

    def setUp(mock_args):
        for exp in [EXP_NAME1, EXP_NAME3]:
            config_loc = os.path.join(EXP_PATH, exp)
            print('Creating {} containing {}...'.format(config_loc, CONFIG_DIETNET))
            os.makedirs(config_loc, exist_ok=True)
            copyfile(os.path.join('tests', CONFIG_DIETNET), 
                     os.path.join(config_loc, CONFIG_DIETNET))
        for exp in [EXP_NAME2, EXP_NAME4]:
            config_loc = os.path.join(EXP_PATH, exp)
            print('Creating {} containing {}...'.format(config_loc, CONFIG_MLP))
            os.makedirs(config_loc, exist_ok=True)
            copyfile(os.path.join('tests', CONFIG_MLP), 
                     os.path.join(config_loc, CONFIG_MLP))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(exp_path=EXP_PATH,
                                                exp_name=EXP_NAME1,
                                                which_fold=0,
                                                config=CONFIG_DIETNET,
                                                model='Dietnet', #'Mlp'
                                                dataset='dataset_407325inds_408940snps_4classes.hdf5',
                                                partition='partition_407325inds_first2000only.npz',
                                                embedding='embedding_407325inds_408940snps_4classes.npz',
                                                input_features_means='input_features_means_407325inds_408940snps.npz',
                                                normalize=False,
                                                task='regression', #'regression'
                                                param_init=None,
                                                comet_ml=None,
                                                comet_ml_project_name=None,
                                                optimization=None)
               )

    def test_main_dietnet_regression(mock_args1, mock_args2):
        result = main()

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(exp_path=EXP_PATH,
                                                exp_name=EXP_NAME2,
                                                which_fold=0,
                                                config=CONFIG_MLP,
                                                model='Mlp',
                                                dataset='dataset_407325inds_408940snps_4classes.hdf5',
                                                partition='partition_407325inds_first2000only.npz',
                                                embedding='embedding_407325inds_408940snps_4classes.npz',
                                                input_features_means='input_features_means_407325inds_408940snps.npz',
                                                normalize=False,
                                                task='regression', #'regression'
                                                param_init=None,
                                                comet_ml=None,
                                                comet_ml_project_name=None,
                                                optimization=None)
               )

    def test_main_mlp_regression(mock_args1, mock_args2):
        result = main()

    def tearDown(self):
        for exp in [EXP_NAME1, EXP_NAME3]:
            config_loc = os.path.join(EXP_PATH, exp)
            print('Deleting {}...'.format(config_loc))
            rmtree(config_loc)
        for exp in [EXP_NAME2, EXP_NAME4]:
            config_loc = os.path.join(EXP_PATH, exp)
            print('Deleting {}...'.format(config_loc))
            rmtree(config_loc)

if __name__ == '__main__':
    unittest.main()
