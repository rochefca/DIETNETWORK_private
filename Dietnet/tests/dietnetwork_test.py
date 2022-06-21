import unittest
import os

import yaml
import sklearn
import numpy as np
import torch
import h5py

import helpers.dataset_utils as du
#import helpers.model as model
import helpers.mainloop_utils as mlu
import helpers.log_utils as lu
from helpers.model_handlers import dietNetworkHandler, MlpHandler

class TestDietNetwork(unittest.TestCase):

    def setUp(self):
        
        # Create dir where training info will be saved
        """
        The directory will be created in exp_path/exp_name with the name
        exp_name_foldi where i is the number of the fold
        """
        exp_path = '/lustre06/project/6065672/sciclun4/Experiments/Dietnet_exp'
        exp_name = 'TMP'
        which_fold = 0

        out_dir = lu.create_out_dir(exp_path, exp_name, which_fold)

        config = {}
        # Hyperparameters
        f = open(os.path.join('tests', 'config_dietnet.yaml'), 'r')
        config_hyperparams = yaml.load(f, Loader=yaml.FullLoader)

        config['params'] = config_hyperparams

        # Add fold to config hyperparams
        config['params']['fold'] = which_fold

        # Specifics
        specifics = {}
        specifics['model'] = 'Dietnet'
        specifics['exp_path'] = '/lustre06/project/6065672/sciclun4/Experiments/Dietnet_exp'
        specifics['exp_name'] = 'TMP'
        specifics['out_dir'] = out_dir
        specifics['partition'] = 'partition_407325inds_first2000only.npz'
        specifics['dataset'] = 'dataset_407325inds_408940snps_4classes.hdf5'
        specifics['embedding'] = 'embedding_407325inds_408940snps_4classes.npz'
        specifics['normalize'] = False
        specifics['input_features_means'] = 'input_features_means_407325inds_408940snps.npz'
        specifics['task'] = 'regression'
        specifics['param_init'] = None

        config['specifics'] = specifics

        # ----------------------------------------
        #               SET DEVICE
        # ----------------------------------------
        print('\n --- Setting device ---')
        print('Cuda available:', torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device:', device)

        # ----------------------------------------
        #               FIX SEED
        # ----------------------------------------
        seed = 42
        #torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type=='cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # ----------------------------------------
        #        LOAD INPUT FEATURES MEANS
        # ----------------------------------------
        print('\n --- Loading input features mean ---')

        # Mean and sd per feature computed on training set
        input_features_means = np.load(os.path.join(
            config['specifics']['exp_path'],
            config['specifics']['input_features_means'])
            )

        mus = input_features_means['means_by_fold'][config['params']['fold']]
        # Send mus to device
        mus = torch.from_numpy(mus).float().to(device)
        if 'sd_by_fold' in input_features_means.files:
            sigmas = input_features_means['sd_by_fold'][config['params']['fold']]
            ##sigmas = preprocess_params['sd_by_fold'][config['params']['fold']]
            sigmas = torch.from_numpy(sigmas).float().to(device)
        else:
            sigmas = None

        # ----------------------------------------
        #           LOAD FOLD INDEXES
        # ----------------------------------------
        print('\n --- Loading fold indexes of train, valid and test sets ---')
        all_folds_idx = np.load(os.path.join(
            config['specifics']['exp_path'],
            config['specifics']['partition']),
            allow_pickle=True)

        fold_idx = all_folds_idx['folds_indexes'][config['params']['fold']]

        # ----------------------------------------
        #       MAKE TRAIN, VALID, TEST SETS
        # ----------------------------------------
        print('\n --- Making train, valid, test sets classes ---')

        # Dataset hdf5 file
        dataset_file = os.path.join(
                config['specifics']['exp_path'],
                config['specifics']['dataset'])

        du.FoldDataset.dataset_file = dataset_file
        du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')

        # Label conversion depending on task
        if config['specifics']['task'] == 'classification':
            du.FoldDataset.label_type = np.int64
        elif config['specifics']['task'] == 'regression':
            du.FoldDataset.label_type = np.float32

        train_set = du.FoldDataset(fold_idx[0])
        print('training set:', len(train_set))
        valid_set = du.FoldDataset(fold_idx[1])
        print('valid set:', len(valid_set))
        test_set = du.FoldDataset(fold_idx[2])
        print('test set:', len(test_set))

        #  Model now encapsulated by modelHandler object
        if config['specifics']['model'] == 'Dietnet':
            mod_handler = dietNetworkHandler(config, device)
        elif config['specifics']['model'] == 'Mlp':
            mod_handler = MlpHandler(config, device)
        else:
            raise Exception('{} Not implemented yet!'.format(config['specifics']['model']))

        # ----------------------------------------
        #               OPTIMIZATION
        # ----------------------------------------
        # Loss
        if config['specifics']['task'] == 'classification':
            criterion = nn.CrossEntropyLoss()
        elif config['specifics']['task'] == 'regression':
            criterion = nn.MSELoss()

        print('For loss we use:', criterion)

        # Optimizer
        lr = config['params']['learning_rate']
        optimizer = torch.optim.Adam(mod_handler.get_trainable_parameters(), lr=lr)

        # Max nb of epochs
        n_epochs = config['params']['epochs']

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

    def test_DietNetwork_classification(self):
        pass

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
