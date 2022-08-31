import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers.model as model


class modelHandler:
    """
    This class encapsulates our Neural Network
    (which itself is an instance of torch.module)
    This class contains methods for model initialization and forward/reverse pass
    """

    def __init__(self, model, task_handler):
        self.model = model
        self.task_handler = task_handler

    def forward(self, x):
        return self.model(x)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def train_mode(self):
        raise NotImplementedError

    def eval_mode(self):
        raise NotImplementedError

    def get_parameters(self):
        return self.model.parameters()

    def log_weight_initialization(self):
        raise NotImplementedError

    def get_exp_identifier(self, config, task, fold):
        raise NotImplementedError


class DietNetworkHandler(modelHandler):

    def __init__(self, task_handler, fold, emb_filename, device,
                 dataset_filename, config, param_init):
        # Make the Dietnet model
        dn_model = model.DietNetwork(fold, emb_filename, device,
                                     dataset_filename, config, param_init)

        super(DietNetworkHandler, self).__init__(dn_model, task_handler)


    def get_exp_identifier(self, config, fold):
        exp_identifier =  self.task_handler.name \
                + '_auxu_' \
                    + str(config['nb_hidden_u_aux'])[1:-1].replace(', ','_') \
                + '_mainu_' \
                    + str(config['nb_hidden_u_aux'][-1]) + '_' \
                    + str(config['nb_hidden_u_main'])[1:-1].replace(', ','_') \
                + '_inpdrop_' + str(config['input_dropout']) \
                + '_maindrop_' + str(config['dropout_main']) \
                + '_lr_' + str(config['learning_rate']) \
                + '_lra_' + str(config['learning_rate_annealing']) \
                + '_uniform_init_limit_' + str(config['uniform_init_limit']) \
                + '_epochs_' + str(config['epochs']) \
                + '_patience_' + str(config['patience']) \
                + '_seed_' + str(config['seed']) \
                + '_fold' + str(fold)

        return exp_identifier


class MlpHandler(modelHandler):

    def __init__(self, task_handler, dataset_filename, config):
        # Make the MLP model
        mlp_model = model.Mlp(task_handler, dataset_filename, config)

        super(MlpHandler, self).__init__(mlp_model, task_handler)


    def get_exp_identifier(self, config, fold):
        exp_identifier =  self.task_handler.name \
                + '_mlp' \
                + '_layers_' \
                    + str(config['nb_hidden_u'])[1:-1].replace(', ','_') \
                + '_inpdrop_' + str(config['input_dropout']) \
                + '_drop_' + str(config['dropout']) \
                + '_lr_' + str(config['learning_rate']) \
                + '_lra_' + str(config['learning_rate_annealing']) \
                + '_epochs_' + str(config['epochs']) \
                + '_patience_' + str(config['patience']) \
                + '_seed_' + str(config['seed']) \
                + '_fold' + str(fold)

        return exp_identifier


        """
        # ----------------------------------------
        #               MAKE MODEL
        # ----------------------------------------

        # load emb to infer number of features
        print('Loading embedding')
        emb = du.load_embedding(os.path.join(
            config['specifics']['exp_path'],
            config['specifics']['embedding']),
            config['params']['fold'])
        # Main net input size (nb of features)
        n_feats = emb.size()[0] # input of main net
        del emb

        # Main net input size (nb of features)
        n_hidden_u = config['params']['n_hidden_u']

        # Main net output size (nb targets)
        if config['specifics']['task'] == 'classification':

            dataset_file = os.path.join(config['specifics']['exp_path'],
                                        config['specifics']['dataset'])
            with h5py.File(dataset_file, 'r') as f:
                n_targets = len(f['label_names'])
        elif config['specifics']['task'] == 'regression':
            n_targets = 1

        print('\n***Nb features in models***')
        print('n_feats:', n_feats)
        print('n_targets:', n_targets)

        # Model init
        print('Initiating the model')
        model_init_start_time = time.time()
        linear_model = model.Mlp(
                n_feats=n_feats,
                n_hidden_u=n_hidden_u,
                n_targets=n_targets,
                input_dropout=config['params']['input_dropout'])
        print('Model initiated in: ', time.time()-model_init_start_time, 'seconds')

        # Data parallel: this is not implemented yet
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            linear_model = nn.DataParallel(linear_model)

        # Note: runs script in single GPU mode only!
        print('Sending model to device')
        linear_model.to(device)
        #print(summary(comb_model.feat_emb, input_size=(294427,1,1,78)))
        #print(summary(comb_model.disc_net, input_size=[(138,1,1,294427),(100,294427)]))


        self.model = linear_model
        """
