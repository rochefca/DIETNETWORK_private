import os
import argparse
import sys
import time
import yaml
import pprint

import h5py

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dataset_utils as du
import model as model
import mainloop_utils as mlu
import log_utils as lu


class modelHandler:
    """
    This class encapsulates our Neural Network 
    (which itself is an instance of torch.module)
    This class contains methods for model initialization and forward/reverse pass
    """

    def __init__(self, config, device):
        raise NotImplementedError

    def forwardpass(self, x):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
    
    def load(self):
        raise NotImplementedError

    def train_mode(self):
        raise NotImplementedError
    
    def eval_mode(self):
        raise NotImplementedError
        
    def get_trainable_parameters(self):
        raise NotImplementedError
    
    def log_weight_initialization(self):
        raise NotImplementedError


class dietNetworkHandler(modelHandler):

    def __init__(self, config, device):

        # ----------------------------------------
        #             LOAD EMBEDDING
        # ----------------------------------------
        print('Loading embedding')
        emb = du.load_embedding(os.path.join(
            config['specifics']['exp_path'],
            config['specifics']['embedding']),
            config['params']['fold'])

        # Send to device
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
        if config['specifics']['task'] == 'classification':
            with h5py.File(dataset_file, 'r') as f:
                n_targets = len(f['label_names'])
        elif config['specifics']['task'] == 'regression':
            n_targets = 1

        print('\n***Nb features in models***')
        print('n_feats_emb:', n_feats_emb)
        print('n_feats:', n_feats)
        print('n_targets:', n_targets)

        # Model init
        print('Initiating the model')
        model_init_start_time = time.time()
        comb_model = model.CombinedModel(
                n_feats=n_feats_emb,
                n_hidden_u_aux=config['params']['nb_hidden_u_aux'],
                n_hidden_u_main=config['params']['nb_hidden_u_aux'][-1:] \
                                +config['params']['nb_hidden_u_main'],
                n_targets=n_targets,
                param_init=config['specifics']['param_init'],
                input_dropout=config['params']['input_dropout'])
        print('Model initiated in: ', time.time()-model_init_start_time, 'seconds')

        # Data parallel: this is not implemented yet
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            comb_model.disc_net = nn.DataParallel(comb_model.disc_net)

        # Note: runs script in single GPU mode only!
        print('Sending model to device')
        comb_model.to(device)
        #print(summary(comb_model.feat_emb, input_size=(294427,1,1,78)))
        #print(summary(comb_model.disc_net, input_size=[(138,1,1,294427),(100,294427)]))

        self.model = comb_model
        self.emb = emb
    
    def get_trainable_parameters(self):
        """
        Get trainable parameters (for torch optimizer)
        """
        return self.model.parameters()
    
    def forwardpass(self, x):
        return self.model(self.emb, x)
    
    def save(self, out_dir, filename):
        lu.save_model_params(out_dir, self.model, filename)
    
    def load(self, torch_weight_dict):
        self.model.load_state_dict(torch_weight_dict)
        
    def log_weight_initialization(self, experiment):
        # Log weights initialisation values to comet-ml

        # Layers in aux net
        for i,layer in enumerate(self.model.feat_emb.hidden_layers):
            layer_name = 'auxNet_weights_layer' + str(i)
            experiment.log_histogram_3d(layer.weight.cpu().detach().numpy(),
                                        name=layer_name,
                                        step=0)
            """
            layer_name = 'auxNet_bias_layer' + str(i)
            experiment.log_histogram_3d(layer.bias.cpu().detach().numpy(),
                                        name=layer_name,
                                        step=0)
            """

        # Layers in main net
        experiment.log_histogram_3d(
                self.model.fatLayer_weights.cpu().detach().numpy(),
                name='mainNet_fatLayer',
                step=0)

        for i,layer in enumerate(self.model.disc_net.hidden_layers):
            layer_name = 'mainNet_layer' + str(i+1)
            experiment.log_histogram_3d(layer.weight.cpu().detach().numpy(),
                                        name=layer_name,
                                        step=0)
            layer_name = 'mainNet_bias_layer' + str(i)
            experiment.log_histogram_3d(layer.bias.cpu().detach().numpy(),
                                        name=layer_name,
                                        step=0)
    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()


class MlpHandler(modelHandler):

    def __init__(self, config, device):

        # ----------------------------------------
        #               MAKE MODEL
        # ----------------------------------------

        # Main net input size (nb of features)
        n_feats = config['params']['num_input_features']
        n_hidden_u = config['params']['n_hidden_u']

        # Main net output size (nb targets)
        if config['specifics']['task'] == 'classification':
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

    def get_trainable_parameters(self):
        """
        Get trainable parameters (for torch optimizer)
        """
        return self.model.parameters()
    
    def forwardpass(self, x):
        return self.model(x)
    
    def save(self, out_dir, filename):
        lu.save_model_params(out_dir, self.model, filename)
    
    def load(self, torch_weight_dict):
        self.model.load_state_dict(torch_weight_dict)
        
    def log_weight_initialization(self, experiment):
        # Log weights initialisation values to comet-ml

        # Layers in net
        for i,layer in enumerate(self.model):
            layer_name = 'weights_layer' + str(i)
            experiment.log_histogram_3d(layer.weight.cpu().detach().numpy(),
                                        name=layer_name,
                                        step=0)
    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
