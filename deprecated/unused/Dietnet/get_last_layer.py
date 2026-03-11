import argparse
import os
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

import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu


def main():
    args = parse_args()

    # Load info from full config file
    full_config_file = os.path.join(args.exp_path, args.exp_name,
                                    args.exp_name+'_fold'+str(args.which_fold),
                                    args.full_config)

    full_config_file_handle = open(full_config_file, 'r')
    full_config = yaml.load(full_config_file_handle, Loader=yaml.FullLoader)

    # Add fold number to full config
    full_config['specifics']['fold'] = args.which_fold

    pprint.pprint(full_config)

    # ----------------------------------------
    #               SET DEVICE
    # ----------------------------------------
    print('Cuda available:', torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # ----------------------------------------
    #           LOAD MEAN and SD
    # ----------------------------------------
    # Load mean and sd features computed on training set
    print('loading preprocessing parameters')
    preprocess_params = np.load(os.path.join(
        full_config['specifics']['exp_path'],
        full_config['specifics']['preprocess_params']))

    mus = preprocess_params['means_by_fold'][full_config['specifics']['fold']]
    sigmas = preprocess_params['sd_by_fold'][full_config['specifics']['fold']]

    # Send mus and sigmans to device
    mus = torch.from_numpy(mus).float().to(device)
    sigmas = torch.from_numpy(sigmas).float().to(device)

    # ----------------------------------------
    #           LOAD FOLD INDEXES
    # ----------------------------------------
    print('Loading fold indexes split into train, valid, test sets')
    all_folds_idx = np.load(os.path.join(
        full_config['specifics']['exp_path'],
        full_config['specifics']['folds_indexes']),
        allow_pickle=True)

    fold_idx = all_folds_idx['folds_indexes'][full_config['specifics']['fold']]

    # ----------------------------------------
    #       LOAD TRAIN, VALID, TEST SETS
    # ----------------------------------------
    print('Making train, valid, test sets classes')

    # Dataset hdf5 file
    dataset_file = os.path.join(
            full_config['specifics']['exp_path'],
            full_config['specifics']['dataset'])

    du.FoldDataset.dataset_file = dataset_file
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')

    # Label conversion
    du.FoldDataset.label_type = np.int64

    train_set = du.FoldDataset(fold_idx[0])
    print('training set:', len(train_set))
    valid_set = du.FoldDataset(fold_idx[1])
    print('valid set:', len(valid_set))
    test_set = du.FoldDataset(fold_idx[2])
    print('test set:', len(test_set))

    # ----------------------------------------
    #             LOAD EMBEDDING
    # ----------------------------------------
    print('Loading embedding')
    emb = du.load_embedding(os.path.join(
        full_config['specifics']['exp_path'],
        full_config['specifics']['embedding']),
        full_config['specifics']['fold'])

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
    n_feats_emb = emb.size()[1] # input of aux net

    # Main net input size
    n_feats = emb.size()[0] # input of main net

    # Main net output size (nb targets)
    with h5py.File(dataset_file, 'r') as f:
        n_targets = len(f['label_names'])

    print('\n***Nb features in models***')
    print('n_feats_emb:', n_feats_emb)
    print('n_feats:', n_feats)
    print('n_targets:', n_targets)

    # Model init
    comb_model = model.CombinedModel(
            n_feats=n_feats_emb,
            n_hidden_u_aux=full_config['params']['nb_hidden_u_aux'],
            n_hidden_u_main=full_config['params']['nb_hidden_u_aux'][-1:] \
                    +full_config['params']['nb_hidden_u_main'],
            n_targets=n_targets,
            param_init=None,
            input_dropout=full_config['params']['input_dropout'])

    comb_model.to(device)

    # Load weights
    model_weights_path = os.path.join(full_config['specifics']['out_dir'], 'model_params.pt')
    comb_model.load_state_dict(torch.load(model_weights_path))

    # ----------------------------------------
    #             BATCH GENERATORS
    # ----------------------------------------
    batch_size = full_config['params']['batch_size']
    print('Batch size:', batch_size)

    train_generator = DataLoader(train_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0)

    valid_generator = DataLoader(valid_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0)

    test_generator = DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)

    # ----------------------------------------
    #         GET LAST HIDDEN LAYER
    # ----------------------------------------
    # Put model in eval mode
    comb_model.eval()
    with h5py.File(dataset_file, 'r') as f:
        label_names = np.array(f['label_names']).astype(np.str_)

    # Getting layers for samples in training set
    print('Getting layers training set', flush=True)
    samples, ys, score, pred, acc, before_last_layers, out_layers = mlu.get_last_layers(
            comb_model, device, train_generator, len(train_set), mus, sigmas, emb, full_config['specifics']['task'])

    last_layers_file = os.path.join(full_config['specifics']['out_dir'], 'last_layers_train.npz')
    np.savez(last_layers_file,
             samples=samples,
             preds=pred.cpu(),
             scores=score.cpu(),
             labels=ys,
             label_names=label_names,
             before_last=np.array(before_last_layers),
             last=np.array(out_layers))
    print('train accuracy:', str(acc), flush=True)

    # Getting layers for samples in valid set
    print('Getting layers valid set', flush=True)
    samples, ys, score, pred, acc, before_last_layers, out_layers = mlu.get_last_layers(
            comb_model, device, valid_generator, len(valid_set), mus, sigmas, emb, full_config['specifics']['task'])

    last_layers_file = os.path.join(full_config['specifics']['out_dir'], 'last_layers_valid.npz')
    np.savez(last_layers_file,
             samples=samples,
             preds=pred.cpu(),
             scores=score.cpu(),
             labels=ys,
             label_names=label_names,
             before_last=np.array(before_last_layers),
             last=np.array(out_layers))
    print('valid accuracy:', str(acc), flush=True)

    # Getting layers for samples in test set
    print('Getting layers test set', flush=True)
    test_samples, test_ys, score, pred, acc, before_last_layers, out_layers = mlu.get_last_layers(comb_model, device,
            test_generator, len(test_set), mus, sigmas, emb,
            full_config['specifics']['task'])

    last_layers_file = os.path.join(full_config['specifics']['out_dir'], 'last_layers_test.npz')
    np.savez(last_layers_file,
             samples=test_samples,
             preds=pred.cpu(),
             scores=score.cpu(),
             labels=test_ys,
             label_names=label_names,
             before_last=np.array(before_last_layers),
             last=np.array(out_layers))
    print('test accuracy:', str(acc), flush=True)



def parse_args():
    parser = argparse.ArgumentParser(
            description=('Get neurons values of last dietnet hidden layer')
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
                  'exp-path.')
            )

    parser.add_argument(
            '--full_config',
            default='full_config.yaml',
            help='Yaml file returned in dietnet training phase'
            )

    parser.add_argument(
            '--preprocess-params',
            type=str,
            default='preprocessing_params.npz',
            help='Normalization parameters obtained with get_preprocessing_params.py'
            )

    parser.add_argument(
            '--which-fold',
            type=int,
            default=0
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
