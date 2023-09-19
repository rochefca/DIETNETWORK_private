import os
import argparse
from pathlib import Path, PurePath
import time
import sys
import yaml
import pprint

import h5py

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import helpers.test_indep_utils as tu
import helpers.dataset_utils as du
import helpers.log_utils as lu
import helpers.model as model
import helpers.mainloop_utils as mlu
from helpers.model_handlers import DietNetworkHandler, MlpHandler
from helpers.task_handlers import ClassificationHandler, RegressionHandler


def test():
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


    # ---------------------------------------------------------------
    #                       ---- TEST DATA ----
    # ---------------------------------------------------------------
    print('\n---\nLoading test data')
    stime = time.time()

    # Dataset
    du.IndepTestDataset.dataset_file = args.test_dataset
    du.IndepTestDataset.f = h5py.File(args.test_dataset, 'r')

    print('Loading input features')
    du.IndepTestDataset.data_x = np.array(du.IndepTestDataset.f['inputs'], dtype=np.int8)

    print('\nLoaded {} genotypes of {} samples'.format(
          du.IndepTestDataset.data_x.shape[1],
          du.IndepTestDataset.data_x.shape[0]))

    print('Loaded test data in {} seconds'.format(time.time()-stime))

    # Init IndepTestDataset with set indexes being 0 to nb of samples in test set
    test_set = du.IndepTestDataset(
            [i for i in range(du.IndepTestDataset.data_x.shape[0])])

    print('data x:', du.IndepTestDataset.data_x)
    print('nb of samples:', len(test_set))

    print('---\n')
    
    # Matching genotypes
    if args.matched_test_dataset is None:
        # ----------------------------------------
        #       SCALE GENOTYPES IN TEST SET
        # ----------------------------------------
        # We remove SNPs that in test set but not in train set
        # For SNPs in train set but not in test set, we put a missing value
        # We scale non-missing genotypes in test set in proportion with the
        # amount of SNPs in train set that are missing in test set
        print('\n---\nMatching snps in test set according to snps in train set\n')

        train_f = h5py.File(args.train_dataset, 'r')

        # Adapt genotype values in test set based on snp used in train set
        matched_genotypes, scale = tu.match_input_features(
                du.IndepTestDataset.data_x,
                np.array(du.IndepTestDataset.f['snp_names']),
                np.array(train_f['snp_names']))
    
        print('\nFinal matched genotypes: {} samples with test genotypes matched to {} train genotypes'.format(
                matched_genotypes.shape[0], matched_genotypes.shape[1]))

        du.IndepTestDataset.data_x = matched_genotypes
    
        # Saving the matched dataset
        print('Saving matched dataset to: {}'.format('dataset_test_tmp_matched.h5py'))
        print(list(du.IndepTestDataset.f.keys()))
        matched_f = h5py.File('dataset_test_tmp_matched.h5py', 'w')
        matched_f.create_dataset('inputs', data=matched_genotypes)
        matched_f.create_dataset('samples', data=du.IndepTestDataset.f['samples'])
        matched_f.create_dataset('snp_names', data=du.IndepTestDataset.f['snp_names'])
        matched_f.create_dataset('scale', data=np.array([scale]))
        matched_f.close()

        print('---\n')
    
    else:
        matched_f = h5py.File(args.matched_test_dataset, 'r')
        du.IndepTestDataset.data_x = np.array(matched_f['inputs'], dtype=np.int8)
        scale = matched_f['scale'][0]
        print('scale:', scale)

    # ---------------------------------------------------------------
    #                 ---- INFO FROM TRAINING PHASE ----
    # ---------------------------------------------------------------
    print('\n---\nLoading info from training phase')
    
    # Which training fold
    fold = args.which_fold
    print('Training fold:', fold)
    
    # ----------------------------------------
    #               TASK HANDLER
    # ----------------------------------------
    # Task : clasification or regression
    # Note: dataset param is used to get 'class_label_names'
    # and set the nb of classes in the classification task.
    # This is why we give train instead of test dataset
    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
        task_handler = ClassificationHandler(args.train_dataset, criterion)

    elif args.task == 'regression':
        criterion = nn.MSELoss()
        task_handler = RegressionHandler(args.train_dataset, criterion)

    print('Task:', task_handler.name)
    
    # ----------------------------------------
    #               INP FEAT STATS
    # ----------------------------------------
    # Input features statistics
    inp_feat_stats = np.load(args.input_features_stats)

    mus = inp_feat_stats['means_by_fold'][fold]
    # Send to GPU
    mus = torch.from_numpy(mus).float().to(device)

    if 'sd_by_fold' in inp_feat_stats.files:
        sigmas = inp_feat_stats['sd_by_fold'][fold]
        print('Input features stats: {} means and {} standard deviations'.format(
              len(mus), len(sigmas)))

        # Send to GPU
        sigmas = torch.from_numpy(sigmas).float().to(device)
    else:
        sigmas = None
        print('Input feautres stats: {} means'.format(len(mus)))

    # ----------------------------------------
    #                 MODEL
    # ----------------------------------------
    param_init = None

    # Model architecture (Dietnet or Mlp)
    if args.model == 'Dietnet':
        model_handler = DietNetworkHandler(task_handler, fold,
                args.embedding, device, args.train_dataset, config, param_init)
    elif args.model == 'Mlp':
        model_handler = MlpHandler(task_handler, args.train_dataset, config)
    else:
        raise Exception('{} is not a recognized model'.format(
            args.model))

    # Send mmodel to GPU
    model_handler.model.to(device)

    print('\nModel:', model_handler.model)
    
    # Loading trained model parameters
    checkpoint = torch.load(args.model_params)
    model_handler.model.load_state_dict(checkpoint['model_state_dict'])
    print('\nLoaded model parameters from {} at epoch {}'.format(
          args.model_params, checkpoint['epoch']))
    print('---\n')



    # ---------------------------------------------------------------
    #                 ---- TEST SET UP ----
    # ---------------------------------------------------------------
    print('\n---\nTest set up')
    # Dir where to save test results
    exp_identifier = model_handler.get_exp_identifier(config, fold)
    results_dirname = 'TEST_RESULTS_' + exp_identifier
    results_fullpath = os.path.join(args.test_path, results_dirname)
    lu.create_dir(results_fullpath)
    print('Test results will be save to {}'.format(results_fullpath))
    
    # Batch generator
    test_generator = DataLoader(test_set,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=0)
    
    for i,(idx,x) in enumerate(test_generator):
        print('Batch', i)
        print('samples:', idx)
        print('xs:', x)

    # Test step
    model_handler.model.eval()
    test_results = mlu.indep_test_step(model_handler,
                                  device,
                                  test_set,
                                  test_generator,
                                  mus, sigmas, args.normalize,
                                  results_fullpath, 0, scale)
    
    # Save test results
    np.savez('test_results.npz',
             samples=test_results['samples'],
             preds=test_results['preds'],
             scores=test_results['scores'])
    print('Test results:', test_results)
    
    

    
    sys.exit()
    


    # Load data from external dataset
    parsed_data_filename = args.test_name + '_dataset.npz'

    # If data was already parsed and saved to file
    if Path(PurePath(args.exp_path, parsed_data_filename)).exists():
        print('Loading test data from',
              PurePath(args.exp_path, parsed_data_filename))

        data = np.load(PurePath(args.exp_path, parsed_data_filename))
        samples = data['samples']
        train_snps = data['snp_names']
        formatted_genotypes = data['inputs']
        feature_scaling = data['feature_scaling'][0]

        print('Loaded', str(formatted_genotypes.shape[1]), 'genotypes of',
              str(formatted_genotypes.shape[0]), 'individuals.')

    # Parse data and match test input features with those from train set
    else:
        # Parse data (here missing genotypes at the ind level are -1)
        samples, snps, genotypes = du.load_genotypes(args.genotypes)
        # Match snps of training and test sets
        train_dataset = np.load(PurePath(args.exp_path, args.dataset))
        train_snps = train_dataset['snp_names'] # SNPs used at training time
        formatted_genotypes, feature_scaling = tu.match_features(genotypes,
                                                                 snps,
                                                                 train_snps)
        # Save data
        print('Saving parsed genotypes and matched input features to',
                PurePath(args.exp_path, parsed_data_filename))
        np.savez(PurePath(args.exp_path, parsed_data_filename),
                 inputs=formatted_genotypes,
                 snp_names=train_snps,
                 samples=samples,
                 feature_scaling=np.array([feature_scaling]))

    # Load training fold specific data
    train_dir = tu.get_train_dir(args.exp_path, args.exp_name, args.which_fold)
    train_data = np.load(PurePath(train_dir, 'additional_data.npz'))
    # Mu and sigma for feature normalization
    mus = train_data['norm_mus']
    sigmas = train_data['norm_sigmas']
    # Trained model parameters
    model_params = PurePath(train_dir, 'model_params.pt')
    # Embedding used to train model
    emb = du.load_embedding(PurePath(args.exp_path, args.embedding),
                                  args.which_fold)

    # Put data on GPU
    formatted_genotypes = torch.from_numpy(formatted_genotypes).to(device)
    emb = (emb.to(device)).float()
    mus = (torch.from_numpy(mus).to(device)).float()
    sigmas = (torch.from_numpy(sigmas).to(device)).float()

    # Make test set: Do feature normalization later by batch for memory issues)
    test_set = tu.TestDataset(formatted_genotypes, samples)

    # Embedding normalization
    emb_norm = (emb ** 2).sum(0) ** 0.5
    emb = emb/emb_norm

    # ---Build model---
    # Input size
    n_feats_emb = emb.size()[1] # input of aux net
    n_feats = emb.size()[0] # input of main net
    # Hidden layers size
    emb_n_hidden_u = 100
    discrim_n_hidden1_u = 100
    discrim_n_hidden2_u = 100
    # Output layer
    n_targets = train_data['label_names'].shape[0]
    print('nb targets:', n_targets)

    comb_model = model.CombinedModel(
                 n_feats=n_feats_emb,
                 n_hidden_u=emb_n_hidden_u,
                 n_hidden1_u=discrim_n_hidden1_u,
                 n_hidden2_u=discrim_n_hidden2_u,
                 n_targets=n_targets,
                 param_init=None,
                 input_dropout=0.)

    # Set model parameters
    comb_model.load_state_dict(torch.load(Path(model_params)))

    comb_model.to(device)

    # Put model in eval mode
    comb_model.eval()
    discrim_model = lambda x: comb_model(emb, x)

    # Data generator
    batch_size = 138
    test_generator = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Evaluate
    start_time = time.time()

    for i, (x_batch, samples_batch) in enumerate(test_generator):
        x_batch = x_batch.float()
        # Replace missing values
        du.replace_missing_values(x_batch, mus)
        # Normalize input feature
        x_batch_normed = du.normalize(x_batch, mus, sigmas)
        # Scaling (Scale non-missing (non-zeros) values
        x_batch_normed *= feature_scaling

        # Forward pass in model
        out = discrim_model(x_batch_normed)
        # Get scores and prediction
        score, pred = mlu.get_predictions(out)
        if i == 0:
            test_pred = pred
            test_score = score
        else:
            test_pred = torch.cat((test_pred,pred), dim=-1)
            test_score = torch.cat((test_score,score), dim=0)

        print('Tested', str(i*batch_size), 'out of', str(len(samples)),
                'individuals')
    # End test
    test_time = time.time() - start_time
    print('Tested', str(len(test_pred)), 'individuals in', str(test_time),
            'seconds.')

    # Save results
    tu.save_test_results(train_dir, args.test_name,
                         samples, test_score, test_pred,
                         train_data['label_names'])


def parse_args():
    parser = argparse.ArgumentParser(
            description='Test a trained model in another dataset'
            )

    parser.add_argument(
            '--model-params',
            type=str,
            required=True,
            help='Pt file of params of the trained model.'
            )

    parser.add_argument(
            '--train-dataset',
            type=str,
            required=True,
            help=('Dataset used to train the model. '
                  'Provide full path')
            )
    
    parser.add_argument(
            '--normalize',
            action='store_true'
            )

    parser.add_argument(
            '--test-path',
            type=str,
            required=True,
            help='Where to save the test results'
            )

    parser.add_argument(
            '--test-dataset',
            type=str,
            required=True,
            help=('Hdf5 dataset of test samples and their genotypes. '
                  'Provide full path')
            )
    
    parser.add_argument(
            '--matched-test-dataset',
            type=str,
            help='Dataset already matched on input features'
    )

    parser.add_argument(
            '--config',
            type=str,
            required=True,
            help=('The config file used for training the model. '
                  'Provide full path')
            )
    
    parser.add_argument(
            '--task',
            choices = ['classification', 'regression'],
            required=True,
            help='Type of prediction : classification or regression'
            )

    parser.add_argument(
            '--model',
            type=str,
            choices=['Dietnet', 'Mlp'],
            default='Dietnet',
            help='Model architecture. Default: %(default)s'
            )

    parser.add_argument(
            '--input-features-stats',
            type=str,
            help = ('a')
            )

    parser.add_argument(
            '--embedding',
            required = True,
            help=('Filename of embedding returned by generate_embedding.py '
                  'and used at training time. The file must be in directory '
                  'specified with exp-path. Default: %(default)s')
            )

    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help=('Trained model of which fold to test (1st fold is 0). '
                  'Default: %(default)i')
            )

    return parser.parse_args()


if __name__ == '__main__':
    test()
