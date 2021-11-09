import argparse
import os
import sys
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
from torchinfo import summary

# Multi gpu
import torch.distributed as dist
import torch.utils.data.distributed

import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu
import helpers.log_utils as lu


def main():
    args = parse_args()

    # Create dir where training info will be saved
    """
    The directory will be created in exp_path/exp_name withe the name exp_name_foldi
    where i is the number of the fold
    """
    out_dir = lu.create_out_dir(args.exp_path, args.exp_name, args.which_fold)

    # Create the full config
    """
    The full config contains 2 level info
        - hyperparams : provided in the config file
        - specifics : paths and files used in the training process
                      (specified with command line arguments)
    """
    config = {}

    # Load config file of hyperparams
    f = open(os.path.join(args.exp_path, args.exp_name, args.config))
    config_hyperparams = yaml.load(f, Loader=yaml.FullLoader)

    config['params'] = config_hyperparams

    # Add fold to config hyperparams
    config['params']['fold'] = args.which_fold

    # Add path (parsed on command line) to full config
    specifics = {} # Add paths to config dict
    specifics['exp_path'] = args.exp_path
    specifics['exp_name'] = args.exp_name
    specifics['out_dir'] = out_dir
    specifics['folds_indexes'] = args.folds_indexes
    specifics['dataset'] = args.dataset
    specifics['embedding'] = args.embedding
    specifics['preprocess_params'] = args.preprocess_params
    specifics['param_init'] = args.param_init

    config['specifics'] = specifics

    # This is the full configurations for the training
    pprint.pprint(config)

    # Save experiment configurations (out_dir/full_config.log)
    lu.save_exp_params(config['specifics']['out_dir'],'full_config.log', config)

    # Training
    train(config)


def train(config):
    # ----------------------------------------
    #               COMET PROJECT
    # ----------------------------------------
    experiment = Experiment(project_name="dietnet_1000G_cometml") # init exp
    experiment.log_parameters(config['params']) # log hyperparams
    experiment.log_others(config['specifics']) # log specifics

    # ----------------------------------------
    #               SET GPU
    # ----------------------------------------
    """
    print('Cuda available:', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    """
    # Number of gpus available on each node
    ngpus_per_node = torch.cuda.device_count()
    print('Nb of gpus available per node:', ngpus_per_node)

    # Local rank : id of the process on each node
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    print('local rank:', local_rank)

    # Rank : Combinaison of node and local rank
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
    print('Rank:', rank)

    # Available gpus on each node (ex for 2 gpus per node we have : [0,1])
    available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',',""))
    print('Available gpus:', available_gpus)

    # Set device
    current_device = int(available_gpus[local_rank])
    print('current device:', current_device)
    torch.cuda.set_device(current_device)

    # Init the process group
    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    dist.init_process_group(backend='gloo', init_method='tcp://'+os.environ.get("MASTER_ADDR")+':3456', world_size=int(os.environ.get("SLURM_NTASKS")), rank=rank)
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))

    # ----------------------------------------
    #               FIX SEED
    # ----------------------------------------
    seed = config['params']['seed']
    #torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    #           LOAD MEAN and SD
    # ----------------------------------------
    print('loading preprocessing parameters')
    # Mean and sd per feature computed on training set
    preprocess_params = np.load(os.path.join(
        config['specifics']['exp_path'],
        config['specifics']['preprocess_params'])
        )
    mus = preprocess_params['means_by_fold'][config['params']['fold']]
    sigmas = preprocess_params['sd_by_fold'][config['params']['fold']]

    # Send mus and sigmans to GPU
    mus = torch.from_numpy(mus).float().to(device)
    sigmas = torch.from_numpy(sigmas).float().to(device)

    # ----------------------------------------
    #           LOAD FOLD INDEXES
    # ----------------------------------------
    print('Loading fold indexes split into train, valid, test sets')
    all_folds_idx = np.load(os.path.join(
        config['specifics']['exp_path'],
        config['specifics']['folds_indexes']),
        allow_pickle=True)

    fold_idx = all_folds_idx['folds_indexes'][config['params']['fold']]

    # ----------------------------------------
    #       LOAD TRAIN, VALID, TEST SETS
    # ----------------------------------------
    print('Making train, valid, test sets classes')

    # Dataset hdf5 file
    dataset_file = os.path.join(
            config['specifics']['exp_path'],
            config['specifics']['dataset'])

    du.FoldDataset.dataset_file = dataset_file
    du.FoldDataset.f = h5py.File(du.FoldDataset.dataset_file, 'r')

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
        config['specifics']['exp_path'],
        config['specifics']['embedding']),
        config['params']['fold'])

    # Send to GPU
    #emb = emb.to(device)
    #emb = emb.float()

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
    with h5py.File(dataset_file, 'r') as f:
        n_targets = len(f['label_names'])

    print('\n***Nb features in models***')
    print('n_feats_emb:', n_feats_emb)
    print('n_feats:', n_feats)
    print('n_targets:', n_targets)

    # Model init
    comb_model = model.CombinedModel(
            n_feats=n_feats_emb,
            n_hidden_u_aux=config['params']['nb_hidden_u_aux'],
            n_hidden_u_main=config['params']['nb_hidden_u_aux'][-1:] \
                            +config['params']['nb_hidden_u_main'],
            n_targets=n_targets,
            param_init=config['specifics']['param_init'],
            input_dropout=config['params']['input_dropout'])

    # Note: runs script in single GPU mode only!
    #comb_model.to(device)
    comb_model.cuda()
    comb_model = torch.nn.parallel.DistributedDataParallel(comb_model, device_ids=[current_device])
    print('From Rank: {}, ==> Preparing data..'.format(rank))

    #print(summary(comb_model.feat_emb, input_size=(294427,1,1,78)))
    #print(summary(comb_model.disc_net, input_size=[(138,1,1,294427),(100,294427)]))

    # ----------------------------------------
    #               OPTIMIZATION
    # ----------------------------------------
    # Loss
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    lr = config['params']['learning_rate']
    optimizer = torch.optim.Adam(comb_model.parameters(), lr=lr)

    # Max nb of epochs
    n_epochs = config['params']['epochs']

    # ----------------------------------------
    #             BATCH GENERATORS
    # ----------------------------------------
    batch_size = config['params']['batch_size']

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    train_generator = DataLoader(train_set, sampler=train_sampler,
                                 batch_size=batch_size, num_workers=0)
    valid_generator = DataLoader(valid_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0)
    test_generator = DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)

    """
    for x_batch, y_batch, _ in train_generator:
        print(summary(model=comb_model, input_data=[emb, x_batch]))
        break
    """
    # Save model summary
    lu.save_model_summary(config['specifics']['out_dir'], comb_model, criterion, optimizer)

    # Monitoring: Epoch loss and accuracy
    train_losses = []
    train_acc = []
    valid_losses = []
    valid_acc = []

    # this is the discriminative model!
    discrim_model = mlu.create_disc_model(comb_model, emb, device)

    # Monitoring: validation baseline
    min_loss, best_acc = mlu.eval_step(device, valid_generator, len(valid_set),
                                       discrim_model, criterion, mus, sigmas)
    print('baseline loss:',min_loss, 'baseline acc:', best_acc)

    # Monitoring: Nb epoch without improvement after which to stop training
    patience = 0
    max_patience = config['params']['patience']
    has_early_stoped = False

    with experiment.train():
        total_time = 0
        for epoch in range(n_epochs):
            print('Epoch {} of {}'.format(epoch+1, n_epochs), flush=True)
            start_time = time.time()

            train_sampler.set_epoch(epoch)

            # ---Training---
            comb_model.train()

            # Monitoring: Minibatch loss and accuracy
            train_minibatch_mean_losses = []
            train_minibatch_n_right = [] #nb of good classifications

            #b = 0
            for x_batch, y_batch, _ in train_generator:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                x_batch.float()
                # Replace missing values
                du.replace_missing_values(x_batch, mus)
                # Normalize
                x_batch = du.normalize(x_batch, mus, sigmas)

                optimizer.zero_grad()

                # Forward pass
                discrim_model_out = comb_model(emb, x_batch)

                # Get prediction (softmax)
                _, pred = mlu.get_predictions(discrim_model_out)

                # Compute loss
                loss = criterion(discrim_model_out, y_batch)
                # Compute gradients
                loss.backward()

                # Optim
                optimizer.step()

                # Monitoring: Minibatch
                train_minibatch_mean_losses.append(loss.item())
                train_minibatch_n_right.append(((y_batch - pred) ==0).sum().item())

                #b += len(y_batch)
                #print('completed batch', b, 'samples passed')

            # Monitoring: Epoch
            epoch_loss = np.array(train_minibatch_mean_losses).mean()
            train_losses.append(epoch_loss)

            epoch_acc = mlu.compute_accuracy(train_minibatch_n_right,
                                             len(train_set))
            train_acc.append(epoch_acc)
            print('train loss:', epoch_loss, 'train acc:', epoch_acc, flush=True)

            # Comet
            #experiment.log_metric("train_accuracy", epoch_acc, step=epoch)


            # ---Validation---
            """
            comb_model = comb_model.eval()
            epoch_loss, epoch_acc = mlu.eval_step(device, valid_generator, len(valid_set),
                                                  discrim_model, criterion, mus, sigmas)

            valid_losses.append(epoch_loss)
            valid_acc.append(epoch_acc)
            print('valid loss:', epoch_loss, 'valid acc:', epoch_acc,flush=True)

            # Comet
            #experiment.log_metric("valid_accuracy", epoch_acc, step=epoch)
            """
            # Early stop
            if mlu.has_improved(best_acc, epoch_acc, min_loss, epoch_loss):
                patience = 0
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                if epoch_loss < min_loss:
                    min_loss = epoch_loss
                # Save model parameters (for later inference)
                print('best acc achieved: {} (loss {}) at epoch {} saving model ...'.format(best_acc, epoch_loss, epoch))
                lu.save_model_params(config['specifics']['out_dir'], comb_model)
            else:
                patience += 1

            if patience >= max_patience:
                has_early_stoped = True
                n_epochs = epoch - patience
                break # exit training loop

            # Anneal laerning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = \
                        param_group['lr'] * config['params']['learning_rate_annealing']

            end_time = time.time()
            total_time += end_time-start_time
            print('time:', end_time-start_time, flush=True)

        # Finish training
        print('Early stoping:', has_early_stoped)

    # ---Test---
    with experiment.test():
        #  reload weights from early stopped model
        discrim_model = mlu.load_model(os.path.join(config['specifics']['out_dir'], 'model_params.pt'),
                                       emb,
                                       device,
                                       n_feats=n_feats_emb,
                                       n_hidden_u_aux=config['params']['nb_hidden_u_aux'],
                                       n_hidden_u_main=config['params']['nb_hidden_u_aux'][-1:]+config['params']['nb_hidden_u_main'],
                                       n_targets=n_targets,
                                       input_dropout=config['params']['input_dropout'])

        comb_model = comb_model.eval()
        score, pred, acc = mlu.test(device, test_generator, len(test_set), discrim_model, mus, sigmas)

        print('Final accuracy:', str(acc))
        print('total running time:', str(total_time))

        # Comet
        #experiment.log_metric("accuracy", acc)

    ## TO DO
    """
    # Save results
    lu.save_results(config['out_dir'],
                    test_set.samples,
                    test_set.ys,
                    data['label_names'],
                    score, pred,
                    n_epochs)

    # Save additional data
    lu.save_additional_data(config['out_dir'],
                            train_set.samples, valid_set.samples,
                            test_set.samples, test_set.ys,
                            pred, score,
                            data['label_names'], data['snp_names'],
                            mus, sigmas)
    """
def parse_args():
    parser = argparse.ArgumentParser(
            description=('Preprocess features for main network '
                         'and train model for a given fold')
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

    parser.add_argument(
            '--config',
            type=str,
            default='config.yaml',
            help='Yaml file of hyperparameter. Default: %(default)s'
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
            '--folds-indexes',
            type=str,
            default='folds_indexes.npz',
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
        '--preprocess-params',
        type=str,
        default='preprocessing_params.npz',
        help='Normalization parameters obtained with get_preprocessing_params.py'
        )

    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    parser.add_argument(
            '--param-init',
            type=str,
            help='File of parameters initialization values'
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
