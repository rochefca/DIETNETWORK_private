import argparse
import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import wandb

import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu
import helpers.log_utils as lu


def main():
    args = parse_args()

    # Directory to save experiment info
    out_dir = lu.create_out_dir(args.exp_path, args.exp_name, args.which_fold)

    # Save experiment parameters
    lu.save_exp_params(out_dir, args)

    # Set GPU
    print('Cuda available:', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Fix seed
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print('Seed:', str(seed))

    # Get fold data (indexes and samples are np arrays, x,y are tensors)
    data = du.load_data(os.path.join(args.exp_path,args.dataset))
    folds_indexes = du.load_folds_indexes(
            os.path.join(args.exp_path,args.folds_indexes))
    (train_indexes, valid_indexes, test_indexes,
     x_train, y_train, samples_train,
     x_valid, y_valid, samples_valid,
     x_test, y_test, samples_test) = du.get_fold_data(args.which_fold,
                                        folds_indexes, data)

    # Convert np array to torch tensors
    x_train, x_valid, x_test = torch.from_numpy(x_train), \
            torch.from_numpy(x_valid), torch.from_numpy(x_test)
    y_train, y_valid, y_test = torch.from_numpy(y_train), \
            torch.from_numpy(y_valid), torch.from_numpy(y_test)

    # Put data on GPU
    x_train, x_valid, x_test = x_train.to(device), x_valid.to(device), \
            x_test.to(device)
    x_train, x_valid, x_test = x_train.float(), x_valid.float(), \
            x_test.float()

    y_train, y_valid, y_test = y_train.to(device), y_valid.to(device), \
            y_test.to(device)

    # Compute mean and sd of training set for normalization
    mus, sigmas = du.compute_norm_values(x_train)

    # Replace missing values
    du.replace_missing_values(x_train, mus)
    du.replace_missing_values(x_valid, mus)
    du.replace_missing_values(x_test, mus)

    # Normalize
    x_train_normed = du.normalize(x_train, mus, sigmas)
    x_valid_normed = du.normalize(x_valid, mus, sigmas)
    x_test_normed = du.normalize(x_test, mus, sigmas)

    # Make fold final dataset
    train_set = du.FoldDataset(x_train_normed, y_train, samples_train)
    valid_set = du.FoldDataset(x_valid_normed, y_valid, samples_valid)
    test_set = du.FoldDataset(x_test_normed, y_test, samples_test)

    # Load embedding
    emb = du.load_embedding(os.path.join(args.exp_path,args.embedding),
                            args.which_fold)
    emb = emb.to(device)
    emb = emb.float()

    # Normalize embedding
    #emb_norm = (emb ** 2).sum(0) ** 0.5
    #emb = emb/emb_norm

    # Alternative embedding
    #print('Other normalization for embedding')
    #mean_by_col = emb.mean(0)
    #std_by_col = emb.std(0)

    #emb = (emb - mean_by_col)/std_by_col

    # Instantiate model
    # Input size
    if len(emb.size()) == 1:
        n_feats_emb = 1 # input of aux net, 1 value per SNP
        emb = torch.unsqueeze(emb, dim=1) # match size in Linear fct (nb_snpsx1)
    else:
        n_feats_emb = emb.size()[1] # input of aux net

    n_feats = emb.size()[0] # input of main net

    # Hidden layer size aux net
    emb_n_hidden_u = [int(i.strip()) for i in args.emb_n_hidden_u.strip('[').strip(']').split(',')]
    # Hidden layer size main net
    discrim_n_hidden_u = [emb_n_hidden_u[-1]] # first hidden layer must match size of aux output layer
    discrim_n_hidden_u += [int(i.strip()) for i in args.discrim_n_hidden_u.strip('[').strip(']').split(',')]
    print(emb_n_hidden_u)
    print(discrim_n_hidden_u)
    # Output layer main net
    n_targets = max(torch.max(train_set.ys).item(),
                    torch.max(valid_set.ys).item(),
                    torch.max(test_set.ys).item()) + 1 #0-based encoding

    print('\n***Nb features in models***')
    print('n_feats_emb:', n_feats_emb)
    print('n_feats:', n_feats)

    # --- CONFIG DICT W&B ---
    config_dict = dict(
            batch_size=138,
            n_hidden_u_aux=emb_n_hidden_u,
            n_hidden_u_main=discrim_n_hidden_u,
            input_dropout=args.input_dropout,
            patience=args.patience,
            learning_rate=args.learning_rate,
            learning_rate_annealing=args.learning_rate_annealing,
            epochs=args.epochs)

    print('wandb.init')
    wandb.init(config=config_dict)
    # Match log and executions
    config = wandb.config

    comb_model = model.CombinedModel(
                 n_feats=n_feats_emb,
                 n_hidden_u_aux=config.n_hidden_u_aux,
                 n_hidden_u_main=config.n_hidden_u_main,
                 n_targets=n_targets,
                 param_init=args.param_init,
                 input_dropout=config.input_dropout)

    #  Note: runs script in single GPU mode only!
    comb_model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    lr = config.learning_rate
    optimizer = torch.optim.Adam(comb_model.parameters(), lr=lr)

    # Training loop hyper param
    n_epochs = config.epochs
    batch_size = config.batch_size

    # Minibatch generators
    train_generator = DataLoader(train_set,
                                 batch_size=config.batch_size)
    valid_generator = DataLoader(valid_set,
                                 batch_size=config.batch_size,
                                 shuffle=False)
    test_generator = DataLoader(test_set,
                                batch_size=config.batch_size,
                                shuffle=False)

    # Save model summary
    lu.save_model_summary(out_dir, comb_model, criterion, optimizer)

    # Monitoring: Epoch loss and accuracy
    train_losses = []
    train_acc = []
    valid_losses = []
    valid_acc = []

    # this is the discriminative model!
    discrim_model = mlu.create_disc_model(comb_model, emb, device)

    # Monitoring: validation baseline
    min_loss, best_acc = mlu.eval_step(valid_generator, len(valid_set),
                                       discrim_model, criterion)
    print('baseline loss:',min_loss, 'baseline acc:', best_acc)

    # Monitoring: Nb epoch without improvement after which to stop training
    patience = 0
    max_patience = config.patience
    has_early_stoped = False

    wandb.watch(comb_model, criterion, log='all', log_freq=100)
    total_time = 0
    for epoch in range(n_epochs):
        print('Epoch {} of {}'.format(epoch+1, n_epochs), flush=True)
        start_time = time.time()

        # ---Training---
        comb_model.train()

        # Monitoring: Minibatch loss and accuracy
        train_minibatch_mean_losses = []
        train_minibatch_n_right = [] #nb of good classifications

        for x_batch, y_batch, _ in train_generator:
            optimizer.zero_grad()

            # Forward pass
            discrim_model_out = comb_model(emb, x_batch)

            # Get prediction (softmax)
            _, pred = mlu.get_predictions(discrim_model_out)

            # Compute loss
            loss = criterion(discrim_model_out, y_batch)
            # Compute gradients in discrim net
            loss.backward()

            # Optim
            optimizer.step()

            # Monitoring: Minibatch
            train_minibatch_mean_losses.append(loss.item())
            train_minibatch_n_right.append(((y_batch - pred) ==0).sum().item())

        # Monitoring: Epoch
        epoch_loss = np.array(train_minibatch_mean_losses).mean()
        train_losses.append(epoch_loss)

        epoch_acc = mlu.compute_accuracy(train_minibatch_n_right,
                                         len(train_set))
        train_acc.append(epoch_acc)
        print('train loss:', epoch_loss, 'train acc:', epoch_acc, flush=True)

        # W&B
        wandb.log({'epoch':epoch, 'train_loss':epoch_loss, 'train_acc':epoch_acc}, step=epoch)


        # ---Validation---
        comb_model = comb_model.eval()
        epoch_loss, epoch_acc = mlu.eval_step(valid_generator, len(valid_set),
                                              discrim_model, criterion)

        valid_losses.append(epoch_loss)
        valid_acc.append(epoch_acc)
        print('valid loss:', epoch_loss, 'valid acc:', epoch_acc,flush=True)

        # W&B
        wandb.log({'valid_loss':epoch_loss, 'valid_acc':epoch_acc}, step=epoch)

        # Early stop
        if mlu.has_improved(best_acc, epoch_acc, min_loss, epoch_loss):
            patience = 0
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            if epoch_loss < min_loss:
                min_loss = epoch_loss
            # Save model parameters (for later inference)
            print('best acc achieved: {} (loss {}) at epoch {} saving model ...'.format(best_acc, epoch_loss, epoch))
            lu.save_model_params(out_dir, comb_model)
        else:
            patience += 1

        if patience >= max_patience:
            has_early_stoped = True
            n_epochs = epoch - patience
            break # exit training loop

        # Anneal laerning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = \
                    param_group['lr'] * args.learning_rate_annealing

        end_time = time.time()
        total_time += end_time-start_time
        print('time:', end_time-start_time, flush=True)

    # Finish training
    print('Early stoping:', has_early_stoped)

    # ---Test---

    #  reload weights from early stopped model
    discrim_model = mlu.load_model(os.path.join(out_dir, 'model_params.pt'),
                                   emb,
                                   device,
                                   n_feats=n_feats_emb,
                                   n_hidden_u_aux=emb_n_hidden_u,
                                   n_hidden_u_main=discrim_n_hidden_u,
                                   n_targets=n_targets,
                                   input_dropout=args.input_dropout)

    comb_model = comb_model.eval()
    score, pred, acc = mlu.test(test_generator, len(test_set), discrim_model)

    print('Final accuracy:', str(acc))
    print('total running time:', str(total_time))

    # Save results
    lu.save_results(out_dir,
                    test_set.samples,
                    test_set.ys,
                    data['label_names'],
                    score, pred,
                    n_epochs)

    # Save additional data
    lu.save_additional_data(out_dir,
                            train_set.samples, valid_set.samples,
                            test_set.samples, test_set.ys,
                            pred, score,
                            data['label_names'], data['snp_names'],
                            mus, sigmas)


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
            '--dataset',
            type=str,
            default='dataset.npz',
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
            '--which-fold',
            type=int,
            default=0,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    parser.add_argument(
            '--train-valid-ratio',
            type=float,
            default=0.75,
            help=('Ratio (between 0-1) for split of train and valid sets. '
                  'For example, 0.75 will use 75%% of data for training '
                  'and 25%% of data for validation. Default: %(default).2f')
            )

    parser.add_argument(
            '--seed',
            type=int,
            default=23,
            help=('Fix feed for shuffle of data before the split into train '
                  'and valid sets. Default: %(default)i')
            )

    parser.add_argument(
            '--emb-n-hidden-u',
            type=str,
            default='[100,100]',
            help=('Number of neurons in every layers of auxiliary network. '
                'Default: %(default)s')
            )

    parser.add_argument(
            '--discrim-n-hidden-u',
            type=str,
            default='[100]',
            help=('Number of neurons in every layers of main network excluding 1st hidden layer. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--patience',
            type=int,
            default=1000,
            help=('Number of epochs without validation improvement after '
                  'which to stop training. Default: %(default)i')
            )

    parser.add_argument(
            '--learning-rate',
            '-lr',
            type=float,
            default=0.00003,
            help='Learning rate. Default: %(default)f'
            )

    parser.add_argument(
            '--learning-rate-annealing',
            '-lra',
            type=float,
            default=0.999,
            help='Learning rate annealing. Default: %(default)f'
            )

    parser.add_argument(
            '--epochs',
            type=int,
            default=20000,
            help='Max number of epochs. Default: %(default)i'
            )

    parser.add_argument(
            '--input-dropout',
            type=float,
            default=0.0,
            help=('Input dropout. The number, between 0 and 1, indicates '
                  'the probability of an element to be zeroed. '
                  'Default: %(default)f')
            )

    parser.add_argument(
            '--param-init',
            type=str,
            help='File of parameters initialization values'
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
