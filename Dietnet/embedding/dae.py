import os
import argparse
import time

import numpy as np
import h5py

import torch
from torch.utils.data import DataLoader

import dae_helpers.dae_helpers as daeh
import dae_helpers.mainloop_utils as dae_mlu

def main():
    NB_GENOTYPES = 3

    # Monitoring time for whole experiment
    exp_start_time = time.time()

    # Parse command line arguments
    args = parse_args()

    # Set device
    print('\n---\nSetting device')
    print('Cuda available:', torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    print('---\n')

    # Fix seed
    seed = 23
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    #                   DATA
    # ----------------------------------------
    # Fold indices
    fold = args.which_fold
    indices_byfold = np.load(args.partition, allow_pickle=True)
    fold_indices = indices_byfold['folds_indexes'][fold]

    daeh.FoldDataset.dataset_file = args.dataset
    daeh.FoldDataset.f = h5py.File(daeh.FoldDataset.dataset_file, 'r')

    # Make datasets
    train_set = daeh.FoldDataset(fold_indices[0])
    valid_set = daeh.FoldDataset(fold_indices[1])
    test_set = daeh.FoldDataset(fold_indices[2])

    print('Loaded train ({} samples), valid ({} samples) and '
          'test ({} samples) sets'.format(
              len(train_set), len(valid_set), len(test_set)))

    # ----------------------------------------
    #                   MODEL
    # ----------------------------------------
    print('\n---\nMaking model')

    n_input_feats = train_set.__getitem__(0)[0].shape[0]
    print('Nb input features:', n_input_feats)

    NB_HIDDEN_U = [1000,1000]
    BOTTLENECK = 20000
    encoder = daeh.DAEencoder(n_input_feats, NB_HIDDEN_U, BOTTLENECK)
    decoder = daeh.DAEdecoder(n_input_feats, NB_HIDDEN_U, BOTTLENECK)

    print(encoder)
    print(decoder)

    # Loss : k-dimensional cross entropy
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    lr = 0.001
    LRA = 0.999
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=lr)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=lr)

    print('Learning rate: {} and learning rate annealing: {}'.format(lr, LRA))

    print('---')


    # ----------------------------------------
    #               TRAINING
    # ----------------------------------------
    # Batch generators
    BS = 138
    train_generator = DataLoader(train_set, batch_size=BS, num_workers=0)
    valid_generator = DataLoader(valid_set, batch_size=BS,
                                 shuffle=False, num_workers=0)
    test_generator = DataLoader(test_set, batch_size=BS,
                                shuffle=False, num_workers=0)

    # Baseline
    print('Computing baseline')
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    # Where to save baseline results
    results_filename = 'baseline_results'
    results_fullpath = os.path.join(
            args.exp_path, args.exp_name, results_filename)
    baseline_loss, baseline_reconst_acc = dae_mlu.valid_step(results_fullpath,
            valid_generator, encoder, decoder, criterion, device)

    print('Baseline loss: {} baseline reconstruction acc: {}'.format(
                                        baseline_loss, baseline_reconst_acc))

    # For monitoring model progress
    best_loss = baseline_loss
    best_reconst_acc = baseline_reconst_acc
    patience = 0
    MAX_PATIENCE = 1000
    max_patience = MAX_PATIENCE

    # For monitoring best model
    best_model_filename = 'model_best.pt'
    best_model_fullpath = os.path.join(args.exp_path, args.exp_name,
                                       best_model_filename)

    # For monitoring last model
    last_model_filename = 'model_last.pt'
    last_model_fullpath = os.path.join(args.exp_path, args.exp_name,
                                       last_model_filename)

    EPOCHS= args.n_epochs
    print('\nStarting training for {} epochs with max patience of {}'.format(
          EPOCHS, max_patience))

    for epoch in range(1, EPOCHS+1):
        print('Epoch {} of {}'.format(epoch, EPOCHS), flush=True)
        epoch_start_time = time.time()

        # Send model to GPU
        encoder.to(device)
        decoder.to(device)

        # --- Train step ---
        encoder.train()
        decoder.train()

        # Where to save epoch reconstruction results
        results_filename = 'epoch' + str(epoch) + '_train_results'
        results_fullpath = os.path.join(
                args.exp_path, args.exp_name, results_filename)

        train_loss, train_reconst_acc = dae_mlu.train_step(results_fullpath,
                train_generator, encoder, decoder,
                optimizer_encoder, optimizer_decoder, criterion, device)

        # --- Valid step ---
        encoder.eval()
        decoder.eval()

        # Where to save epoch reconstruction results
        results_filename = 'epoch' + str(epoch) + '_valid_results'
        results_fullpath = os.path.join(
                args.exp_path, args.exp_name, results_filename)

        valid_loss, valid_reconst_acc = dae_mlu.valid_step(results_fullpath,
                valid_generator, encoder, decoder, criterion, device)

        print('Train loss: {} Train reconstruction acc: {}\n'
              'Valid loss: {} Valid reconstruction acc {}'.format(
               train_loss, train_reconst_acc, valid_loss, valid_reconst_acc))

        # --- Save model ---
        # Put model on cpu, otherwise torch.save causes a memory error
        encoder.cpu()
        decoder.cpu()
        torch.save({'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'train_loss': train_loss,
            'train_reconst_acc': train_reconst_acc,
            'valid_loss': valid_loss,
            'valid_reoncst_acc': valid_reconst_acc}, last_model_fullpath)

        # --- Update best results ---
        # Check if this model is better than previous best model and
        has_improved = False
        if valid_reconst_acc > best_reconst_acc:
            has_improved = True
            best_reconst_acc = valid_reconst_acc
        elif valid_reconst_acc == best_reconst_acc and valid_loss < best_loss:
            has_improved=True
            best_loss = valid_loss

        # Update patience and save best model
        if has_improved:
            patience = 0
            torch.save({'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'train_loss': train_loss,
                'train_reconst_acc': train_reconst_acc,
                'valid_loss': valid_loss,
                'valid_reoncst_acc': valid_reconst_acc}, best_model_fullpath)

            print('Saving best model achieved at epoch {} with loss {} and '
                  'reconstruction acc {}'.format(epoch, valid_loss,
                      valid_reconst_acc))
        else:
            patience += 1

        # --- Lr annealing ---
        for param_group in optimizer_encoder.param_groups:
            param_group['lr'] = param_group['lr'] * LRA
        for param_group in optimizer_decoder.param_groups:
            param_group['lr'] = param_group['lr'] * LRA

        # --- Early stopping  ---
        if patience >= max_patience:
            print('Early stopping, exit training loop')
            break

        print('Executed epoch in {} seconds'.format(
               time.time() - epoch_start_time))

    # ----------------------------------------
    #                   TEST
    # ----------------------------------------
    print('\n---\nTest')
    test_start_time = time.time()

    print('Loading best model achieved')
    # Load best achieved model
    checkpoint = torch.load(best_model_fullpath)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.to(device)
    decoder.to(device)

    print('Loaded best model (epoch {})'.format(checkpoint['epoch']))

    # Test step
    print('Computing test')
    encoder.eval()
    decoder.eval()

    # Where to save test results
    results_fullpath = os.path.join(
            args.exp_path, args.exp_name, 'test_results')

    test_loss, test_reconst_acc = dae_mlu.valid_step(results_fullpath,
            test_generator, encoder, decoder, criterion, device)

    print('Test loss: {} Test reconstruction accuracy: {}'.format(
          test_loss, test_reconst_acc))

    print('Excuted test in {} seconds\n---\n'.format(
           time.time()-test_start_time))

    print('End of execution')
    print('Experiment was executed in {} seconds\n'.format(
           time.time()-exp_start_time))

def parse_args():
    parser = argparse.ArgumentParser(
            description=('Train Denoising Auto Encoder (DAE) SNP to vec')
            )

    # Temporary
    parser.add_argument(
            '--n-epochs',
            type=int,
            default=5
            )
    # Path
    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory where to save DAE results'
            )

    parser.add_argument(
            '--exp-name',
            type=str,
            required=True,
            help='Experiment name. Results will be saved to exp-path/exp-name'
            )

    # Files
    parser.add_argument(
            '--dataset',
            type=str,
            required=True,
            help=('Hdf5 dataset created with create_dataset.py '
                  'Provide full path')
            )

    parser.add_argument(
            '--partition',
            type=str,
            required=True,
            help=('Npz dataset partition returned by partition_data.py '
                  'Provide full path')
            )

    # Fold
    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    return parser.parse_args()




if __name__=='__main__':
    main()
