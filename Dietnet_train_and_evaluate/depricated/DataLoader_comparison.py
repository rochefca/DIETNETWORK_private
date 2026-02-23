import argparse
import time
import numpy as np
import pandas as pd
from pandas_plink import read_plink
import dask.array as da
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class DaskGenotypeDataset(Dataset):
    def __init__(self, plink_file, batch_size, chunksize):
        print('Read plink files')
        start_time = time.time()
        ref_bim, ref_fam, ref_bed = read_plink(plink_file)
        print('Read plink files:', time.time()-start_time, 'seconds')
        
        self.X = ref_bed.rechunk(chunksize).T
        self.n_samples = self.X.shape[0]
        self.batch_size = batch_size
        
        print('self.X:', self.X)
        print('self.n_samples:', self.n_samples)

    def __len__(self):
        # Returns nb of batches
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = min(start + self.batch_size, self.n_samples) # last batch can be smaller

        # Load batch
        batch = self.X[start:stop].compute()  # np array

        return torch.from_numpy(batch).float()


class Hdf5GenotypeDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        with h5py.File(hdf5_file, 'r') as f:
            self.nsamples = f['samples'].shape[0]
        
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            batch = f['inputs'][idx,:]
            
        return torch.from_numpy(batch).float()
        

def main():
    args = parse_args()
    
    chunksizes = [(229_986, args.batch_size),(100_000, args.batch_size), (200_000, args.batch_size), (50_000, args.batch_size), (1024, args.batch_size),
                  (229_986, args.batch_size*2),(100_000, args.batch_size*2), (200_000, args.batch_size*2), (50_000, args.batch_size*2), (1024, args.batch_size*2),
                  (229_986, args.batch_size*10),(100_000, args.batch_size*10), (200_000, args.batch_size*10), (50_000, args.batch_size*10), (1024, args.batch_size*10)]
    for c in chunksizes:
        # Pytorch Dask array Dataset
        dataset = DaskGenotypeDataset(args.plink_file, args.batch_size, c)
        
        # Pytorch Dataloader
        loader = DataLoader(dataset,
        batch_size=None,   # IMPORTANT: Dataset already returns batches
        shuffle=False, num_workers=0)
        
        # Iterating batches
        start_time = time.time()
        for batch in loader:
                print(batch.shape)
        print('Iterating batches:', 'chunksize:', dataset.X.chunksize, time.time()-start_time, 'seconds')
    
    # Pytorch Hdf5 Dataset
    dataset = Hdf5GenotypeDataset(args.hdf5_file)
    
    # Pytorch Dataloader
    loader = DataLoader(dataset, batch_size=args.batch_size,shuffle=False, num_workers=0)
    
    # Iterating batches
    start_time = time.time()
    for batch in loader:
        pass
    print('Iterating batches (hdf5):', time.time()-start_time, 'seconds')


def parse_args():
    parser = argparse.ArgumentParser(
            description='Test a trained model in another dataset'
            )
    
    parser.add_argument(
            '--plink-file',
            type=str,
            required=True,
            help='Plink file of dataset used to train dietnet'
            )
    
    parser.add_argument(
            '--hdf5-file',
            type=str,
            required=True,
            help='Hdf5 file of dataset used to train dietnet'
            )
    
    parser.add_argument(
            '--batch-size',
            type=int,
            required=True,
            help='Nb of samples per batch'
            )

    """
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
            '--test-name',
            type=str,
            required=True
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
    """
    return parser.parse_args()


if __name__ == '__main__':
    main()