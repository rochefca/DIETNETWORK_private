import argparse
import time
import numpy as np
import pandas as pd
from pandas_plink import read_plink
import dask.array as da
import torch
from torch.utils.data import Dataset, DataLoader


class DaskGenotypeDataset(Dataset):
    def __init__(self, dask_array, batch_size):
        """
        dask_array: Dask array of genotypes, shape: n_samples x n_snps
        batch_size matches chunk size of dask array
        """
        self.X = dask_array
        self.batch_size = batch_size
        self.n_samples = dask_array.shape[0]

    def __len__(self):
        # Returns nb of batches
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = min(start + self.batch_size, self.n_samples) # last batch can be smaller

        # Load batch
        batch = self.X[start:stop].compute()  # np array

        return torch.from_numpy(batch).float()


def main():
    args = parse_args()
    
    # Reference dataset
    print('Reference dataset')
    start_time = time.time()
    ref_bim, ref_fam, ref_bed = read_plink(args.ref_plink)
    print('Reference dataset:', time.time()-start_time, 'seconds')
    
    # Query SNPs
    print('Query SNPs')
    start_time = time.time()
    query_bim = pd.read_csv(args.query_bim, sep="\t", header=None,
                            names=["chrom", "snp", "cm", "pos", "a1", "a2"])
    print('Query SNPs:', time.time()-start_time, 'seconds')
    
    ref_snps = ref_bim.snp.values
    query_snps = query_bim.snp.values

    # Mask: True if SNP NOT in query (shape 1 x nb snps)
    print('Matching Reference and Query SNPs')
    start_time = time.time()
    mask_missing = ~np.in1d(ref_bim.snp.values, query_bim.snp.values)
    print('Matching Reference and Query SNPs:', time.time()-start_time, 'seconds')
    
    # Turning mask_missing vector into a dask array
    print('Making missing dask array')
    start_time = time.time()
    mask_missing_da = da.from_array(mask_missing[:, None],  # shape (n_snps, 1)
                                    chunks=(ref_bed.chunks[0], 1))

    ref_bed_querymatched = da.where(mask_missing_da, -1.0, ref_bed)
    print('Making missing dask array:', time.time()-start_time)
    print(ref_bed_querymatched)

    # To do for performance
    ref_bed_querymatched = ref_bed_querymatched.rechunk((50_000, 128))
    
    # Lazy transpose
    X = ref_bed_querymatched.T
    
    # Pytorch Dataset
    print('Making Pytorch Dataset')
    start_time = time.time()
    batch_size=256
    dataset = DaskGenotypeDataset(X, batch_size)
    print('Making Pytorch Dataset:', time.time()-start_time, 'seconds')
    
    # Pytorch Dataloader
    print('Making Pytorch DataLoader')
    start_time = time.time()
    loader = DataLoader(dataset,
    batch_size=None,   # IMPORTANT: Dataset already returns batches
    shuffle=False, num_workers=0)
    print('Making Pytorch DataLoader:', time.time()-start_time, 'seconds')
    
    print('Iterating batches (calling .compute)')
    start_time = time.time()
    for batch in loader:
        print(batch.shape)
    print('Iterating batches:', time.time()-start_time, 'seconds')


def parse_args():
    parser = argparse.ArgumentParser(
            description='Test a trained model in another dataset'
            )
    
    parser.add_argument(
            '--ref-plink',
            type=str,
            required=True,
            help='Plink file of dataset used to train dietnet'
            )
    
    parser.add_argument(
            '--query-bim',
            type=str,
            required=True,
            help='Plink bim file of query dataset'
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