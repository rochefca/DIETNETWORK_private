import argparse
import os
import time
from datetime import datetime
import multiprocessing as mp

import numpy as np

import helpers.create_dataset_utils as cdu


# Additive encoding : genotypes 0, 1 and 2
NB_POSSIBLE_GENOTYPES = 3


def main():
    start_time = time.time()
    args = parse_args()

    #----------------------------------------------
    #               PARSE GENOTYPES
    #----------------------------------------------
    print('\n---')
    genotype_start_time = time.time()

    # Multiprocessing to load lines in parallel
    ncpus = args.ncpus
    print('Parsing genotypes using {} cpu(s)'.format(ncpus))
    pool = mp.Pool(processes=ncpus) #this starts ncpus nb of processes

    # Results returned by processes
    results = []

    # Every line is read by a different process
    with open(args.genotypes, 'r') as f:
        header_line = next(f)
        for line in f:
            # Add sample, sample's genotype accros all SNPs
            results.append(pool.apply_async(cdu.parse_genotypes, args=(line,)))

    # Get parsed samples and genotypes
    print('Parsed genotypes, getting the results')
    results_values = [p.get() for p in results]
    samples = np.array([i[0] for i in results_values])
    genotypes = np.array([i[1] for i in results_values])

    # Get snps names from header line
    snps = np.array([i.strip() for i in header_line.split('\t')[1:]], dtype='S')

    print('Parsed {} genotypes of {} samples in {} seconds'.format(
          len(snps), len(samples), time.time() - genotype_start_time))
    print('---\n')

    #----------------------------------------------
    #             LOAD CLASS LABELS
    #----------------------------------------------
    # Loading class labels from file
    print('\n---\nLoading class labels')
    samples_in_labels, labels = cdu.load_labels(args.class_labels)
    print('Loaded {} class labels of {} samples'.format(
          len(labels), len(samples_in_labels)))

    # Matching class labels with genotypes
    print('\nMatching class labels and genotypes using sample ids')
    ordered_labels = cdu.order_labels(samples, samples_in_labels, labels)
    print('Matched genotypes and class labels of {} samples'.format(
          len(ordered_labels)))

    # Encode class labels as numerical values
    print('\nNumeric encoding of class labels')
    class_label_names, \
    encoded_class_labels = cdu.numeric_encode_labels(ordered_labels)
    print('Encoded {} classes: {}\n---\n'.format(
          len(class_label_names), class_label_names))

    #----------------------------------------------
    #       COMPUTE GENOTYPE CLASS FREQUENCY
    #----------------------------------------------
    # Total number of classes
    nb_class = len(class_label_names)
    class_to_count = {}

    # Compute sum of genotypes (0-1-2) per class
    genotypes = genotypes.transpose() # rows are snps, col are inds
    embedding = np.zeros((genotypes.shape[0],nb_class*NB_POSSIBLE_GENOTYPES))
    encoded_class_labels = np.array(encoded_class_labels)
    for c in range(nb_class):
        # Select genotypes for samples of same class
        class_genotypes = genotypes[:,encoded_class_labels==c]
        nb = class_genotypes.shape[1] #nb of samples in that class
        #print('Class:', c, 'NB:', nb)
        class_to_count[c] = nb
        for genotype in range(NB_POSSIBLE_GENOTYPES):
            col = NB_POSSIBLE_GENOTYPES*c+genotype
            embedding[:,col] = (class_genotypes == genotype).sum(axis=1)/nb

    print('Computed embedding of {} classes with following number of '
          'samples per class {}'.format(nb_class, class_to_count))
    print('---')

    # Write embedding to file
    if args.out is not None:
        emb_filename = args.out
    else:
        emb_filename = 'genotype_class_freq_embedding_' \
                       + datetime.now().strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    emb_fullpath = os.path.join(args.exp_path, emb_filename)

    print('\n---\nEmbeddings generated in {} seconds'.format(
        time.time()-start_time))

    print('Saving results to {}'.format(emb_fullpath))
    np.savez(emb_fullpath, emb=embedding, label_names=class_label_names)

    print('---\n')


def parse_args():
    parser = argparse.ArgumentParser(
            description='Generate embedding'
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory where embedding will be saved. '
            )

    parser.add_argument(
            '--genotypes',
            type=str,
            required=True,
            help=('File of samples and their genotypes '
                  '(tab-separated format, one sample per line). '
                  'Missing genotypes can be encoded with NA, .\. or -1. '
                  'Provide full path')
            )

    parser.add_argument(
            '--class-labels',
            type=str,
            required=True,
            help=('File of samples and their class labels '
                  '(tab-separated format, one sample per line). '
                  'Provide full path')
            )

    parser.add_argument(
            '--ncpus',
            type=int,
            default=1,
            help=('Number of cpus available to load genotypes in parallel '
                  'Default:%(default)i')
            )

    parser.add_argument(
            '--out',
            type=str,
            help=('Optional filename for embedding file. If not provided '
                  'the file will be named '
                  'genotype_class_freq_embedding_date')
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
