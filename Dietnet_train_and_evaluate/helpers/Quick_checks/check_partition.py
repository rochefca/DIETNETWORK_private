import argparse

import numpy as np


def main():
    args = parse_args()

    # Load partition
    data = np.load(args.partition, allow_pickle=True)
    partition = data['folds_indexes']

    # Check for leakage between train-valid-test sets
    print('\n---\n')
    print('Checking leakage between train-valid-test sets in each fold')
    print('ALL values must be 0 for no leakage\n')
    for fold in range(len(partition)):
        print('***')
        print('FOLD:', fold)
        train = np.array(partition[fold][0])
        valid = np.array(partition[fold][1])
        test = np.array(partition[fold][2])

        print('train-valid:', len(np.intersect1d(train,valid)))

        print('train-test:', len(np.intersect1d(train,test)))

        print('valid-test:', len(np.intersect1d(valid,test)))

    print('\n---\n')


    # Check for redundancy train, valid and test sets across folds
    print('\n---\n')
    print('Checking redundancy between train, valid and test sets across folds')
    print('There should be:')
    print('- no redundancy between test sets')
    print('- very little to no redundancy between valid sets')
    print('- some, but not complete, redundancy between train sets\n')
    for i in range(len(partition)):
        print('***')
        for j in range(i+1,len(partition)):
            print('Intersect folds {} and {}'.format(i,j))

            # Train sets
            train_i = np.array(partition[i][0])
            train_j = np.array(partition[j][0])
            print('Train:', len(np.intersect1d(train_i, train_j)))

            # Valid sets
            valid_i = np.array(partition[i][1])
            valid_j = np.array(partition[j][1])
            print('Valid:', len(np.intersect1d(valid_i, valid_j)))

            # Test sets
            test_i = np.array(partition[i][2])
            test_j = np.array(partition[j][2])
            print('Test:', len(np.intersect1d(test_i, test_j)))


def parse_args():
    parser = argparse.ArgumentParser(
            description='Quick check of overlap between folds.'
            )

    parser.add_argument(
            '--partition',
            type=str,
            required=True,
            help=('Npz dataset partition returned by partition_data.py '
                  'Provide full path')
            )

    return parser.parse_args()

if __name__ == '__main__':
    main()
