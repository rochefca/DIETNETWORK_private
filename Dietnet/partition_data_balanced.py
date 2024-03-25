import argparse
import os
import h5py

import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.utils import shuffle


def make_folds(
    features,
    labels=None,
    n_splits=5,
    train_ratio=0.75,
    valid_ratio=0.5,
    balance_class=True,
    random_state=None,
):
    """
    Create folds for cross-validation.

    Args:
        features (array-like): Input features.
        labels (array-like): Target labels (optional).
        n_splits (int): Number of folds (default: 5).
        train_ratio (float): Proportion of the dataset to include in the training set (default: 0.75).
        valid_ratio (float): Proportion of the test set (1 - train_ratio) to include in the validation set versus the final test set (default: 0.5).
        balance_class (bool): Whether to balance the folds based on labels (default: True).
        random_state (int): Random seed for reproducibility (default: None).

    Returns:
        list: List of tuples, where each tuple contains (train_indices, test_indices, valid_indices) for each fold.
    """
    test_ratio = 1 - train_ratio
    splitter_outer = (
        StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_ratio, random_state=random_state
        )
        if balance_class
        else ShuffleSplit(
            n_splits=n_splits, test_size=test_ratio, random_state=random_state
        )
    )
    splitter_inner = (
        StratifiedShuffleSplit(
            n_splits=2, test_size=valid_ratio, random_state=random_state
        )
        if balance_class
        else ShuffleSplit(n_splits=2, test_size=0.5, random_state=random_state)
    )

    folds = []
    for train_index, test_valid_index in splitter_outer.split(features, labels):
        features_test_valid, labels_test_valid = (
            features[test_valid_index],
            labels[test_valid_index] if labels is not None else None,
        )
        test_index, valid_index = next(
            splitter_inner.split(features_test_valid, labels_test_valid)
        )
        folds.append(
            (train_index, test_valid_index[valid_index], test_valid_index[test_index])
        )

    return folds


def get_infos(data, indices, save_dicts):
    """
    Get information about the partitioning.

    Args:
        data (tuple): Tuple containing (features, labels_encoded).
        indices (list): List of indices for train, valid, and test sets.
        save_dicts (tuple): Tuple containing (output_dict, samples_dict).

    Returns:
        tuple: Tuple containing (output_dict, samples_dict).
    """
    features, labels_encoded = data
    output_dict, samples_dict = save_dicts
    fold_index, train_index, valid_index, test_index = indices

    for set_name, set_index in zip(
        ["train", "valid", "test"], [train_index, valid_index, test_index]
    ):
        unique_values, counts = np.unique(labels_encoded[set_index], return_counts=True)
        output_dict[f"{set_name}_{fold_index}"] = dict(zip(unique_values, counts))
        result = np.isin(features, features[set_index]).astype(int)
        samples_dict[f"{set_name}_{fold_index}"] = dict(zip(features, result))

    return output_dict, samples_dict


def main():
    """
    Partition data into folds.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    fold_fullpath = os.path.join(
        args.exp_path,
        f"folds/nfold{args.nb_folds}_seed{args.seed}_train{args.train_ratio}_valid{args.valid_ratio}",
    )
    partition_fullpath = os.path.join(fold_fullpath, args.out)
    dataset_fullpath = os.path.join(args.exp_path, args.dataset)
    os.makedirs(fold_fullpath, exist_ok=True)

    with h5py.File(dataset_fullpath, "r") as dataset_file:
        features = np.array(dataset_file["samples"])
        label_names = np.array(dataset_file["class_label_names"])
        labels = np.array(
            [label_names[int(label)] for label in dataset_file["class_labels"]]
        )

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    if args.shuffle:
        features, labels_encoded = shuffle(
            features, labels_encoded, random_state=args.seed
        )

    folds = make_folds(
        features,
        labels_encoded,
        n_splits=args.nb_folds,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        balance_class=args.balance_class,
        random_state=args.seed,
    )

    samples, indices = [], []
    samples_dict, output_dict = (
        {"labels": dict(zip(features, labels))},
        {"labels": dict(zip(labels_encoded, labels))},
    )

    for i, (train_index, test_index, valid_index) in enumerate(folds):
        indices.append([train_index, valid_index, test_index])
        samples.append(
            [features[train_index], features[valid_index], features[test_index]]
        )

        if args.save_infos:
            output_dict, samples_dict = get_infos(
                (features, labels_encoded),
                (i, train_index, valid_index, test_index),
                (output_dict, samples_dict),
            )

    if args.save_infos:
        os.makedirs(f"{fold_fullpath}/save-infos", exist_ok=True)
        balance_tag = "balanced" if args.balance_class else "unbalanced"
        pd.DataFrame(samples_dict).to_csv(
            f"{fold_fullpath}/save-infos/samples_{balance_tag}.csv"
        )
        pd.DataFrame(output_dict).to_csv(
            f"{fold_fullpath}/save-infos/splits_{balance_tag}.csv"
        )

    if args.verbose:
        for i, fold_indices in enumerate(indices):
            print(f"  FOLD: {i}")
            print(f"  - Train: {len(fold_indices[0])} samples")
            print(f"  - Valid: {len(fold_indices[1])} samples")
            print(f"  - Test: {len(fold_indices[2])} samples")
            print(
                f"Fold number of samples: {len(fold_indices[0]) + len(fold_indices[1]) + len(fold_indices[2])}"
            )
            print("***")

    np.savez(
        partition_fullpath,
        folds_indexes=np.array(indices, dtype=object),
        folds_samples=np.array(samples, dtype=object),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Partition data into folds.")

    parser.add_argument(
        "--exp-path", type=str, help="Path to directory where partition will be written"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Hdf5 dataset created with create_dataset.py. Provide full path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Seed for fixing random shuffle of samples before partitioning. Default: %(default)i",
    )
    parser.add_argument(
        "--nb-folds",
        type=int,
        default=5,
        help="Number of folds. Use 1 for no folds. Default: %(default)i",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Ratio (between 0-1) for split of train and valid sets. For example, 0.75 will use 75%% of data for training and 25%% of data for validation. Default: %(default).2f",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.5,
        help="Ratio (between 0-1) for split of train and valid sets. For example, 0.75 will use 75%% of data for training and 25%% of data for validation. Default: %(default).2f",
    )
    parser.add_argument(
        "--balance-class",
        action="store_true",
        help="Balance the folds based on class labels",
        default=False,
    )
    parser.add_argument(
        "--save-infos",
        action="store_true",
        help="Save information about the partitioning",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log information about the partitioning",
        default=False,
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before partitioning",
        default=False,
    )
    parser.add_argument(
        "--out",
        help="Optional filename for dataset partition. If not provided, the file will be named partition_datasetFilename_date",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main()
