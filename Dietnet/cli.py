"""
Command-line interface for DietNetwork.

This module provides a unified CLI for training and inference with DietNetwork.
"""
import os
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.2.0")
def main():
    """
    DietNetwork: Deep learning for genetic ancestry classification.

    A PyTorch implementation of DietNetwork (https://arxiv.org/abs/1611.09340)
    for population genetics and ancestry inference.
    """
    pass


@main.command()
@click.option(
    '--exp-path',
    type=click.Path(exists=True),
    required=True,
    help='Path to directory containing dataset, partitions, and embeddings.'
)
@click.option(
    '--exp-name',
    type=str,
    required=True,
    help='Name of experiment directory where results will be saved.'
)
@click.option(
    '--config',
    type=str,
    default='config.yaml',
    help='YAML config file with hyperparameters (default: config.yaml).'
)
@click.option(
    '--dataset',
    type=str,
    default='dataset.hdf5',
    help='HDF5 dataset file (default: dataset.hdf5).'
)
@click.option(
    '--partition',
    type=str,
    default='partitioned_idx.npz',
    help='Partition indexes file (default: partitioned_idx.npz).'
)
@click.option(
    '--embedding',
    type=str,
    default='embedding.npz',
    help='Genotype frequency embedding file (default: embedding.npz).'
)
@click.option(
    '--input-features-means',
    type=str,
    default='input_features_means.npz',
    help='Input feature means for missing value imputation (default: input_features_means.npz).'
)
@click.option(
    '--label-file',
    type=str,
    default=None,
    help='Label file (TSV format, required for PLINK datasets).'
)
@click.option(
    '--which-fold',
    type=int,
    required=True,
    help='Which fold to train on (0-indexed).'
)
@click.option(
    '--task',
    type=click.Choice(['classification', 'regression']),
    default='classification',
    help='Task type (default: classification).'
)
@click.option(
    '--normalize/--no-normalize',
    default=True,
    help='Apply normalization to input features (default: normalize).'
)
@click.option(
    '--param-init',
    type=click.Path(exists=True),
    default=None,
    help='Path to pre-trained parameter initialization (.npz from Theano).'
)
@click.option(
    '--comet-ml/--no-comet-ml',
    default=False,
    help='Enable Comet.ML experiment tracking (default: disabled).'
)
@click.option(
    '--comet-ml-project-name',
    type=str,
    default=None,
    help='Comet.ML project name.'
)
@click.option(
    '--optimization/--no-optimization',
    default=False,
    help='Run in hyperparameter optimization mode (default: disabled).'
)
def train(exp_path, exp_name, config, dataset, partition, embedding,
          input_features_means, label_file, which_fold, task, normalize, param_init,
          comet_ml, comet_ml_project_name, optimization):
    """
    Train a DietNetwork model on a specific fold.

    Example:
        dietnet train --exp-path ./data --exp-name experiment1 --which-fold 0
    """
    from Dietnet import train as train_module

    # Build arguments object to pass to original train script
    class Args:
        pass

    args = Args()
    args.exp_path = exp_path
    args.exp_name = exp_name
    args.config = config
    args.dataset = dataset
    args.partition = partition
    args.embedding = embedding
    args.input_features_means = input_features_means
    args.label_file = label_file
    args.which_fold = which_fold
    args.task = task
    args.normalize = normalize
    args.param_init = param_init
    args.comet_ml = comet_ml
    args.comet_ml_project_name = comet_ml_project_name
    args.optimization = optimization

    # Call the original train logic
    train_module.main_with_args(args)


@main.command()
# NEW MODEL PACKAGE APPROACH (Recommended)
@click.option(
    '--model',
    type=str,
    default=None,
    help='Model preset (e.g., 1kgp_default) or path to model package directory.'
)
@click.option(
    '--plink-prefix',
    type=str,
    default=None,
    help='PLINK file prefix for test data (without .bed/.bim/.fam).'
)
@click.option(
    '--output',
    type=str,
    default=None,
    help='Output TSV file for predictions.'
)
# LEGACY HDF5 APPROACH (Deprecated)
@click.option(
    '--test-dataset',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] HDF5 file containing test data.'
)
@click.option(
    '--train-dataset',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] HDF5 file containing training data (for SNP alignment).'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] YAML config file used for training.'
)
@click.option(
    '--embedding',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] Genotype frequency embedding file from training.'
)
@click.option(
    '--input-features-stats',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] Input feature statistics from training (for normalization).'
)
@click.option(
    '--model-params',
    type=click.Path(exists=True),
    default=None,
    help='[DEPRECATED] Trained model checkpoint (.pt file).'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default=None,
    help='[DEPRECATED] Directory where predictions will be saved.'
)
@click.option(
    '--output-name',
    type=str,
    default='predictions',
    help='[DEPRECATED] Base name for output files (default: predictions).'
)
@click.option(
    '--which-fold',
    type=int,
    default=None,
    help='[DEPRECATED] Which fold was used for training (0-indexed).'
)
# COMMON OPTIONS
@click.option(
    '--seeds',
    type=int,
    multiple=True,
    default=None,
    help='Seeds to use (default: all in model package).'
)
@click.option(
    '--folds',
    type=int,
    multiple=True,
    default=None,
    help='Folds to use (default: all in model package).'
)
@click.option(
    '--batch-size',
    type=int,
    default=128,
    help='Batch size for inference (default: 128).'
)
@click.option(
    '--device',
    type=str,
    default=None,
    help='Device: cpu or cuda (default: auto-detect).'
)
@click.option(
    '--num-workers',
    type=int,
    default=0,
    help='DataLoader workers (default: 0).'
)
@click.option(
    '--force-download',
    is_flag=True,
    help='Force re-download of model preset (only for --model presets).'
)
@click.option(
    '--task',
    type=click.Choice(['classification', 'regression']),
    default='classification',
    help='Task type (default: classification).'
)
@click.option(
    '--normalize/--no-normalize',
    default=True,
    help='Apply normalization (default: True).'
)
def predict(model, plink_prefix, output, test_dataset, train_dataset, config,
            embedding, input_features_stats, model_params, output_dir, output_name,
            which_fold, seeds, folds, batch_size, device, num_workers, force_download,
            task, normalize):
    """
    Run inference on a test dataset using a trained model.

    NEW APPROACH (Recommended):
        Use model packages with PLINK data:

        \b
        # Using a preset model
        dietnet predict --model 1kgp_default \\
                        --plink-prefix /path/to/test_data \\
                        --output predictions.tsv

        \b
        # Using a local model package
        dietnet predict --model ./pretrained_1000g \\
                        --plink-prefix /path/to/test_data \\
                        --output predictions.tsv

    LEGACY APPROACH (Deprecated):
        The HDF5-based approach is deprecated. Please use model packages instead.
    """
    import warnings
    import sys
    import os
    import subprocess
    from pathlib import Path

    # Detect which approach is being used
    using_model_package = model is not None
    using_legacy_hdf5 = model_params is not None

    if using_model_package and using_legacy_hdf5:
        click.echo("ERROR: Cannot use both --model and --model-params. Choose one approach.", err=True)
        sys.exit(1)

    if not using_model_package and not using_legacy_hdf5:
        click.echo("ERROR: Must provide either --model (new approach) or --model-params (legacy).", err=True)
        click.echo("See 'dietnet predict --help' for usage examples.", err=True)
        sys.exit(1)

    # ====================
    # NEW MODEL PACKAGE APPROACH
    # ====================
    if using_model_package:
        # Validate required arguments
        if not plink_prefix:
            click.echo("ERROR: --plink-prefix is required when using --model", err=True)
            sys.exit(1)
        if not output:
            click.echo("ERROR: --output is required when using --model", err=True)
            sys.exit(1)

        from Dietnet.model_manager import get_model_path
        from Dietnet.pretrained_models import PRETRAINED_MODELS

        # Resolve model path
        if model in PRETRAINED_MODELS:
            # Model preset - download if needed
            click.echo(f"Using model preset: {model}")
            model_dir = get_model_path(model, force_download=force_download)
        else:
            # Local path
            model_dir = Path(model)
            if not model_dir.exists():
                click.echo(f"ERROR: Model directory not found: {model_dir}", err=True)
                sys.exit(1)
            click.echo(f"Using local model: {model_dir}")

        # Determine device
        if device is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Build command for predict_with_plink.py
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'predict_with_plink.py'),
            '--model-dir', str(model_dir),
            '--plink-prefix', plink_prefix,
            '--output', output,
            '--batch-size', str(batch_size),
            '--device', device,
            '--num-workers', str(num_workers)
        ]

        if seeds:
            cmd.extend(['--seeds'] + [str(s) for s in seeds])
        if folds:
            cmd.extend(['--folds'] + [str(f) for f in folds])

        # Run inference
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    # ====================
    # LEGACY HDF5 APPROACH (DEPRECATED)
    # ====================
    else:
        # Show deprecation warning
        warnings.warn(
            "\n"
            "═══════════════════════════════════════════════════════════════\n"
            "DEPRECATION WARNING: HDF5-based inference is deprecated!\n"
            "═══════════════════════════════════════════════════════════════\n"
            "\n"
            "The HDF5-based inference approach (--model-params, --test-dataset)\n"
            "is deprecated and will be removed in a future version.\n"
            "\n"
            "Please use the new model package approach instead:\n"
            "\n"
            "  dietnet predict --model 1kgp_default \\\n"
            "                  --plink-prefix /path/to/data \\\n"
            "                  --output predictions.tsv\n"
            "\n"
            "Benefits:\n"
            "  • Simpler: One --model flag instead of 6 separate files\n"
            "  • Faster: Works directly with PLINK files (no HDF5 conversion)\n"
            "  • Scalable: Handles large datasets with memory mapping\n"
            "  • Portable: Download pretrained models automatically\n"
            "\n"
            "═══════════════════════════════════════════════════════════════\n",
            DeprecationWarning,
            stacklevel=2
        )

        # Validate required legacy arguments
        required_legacy = {
            'test-dataset': test_dataset,
            'train-dataset': train_dataset,
            'config': config,
            'embedding': embedding,
            'input-features-stats': input_features_stats,
            'model-params': model_params,
            'output-dir': output_dir,
            'which-fold': which_fold
        }

        missing = [name for name, value in required_legacy.items() if value is None]
        if missing:
            click.echo(f"ERROR: Missing required legacy arguments: {', '.join('--' + m for m in missing)}", err=True)
            sys.exit(1)

        click.echo(f"Loading model from {model_params}...")
        click.echo(f"Processing test data from {test_dataset}...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Build arguments for test script
        class Args:
            pass

        args = Args()
        args.test_dataset = test_dataset
        args.train_dataset = train_dataset
        args.config = config
        args.embedding = embedding
        args.input_features_stats = input_features_stats
        args.model_params = model_params
        args.test_path = output_dir
        args.test_name = output_name
        args.which_fold = which_fold
        args.task = task
        args.normalize = normalize

        # Import and run test logic
        from Dietnet import test_external_dataset as test_module
        test_module.test_with_args(args)

        click.echo(f"✓ Predictions saved to {output_dir}/{output_name}_results.npz")


@main.command()
@click.option(
    '--genotypes',
    type=click.Path(exists=True),
    required=True,
    help='Tab-separated genotype file or PLINK prefix.'
)
@click.option(
    '--labels',
    type=click.Path(exists=True),
    required=True,
    help='Tab-separated label file (sample_id<tab>label).'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    required=True,
    help='Output directory for dataset.'
)
@click.option(
    '--output-name',
    type=str,
    default='dataset.hdf5',
    help='Output filename (default: dataset.hdf5).'
)
@click.option(
    '--task',
    type=click.Choice(['classification', 'regression']),
    default='classification',
    help='Task type (default: classification).'
)
@click.option(
    '--class-labels',
    type=click.Path(exists=True),
    default=None,
    help='Class labels file (required for regression task).'
)
@click.option(
    '--parallel-loading/--no-parallel-loading',
    default=False,
    help='Use parallel processing for loading (default: disabled).'
)
@click.option(
    '--ncpus',
    type=int,
    default=4,
    help='Number of CPUs for parallel loading (default: 4).'
)
def create_dataset(genotypes, labels, output_dir, output_name, task,
                   class_labels, parallel_loading, ncpus):
    """
    Create HDF5 dataset from genotype and label files.

    Parses genotype data (text or PLINK format) and creates an HDF5 dataset
    suitable for training.

    Example:
        dietnet create-dataset --genotypes data/snps.txt --labels data/labels.txt \\
                               --output-dir ./processed
    """
    os.makedirs(output_dir, exist_ok=True)

    class Args:
        pass

    args = Args()
    args.genotypes = genotypes
    args.labels = labels
    args.exp_path = output_dir
    args.out = output_name
    args.task = task
    args.class_labels = class_labels
    args.parallel_loading = parallel_loading
    args.ncpus = ncpus

    from Dietnet import create_dataset as create_module
    create_module.create_dataset_with_args(args)

    click.echo(f"✓ Dataset created: {output_dir}/{output_name}")


@main.command()
@click.option(
    '--exp-path',
    type=click.Path(exists=True),
    required=True,
    help='Path to directory containing dataset.'
)
@click.option(
    '--dataset',
    type=str,
    default='dataset.hdf5',
    help='Dataset filename (default: dataset.hdf5).'
)
@click.option(
    '--output-name',
    type=str,
    default='partitioned_idx.npz',
    help='Output filename (default: partitioned_idx.npz).'
)
@click.option(
    '--nb-folds',
    type=int,
    default=5,
    help='Number of cross-validation folds (default: 5).'
)
@click.option(
    '--train-valid-ratio',
    type=float,
    default=0.8,
    help='Train/validation split ratio (default: 0.8).'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducibility (default: 42).'
)
def partition(exp_path, dataset, output_name, nb_folds, train_valid_ratio, seed):
    """
    Partition dataset into cross-validation folds.

    Creates train/valid/test splits for k-fold cross-validation.

    Example:
        dietnet partition --exp-path ./processed --nb-folds 5
    """
    class Args:
        pass

    args = Args()
    args.exp_path = exp_path
    args.dataset = dataset
    args.out = output_name
    args.nb_folds = nb_folds
    args.train_valid_ratio = train_valid_ratio
    args.seed = seed

    from Dietnet import partition_data as partition_module
    partition_module.partition_data_with_args(args)

    click.echo(f"✓ Partitions created: {exp_path}/{output_name}")


@main.command()
@click.option(
    '--exp-path',
    type=click.Path(exists=True),
    required=True,
    help='Path to directory containing dataset and partitions.'
)
@click.option(
    '--dataset',
    type=str,
    default='dataset.hdf5',
    help='Dataset filename (default: dataset.hdf5).'
)
@click.option(
    '--partition',
    type=str,
    default='partitioned_idx.npz',
    help='Partition file (default: partitioned_idx.npz).'
)
@click.option(
    '--output-name',
    type=str,
    default='embedding.npz',
    help='Output filename (default: embedding.npz).'
)
@click.option(
    '--task',
    type=click.Choice(['classification', 'regression']),
    default='classification',
    help='Task type (default: classification).'
)
@click.option(
    '--label-file',
    type=str,
    default=None,
    help='Label file (TSV format, required for PLINK datasets).'
)
def generate_embedding(exp_path, dataset, partition, output_name, task, label_file):
    """
    Generate genotype frequency embeddings for each fold.

    Computes embeddings from training data, excluding missing values.

    Example:
        dietnet generate-embedding --exp-path ./processed
    """
    class Args:
        pass

    args = Args()
    args.exp_path = exp_path
    args.dataset = dataset
    args.partition = partition
    args.out = output_name
    args.task = task
    args.label_file = label_file
    args.include_valid = False
    args.only_valid = False
    args.emb_class_label = 'labels'

    from Dietnet import generate_embedding as embedding_module
    embedding_module.generate_embedding_with_args(args)

    click.echo(f"✓ Embeddings generated: {exp_path}/{output_name}")


@main.command()
def info():
    """Display information about the DietNetwork installation."""
    import torch
    import numpy as np
    import pandas as pd
    import h5py

    click.echo("DietNetwork Installation Info")
    click.echo("=" * 50)
    click.echo(f"DietNetwork version: 0.2.0")
    click.echo(f"Python version: {sys.version.split()[0]}")
    click.echo(f"PyTorch version: {torch.__version__}")
    click.echo(f"NumPy version: {np.__version__}")
    click.echo(f"Pandas version: {pd.__version__}")
    click.echo(f"h5py version: {h5py.__version__}")
    click.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA version: {torch.version.cuda}")
        click.echo(f"GPU count: {torch.cuda.device_count()}")


if __name__ == '__main__':
    main()
