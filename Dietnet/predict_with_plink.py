"""
Inference script for DietNetwork with PLINK preprocessing.

Uses PLINK --real-ref-alleles to ensure consistent allele coding between
training and test datasets.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Dietnet.helpers.model_package import ModelPackage, find_model_packages
from Dietnet.helpers.plink_utils import preprocess_test_data_with_plink, find_plink_binary
from Dietnet.helpers.dataset_utils import InferenceDataset
from Dietnet.helpers import model as model_module


def load_model_from_package(model_package, device='cpu'):
    """Load a DietNetwork model from a model package."""
    # Get model configuration
    config = model_package.metadata['config']
    n_classes = model_package.metadata['n_classes']

    # Get embedding to determine input size
    embedding = model_package.embedding
    if len(embedding.shape) == 1:
        n_feats_emb = 1
    else:
        n_feats_emb = embedding.shape[1]

    n_feats = embedding.shape[0]

    # Create model
    combined_model = model_module.CombinedModel(
        n_feats=n_feats_emb,
        n_hidden_u_aux=config['nb_hidden_u_aux'],
        n_hidden_u_main=config['nb_hidden_u_aux'][-1:] + config['nb_hidden_u_main'],
        n_targets=n_classes,
        param_init=None,
        aux_uniform_init_limit=config.get('uniform_init_limit', 0.02),
        input_dropout=config.get('input_dropout', 0.995)
    )

    # Load state dict
    checkpoint = model_package.load_model_state(device=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    combined_model.load_state_dict(state_dict)
    combined_model.to(device)
    combined_model.eval()

    return combined_model


def predict_single_model(
    model_package,
    plink_prefix,
    device='cpu',
    batch_size=128,
    num_workers=0,
    plink_bin=None,
    temp_dir=None,
    skip_preprocess=False,
    force_preprocess=False
):
    """
    Run inference with a single model using PLINK preprocessing.

    Args:
        model_package: ModelPackage instance
        plink_prefix: Path to test PLINK files (or preprocessed files if skip_preprocess=True)
        device: Device for inference
        batch_size: Batch size
        num_workers: Number of data loader workers
        plink_bin: Path to PLINK binary (auto-detected if None)
        temp_dir: Directory for preprocessed files
        skip_preprocess: If True, assume plink_prefix is already preprocessed
        force_preprocess: If True, rerun preprocessing even if exists

    Returns:
        Tuple of (sample_ids, predictions, probabilities)
    """
    print(f"\n{'='*60}")
    print(f"Running inference: seed {model_package.seed}, fold {model_package.fold}")
    print(f"{'='*60}")

    # Determine which PLINK file to use
    if skip_preprocess:
        # User provided already-preprocessed PLINK file
        preprocessed_plink_prefix = plink_prefix
        print(f"\nUsing provided preprocessed PLINK: {preprocessed_plink_prefix}")
    else:
        # Need to preprocess
        # Check if model has BIM file
        bim_file = model_package.package_dir / 'allpos.bim'
        if not bim_file.exists():
            raise FileNotFoundError(
                f"Model package missing BIM file: {bim_file}\n"
                "Please re-convert models with updated convert_legacy_models.py"
            )

        # Find PLINK binary
        if plink_bin is None:
            plink_bin = find_plink_binary()
        print(f"Using PLINK: {plink_bin}")

        # Preprocess test data with PLINK
        if temp_dir is None:
            temp_dir = Path('.') / 'preprocessed_plink'
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Use fixed output name for preprocessed file (shared across all models)
        output_prefix = temp_dir / "test_preprocessed"

        preprocessed_plink_prefix = preprocess_test_data_with_plink(
            test_plink_prefix=plink_prefix,
            model_bim_file=str(bim_file),
            output_prefix=str(output_prefix),
            plink_bin=plink_bin,
            force=force_preprocess
        )

    # Load model
    print("\nLoading model...")
    model = load_model_from_package(model_package, device=device)

    # Load and normalize embedding
    embedding = torch.from_numpy(model_package.embedding).float().to(device)
    emb_norm = (embedding ** 2).sum(0) ** 0.5
    embedding = embedding / emb_norm
    if len(embedding.size()) == 1:
        embedding = torch.unsqueeze(embedding, dim=1)

    # Create dataset (loads preprocessed PLINK with memory mapping)
    print("\nCreating inference dataset...")
    dataset = InferenceDataset(
        plink_prefix=preprocessed_plink_prefix,
        model_package=model_package,
        use_memmap=True
    )

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != 'cpu')
    )

    # Run inference
    print(f"\nRunning inference on {len(dataset)} samples...")
    all_sample_ids = []
    all_logits = []

    with torch.no_grad():
        for batch_geno, batch_sample_ids in tqdm(loader, desc="Inference"):
            batch_geno = batch_geno.to(device)
            logits = model(embedding, batch_geno)
            all_logits.append(logits.cpu())
            all_sample_ids.extend(batch_sample_ids)

    # Concatenate results
    all_logits = torch.cat(all_logits, dim=0)
    probabilities = F.softmax(all_logits, dim=1).numpy()
    predictions = torch.argmax(all_logits, dim=1).numpy()

    print("✓ Inference complete")

    return all_sample_ids, predictions, probabilities


def predict_ensemble(
    model_packages,
    plink_prefix,
    device='cpu',
    batch_size=128,
    num_workers=0,
    plink_bin=None,
    temp_dir=None,
    skip_preprocess=False,
    force_preprocess=False,
    label_mapping=None
):
    """
    Run ensemble inference with multiple models.

    Args:
        model_packages: List of ModelPackage instances
        plink_prefix: Path to test PLINK files
        device: Device for inference
        batch_size: Batch size
        num_workers: Number of data loader workers
        plink_bin: Path to PLINK binary
        temp_dir: Directory for preprocessed files
        skip_preprocess: Skip preprocessing if True
        force_preprocess: Force re-preprocessing if True
        label_mapping: Label mapping dict

    Returns:
        Tuple of (sample_ids, ensemble_predictions, ensemble_probabilities, agreement_stats)
    """
    all_predictions = []
    all_probabilities = []
    sample_ids = None

    # Preprocess once, reuse for all models
    from pathlib import Path
    if temp_dir is None:
        temp_dir = Path('.') / 'preprocessed_plink'
    else:
        temp_dir = Path(temp_dir)
    preprocessed_prefix = temp_dir / 'test_preprocessed'

    # Run inference with each model
    for i, pkg in enumerate(model_packages, 1):
        print(f"\n[{i}/{len(model_packages)}] Processing seed {pkg.seed}, fold {pkg.fold}...")

        # First model: preprocess and save to common file
        # Subsequent models: reuse preprocessed file
        if i == 1:
            current_plink = plink_prefix
            skip_prep = skip_preprocess
            force_prep = force_preprocess
        else:
            # Reuse preprocessed file from first model
            current_plink = str(preprocessed_prefix)
            skip_prep = True
            force_prep = False

        ids, preds, probs = predict_single_model(
            model_package=pkg,
            plink_prefix=current_plink,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            plink_bin=plink_bin,
            temp_dir=temp_dir,
            skip_preprocess=skip_prep,
            force_preprocess=force_prep
        )

        if sample_ids is None:
            sample_ids = ids
        else:
            # Verify sample IDs match across models
            if ids != sample_ids:
                raise ValueError(f"Sample ID mismatch in model {i}")

        all_predictions.append(preds)
        all_probabilities.append(probs)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)
    all_probabilities = np.array(all_probabilities)  # Shape: (n_models, n_samples, n_classes)

    # Compute ensemble predictions (majority vote)
    ensemble_predictions = []
    agreement_fractions = []

    for i in range(all_predictions.shape[1]):  # For each sample
        sample_preds = all_predictions[:, i]
        unique, counts = np.unique(sample_preds, return_counts=True)
        # Most common prediction
        majority_pred = unique[np.argmax(counts)]
        ensemble_predictions.append(majority_pred)
        # Agreement fraction
        agreement = np.max(counts) / len(sample_preds)
        agreement_fractions.append(agreement)

    ensemble_predictions = np.array(ensemble_predictions)
    agreement_fractions = np.array(agreement_fractions)

    # Average probabilities across models
    ensemble_probabilities = np.mean(all_probabilities, axis=0)

    # Compute agreement statistics
    agreement_stats = {
        'mean_agreement': np.mean(agreement_fractions),
        'min_agreement': np.min(agreement_fractions),
        'max_agreement': np.max(agreement_fractions),
        'agreement_fractions': agreement_fractions,
        'n_models': len(model_packages),
        'seeds': sorted(set(pkg.seed for pkg in model_packages)),
        'folds': sorted(set(pkg.fold for pkg in model_packages))
    }

    print(f"\n✓ Ensemble prediction complete")
    print(f"  Mean agreement: {agreement_stats['mean_agreement']:.2%}")
    print(f"  Agreement range: {agreement_stats['min_agreement']:.2%} - {agreement_stats['max_agreement']:.2%}")

    return sample_ids, ensemble_predictions, ensemble_probabilities, agreement_stats


def save_predictions(
    output_file,
    sample_ids,
    predictions,
    probabilities,
    label_mapping,
    agreement_stats=None
):
    """Save predictions to TSV file."""
    idx_to_label = {idx: label for label, idx in label_mapping.items()}

    df = pd.DataFrame({
        'sample_id': sample_ids,
        'predicted_class': [idx_to_label[idx] for idx in predictions],
        'predicted_idx': predictions,
        'max_probability': np.max(probabilities, axis=1)
    })

    # Add agreement fraction if available (ensemble mode)
    if agreement_stats is not None:
        df['model_agreement'] = agreement_stats['agreement_fractions']

    # Add class probabilities
    for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        df[f'prob_{label}'] = probabilities[:, idx]

    df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✓ Saved predictions to {output_file}")
    print(f"  {len(df)} samples, {len(label_mapping)} classes")

    if agreement_stats is not None:
        print(f"  Ensemble of {agreement_stats['n_models']} models")
        print(f"  Seeds: {agreement_stats['seeds']}")
        print(f"  Folds: {agreement_stats['folds']}")


def main():
    parser = argparse.ArgumentParser(
        description='Run DietNetwork inference with PLINK preprocessing'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directory containing model package(s)'
    )

    parser.add_argument(
        '--plink-prefix',
        type=str,
        required=True,
        help='PLINK file prefix (without .bed/.bim/.fam)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file for predictions (.tsv)'
    )

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        help='Seeds to use (default: all)'
    )

    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=None,
        help='Folds to use (default: all)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size (default: 128)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (default: cuda if available, else cpu)'
    )

    parser.add_argument(
        '--plink-bin',
        type=str,
        default=None,
        help='Path to PLINK binary (default: auto-detect)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of data loader workers (default: 0)'
    )

    parser.add_argument(
        '--temp-dir',
        type=str,
        default='./preprocessed_plink',
        help='Directory for preprocessed PLINK files (default: ./preprocessed_plink)'
    )

    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip PLINK preprocessing (use if plink-prefix is already preprocessed)'
    )

    parser.add_argument(
        '--force-preprocess',
        action='store_true',
        help='Force re-preprocessing even if preprocessed file exists'
    )

    args = parser.parse_args()

    # Find model packages
    model_packages = find_model_packages(
        root_dir=args.model_dir,
        seeds=args.seeds,
        folds=args.folds
    )

    if not model_packages:
        print(f"ERROR: No model packages found in {args.model_dir}")
        sys.exit(1)

    seeds = sorted(set(pkg.seed for pkg in model_packages))
    folds = sorted(set(pkg.fold for pkg in model_packages))
    print(f"Found {len(model_packages)} model(s): seeds {seeds}, folds {folds}")

    # Get label mapping
    label_mapping = model_packages[0].label_mapping

    # Run inference with all models (ensemble)
    if len(model_packages) == 1:
        print("\nRunning inference with single model...")
        sample_ids, predictions, probabilities = predict_single_model(
            model_package=model_packages[0],
            plink_prefix=args.plink_prefix,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            plink_bin=args.plink_bin,
            temp_dir=args.temp_dir,
            skip_preprocess=args.skip_preprocess,
            force_preprocess=args.force_preprocess
        )
        agreement_stats = None
    else:
        print(f"\nRunning ensemble inference with {len(model_packages)} models...")
        sample_ids, predictions, probabilities, agreement_stats = predict_ensemble(
            model_packages=model_packages,
            plink_prefix=args.plink_prefix,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            plink_bin=args.plink_bin,
            temp_dir=args.temp_dir,
            skip_preprocess=args.skip_preprocess,
            force_preprocess=args.force_preprocess,
            label_mapping=label_mapping
        )

    # Save predictions
    save_predictions(
        output_file=args.output,
        sample_ids=sample_ids,
        predictions=predictions,
        probabilities=probabilities,
        label_mapping=label_mapping,
        agreement_stats=agreement_stats
    )

    print("\n✓ Inference complete!")


if __name__ == '__main__':
    main()
