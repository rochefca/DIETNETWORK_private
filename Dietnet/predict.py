"""
Inference script for DietNetwork with PLINK support and model packages.

Runs batched inference on PLINK datasets using converted model packages.
Supports single model or ensemble prediction across multiple seeds/folds.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Dietnet.helpers.model_package import ModelPackage, find_model_packages
from Dietnet.helpers.dataset_utils import InferenceDataset
from Dietnet.helpers import model as model_module


def load_model_from_package(model_package, device='cpu'):
    """
    Load a DietNetwork model from a model package.

    Args:
        model_package: ModelPackage instance
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model in eval mode
    """
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
        param_init=None,  # None = use random init, then load pretrained weights
        aux_uniform_init_limit=config.get('uniform_init_limit', 0.02),
        input_dropout=config.get('input_dropout', 0.995)
    )

    # Load state dict
    checkpoint = model_package.load_model_state(device=device)

    # Handle both direct state_dict and checkpoint format
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    combined_model.load_state_dict(state_dict)

    # Move to device and set to eval mode
    combined_model.to(device)
    combined_model.eval()

    return combined_model


def predict_single_model(
    model_package,
    plink_prefix,
    device='cpu',
    batch_size=128,
    num_workers=0,
    verbose=True
):
    """
    Run inference with a single model.

    Args:
        model_package: ModelPackage instance
        plink_prefix: Path to test PLINK files
        device: Device for inference
        batch_size: Batch size for inference
        num_workers: Number of data loader workers

    Returns:
        Tuple of (sample_ids, predictions, probabilities)
    """
    if verbose:
        print(f"\nRunning inference with model: seed {model_package.seed}, fold {model_package.fold}")

    # Load model
    if verbose:
        print("\nLoading model...")
    model = load_model_from_package(model_package, device=device)

    # Load embedding for this model
    embedding = torch.from_numpy(model_package.embedding).float().to(device)

    # Normalize embedding (same as in training)
    emb_norm = (embedding ** 2).sum(0) ** 0.5
    embedding = embedding / emb_norm

    # Add dimension if needed
    if len(embedding.size()) == 1:
        embedding = torch.unsqueeze(embedding, dim=1)

    # Create dataset
    dataset = InferenceDataset(
        plink_prefix=plink_prefix,
        model_package=model_package,
        use_memmap=True,
        verbose=verbose
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
    if verbose:
        print(f"\nRunning inference on {len(dataset)} samples...")
    all_sample_ids = []
    all_logits = []

    with torch.no_grad():
        for batch_geno, batch_sample_ids in tqdm(
            loader,
            desc="Inference",
            disable=not verbose,
            leave=False
        ):
            # Move to device
            batch_geno = batch_geno.to(device)

            # Forward pass (pass embedding and batch)
            logits = model(embedding, batch_geno)

            # Store results
            all_logits.append(logits.cpu())
            all_sample_ids.extend(batch_sample_ids)

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)

    # Get predictions and probabilities
    probabilities = F.softmax(all_logits, dim=1).numpy()
    predictions = torch.argmax(all_logits, dim=1).numpy()

    return all_sample_ids, predictions, probabilities


def ensemble_predict(
    model_packages,
    plink_prefix,
    device='cpu',
    batch_size=128,
    num_workers=0,
    method='vote'
):
    """
    Run ensemble inference with multiple models.

    Args:
        model_packages: List of ModelPackage instances
        plink_prefix: Path to test PLINK files
        device: Device for inference
        batch_size: Batch size
        num_workers: Number of workers
        method: 'vote' for majority voting or 'average' for probability averaging

    Returns:
        Tuple of (sample_ids, predictions, probabilities, individual_predictions)
    """
    print(f"\n{'='*60}")
    print(f"Running ensemble inference with {len(model_packages)} models")
    print(f"Ensemble method: {method}")
    print(f"{'='*60}")

    all_predictions = []
    all_probabilities = []
    sample_ids = None

    model_bar = tqdm(model_packages, desc="Models", unit="model")
    for i, pkg in enumerate(model_bar):
        model_bar.set_postfix_str(f"seed {pkg.seed} fold {pkg.fold}")
        sids, preds, probs = predict_single_model(
            model_package=pkg,
            plink_prefix=plink_prefix,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=(i == 0)
        )

        if sample_ids is None:
            sample_ids = sids
        else:
            # Verify sample order is consistent
            assert sample_ids == sids, "Sample order mismatch between models!"

        all_predictions.append(preds)
        all_probabilities.append(probs)

    # Stack predictions and probabilities
    all_predictions = np.stack(all_predictions, axis=0)  # Shape: (n_models, n_samples)
    all_probabilities = np.stack(all_probabilities, axis=0)  # Shape: (n_models, n_samples, n_classes)

    # Compute ensemble prediction
    if method == 'vote':
        # Majority voting
        from scipy import stats
        ensemble_predictions, _ = stats.mode(all_predictions, axis=0, keepdims=False)
        ensemble_probabilities = np.mean(all_probabilities, axis=0)
    elif method == 'average':
        # Average probabilities then argmax
        ensemble_probabilities = np.mean(all_probabilities, axis=0)
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    print(f"\n✓ Ensemble inference complete")

    return sample_ids, ensemble_predictions, ensemble_probabilities, all_predictions


def save_predictions(
    output_file,
    sample_ids,
    predictions,
    probabilities,
    label_mapping,
    individual_predictions=None
):
    """
    Save predictions in a compact, human-readable format.

    Single model:
        <sample_id> <prediction>

    Ensemble:
        <sample_id> <labelA>(count) <labelB>(count) ...
    """
    # Invert label mapping
    idx_to_label = {idx: label for label, idx in label_mapping.items()}
    sample_strs = [str(sid) for sid in sample_ids]
    id_width = max(len(sid) for sid in sample_strs)
    lines = []

    if individual_predictions is not None and len(np.atleast_2d(individual_predictions)) > 0:
        votes = np.atleast_2d(individual_predictions)
        for i, sid in enumerate(sample_strs):
            counts = Counter(votes[:, i])
            ordered = sorted(counts.items(), key=lambda item: (-item[1], idx_to_label[item[0]]))
            label_tokens = [f"{idx_to_label[idx]}({count})" for idx, count in ordered]
            lines.append(f"{sid.rjust(id_width)} " + " ".join(label_tokens))
    else:
        for sid, pred in zip(sample_strs, predictions):
            lines.append(f"{sid.rjust(id_width)} {idx_to_label[pred]}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")

    print(f"✓ Saved predictions to {output_path}")
    print(f"  {len(sample_ids)} samples")
    print(f"  {len(label_mapping)} classes")


def main():
    parser = argparse.ArgumentParser(
        description='Run DietNetwork inference on PLINK datasets'
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
        help='Output file for predictions (text)'
    )

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        help='Seeds to use (default: all available)'
    )

    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=None,
        help='Folds to use (default: all available)'
    )

    parser.add_argument(
        '--ensemble-method',
        type=str,
        choices=['vote', 'average'],
        default='vote',
        help='Ensemble method: vote or average (default: vote)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for inference (default: 128)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference (default: cuda if available, else cpu)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of data loader workers (default: 0)'
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

    # Show compact summary
    seeds = sorted(set(pkg.seed for pkg in model_packages))
    folds = sorted(set(pkg.fold for pkg in model_packages))
    print(f"Found {len(model_packages)} model(s): seeds {seeds}, folds {folds}")

    # Get label mapping from first package (should be same for all)
    label_mapping = model_packages[0].label_mapping

    # Run inference
    if len(model_packages) == 1:
        # Single model
        sample_ids, predictions, probabilities = predict_single_model(
            model_package=model_packages[0],
            plink_prefix=args.plink_prefix,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        individual_preds = None
    else:
        # Ensemble
        sample_ids, predictions, probabilities, individual_preds = ensemble_predict(
            model_packages=model_packages,
            plink_prefix=args.plink_prefix,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            method=args.ensemble_method
        )

    # Save predictions
    save_predictions(
        output_file=args.output,
        sample_ids=sample_ids,
        predictions=predictions,
        probabilities=probabilities,
        label_mapping=label_mapping,
        individual_predictions=individual_preds
    )

    print("\n✓ Inference complete!")


if __name__ == '__main__':
    main()
