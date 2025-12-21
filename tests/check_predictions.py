#!/usr/bin/env python3
"""
Compare DietNetwork predictions to true labels and compute accuracy.
"""

import pandas as pd
import numpy as np
import sys

def check_predictions(predictions_file, labels_file):
    # Load predictions
    preds = pd.read_csv(predictions_file, sep='\t')
    print(f"Loaded {len(preds)} predictions from {predictions_file}")
    
    # Load true labels (skip header if present)
    labels = pd.read_csv(labels_file, sep='\t')
    # If first row looks like a header, it was already skipped by pandas
    # Otherwise, set column names
    if labels.columns[0] != 'sample_id':
        labels.columns = ['sample_id', 'true_label']
    else:
        # Rename second column if needed
        labels = labels.rename(columns={labels.columns[1]: 'true_label'})

    print(f"Loaded {len(labels)} true labels from {labels_file}")
    
    # Merge on sample_id
    merged = preds.merge(labels, on='sample_id', how='inner')
    print(f"\nMatched {len(merged)} samples")
    
    if len(merged) == 0:
        print("ERROR: No matching samples found!")
        return
    
    # Compute accuracy
    correct = (merged['predicted_class'] == merged['true_label']).sum()
    accuracy = correct / len(merged) * 100
    
    print(f"\n{'='*60}")
    print(f"ACCURACY: {correct}/{len(merged)} = {accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Show per-class accuracy
    print("\nPer-class accuracy:")
    for label in sorted(merged['true_label'].unique()):
        subset = merged[merged['true_label'] == label]
        label_correct = (subset['predicted_class'] == subset['true_label']).sum()
        label_acc = label_correct / len(subset) * 100
        print(f"  {label:12s}: {label_correct:4d}/{len(subset):4d} = {label_acc:6.2f}%")
    
    # Show confusion for misclassified samples
    misclassified = merged[merged['predicted_class'] != merged['true_label']]
    if len(misclassified) > 0:
        print(f"\nMisclassified samples: {len(misclassified)}")
        print("\nMost common confusions:")
        confusion = misclassified.groupby(['true_label', 'predicted_class']).size().reset_index(name='count')
        confusion = confusion.sort_values('count', ascending=False).head(10)
        for _, row in confusion.iterrows():
            print(f"  {row['true_label']} → {row['predicted_class']}: {row['count']} samples")
    else:
        print("\n✓ Perfect predictions! No misclassifications.")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python check_predictions.py <predictions.tsv> <labels.tsv>")
        sys.exit(1)
    
    check_predictions(sys.argv[1], sys.argv[2])
