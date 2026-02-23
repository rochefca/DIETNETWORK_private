#!/bin/bash
import argparse
from collections import Counter
import numpy as np
import pandas as pd


def main():
    dn_label_order = ['ACB', 'ASW', 'BEB', 'CDX', 'CEUGBR',
                      'CHB', 'CHS', 'CLM', 'ESN', 'FIN',
                      'GIH', 'GWD', 'IBS', 'JPT', 'KHV',
                      'LWK', 'MSL', 'MXL', 'PEL', 'PJL',
                      'PUR', 'STUITU', 'TSI', 'YRI']
    
    args = parse_args()
    
    d_preds = dict()
    d_scores = dict()
    
    for fold in range(1,6):
        for rep in range(1,4):
            # Load results for this model
            model_results = np.load(args.dietnet_results.format(fold=fold, rep=rep))
            
            if 'samples' not in d_preds.keys():
                d_preds['samples'] = model_results['samples']
                d_scores['samples'] = model_results['samples']
                
            d_preds['model{fold}_{rep}'.format(fold=fold, rep=rep)] = model_results['preds'].astype(np.int8)
            
            for i,pop in enumerate(dn_label_order):
                d_scores['{pop}_model{fold}_{rep}'.format(pop=pop,fold=fold,rep=rep)] = model_results['scores'][:,i]
            
    # Put predictions in df and map from indices to population labels
    df_preds = pd.DataFrame(d_preds)
    for col in df_preds.columns:
        if col.startswith('model'):
            df_preds[col] = df_preds[col].apply(lambda x: dn_label_order[x])
    
    # Create a column with counts of each predicted population across all models
    def get_pop_counts(row):
        model_cols = [col for col in df_preds.columns if col.startswith('model')]
        pops = row[model_cols].values
        
        counts = Counter(pops)
        result = ','.join([f'{pop}({count})' for pop, count in sorted(counts.items())])
        return result
    
    df_preds['all_models'] = df_preds.apply(get_pop_counts, axis=1)
    
    # Score df
    df_scores = pd.DataFrame(d_scores)
    
    # Save df of results to file
    df_preds.to_csv(args.out+'_dietnet_predictions.txt', index=False, sep='\t')
    df_scores.to_csv(args.out+'_dietnet_scores.txt', index=False, sep='\t')


def parse_args():
    parser = argparse.ArgumentParser(description='Group model predictions by sample and save to text file.')
    
    parser.add_argument(
        '--dietnet-results',
        type=str,
        required=True,
        help='Npz file of results for 15 models. Provide full path'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output name for model grouped predictions and scores. Provide full'
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    main()