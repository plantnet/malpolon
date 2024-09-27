"""This script computes metrics off of model inference predictions.

It computes the Precision, Recall, F1-score (micro, samples and macro)
for the top-k predictions of a model inference predictions (in a CSV);
as well as the AUC (micro, samples and macro) for all the probabilities
(not just the top-k).

Author: Theo Larcher <theo.larcher@inria.fr>
        Alexis Joly <alexis.joly@inria.fr>
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, top_k_accuracy_score)

# Constant variables
N_CLS = 174
TOP_K = [1, 3, 5]
TASK = 'multiclass'

# Print colors
INFO = '\033[93m'
RESET = '\033[0m'
LINK = '\033[94m'
BOLD = "\033[1m"

# 0. Load data
df = pd.read_csv('GLC24_habitat_predictions_multiclass_val-dataset.csv')
df['target_habitat_id'] = df['target_habitat_id'].astype(str)
df_gt = df.copy()
df_preds = df.copy()

for rowi, row in deepcopy(df_gt).iterrows():
    tsi = np.array(row['target_habitat_id'].split()).astype(int)  # Split the predictions string by space and convert to int
    inds = np.where(tsi > (N_CLS - 1))[0]
    vals = tsi[inds]
    if inds.size > 0:
        df_gt = df_gt.drop(rowi)
        df_preds = df_preds.drop(rowi)
        print(f"obs {rowi} of surveyId {row['surveyId']} removed because target_habitat_id value {vals} out of range")

targets = df_gt['target_habitat_id']
targets = [list(map(int, str(x).split())) for x in targets]

preds = df_preds['predictions']
preds = np.array([list(map(int, str(x).split())) for x in preds])

probas = df_preds['probas']
probas = np.array([list(map(float, str(x).split())) for x in probas])

all_targets_oh = np.zeros((len(df_gt), N_CLS))
all_probas = np.zeros_like(probas)
all_predictions_topk_oh = np.zeros((len(df_preds), N_CLS))


if TASK == 'multilabel':
    # 1. Convert data to usable types and compute one-hot encodings
    res = pd.DataFrame(columns=['Precision_micro', 'Recall_micro', 'F1_micro',
                                'Precision_samples', 'Recall_samples', 'F1_samples',
                                'Precision_macro', 'Recall_macro', 'F1_macro',
                                'AUC_micro', 'AUC_samples', 'AUC_macro'])

    idx = np.arange(len(all_predictions_topk_oh)).reshape(-1, 1)
    all_targets_oh[idx, targets] = 1  # One-hot encode the targets
    all_probas = probas[idx, np.argsort(preds, axis=1)]  # Sort the probabilities in class order

    # 2. Compute Precision / Recall / F1-score
    print('\nComputing Precision, Recall, F1-scores...')
    prfs = {}
    for topk in TOP_K:
        for avg in ['micro', 'samples', 'macro']:
            prf = precision_recall_fscore_support(all_targets_oh, all_predictions_topk_oh[:, :topk], average=avg, zero_division=np.nan)[:3]
            prfs[f'Precision_{avg}_top-{topk}'] = prf[0]
            prfs[f'Recall_{avg}_top-{topk}'] = prf[1]
            prfs[f'F1_{avg}_top-{topk}'] = prf[2]
            print(f"Top-{topk} {avg.upper()}: Precision, Recall, F1", prf)

    print('\nComputing Accuracy...')
    acc = accuracy_score(targets, preds)

    # 3. Compute AUCs
    print('\nComputing AUCs...')
    # Find rows and columns with all zeros in both arrays, that is to say
    # species that are never observed in any plot according to the ground truth
    zero_cols_targets = np.all(all_targets_oh == 0, axis=0)
    ones_cols_targets = np.all(all_targets_oh == 1, axis=0)
    zero_cols = zero_cols_targets | ones_cols_targets
    # Filter out rows and columns containing only zeros
    filtered_targets = all_targets_oh[:][:, ~zero_cols]
    filtered_probas = all_probas[:][:, ~zero_cols]

    aucs = {}
    for avg in ['micro', 'samples', 'macro']:
        auc = roc_auc_score(filtered_targets, filtered_probas, average=avg)
        aucs[f'AUC_{avg}'] = auc
        print(f"{avg.upper()}: AUC", auc)

    # 4. Save results
    res.loc[0] = prfs | aucs
    res.to_csv('Inference_PRC-AUC.csv', index=False)
    print('\nResults saved to Inference_PRC-AUC.csv')

elif TASK == 'multiclass':
    # 1. Convert data to usable types and compute one-hot encodings
    res = pd.DataFrame(columns=['Accuracy',
                                'Precision_micro', 'Recall_micro', 'F1_micro',
                                'Precision_samples', 'Recall_samples', 'F1_samples',
                                'Precision_macro', 'Recall_macro', 'F1_macro',
                                'AUC_micro', 'AUC_samples', 'AUC_macro'])

    idx = np.arange(len(all_predictions_topk_oh)).reshape(-1, 1)
    all_targets_oh[idx, targets] = 1  # One-hot encode the targets
    all_probas = probas[idx, np.argsort(preds, axis=1)]  # Sort the probabilities in class order

    # 2. Compute Precision / Recall / F1-score
    print(f'{INFO}{BOLD}\nComputing Precision, Recall, F1-scores...{RESET}')
    prfs = {}
    for topk in TOP_K:
        for avg in ['micro', 'samples', 'macro']:
            all_predictions_topk_oh[idx, preds[:, :topk]] = 1  # One-hot encode the top-k predictions
            prf = precision_recall_fscore_support(all_targets_oh, all_predictions_topk_oh, average=avg, zero_division=np.nan)[:3]
            prfs[f'Precision_{avg}_top-{topk}'] = prf[0]
            prfs[f'Recall_{avg}_top-{topk}'] = prf[1]
            prfs[f'F1_{avg}_top-{topk}'] = prf[2]
            print(f"Top-{topk} {avg.upper()}: Precision, Recall, F1", prf)
        print("")

    print(f'{INFO}{BOLD}\nComputing Top-k Accuracy...{RESET}')
    accs = {}
    for topk in TOP_K:
        acc = top_k_accuracy_score(targets, all_probas, k=topk, labels=np.arange(N_CLS))
        accs[f'Accuracy_multiclass_top-{topk}'] = acc
        print(f"Top-{topk} Accuracy_multiclass: {acc}")

    # 4. Save results
    res.loc[0] = prfs | accs
    res.to_csv('Inference_PRC-ACC.csv', index=False)
    print('\nResults saved to Inference_PRC-ACC.csv')
