"""This script computes metrics off of model inference predictions.

It computes the Precision, Recall, F1-score (micro, samples and macro)
for the top-25 predictions of a model inference predictions (in a CSV);
as well as the AUC (micro, samples and macro) for all the probabilities
(not just the top-25).

Author: Theo Larcher <theo.larcher@inria.fr>
        Alexis Joly <alexis.joly@inria.fr>
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

# 0. Load data
df_gt = pd.read_csv('predictions_and_evaluation/GLC24_SOLUTION_FILE.csv')
df_preds = pd.read_csv('predictions_and_evaluation/predictions_GLC24_SOLUTION_FILE.csv', sep=';')
for rowi, row in deepcopy(df_gt).iterrows():
    tsi = np.array(row['target_species_ids'].split()).astype(int)  # Split the predictions string by space and convert to int
    inds = np.where(tsi > 11254)[0]
    vals = tsi[inds]
    if inds.size > 0:
        df_gt = df_gt.drop(rowi)
        df_preds = df_preds.drop(rowi)
        print(f"obs {rowi} of surveyId {row['surveyId']} removed because target_species_ids value {vals} out of range")


# 1. Convert data to usable types and compute one-hot encodings
res = pd.DataFrame(columns=['Precision_micro', 'Recall_micro', 'F1_micro',
                            'Precision_samples', 'Recall_samples', 'F1_samples',
                            'Precision_macro', 'Recall_macro', 'F1_macro',
                            'AUC_micro', 'AUC_samples', 'AUC_macro'])
obs_id = df_gt['surveyId']

targets = df_gt['target_species_ids']
targets = [list(map(int, x.split())) for x in targets]

preds = df_preds['predictions']
preds = np.array([list(map(int, x.split())) for x in preds])

probas = df_preds['probas']
probas = np.array([list(map(float, x.split())) for x in probas])

all_targets_oh = np.zeros((len(df_gt), 11255))
all_probas = np.zeros_like(probas)
all_predictions_top25_oh = np.zeros((len(df_preds), 11255))

for k, (p, t) in tqdm(enumerate(zip(preds, targets)), total=len(targets)):
    all_probas[k] = probas[k][np.argsort(p)]
    for t2 in t:
        all_targets_oh[k, t2] = 1
    for p2 in p[:25]:
        all_predictions_top25_oh[k, p2] = 1

# 2. Compute Precision / Recall / F1-score
print('\nComputing Precision, Recall, F1-scores...')
prfs = {}
for avg in ['micro', 'samples', 'macro']:
    prf = precision_recall_fscore_support(all_targets_oh, all_predictions_top25_oh, average=avg, zero_division=np.nan)[:3]
    prfs[f'Precision_{avg}'] = prf[0]
    prfs[f'Recall_{avg}'] = prf[1]
    prfs[f'F1_{avg}'] = prf[2]
    print(f"{avg.upper()}: Precision, Recall, F1", prf)


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
filtered_predictions_top25 = all_predictions_top25_oh[:][:, ~zero_cols]

aucs = {}
for avg in ['micro', 'samples', 'macro']:
    auc = roc_auc_score(filtered_targets, filtered_probas, average=avg)
    aucs[f'AUC_{avg}'] = auc
    print(f"{avg.upper()}: AUC", auc)


# 4. Save results
res.loc[0] = prfs | aucs
res.to_csv('Inference_PRC-AUC.csv', index=False)
print('\nResults saved to Inference_PRC-AUC.csv')
