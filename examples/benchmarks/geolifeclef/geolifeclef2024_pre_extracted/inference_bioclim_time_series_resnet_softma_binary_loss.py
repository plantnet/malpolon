import csv
import itertools
import os
import ssl

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset  # Import Subset class
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights

df_gt = pd.read_csv('GLC24_SOLUTION_FILE.csv')
df_preds = pd.read_csv('predictions_top25_GLC24_SOLUTION_FILE.csv', sep=';')
obs_id = df_gt['surveyId']

targets = df_gt['target_species_ids']
targets = [list(map(int, x.split())) for x in targets]

preds = df_preds['predictions']
preds = [list(map(int, x.split())) for x in preds]

probas = df_preds['probas']
probas = [list(map(float, x.split())) for x in probas]


# Calculate F1 score
all_targets = list(itertools.chain.from_iterable(targets))
all_predictions = list(itertools.chain.from_iterable(preds))
all_probas = list(itertools.chain.from_iterable(probas))
print("SAMPLE: Precision, Recall, F1")
print(precision_recall_fscore_support(all_targets, all_predictions, average='samples', zero_division='warn'))
print("MACRO: Precision, Recall, F1 ")
print(precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division='warn'))
print("MICRO: Precision, Recall, F1 ")
print(precision_recall_fscore_support(all_targets, all_predictions, average='micro', zero_division='warn'))

# Find rows and columns with all zeros in both arrays
zero_cols_targets = np.all(all_targets == 0, axis=0)
ones_cols_targets = np.all(all_targets == 1, axis=0)

# Combine zero rows and columns from both arrays
zero_cols = zero_cols_targets #& ones_cols_targets

# Filter out rows and columns containing only zeros
filtered_targets = all_targets[:][:,~zero_cols]
filtered_predictions = all_predictions[:][:,~zero_cols]


np.savetxt('filtered_targets.txt', np.sum(filtered_targets, axis=0))

print("micro:",roc_auc_score(filtered_targets, filtered_predictions, average='micro'))
print("samples:",roc_auc_score(filtered_targets, filtered_predictions, average='samples'))
print("macro:",roc_auc_score(filtered_targets, filtered_predictions, average='macro'))
