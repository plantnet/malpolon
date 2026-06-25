import sys
import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from time import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb
import timm
from matplotlib import pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from dino_v2_large_PN22M.models import vit_large
from habitat_models import get_model, MultiHeadModel
# ----------------------------
# Config
# ----------------------------
SPLIT = 'S3'
BASELINE = 'B2'
MODEL = "dinov2_PN22M"  # One of ['mobilenet_v3', 'resnet18', 'resnet50', 'vitb32', 'inception_v3', 'dinov2_vits14', 'vgg16', 'convnext', 'dinov2_PN22M']

INFERENCE = False
INFERENCE_SUFFIX = ''
TRAIN_SUFFIX = ''
MULTILABEL_CORRESPONDANCE_STRATEGY = 'ml'  # One of ['soft_ml', 'ml']
LOSS_FUNCTION = 'CE_soft_ml'  # One of ['CE', 'CE_soft_ml', 'KL_divergence']
LABEL_SMOOTHING = 0.0  # Float in [0, 1]

EUNIS_LVL = ['1', '2']  # List of values in [1, 2, 3, 3_4]
EUNIS_LVL_WEIGHTS = [0.5, 0.85]
NUM_UNIQUE_CLASSES = {'1': 6, '2': 25, '3': 0, '4': 0, '3_4': 0} # Values in [9, 35, 209, 11, 215] If None, inferred from the dataset
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_FPS = {
    'CSV_S1_TRAIN': "metadata_labels_merged_S1_stratified_split-10.33%_train.csv",
    'CSV_S1_TEST': "metadata_labels_merged_S1_stratified_split-10.33%_test.csv",
    'CSV_S2_TRAIN': "metadata_labels_merged_gps_only_S2_encoded_train-0.00225deg.csv",
    'CSV_S2_TEST': "metadata_labels_merged_gps_only_S2_encoded_test-0.00225deg.csv",
    'CSV_S1BIS_TRAIN': f'metadata_labels_merged_S1bis-10%_train{TRAIN_SUFFIX}.csv',
    'CSV_S1BIS_TEST': f'metadata_labels_merged_S1bis-10%_test{INFERENCE_SUFFIX}.csv',  # "metadata_labels_merged_S1bis-10%_test.csv"
    'CSV_S0BIS_TRAIN': f'metadata_labels_merged_S0bis-10%_train{TRAIN_SUFFIX}.csv',
    'CSV_S0BIS_TEST': f'metadata_labels_merged_S0bis-10%_test{INFERENCE_SUFFIX}.csv',
    'CSV_S3_TRAIN': f'../LUCAS_habitats/data/output/csv/LUCAS_metadata_labels_merged_S3-10%_extended_train{TRAIN_SUFFIX}.csv',
    'CSV_S3_TEST': f'../LUCAS_habitats/data/output/csv/LUCAS_metadata_labels_merged_S3-10%_extended_test{INFERENCE_SUFFIX}.csv',
}

CSV_FILE = CSV_FPS[f'CSV_{SPLIT}_TRAIN']
CSV_FILE_TEST =  CSV_FPS[f'CSV_{SPLIT}_TEST'] # 'baselines/B1_freq/metadata_labels_merged_S1_stratified_split-10.33%_test_1-to-1_enc.csv'
ROOT_DIR = "../LUCAS_habitats/"
IMAGE_DIR = os.path.join(ROOT_DIR, "")  # No need because for LUCAS data the image paths are already specified in the CSV files as relative paths to the root dir, but this variable can be useful if we want to add a common prefix to the image paths specified in the CSV files.
OUTPUT_DIR = os.path.join(ROOT_DIR, f"baselines/{BASELINE}_{SPLIT}_{MODEL}_multihead/")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'inference/'), exist_ok=True)

TRAIN_METRICS = os.path.join(OUTPUT_DIR, "train_metrics.csv")
VAL_METRICS = os.path.join(OUTPUT_DIR, "val_metrics.csv")
TEST_METRICS = os.path.join(OUTPUT_DIR, f"inference/test_metrics{INFERENCE_SUFFIX}.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, f"inference/test_predictions{INFERENCE_SUFFIX}.csv")

BEST_MODEL_PATH = os.path.join(f"{OUTPUT_DIR}", "best_model.pth")
LAST_MODEL_PATH = os.path.join(f"{OUTPUT_DIR}", "last_model.pth")

TIME_STAMP_START = time()

writer = wandb.init(
    entity="tlarcher-phd-jrc",
    project='habitats_floraveg',
    name=f'{OUTPUT_DIR} (train {TRAIN_SUFFIX}, test {INFERENCE_SUFFIX})',  #'Unique surveyId spatial split 0.06min, dropout',
    notes="B2: Custom loss & metrics adapted for soft multilabelling.\n"
          "S1bis: Split over unique FLoraveg IDs (no leakeage) Stratified 1-to-k soft multilabels.",
    config={'MULTILABEL_CORRESPONDANCE_STRATEGY': MULTILABEL_CORRESPONDANCE_STRATEGY,
            'LOSS_FUNCTION': LOSS_FUNCTION,
            'LABEL_SMOOTHING': LABEL_SMOOTHING,
            'MODEL': MODEL,
            'EUNIS_LVL': EUNIS_LVL,
            'EUNIS_LVL_WEIGHTS': EUNIS_LVL_WEIGHTS,
            'NUM_UNIQUE_CLASSES': NUM_UNIQUE_CLASSES,
            'BATCH_SIZE': BATCH_SIZE,
            'EPOCHS': EPOCHS,
            'LR': LR,
            'NUM_WORKERS': NUM_WORKERS,
            'DEVICE': DEVICE,
            'CSV_FILE': CSV_FILE,
            'CSV_FILE_TEST': CSV_FILE_TEST,
            'OUTPUT_DIR': OUTPUT_DIR,
            'TRAIN_METRICS': TRAIN_METRICS,
            'VAL_METRICS': VAL_METRICS,
            'PREDICTIONS_PATH': PREDICTIONS_PATH,
            'BEST_MODEL_PATH': BEST_MODEL_PATH,
            'LAST_MODEL_PATH': LAST_MODEL_PATH,
            },
    job_type='train' if INFERENCE else 'train',
    mode='disabled',  # any of "online", "offline", "disabled"
)
# Print wandb config
print("[INFO] Wandb config:")
for key, value in writer.config.items():
    print(f"{key}: {value}")
# ----------------------------
# Dataset
# ----------------------------

class TestHabitatDataset(Dataset):
    def __init__(self, dataframe, image_dir, n_u_classes=None, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.max_unique_classes = n_u_classes
        self.labels = self.df['label']
        self.n_classes = dataframe['label'].nunique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lucas_id = row['point_id']

        img_path = os.path.join(
            self.image_dir,
            row["filepath"].strip()
        )

        image = Image.open(img_path).convert("RGB")

        # label = self.df[self.df['point_id'] == row["point_id"]]['label'].values.tolist()
        # print(f'Dataset Label: {label}')
        label_enc = row['label']
        label = row['habitats_code_lvl2']

        if self.transform:
            image = self.transform(image)

        return image, label_enc, [-1], label, lucas_id

class HabitatDatasetSoftMultilabels(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.df['label'] = self.df['habitats_code_ID_lvl2'].copy()
        self.lucas_ids = dataframe['point_id'].value_counts()
        self.labels = self.df['label']
        self.image_dir = image_dir
        self.transform = transform
        self.n_classes = dataframe['label'].nunique()
        self.label_encoding_table = dict(zip(self.df["habitats_code_ID_lvl2"], self.df["habitats_code_lvl2"]))

    def __len__(self):
        return len(self.lucas_ids)

class HabitatDatasetMultilabels(HabitatDatasetSoftMultilabels):
    """Same as HabitatDatasetSoftMultilabels but assumes data is pre-formated for multi-labelling.

    The CSV occurrences files are expected to contain 1 row per site (i.e. per unique id_floraveg).
    The labels columns are to contain strings of labels separated by a semi-colon (e.g. "N1H;N1J;Q51").
    Other columns are to be formated in the same way.
    
    The __getitem__ function returns a one-hot tensor based on the encoded labels (also expected to
    be in the same format as regular labels).
    """
    def __init__(
        self,
        dataframe,
        image_dir,
        col_code_ID='habitats_code_ID_lvl2',
        col_code='habitats_code_lvl2',
        col_id='point_id',
        col_filepath='filepath',
        n_u_classes_lvl1=None,
        n_u_classes_lvl2=None,
        transform=None
    ):
        super().__init__(dataframe, image_dir, transform)
        self.col_code_IDs = []
        self.col_codes = []
        self.col_code_ID_ohs = []
        self.col_id = col_id
        self.col_filepath = col_filepath
        self.fp_suffix_ok = ['N', 'S', 'E', 'W', 'A', 'P', 'Q', 'U',        # 2018 suffixes
                             'TransectStart', 'TransectEnd', 'UnclearAge']  # 2022 suffixes
        self.label_encoding_table = {'1': {}, '2': {}}
        self.n_u_classes = {'1': n_u_classes_lvl1,
                            '2': n_u_classes_lvl2,}
        
        for eunis_lvl in self.n_u_classes.keys():
            col_code_ID = 'habitats_code_ID' + f'_lvl{eunis_lvl}'
            col_code = 'habitats_code' + f'_lvl{eunis_lvl}'
            col_code_ID_oh = 'habitats_code_ID_oh' + f'_lvl{eunis_lvl}'
            self.col_code_ID_ohs.append(col_code_ID_oh)
            self.col_code_IDs.append(col_code_ID)
            self.col_codes.append(col_code)

            self.df[col_code_ID] = self.df[col_code_ID].astype(str)
            self.df[col_code] = self.df[col_code].astype(str)
            self.df[col_code_ID_oh] = self.df[col_code_ID_oh].apply(lambda x: [int(i) for i in x.replace('[', '').replace(']', '').split(' ')])

            # To build the corresponding table between habitats code and their encoded values. Useful to compute one-hot labels if not provided.
            # Also used in inference export to decode the predicted labels back to their habitat code format.
            for hc, hcid in zip(self.df[col_code], self.df[col_code_ID]):
                labels = [str(i) for i in hc.split(';')]
                labels_enc = [int(i) for i in hcid.split(';')]
                self.label_encoding_table[eunis_lvl].update(dict(zip(labels_enc, labels)))

    def __getitem__(self, idx):
        """Return the image and the labels for all EUNIS levels.
        
        Views are randomly picked among the selected ones. In average, all views are seen during
        training and validation given a sufficiently high number of epochs.
        """
        row = self.df.iloc[idx]
        survey_id = row[self.col_id]
        fps = row[self.col_filepath].strip().split(';')
        fps = [fp for fp in fps if Path(fp).stem.endswith(tuple(self.fp_suffix_ok))]

        img_path = os.path.join(
            self.image_dir,
            np.random.choice(fps).strip()
        )

        image = Image.open(img_path).convert("RGB")
        # Labels
        labels_lvl1 = row[f'{self.col_codes[0]}']
        labels_lvl2 = row[f'{self.col_codes[1]}']
        # Labels pre-encoded
        labels_enc_lvl1 = row[f'{self.col_code_IDs[0]}']
        labels_enc_lvl2 = row[f'{self.col_code_IDs[1]}']
        # Labels pre-encoded and one-hot encoded
        labels_enc_oh_lvl1 = torch.tensor(row[f'{self.col_code_ID_ohs[0]}'])
        labels_enc_oh_lvl2 = torch.tensor(row[f'{self.col_code_ID_ohs[1]}'])

        if self.transform:
            image = self.transform(image)

        return (image,
                labels_enc_oh_lvl1, labels_enc_oh_lvl2, torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]),
                labels_enc_lvl1, labels_enc_lvl2, torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]),
                labels_lvl1, labels_lvl2, torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]),
                survey_id)
    
    def labels_value_counts(self, label_id):
        """This method computes the number of samples per unique label in a multilabel dataframe"""
        return self.label_value_counts[label_id]

# ----------------------------
# Load CSV
# ----------------------------

def sample_onehot_encode(labels, n_labels):
    t = torch.zeros(n_labels, dtype=int)
    t[torch.tensor(labels)] = 1
    return t

def batch_onehot_encode(labels, n_classes):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, n_classes, device=labels.device)
    out.scatter_(1, labels.long(), 1)

    return out

df = pd.read_csv(CSV_FILE)
df_test = pd.read_csv(CSV_FILE_TEST)

lvl_suffix = '_lvl2'  # '' if EUNIS_LVL == '3_4' else '_lvl'+str(EUNIS_LVL)
df['label'] = df[f'habitats_code_ID{lvl_suffix}']
df_test['label'] = df[f'habitats_code_ID{lvl_suffix}']

if not NUM_UNIQUE_CLASSES:  # Only set based on data if not manually set at the begining of the config section
    raise NotImplementedError(f"No auto-computation of NUM_UNIQUE_CLASSES in this multi-head version. Please set it manually in the config section.")
print("[INFO] Unique classes per EUNIS level:", NUM_UNIQUE_CLASSES)

if MULTILABEL_CORRESPONDANCE_STRATEGY in ['ml']:
    label_value_counts = {}
    for fid, hc, hcid in zip(df["point_id"], df[f"habitats_code{lvl_suffix}"], df[f"habitats_code_ID{lvl_suffix}"]):
        labels = [str(i) for i in hc.split(';')]
        for label in labels:
            label_value_counts[label] = label_value_counts[label] + 1 if label in label_value_counts.keys() else 1
    df_labels_value_count = pd.DataFrame({f'habitats_code_ID{lvl_suffix}': list(label_value_counts.keys()), 'count': list(label_value_counts.values())})
    habitats_single_occurrence = df_labels_value_count[df_labels_value_count['count']<=1]

    habitats_counts = df['label'].value_counts()
    habitats_single_occurrence = habitats_counts[habitats_counts == 1]
else:
    habitats_counts = df['label'].value_counts()
    habitats_single_occurrence = habitats_counts[habitats_counts == 1]

if len(habitats_single_occurrence) > 1:
    df_habitats_single_occurrence = df[df['label'].isin(habitats_single_occurrence.index)]
    df = df[~df['label'].isin(habitats_single_occurrence.index)]
    print(f'[WARNING]: The following habitat codes were excluded from validation and kept in train because they only have one occurrence: {habitats_single_occurrence.index.tolist()}')

# ----------------------------
# Train / Val split
# ----------------------------

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["label"],
    random_state=42
)


train_df = pd.concat([train_df, df_habitats_single_occurrence])

# ----------------------------
# Transforms
# ----------------------------
# To check the correct values of resize and centercrop, call torchvision.models.<model>_Weights.IMAGENET1K_V1.transforms()
# The exact name of the class can be found on the doc page of each specific model, ex: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html#torchvision.models.inception_v3
MODEL_STATS_IMAGENET = {
    "mean": (0.442, 0.469, 0.326),
    "std": (0.229,0.224,0.225),
}
MODEL_STATS_PN22M = {
    "mean": (0.485,0.456,0.406),
    "std": (0.235, 0.221, 0.232),
}
if MODEL == 'resnet18':
    model_specific_transforms = [transforms.Resize(256),
                                 transforms.CenterCrop(224),]
    MODEL_STATS = MODEL_STATS_IMAGENET
elif MODEL == 'resnet50':
    model_specific_transforms = [transforms.Resize(232),  # transforms for IMAGENET1K_V2 are different from V1
                                 transforms.CenterCrop(224),]
    MODEL_STATS = MODEL_STATS_IMAGENET
elif MODEL == 'dinov2_vits14':
    model_specific_transforms = [transforms.Resize(520),
                                 transforms.CenterCrop(518),]
    MODEL_STATS = MODEL_STATS_IMAGENET
elif MODEL == 'dinov2_PN22M':
    model_specific_transforms = [transforms.Resize((224, 224)),
                                 ]
    MODEL_STATS = MODEL_STATS_PN22M
elif MODEL == 'convnext':
    model_specific_transforms = [transforms.Resize(236),
                                 transforms.CenterCrop(224),]
    MODEL_STATS = MODEL_STATS_IMAGENET
elif MODEL == 'vgg16':
    model_specific_transforms = [transforms.Resize(256),
                                 transforms.CenterCrop(224),]
    MODEL_STATS = MODEL_STATS_IMAGENET
elif MODEL == 'vitb32':
    model_specific_transforms = [transforms.Resize(224),
                                 transforms.CenterCrop(224),]
    MODEL_STATS = MODEL_STATS_IMAGENET
elif MODEL == 'mobilenet_v3':
    model_specific_transforms = [transforms.Resize(232),
                                 transforms.CenterCrop(224),]
    MODEL_STATS = MODEL_STATS_IMAGENET
elif MODEL == 'inception_v3':
    model_specific_transforms = [transforms.Resize(342),
                                 transforms.CenterCrop(299),]
    MODEL_STATS = MODEL_STATS_IMAGENET
else:
    model_specific_transforms = []
    print(f'[ERROR] Unknown MODEL: {MODEL}, no resize transform applied !')

train_tf = transforms.Compose(
    model_specific_transforms + 
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MODEL_STATS['mean'],
            std=MODEL_STATS['std']
        )
    ]
)

val_tf = transforms.Compose(
    model_specific_transforms + 
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MODEL_STATS['mean'],
            std=MODEL_STATS['std']
        )
    ]
)


# ----------------------------
# Datasets
# ----------------------------
match MULTILABEL_CORRESPONDANCE_STRATEGY:
    case 'soft_ml':
        dataset = HabitatDatasetSoftMultilabels
        test_dataset = HabitatDatasetSoftMultilabels
    case 'ml':
        dataset = HabitatDatasetMultilabels
        test_dataset = HabitatDatasetMultilabels
    case _:
        print(f'[ERROR] Unknown MULTILABEL_CORRESPONDANCE_STRATEGY: {MULTILABEL_CORRESPONDANCE_STRATEGY}')

train_dataset = dataset(train_df,
                        IMAGE_DIR,
                        n_u_classes_lvl1=NUM_UNIQUE_CLASSES['1'], n_u_classes_lvl2=NUM_UNIQUE_CLASSES['2'],
                        transform=train_tf)
val_dataset = dataset(val_df,
                      IMAGE_DIR,
                      n_u_classes_lvl1=NUM_UNIQUE_CLASSES['1'], n_u_classes_lvl2=NUM_UNIQUE_CLASSES['2'],
                      transform=val_tf)
test_dataset = test_dataset(df_test,
                            IMAGE_DIR,
                            n_u_classes_lvl1=NUM_UNIQUE_CLASSES['1'], n_u_classes_lvl2=NUM_UNIQUE_CLASSES['2'],
                            transform=val_tf)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)
print("[INFO] Train size:", len(train_loader.dataset))
print("[INFO] Number of classes in the training set:", train_loader.dataset.n_classes)
print("[INFO] Val size:", len(val_loader.dataset))
print(f"[INFO] Number of classes in the validation set: {val_loader.dataset.n_classes} ({len(set(val_loader.dataset.labels) & set(train_loader.dataset.labels)) / train_loader.dataset.n_classes * 100:.2f}% overlap with train)")
print("[INFO] Test size:", len(test_loader.dataset))
print(f"[INFO] Number of classes in the test set: {test_loader.dataset.n_classes} ({len(set(test_loader.dataset.labels) & set(train_loader.dataset.labels)) / train_loader.dataset.n_classes * 100:.2f}% overlap with train)")

# ----------------------------
# Model
# ----------------------------
model = get_model(MODEL, 10).to(DEVICE)  # Doesn't mater what number of classes we put here since we are going to replace the head with a custom one adapted for multilabels. This is just to re-use code written to instantiate models.
model = MultiHeadModel(model, MODEL, EUNIS_LVL,
                       NUM_UNIQUE_CLASSES['1'], NUM_UNIQUE_CLASSES['2'], NUM_UNIQUE_CLASSES['3'], NUM_UNIQUE_CLASSES['4'], NUM_UNIQUE_CLASSES['3_4']).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

def get_criterion(logits, labels):
    match LOSS_FUNCTION:
        case 'CE':
            criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
            loss = criterion(logits, labels)
        case 'CE_soft_ml':  #  Expects class probabilities
            criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
            ones_per_row = labels.sum(dim=1, keepdim=True)
            labels = labels / ones_per_row.clamp(min=1)
            for row in labels:
                non_zero_values = row[row != 0]
            loss = criterion(logits, labels)
        case 'KL_divergence':
            criterion = F.kl_div(
            F.log_softmax(logits, dim=1),
            labels,
            reduction="batchmean"
        )
    return loss
    
# ----------------------------
# Metrics
# ----------------------------

def top1_soft_multilabels_accuracy(y_true, y_pred):
    top_indices = np.argmax(y_pred, axis=1)
    result = (y_true[np.arange(y_true.shape[0]), top_indices] == 1).astype(int)
    return result.mean()

# def topk_soft_multilabels_accuracy(y_true, y_pred, k):
#     top_indices = np.argsort(y_pred, axis=1)[:, -k:]
#     # result = (y_true[np.arange(y_true.shape[0]), top_indices] == 1).astype(int)
#     return result.mean()

def get_confusion_matrix(all_labels, all_preds, num_classes, normalize=True):
    
    # The confusion matrix must be computed by hand because labels are in multilabel format while the targets are multiclass
    cm = np.zeros((num_classes, num_classes), dtype=np.float32)
    # all_labels_weighted = all_labels / all_labels.sum(axis=1, keepdims=True)
    for i in range(len(all_preds)):
        pred_class = all_preds[i]
        # distribute contribution
        for true_class in range(num_classes):
            weight = all_labels[i, true_class]
            if weight > 0:
                cm[true_class, pred_class] += weight

    if normalize:
        # [INFO]: since the inference set does not contain all the labels of the dataset, some rows will contain NaNs and will be displayed white by default
        cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

    return cm, cm_norm
    
def compute_metrics(y_true_lvl, y_pred_lvl, y_prob_lvl):
    if MULTILABEL_CORRESPONDANCE_STRATEGY in ['soft_ml', 'ml']:
        acc_top1_softml = {'1': -1, '2': -1, '3': -1, '4': -1, '3_4': -1}
        cm = {'1': None, '2': None, '3': None, '4': None, '3_4': None}
        cm_norm = {'1': None, '2': None, '3': None, '4': None, '3_4': None}
        for lvl in EUNIS_LVL:
            y_true, y_prob, y_pred = np.array(y_true_lvl[lvl]), np.array(y_prob_lvl[lvl]), np.array(y_pred_lvl[lvl])

            acc_top1_softml[lvl] = top1_soft_multilabels_accuracy(y_true, y_prob)
            cm[lvl], cm_norm[lvl] = get_confusion_matrix(y_true, y_pred, NUM_UNIQUE_CLASSES[lvl], normalize=True)
        return [acc_top1_softml, cm, cm_norm]
    else:
        raise NotImplementedError(f"Unknown MULTILABEL_CORRESPONDANCE_STRATEGY: {MULTILABEL_CORRESPONDANCE_STRATEGY}")

# ----------------------------
# Epoch runner
# ----------------------------

def run_epoch(loader, split, epoch_nb, training=True):

    if training:
        model.train()
    else:
        model.eval()

    all_labels_enc_oh = {'1': [], '2': [], '3': [], '4': [], '3_4': []}
    all_preds = {'1': [], '2': [], '3': [], '4': [], '3_4': []}
    all_probs = {'1': [], '2': [], '3': [], '4': [], '3_4': []}

    running_loss = 0

    for (images,
         labels_enc_oh_lvl1, labels_enc_oh_lvl2, labels_enc_oh_lvl3, labels_enc_oh_lvl4, labels_enc_oh_lvl3_4,
         labels_enc_lvl1, labels_enc_lvl2, labels_enc_lvl3, labels_enc_lvl4, labels_enc_lvl3_4,
         labels_lvl1, labels_lvl2, labels_lvl3, labels_lvl4, labels_lvl3_4,
         survey_id
    ) in tqdm(loader):

        labels_enc_ohs = {'1': labels_enc_oh_lvl1,
                          '2': labels_enc_oh_lvl2,
                          '3': labels_enc_oh_lvl3,
                          '4': labels_enc_oh_lvl4,
                          '3_4': labels_enc_oh_lvl3_4}
        probs = {'1': None, '2': None, '3': None, '4': None, '3_4': None}
        preds = {'1': None, '2': None, '3': None, '4': None, '3_4': None}
        images = images.to(DEVICE)
        # Only keep labels selected by EUNIS_LVL config constant and move to device
        for k, v in labels_enc_ohs.items():
            if k in EUNIS_LVL:
                labels_enc_ohs[k] = v.to(DEVICE)
            else:
                labels_enc_ohs[k] = None

        if training:
            optimizer.zero_grad()

        if MODEL == 'inception_v3' and split == 'train':
            outputs, aux_output = model(images)
        else:
            outputs = model(images)

        # loss = get_criterion(outputs, labels_enc_oh)
        loss = 0

        for lvl, weight in zip(EUNIS_LVL, EUNIS_LVL_WEIGHTS):
            loss += get_criterion(outputs[lvl], labels_enc_ohs[lvl])

        if training:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        for lvl in EUNIS_LVL:
            probs[lvl] = torch.softmax(outputs[lvl], dim=1)
            preds[lvl] = torch.argmax(probs[lvl], dim=1)

            all_labels_enc_oh[lvl].extend(labels_enc_ohs[lvl].cpu().numpy())
            all_preds[lvl].extend(preds[lvl].cpu().numpy())
            all_probs[lvl].extend(probs[lvl].detach().cpu().numpy())


    metrics = compute_metrics(
        all_labels_enc_oh,
        all_preds,
        all_probs
    )

    loss = running_loss / len(loader)
    
    wandb.log({
        f"loss (epoch)/{split}": loss,
    }, step=epoch_nb)
    if MULTILABEL_CORRESPONDANCE_STRATEGY in ['soft_ml', 'ml']:
        for lvl in EUNIS_LVL:
            wandb.log({
                f"acc_top1_softml (epoch)/{split}/eunis_lvl_{lvl}": metrics[0][lvl],
            }, step=epoch_nb)

    return loss, metrics


# ----------------------------
# Training
# ----------------------------
if not INFERENCE:
    # ----------------------------
    # Init logs
    # ----------------------------

    with open(TRAIN_METRICS,"w") as f:
        f.write("epoch,loss,accuracy,precision,recall,f1,auroc\n")

    with open(VAL_METRICS,"w") as f:
        f.write("epoch,loss,accuracy,precision,recall,f1,auroc\n")

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        print("\nEpoch",epoch)

        train_loss, train_metrics_lvl = run_epoch(
            train_loader,
            'train',
            epoch,
            training=True
        )

        val_loss, val_metrics_lvl = run_epoch(
            val_loader,
            'val',
            epoch,
            training=False
        )

        print("\nTRAIN")
        print("Loss:",train_loss)
        if MULTILABEL_CORRESPONDANCE_STRATEGY in ['soft_ml', 'ml']:
            for lvl in EUNIS_LVL:
                train_metrics = (train_metrics_lvl[0][lvl], train_metrics_lvl[1][lvl], train_metrics_lvl[2][lvl])
                print(f"Acc_top1_soft_multilabels_lvl-{lvl}:", train_metrics[0])
                with open(TRAIN_METRICS,"a") as f:
                    f.write(
                        f"{epoch},{train_loss},eunis_lvl-{lvl}{train_metrics[0]}\n"
                    )
        else:
            raise NotImplementedError(f"Unknown MULTILABEL_CORRESPONDANCE_STRATEGY: {MULTILABEL_CORRESPONDANCE_STRATEGY} (train metrics logging)")

        print("\nVAL")
        print("Loss:",val_loss)
        if MULTILABEL_CORRESPONDANCE_STRATEGY in ['soft_ml', 'ml']:
            for lvl in EUNIS_LVL:
                val_metrics = (train_metrics_lvl[0][lvl], train_metrics_lvl[1][lvl], train_metrics_lvl[2][lvl])
                print(f"Acc_top1_soft_multilabel_lvl-{lvl}", val_metrics[0])
                # Save logs
                with open(VAL_METRICS,"a") as f:
                    f.write(
                        f"{epoch},{val_loss},eunis_lvl-{lvl}{val_metrics[0]}\n"
                    )
        else:
            raise NotImplementedError(f"Unknown MULTILABEL_CORRESPONDANCE_STRATEGY: {MULTILABEL_CORRESPONDANCE_STRATEGY} (val metrics logging)")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, BEST_MODEL_PATH)
            print("Best model saved.")


        # Save last model
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, LAST_MODEL_PATH)
        print(f"Epoch n°{epoch} model saved.")

    print("\nLast model saved.")
    print("Training finished.")
    print(f'Elapsed time for training: {(time() - TIME_STAMP_START):.2f}s')

# ----------------------------
# Final Test Evaluation
# ----------------------------

print("\nTEST EVALUATION")
TIME_STAMP_TEST_EVAL = time()

# Load best model
checkpoint = torch.load(BEST_MODEL_PATH)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_metrics_lvl = run_epoch(
    test_loader,
    'test',
    0,
    training=False
)

print("\nTEST")
print("Loss:",test_loss)
if MULTILABEL_CORRESPONDANCE_STRATEGY in ['soft_ml', 'ml']:
    csv_header = 'loss,'
    csv_values = f"{test_loss},"
    for lvl in EUNIS_LVL:
        test_metrics = (test_metrics_lvl[0][lvl], test_metrics_lvl[1][lvl], test_metrics_lvl[2][lvl])
        
        print(f"Acc_top1_soft_multilabel_eunis_lvl-{lvl}", test_metrics[0])
        csv_header += f'Acc_top1_soft_multilabel_eunis_lvl-{lvl},'
        csv_values += f"{test_metrics[0]},"

        # Save confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        im0 = axes[0].imshow(test_metrics[1])
        axes[0].set_xlabel("Predicted class")
        axes[0].set_ylabel("True class")
        axes[0].set_title(f"Confusion Matrix (EUNIS lvl-{lvl})")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(test_metrics[2])
        axes[1].set_xlabel("Predicted class")
        axes[1].set_ylabel("True class")
        axes[1].set_title(f"Confusion Matrix (normalized) (EUNIS lvl-{lvl})")
        fig.colorbar(im1, ax=axes[1])
        plt.tight_layout()
        plt.savefig(f'{Path(TEST_METRICS).parent}/confusion_matrix_eunis_lvl-{lvl}.png')
        plt.close()
    # Save logs
    with open(TEST_METRICS,"w") as f:
        f.write(f"{csv_header}\n")
        f.write(
            f"{csv_values}\n"
        )


else:
    raise NotImplementedError(f"Unknown MULTILABEL_CORRESPONDANCE_STRATEGY: {MULTILABEL_CORRESPONDANCE_STRATEGY} (test metrics logging)")

print("\nTest evaluation saved.")
print(f'Elapsed time for test evaluation: {(time() - TIME_STAMP_TEST_EVAL):.2f}s')

wandb.finish()
# ----------------------------
# Export Test Predictions
# ----------------------------

print("\nExporting test predictions...")
TIME_STAMP_INFERENCE = time()

model.eval()

res = {'1': None, '2': None, '3': None, '4': None, '3_4': None}
# all_preds = {'1': [], '2': [], '3': [], '4': [], '3_4': []}
# all_probs = {'1': [], '2': [], '3': [], '4': [], '3_4': []}
# all_labels_enc = {'1': [], '2': [], '3': [], '4': [], '3_4': []}
# all_labels_enc_softml_str = {'1': [], '2': [], '3': [], '4': [], '3_4': []}

with torch.no_grad():
    for (images,
         labels_enc_oh_lvl1, labels_enc_oh_lvl2, labels_enc_oh_lvl3, labels_enc_oh_lvl4, labels_enc_oh_lvl3_4,
         labels_enc_lvl1, labels_enc_lvl2, labels_enc_lvl3, labels_enc_lvl4, labels_enc_lvl3_4,
         labels_lvl1, labels_lvl2, labels_lvl3, labels_lvl4, labels_lvl3_4,
         survey_id
    ) in tqdm(test_loader):
        labels_enc_ohs = {'1': labels_enc_oh_lvl1, '2': labels_enc_oh_lvl2, '3': labels_enc_oh_lvl3, '4': labels_enc_oh_lvl4, '3_4': labels_enc_oh_lvl3_4}
        labels_encs = {'1': labels_enc_lvl1, '2': labels_enc_lvl2, '3': labels_enc_lvl3, '4': labels_enc_lvl4, '3_4': labels_enc_lvl3_4}
        labels_lvls = {'1': labels_lvl1, '2': labels_lvl2, '3': labels_lvl3, '4': labels_lvl4, '3_4': labels_lvl3_4}
        probs, preds = None, None

        images = images.to(DEVICE)
        outputs = model(images)

        for lvl in EUNIS_LVL:
            probs = torch.softmax(outputs[lvl], dim=1)
            preds = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1).values

            res[lvl].update({
                sid.item(): {
                    'all_preds': preds[i].cpu().to(torch.int32).item(),
                    'all_preds_confidence': confidence[i].cpu().item(),
                    'all_probs': probs[i].cpu().numpy(),
                    'all_labels': labels_lvls[lvl][i],
                    'all_labels_enc': labels_encs[lvl][i].cpu().numpy(),
                    'all_labels_enc_oh': labels_enc_ohs[lvl][i],
                }
                for i, sid in enumerate(survey_id)
            })

# Decode labels
pred_df = df_test.reset_index(drop=True).copy()
if MULTILABEL_CORRESPONDANCE_STRATEGY in ['soft_ml', 'ml']:
    # Length of dataset = n_unique id_floraveg
    pred_df = pred_df.drop_duplicates(subset=['point_id']).reset_index(drop=True)
    dataset_labels_table = {**train_loader.dataset.label_encoding_table, **test_loader.dataset.label_encoding_table}
    for i in range(len(pred_df)):
        lucas_id = pred_df.loc[i, 'point_id']
        pred_df.loc[i, 'habitats_code_lvl2'] = res[list(res.keys())[0]][lucas_id]['all_labels']
        pred_df.loc[i, 'habitats_code_ID_lvl2'] = res[list(res.keys())[0]][lucas_id]['all_labels_enc_softml_str']
        for lvl in EUNIS_LVL:
            pred_df.loc[i, f'pred_label_lvl{lvl}'] = dataset_labels_table[res[lvl][lucas_id]['all_preds']]
            pred_df.loc[i, f'pred_label_encoded_lvl{lvl}'] = res[lvl][lucas_id]['all_preds']
            pred_df.loc[i, f'pred_confidence_lvl{lvl}'] = res[lvl][lucas_id]['all_preds_confidence']
            pred_df.loc[i, f'valid_prediction_lvl{lvl}'] = str(int(pred_df.loc[i, f'pred_label_encoded_lvl{lvl}'])) in pred_df.loc[i, 'habitats_code_ID_lvl2']
    for lvl in EUNIS_LVL:
        pred_df[f'pred_label_encoded_lvl{lvl}'] = pred_df[f'pred_label_encoded_lvl{lvl}'].astype(int)
    pred_df['label'] = pred_df['habitats_code_ID_lvl2']
else:
    raise NotImplementedError(f"Unknown MULTILABEL_CORRESPONDANCE_STRATEGY: {MULTILABEL_CORRESPONDANCE_STRATEGY} (predictions export)")

pred_df.to_csv(PREDICTIONS_PATH, index=False)

print("Predictions saved:", PREDICTIONS_PATH)
print(f'Elapsed time for inference export: {(time() - TIME_STAMP_INFERENCE):.2f}s')
print(f'Total Elapsed time: {(time() - TIME_STAMP_START):.2f}s')