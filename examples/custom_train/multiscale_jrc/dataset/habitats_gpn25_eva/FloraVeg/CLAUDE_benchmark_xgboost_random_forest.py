import os
import re
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import xgboost
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SPLIT = 'S3'
BASELINE = 'B2'
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_UNIQUE_CLASSES = {'1': 9, '2': 35, '3': 209, '4': 11, '3_4': 215}

METADATA_ROOT_PATH = "metadata/"
INFERENCE_SUFFIX = ''
TRAIN_SUFFIX = ''
CSV_FPS = {
    'CSV_S1_TRAIN': "metadata_labels_merged_S1_stratified_split-10.33%_train.csv",
    'CSV_S1_TEST': "metadata_labels_merged_S1_stratified_split-10.33%_test.csv",
    'CSV_S2_TRAIN': "metadata_labels_merged_gps_only_S2_encoded_train-0.00225deg.csv",
    'CSV_S2_TEST': "metadata_labels_merged_gps_only_S2_encoded_test-0.00225deg.csv",
    'CSV_S1BIS_TRAIN': f'metadata_labels_merged_S1bis-10%_train{TRAIN_SUFFIX}.csv',
    'CSV_S1BIS_TEST': f'metadata_labels_merged_S1bis-10%_test{INFERENCE_SUFFIX}.csv',  # "metadata_labels_merged_S1bis-10%_test.csv"
    'CSV_S0BIS_TRAIN': f'metadata_labels_merged_S0bis-10%_train{TRAIN_SUFFIX}.csv',
    'CSV_S0BIS_TEST': f'metadata_labels_merged_S0bis-10%_test{INFERENCE_SUFFIX}.csv',
    'CSV_S3_TRAIN': f'metadata_labels_merged_S3-10%_extended_train{TRAIN_SUFFIX}.csv',
    'CSV_S3_TEST': f'metadata_labels_merged_S3-10%_extended_test{INFERENCE_SUFFIX}.csv',
}
CSV_FILE_TRAIN = os.path.join(METADATA_ROOT_PATH, CSV_FPS[f'CSV_{SPLIT}_TRAIN'])
CSV_FILE_TEST = os.path.join(METADATA_ROOT_PATH, CSV_FPS[f'CSV_{SPLIT}_TEST'])
HABITATS_CODE_CSV = os.path.join(METADATA_ROOT_PATH, "habitats_code_encoded.csv")
IMAGE_DIR = "Images/"
OUTPUT_DIR = "output_agents/"

# Frozen pre-trained backbone used to turn each image into a fixed-size feature vector
MULTILABEL_CORRESPONDANCE_STRATEGY = 'ml'  # One of ['soft_ml', 'ml']
BACKBONE = "resnet18"  # One of ["dinov2_PN22M", "resnet18"]
BATCH_SIZE = 32
NUM_WORKERS = 4
N_JOBS = -1


CONFIG_RF = {
    'n_estimators':128,
    'max_features': 'sqrt',
    'max_samples': 1.0,
    'max_depth':7,
    'class_weight':'balanced',
    'criterion': 'gini',
    'verbose': 1,
    'n_jobs': -1
}
CONFIG_XGB = {
    'hyperparams': {
        'n_estimators':512,
        'eta': 0.1,
        'n_jobs': -1,
        'max_depth': 0,
        'enable_categorical': True,
        'objective':'multi:softprob',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'verbosity': 1        
    },
    
    'train_params' : {
        'sample_weight' : True,
        'early_stopping_rounds':10,
        'eval_metric': 'mlogloss',
        'maximize': False,
        'min_delta': 1e-3
    }
}
N_ESTIMATORS = 512
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 5
XGB_USE_SAMPLE_WEIGHTS = True  # inverse class-frequency sample weights
RF_MAX_DEPTH = 6
RF_CLASS_WEIGHT = "balanced"

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
if BACKBONE == "dinov2_PN22M":
    IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.235, 0.221, 0.232)
else:
    IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ----------------------------------------------------------------------
# Feature extractor (frozen backbone)
# ----------------------------------------------------------------------

def build_feature_extractor():

    if BACKBONE == "dinov2_PN22M":
        from dino_v2_large_PN22M.models import vit_large

        weights_path = os.path.join("dino_v2_large_PN22M", "ssl_vitl_224.pth")

        def load_state_dict(path):
            sd = torch.load(path, map_location="cpu")["teacher"]
            sd_df = pd.DataFrame({"key": list(sd.keys()), "value": list(sd.values())})
            sd_df = sd_df[sd_df["key"].str.startswith("backbone.")]
            sd_df["key"] = (
                sd_df["key"]
                .str.replace("backbone.", "", regex=False)
                .str.replace(r"blocks\.\d+\.(?=\d+)", "blocks.", regex=True)
            )
            return dict(zip(sd_df["key"], sd_df["value"]))

        model = vit_large(patch_size=16, img_size=224, init_values=0.1, block_chunks=0, num_register_tokens=4)
        model.load_state_dict(load_state_dict(weights_path), strict=True)
        model.eval()
        model.requires_grad_(False)
        model.to(DEVICE)

        def extract(images):
            feats = model.forward_features(images)
            return feats["x_norm_clstoken"] if isinstance(feats, dict) else feats[:, 0]

        return extract

    if BACKBONE == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Identity()
        model.eval()
        model.requires_grad_(False)
        model.to(DEVICE)
        return lambda images: model(images)

    raise ValueError(f"Unknown BACKBONE: {BACKBONE}")

# ----------------------------------------------------------------------
# Load CSVs
# ----------------------------------------------------------------------


def drop_unnamed_columns(df):
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


train_df = drop_unnamed_columns(pd.read_csv(CSV_FILE_TRAIN))
test_df = drop_unnamed_columns(pd.read_csv(CSV_FILE_TEST))

# Dense 0..C-1 label encoding (required by XGBoost / sklearn)
all_code_ids = sorted(set(train_df["habitats_code_ID_lvl3"].tolist()) | set(test_df["habitats_code_ID_lvl3"].tolist()))
code_id_to_label = dict(zip(all_code_ids, range(len(all_code_ids))))
label_to_code_id = dict(zip(range(len(all_code_ids)), all_code_ids))
NUM_CLASSES = len(all_code_ids)

train_df["label"] = train_df["habitats_code_ID_lvl3"].map(code_id_to_label).astype(int)
test_df["label"] = test_df["habitats_code_ID_lvl3"].map(code_id_to_label).astype(int)

habitats_code_df = pd.read_csv(HABITATS_CODE_CSV)
label_to_code = dict(zip(habitats_code_df["code_ID"].astype(int), habitats_code_df["code"].astype(str)))

print("[INFO] Backbone:", BACKBONE, "| Device:", DEVICE)
print("[INFO] Train samples:", len(train_df), "| Test samples:", len(test_df))
print("[INFO] Train classes:", train_df["label"].nunique(), "| Test classes:", test_df["label"].nunique())

# ----------------------------------------------------------------------
# Datasets / DataLoaders
# ----------------------------------------------------------------------


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
        floraveg_id = row['id_floraveg']

        img_path = os.path.join(
            self.image_dir,
            row["filename_photos"].strip()
        )

        image = Image.open(img_path).convert("RGB")

        # label = self.df[self.df['id_floraveg'] == row["id_floraveg"]]['label'].values.tolist()
        # print(f'Dataset Label: {label}')
        label_enc = row['label']
        label = row['habitats_code']

        if self.transform:
            image = self.transform(image)

        return image, label_enc, [-1], label, floraveg_id

class HabitatDatasetSoftMultilabels(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.df['label'] = self.df['habitats_code_ID'].copy()
        self.floraveg_ids = dataframe['id_floraveg'].value_counts()
        self.labels = self.df['label']
        self.image_dir = image_dir
        self.transform = transform
        self.n_classes = dataframe['label'].nunique()
        self.label_encoding_table = dict(zip(self.df["habitats_code_ID"], self.df["habitats_code"]))

    def __len__(self):
        return len(self.floraveg_ids)

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
        col_code_ID='habitats_code_ID',
        col_code='habitats_code',
        col_id='id_floraveg',
        col_filepath='filename_photos',
        n_u_classes_lvl1=None,
        n_u_classes_lvl2=None,
        n_u_classes_lvl3=None,
        n_u_classes_lvl4=None,
        n_u_classes_lvl3_4=None,
        transform=None
    ):
        super().__init__(dataframe, image_dir, transform)
        self.col_code_IDs = []
        self.col_codes = []
        self.col_code_ID_ohs = []
        self.col_id = col_id
        self.col_filepath = col_filepath
        self.label_encoding_table = {'1': {}, '2': {}, '3': {}, '4': {}, '3_4': {}}
        self.n_u_classes = {'1': n_u_classes_lvl1,
                            '2': n_u_classes_lvl2,
                            '3': n_u_classes_lvl3,
                            '4': n_u_classes_lvl4,
                            '3_4': n_u_classes_lvl3_4}

        for eunis_lvl in self.n_u_classes.keys():
            col_code_ID = 'habitats_code_ID' + (f'_lvl{eunis_lvl}' if eunis_lvl != '3_4' else '')
            col_code = 'habitats_code' + (f'_lvl{eunis_lvl}' if eunis_lvl != '3_4' else '')
            col_code_ID_oh = 'habitats_code_ID_oh' + (f'_lvl{eunis_lvl}' if eunis_lvl != '3_4' else '')
            self.col_code_ID_ohs.append(col_code_ID_oh)
            self.col_code_IDs.append(col_code_ID)
            self.col_codes.append(col_code)

            self.df[col_code_ID] = self.df[col_code_ID].astype(str)
            self.df[col_code] = self.df[col_code].fillna('')
            self.df[col_code] = self.df[col_code].astype(str)
            self.df[col_code_ID_oh] = self.df[col_code_ID_oh].apply(lambda x: [int(i) for i in x.replace('[', '').replace(']', '').split(' ')])

            # To build the corresponding table between habitats code and their encoded values. Useful to compute one-hot labels if not provided.
            # Also used in inference export to decode the predicted labels back to their habitat code format.
            for hc, hcid in zip(self.df[col_code], self.df[col_code_ID]):
                labels = [str(i) for i in hc.split(';')]
                labels_enc = [int(i) for i in hcid.split(';')]
                self.label_encoding_table[eunis_lvl].update(dict(zip(labels_enc, labels)))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        survey_id = row[self.col_id]

        img_path = os.path.join(
            self.image_dir,
            row[self.col_filepath].strip()
        )

        image = Image.open(img_path).convert("RGB")
        # Labels
        labels_lvl1 = row[f'{self.col_codes[0]}']
        labels_lvl2 = row[f'{self.col_codes[1]}']
        labels_lvl3 = row[f'{self.col_codes[2]}']
        labels_lvl4 = row[f'{self.col_codes[3]}']
        labels_lvl3_4 = row[f'{self.col_codes[4]}']
        # Labels pre-encoded
        labels_enc_lvl1 = row[f'{self.col_code_IDs[0]}']
        labels_enc_lvl2 = row[f'{self.col_code_IDs[1]}']
        labels_enc_lvl3 = row[f'{self.col_code_IDs[2]}']
        labels_enc_lvl4 = row[f'{self.col_code_IDs[3]}']
        labels_enc_lvl3_4 = row[f'{self.col_code_IDs[4]}']
        # Labels pre-encoded and one-hot encoded
        labels_enc_oh_lvl1 = torch.tensor(row[f'{self.col_code_ID_ohs[0]}'])
        labels_enc_oh_lvl2 = torch.tensor(row[f'{self.col_code_ID_ohs[1]}'])
        labels_enc_oh_lvl3 = torch.tensor(row[f'{self.col_code_ID_ohs[2]}'])
        labels_enc_oh_lvl4 = torch.tensor(row[f'{self.col_code_ID_ohs[3]}'])
        labels_enc_oh_lvl3_4 = torch.tensor(row[f'{self.col_code_ID_ohs[4]}'])

        if self.transform:
            image = self.transform(image)

        return (image,
                labels_enc_oh_lvl1, labels_enc_oh_lvl2, labels_enc_oh_lvl3, labels_enc_oh_lvl4, labels_enc_oh_lvl3_4,
                labels_enc_lvl1, labels_enc_lvl2, labels_enc_lvl3, labels_enc_lvl4, labels_enc_lvl3_4,
                labels_lvl1, labels_lvl2, labels_lvl3, labels_lvl4, labels_lvl3_4,
                survey_id)

    def labels_value_counts(self, label_id):
        """This method computes the number of samples per unique label in a multilabel dataframe"""
        return self.label_value_counts[label_id]


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
                        n_u_classes_lvl1=NUM_UNIQUE_CLASSES['1'], n_u_classes_lvl2=NUM_UNIQUE_CLASSES['2'], n_u_classes_lvl3=NUM_UNIQUE_CLASSES['3'], n_u_classes_lvl4=NUM_UNIQUE_CLASSES['4'], n_u_classes_lvl3_4=NUM_UNIQUE_CLASSES['3_4'],
                        transform=transform)
test_dataset = test_dataset(test_df,
                            IMAGE_DIR,
                            n_u_classes_lvl1=NUM_UNIQUE_CLASSES['1'], n_u_classes_lvl2=NUM_UNIQUE_CLASSES['2'], n_u_classes_lvl3=NUM_UNIQUE_CLASSES['3'], n_u_classes_lvl4=NUM_UNIQUE_CLASSES['4'], n_u_classes_lvl3_4=NUM_UNIQUE_CLASSES['3_4'],
                            transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

extract_features = build_feature_extractor()

# ----------------------------------------------------------------------
# LOOP 1 / 2: TRAIN (feature extraction + training)
# ----------------------------------------------------------------------

print("[INFO] LOOP 1/2: training (feature extraction on the train set)")

feats_batches = []
label_batches = []

for (
    images,
    labels_enc_oh_lvl1, labels_enc_oh_lvl2, labels_enc_oh_lvl3, labels_enc_oh_lvl4, labels_enc_oh_lvl3_4,
    labels_enc_lvl1, labels_enc_lvl2, labels_enc_lvl3, labels_enc_lvl4, labels_enc_lvl3_4,
    labels_lvl1, labels_lvl2, labels_lvl3, labels_lvl4, labels_lvl3_4,
    survey_ids
) in tqdm(train_loader, desc="Train"):
    images = images.to(DEVICE, non_blocking=True)
    with torch.no_grad():
        feats = extract_features(images)
    feats_batches.append(feats.float().cpu().numpy())
    ### === CHANTIER ===
    # Labels are soft-multilabel. For now, try randomly selecting one
    if MULTILABEL_CORRESPONDANCE_STRATEGY == 'mc':
        label_batches.append(np.asarray(labels_enc_lvl3).reshape(-1).astype(np.int64))
    elif MULTILABEL_CORRESPONDANCE_STRATEGY in ['ml', 'soft_ml']:
        label_batches.append(labels_enc_oh_lvl3.numpy())
    ### === CHANTIER ===

X_train = np.concatenate(feats_batches, axis=0)
y_train = np.concatenate(label_batches, axis=0)
if MULTILABEL_CORRESPONDANCE_STRATEGY == 'mc':
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

print(f"[INFO] Train feature matrix shape: {X_train.shape}")

if XGB_USE_SAMPLE_WEIGHTS:
    if MULTILABEL_CORRESPONDANCE_STRATEGY == 'mc':
        class_counts = pd.Series(y_train).value_counts()
        sample_weights = len(y_train) / (class_counts.shape[0] * class_counts.reindex(y_train).to_numpy())
    elif MULTILABEL_CORRESPONDANCE_STRATEGY in ['ml', 'soft_ml']:
        class_counts = pd.Series(y_train.sum(axis=0)).value_counts()
        sample_weights = len(y_train) / (class_counts.shape[0] * class_counts.reindex(y_train.sum(axis=0)).to_numpy())
else:
    sample_weights = None

# Regular multi-class ML models (XGBoost, RandomForest) are trained on the extracted features
# xgboost.set_config(verbosity=2)
# xgb_model = xgboost.XGBClassifier(
#     **CONFIG_XGB['hyperparams'],
# )
rf_model = RandomForestClassifier(
    **CONFIG_RF,
)

# print("[INFO] Training XGBoost (learning_rate=0.1, max_depth=5, sample_weights=True)")
# xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

print("[INFO] Training RandomForest (max_depth=6, class_weight=balanced)")
rf_model.fit(X_train, y_train)


# ----------------------------------------------------------------------
# LOOP 2 / 2: TEST (evaluation on the test set)
# ----------------------------------------------------------------------

def get_all_labels_probas(y_true, predict_proba, classes):
    """
    Parameters
    ----------
    y_true : np.ndarray
        Shape: (batch_size, n_labels)
        Multi-hot ground-truth labels.

    predict_proba : list[np.ndarray]
        Output of RandomForestClassifier.predict_proba().
        Each element has shape:
            (batch_size, n_classes_for_this_label)

    classes : list[np.ndarray]
        model.classes_

    Returns
    -------
    np.ndarray
        Shape: (batch_size, n_labels)

        For each sample/label:
          - if y_true == 1: probability of class 1
          - if y_true == 0: probability of class 0
          - if only one class exists: 0
    """

    batch_size, n_labels = y_true.shape
    result = np.zeros((batch_size, n_labels), dtype=float)

    for label_idx, (proba, label_classes) in enumerate(
        zip(predict_proba, classes)
    ):
        # We need BOTH classes to calculate a meaningful
        # probability for the ground-truth class.
        if len(label_classes) != 2:
            continue

        # Find the columns corresponding to classes 0 and 1
        class_0_idx = np.where(label_classes == 0)[0][0]
        class_1_idx = np.where(label_classes == 1)[0][0]

        # For each label, select P(1) to get a probability matrix.
        # If a given label was not present in the given batch: 0 is put.
        # [Edge case] Theoretically, if a given label only has 1 for each sample of the label, 0 is also put but this particular case cannot happen in our dataset.
        result[:, label_idx] = np.where(
            y_true[:, label_idx] == 1,
            proba[:, class_1_idx],
            proba[:, class_1_idx],
        )

    return result

print("[INFO] LOOP 2/2: test (evaluation on the test set)")

CLASSES = list(range(NUM_CLASSES))
class_names = pd.Series(CLASSES).map(label_to_code_id).map(label_to_code).astype(str).tolist()

true_batches = []
pred_xgb_batches = []
pred_rf_batches = []
probas_rf_batches = []  # Unused for now but useful to see each class binary probabilities. Shape: (list(n_batches), list(n_classes), array(BS, 1 or 2))
survey_id_batches = []
probas_rf_oh_batches = []

for (
    images,
    labels_enc_oh_lvl1, labels_enc_oh_lvl2, labels_enc_oh_lvl3, labels_enc_oh_lvl4, labels_enc_oh_lvl3_4,
    labels_enc_lvl1, labels_enc_lvl2, labels_enc_lvl3, labels_enc_lvl4, labels_enc_lvl3_4,
    labels_lvl1, labels_lvl2, labels_lvl3, labels_lvl4, labels_lvl3_4,
    survey_ids
) in tqdm(test_loader, desc="Test"):
    images = images.to(DEVICE, non_blocking=True)
    with torch.no_grad():
        feats = extract_features(images)
    feats_np = feats.float().cpu().numpy()
    # true_batches.append(np.asarray(labels).reshape(-1).astype(np.int64))
    if MULTILABEL_CORRESPONDANCE_STRATEGY == 'mc':
        true_batches.append(np.asarray(labels_enc_lvl3).reshape(-1).astype(np.int64))
    elif MULTILABEL_CORRESPONDANCE_STRATEGY in ['ml', 'soft_ml']:
        true_batches.append(labels_enc_oh_lvl3.numpy())
    # pred_xgb_batches.append(np.asarray(xgb_model.predict(feats_np)).astype(np.int64))
    pred_rf_batches.append(np.asarray(rf_model.predict(feats_np)).astype(np.int64))
    probas = rf_model.predict_proba(feats_np)
    probas_rf_batches.append(probas)  # list(list(binary probas per batch)) -> (n_batches, n_classes, array(BS, 1 or 2))
    probas_rf_oh_batches.append(get_all_labels_probas(true_batches[-1], probas, rf_model.classes_))
    survey_id_batches.append(np.asarray(survey_ids).reshape(-1))

# y_pred_xgb = np.concatenate(pred_xgb_batches)
y_true = np.concatenate(true_batches)
y_pred_rf = np.concatenate(pred_rf_batches)
y_probas_rf = np.concatenate(probas_rf_oh_batches)
if MULTILABEL_CORRESPONDANCE_STRATEGY == 'mc':
    y_train = le.inverse_transform(y_pred_rf)
survey_ids = np.concatenate(survey_id_batches)

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def top1_soft_multilabels_accuracy(y_true, y_pred):
    top_indices = np.argmax(y_pred, axis=1)
    result = (y_true[np.arange(y_true.shape[0]), top_indices] == 1).astype(int)
    return result.mean()

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

def compute_metrics(y_true, y_pred, y_prob=None, threshold_ml=0.5):
    if MULTILABEL_CORRESPONDANCE_STRATEGY == 'mc':
        accuracy = accuracy_score(y_true, y_pred)
    elif MULTILABEL_CORRESPONDANCE_STRATEGY in ['ml', 'soft_ml']:
        accuracy = accuracy_score(y_true, (y_prob >= threshold_ml).astype(int))
    metrics = {
        "acc_sklearn_ml": accuracy,
        "acc_top_1": top1_soft_multilabels_accuracy(y_true, y_prob),
        "f1_micro": f1_score(y_true, y_pred, average="micro", labels=CLASSES, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", labels=CLASSES, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", labels=CLASSES, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=CLASSES, zero_division=0),
    }
    if MULTILABEL_CORRESPONDANCE_STRATEGY == 'mc':
        cm, cm_norm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    else:
        cm, cm_norm = get_confusion_matrix(y_true, y_pred, NUM_UNIQUE_CLASSES['3'], normalize=True)
    return metrics, cm, cm_norm


def save_results(model_name, model, y_true, y_pred, y_prob, y_pred_str, y_pred_rf_cls, survey_ids):
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    metrics, cm, cm_norm = compute_metrics(y_true, y_pred, y_prob=y_prob)

    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

    pd.DataFrame([metrics]).to_csv(os.path.join(model_dir, "test_metrics.csv"), index=False)

    # cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    # cm_df.index.name = "true_code"
    # cm_df.columns.name = "pred_code"
    # cm_df.to_csv(os.path.join(model_dir, "confusion_matrix.csv"))
    # Save confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    im0 = axes[0].imshow(cm)
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_title("Confusion Matrix (EUNIS lvl-3)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(cm_norm)
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")
    axes[1].set_title("Confusion Matrix (normalized) (EUNIS lvl-3)")
    fig.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(f'{Path(model_dir)}/confusion_matrix_eunis_lvl-3.png')
    plt.close()

    preds_df = pd.DataFrame({
        "survey_id": survey_ids,
        "true_label": test_loader.dataset.df["habitats_code_ID_lvl3"].tolist(),
        "pred_label_str": y_pred_str,
        "pred_label_cls": y_pred_rf_cls,
        "pred_prob": [str(y).replace('\n', '') for y in y_prob]
    })
    preds_df["true_code"] = pd.Series(preds_df["true_label"].tolist()).map(label_to_code_id).map(label_to_code).astype(str)
    preds_df["pred_code"] = pd.Series(preds_df["pred_label"].tolist()).map(label_to_code_id).map(label_to_code).astype(str)
    preds_df.to_csv(os.path.join(model_dir, "test_predictions.csv"), index=False)

    return metrics


# metrics_xgb = save_results("xgboost", xgb_model, y_true, y_pred_xgb, survey_ids)
if MULTILABEL_CORRESPONDANCE_STRATEGY == 'mc':
    y_pred_rf = le.inverse_transform(y_pred_rf)
else:
    y_pred_rf_str = [str(y).replace('\n', '') for y in y_pred_rf]
    y_pred_rf_cls = [';'.join(np.where(y == 1)[0].astype(str).tolist()) for y in y_pred_rf]
metrics_rf = save_results("random_forest", rf_model, y_true, y_pred_rf, y_probas_rf, y_pred_rf_str, y_pred_rf_cls, survey_ids)

summary_df = pd.DataFrame([
#    {"model": "xgboost", **metrics_xgb},
    {"model": "random_forest", **metrics_rf},
])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_summary.csv"), index=False)

print("\n[INFO] Test set metrics")
print(summary_df.to_string(index=False))
print(f"\n[INFO] Results saved in '{os.path.abspath(OUTPUT_DIR)}/'")