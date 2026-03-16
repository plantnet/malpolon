import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb
import timm

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ----------------------------
# Config
# ----------------------------
INFERENCE = True
INFERENCE_SUFFIX = '_1-to-1'
MULTILABEL_CORRESPONDANCE_STRATEGY = 'soft_ml'  # One of ['naive', 'random sampling', 'soft_ml]
LOSS_FUNCTION = 'CE_soft_ml'  # One of ['CE', 'CE_soft_ml', 'KL_divergence']
LABEL_SMOOTHING = 0.0  # Float in [0, 1]

CSV_FILE_TRAIN_FREQ_SPLIT = "metadata_labels_merged_freq_split-10.33%_train.csv"
CSV_FILE_TRAIN_SPATIAL_SPLIT = "metadata_labels_merged_gps_only_S2_train-0.54min.csv"
CSV_FILE_TEST_FREQ_SPLIT = "metadata_labels_merged_freq_split-10.33%_test_1-to-5.csv"
CSV_FILE_TEST_SPATIAL_SPLIT = "metadata_labels_merged_gps_only_S2_test-0.54min.csv"
CSV_S1BIS_TRAIN = "metadata_labels_merged_S1bis-10%_train.csv"
CSV_S1BIS_TEST = f'metadata_labels_merged_S1bis-10%_test{INFERENCE_SUFFIX}.csv'  # "metadata_labels_merged_S1bis-10%_test.csv"

CSV_FILE = CSV_S1BIS_TRAIN
CSV_FILE_TEST = CSV_S1BIS_TEST
IMAGE_DIR = "Images"
OUTPUT_DIR = "baselines/B2_S1bis/"

MODEL = "resnet18"  # One of ['resnet18', 'dinov2_vits14']
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'inference/'), exist_ok=True)

TRAIN_METRICS = os.path.join(OUTPUT_DIR, "train_metrics.csv")
VAL_METRICS = os.path.join(OUTPUT_DIR, "val_metrics.csv")
TEST_METRICS = os.path.join(OUTPUT_DIR, f"inference/test_metrics{INFERENCE_SUFFIX}.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, f"inference/test_predictions{INFERENCE_SUFFIX}.csv")

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_model.pth")

TIME_STAMP_START = time()

writer = wandb.init(
    entity="tlarcher-phd-jrc",
    project='habitats_floraveg',
    name='[INFERENCE] B2_S2bis: ResNet18 (1-to-1)',  #'Unique surveyId spatial split 0.06min, dropout',
    notes="B2: Custom loss & metrics adapted for soft multilabelling.\n"
          "S1bis: Split over unique FLoraveg IDs (no leakeage) Stratified 1-to-k soft multilabels.",
    config={'MULTILABEL_CORRESPONDANCE_STRATEGY': MULTILABEL_CORRESPONDANCE_STRATEGY,
            'LOSS_FUNCTION': LOSS_FUNCTION,
            'LABEL_SMOOTHING': LABEL_SMOOTHING,
            'BATCH_SIZE': BATCH_SIZE,
            'EPOCHS': EPOCHS,
            'LR': LR,
            'NUM_WORKERS': NUM_WORKERS,
            'DEVICE': DEVICE,
            'TRAIN_CSV': CSV_FILE,
            'TEST_CSV': CSV_FILE_TEST,
            'OUTPUT_DIR': OUTPUT_DIR,},
    job_type='inference' if INFERENCE else 'train',
    mode='disabled',  # any of "online", "offline", "disabled"
)
# ----------------------------
# Dataset
# ----------------------------

class TestHabitatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

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

        return image, label_enc, None, label, floraveg_id

class HabitatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
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

        label_enc = row["label"]
        label = row['habitats_code']

        if self.transform:
            image = self.transform(image)

        return image, label_enc, None, label, floraveg_id

class HabitatDatasetRandomlySampleDuplicateLabelsMatching(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.floraveg_ids = dataframe['id_floraveg'].value_counts()
        self.labels = self.df['label']
        self.image_dir = image_dir
        self.transform = transform
        self.n_classes = dataframe['label'].nunique()

    def __len__(self):
        return len(self.floraveg_ids)

    def __getitem__(self, idx):
        floraveg_id = self.floraveg_ids.index[idx]
        df_slice = self.df[self.df['id_floraveg'] == floraveg_id]
        row = df_slice.sample(1).iloc[0].to_dict()

        img_path = os.path.join(
            self.image_dir,
            row["filename_photos"].strip()
        )

        image = Image.open(img_path).convert("RGB")

        label_enc = row["label"]
        label = row['habitats_code']

        if self.transform:
            image = self.transform(image)

        return image, label_enc, None, label, floraveg_id
    
class HabitatDatasetSoftMultilabels(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.df['label'] = self.df['habitats_code_ID'].copy()
        self.floraveg_ids = dataframe['id_floraveg'].value_counts()
        self.labels = self.df['label']
        self.image_dir = image_dir
        self.transform = transform
        self.n_classes = dataframe['label'].max()+1
        self.label_encoding_table = dict(zip(self.df["habitats_code_ID"], self.df["habitats_code"]))

    def __len__(self):
        return len(self.floraveg_ids)

    def __getitem__(self, idx):
        floraveg_id = self.floraveg_ids.index[idx]
        df_slice = self.df[self.df['id_floraveg'] == floraveg_id]
        row = df_slice.sample(1).iloc[0].to_dict()

        img_path = os.path.join(
            self.image_dir,
            row["filename_photos"].strip()
        )

        image = Image.open(img_path).convert("RGB")

        labels_enc = df_slice["label"].values.tolist()
        labels_oh = sample_onehot_encode(labels_enc, self.n_classes)
        labels_enc = ' '.join(str(i) for i in labels_enc)
        labels = ' '.join(str(i) for i in df_slice["habitats_code"].values.tolist())

        if self.transform:
            image = self.transform(image)

        return image, labels_oh, labels_enc, labels, floraveg_id


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

le = LabelEncoder()
le.fit(pd.concat([df, df_test])["habitats_code"])
df["label"] = le.transform(df["habitats_code"])

df_test = df_test[df_test["habitats_code"].isin(le.classes_)].copy()  # Should not change anything
df_test["label"] = le.transform(df_test["habitats_code"])

if MULTILABEL_CORRESPONDANCE_STRATEGY == 'soft_ml':
    NUM_CLASSES = pd.concat([df, df_test])['habitats_code'].nunique()
else:
    NUM_CLASSES = len(le.classes_)
print("[INFO] Total number of classes:", NUM_CLASSES)

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

train_tf = transforms.Compose([
    # transforms.Resize((224,224)),  # Resnet18
    transforms.Resize((518,518)),  # DinoV2
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_tf = transforms.Compose([
    # transforms.Resize((224,224)),  # Resnet18
    transforms.Resize((518,518)),  # DinoV2
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ----------------------------
# Datasets
# ----------------------------
match MULTILABEL_CORRESPONDANCE_STRATEGY:
    case 'naive': 
        dataset = HabitatDataset
        test_dataset = TestHabitatDataset
    case 'random sampling':
        dataset = HabitatDatasetRandomlySampleDuplicateLabelsMatching
        test_dataset = TestHabitatDataset
    case 'soft_ml':
        dataset = HabitatDatasetSoftMultilabels
        test_dataset = HabitatDatasetSoftMultilabels

train_dataset = dataset(train_df, IMAGE_DIR, train_tf)
val_dataset = dataset(val_df, IMAGE_DIR, val_tf)
test_dataset = test_dataset(df_test, IMAGE_DIR, val_tf)

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

print("[INFO] Train size:", len(train_loader))
print("[INFO] Number of classes in the training set:", train_loader.dataset.n_classes)
print("[INFO] Val size:", len(val_loader))
print(f"[INFO] Number of classes in the validation set: {val_loader.dataset.n_classes} ({len(set(val_loader.dataset.labels) & set(train_loader.dataset.labels)) / val_loader.dataset.n_classes * 100:.2f}% overlap with train)")
print("[INFO] Test size:", len(test_loader))

# ----------------------------
# Model
# ----------------------------

match MODEL:
    case 'resnet18':
        print("[INFO] Using ResNet18")
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES,
        )
    case 'dinov2_vits14':
        print("[INFO] Using DINOv2 ViT-S/14")
        model = timm.create_model('timm/vit_small_patch14_dinov2.lvd142m',
                                  pretrained=True,
                                  num_classes=NUM_CLASSES)

model = model.to(DEVICE)

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
    
    
def compute_metrics(y_true, y_pred, y_prob):
    if MULTILABEL_CORRESPONDANCE_STRATEGY == 'soft_ml':
        acc_top1_softml = top1_soft_multilabels_accuracy(y_true, y_prob)
        return [acc_top1_softml]
    else:
        acc = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        try:
            auroc = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro"
            )
        except Exception as e:
            auroc = np.nan
            print(e)

        return [acc, precision, recall, f1, auroc]


# ----------------------------
# Epoch runner
# ----------------------------

def run_epoch(loader, split, training=True):

    if training:
        model.train()
    else:
        model.eval()

    all_labels_enc = []
    all_preds = []
    all_probs = []

    running_loss = 0

    for images, labels_enc, labels_softml_str, labels, floraveg_id in tqdm(loader):

        images = images.to(DEVICE)
        labels_enc = labels_enc.to(DEVICE)

        if training:
            optimizer.zero_grad()

        outputs = model(images)

        # loss = criterion(outputs, labels)
        loss = get_criterion(outputs, labels_enc)

        if training:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        # print(f'Labels: {labels_enc}')
        # print(f'Preds: {preds}')
        # if preds in labels_enc:
        #     preds = labels_enc[0]

        all_labels_enc.extend(labels_enc.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())


    metrics = compute_metrics(
        np.array(all_labels_enc),
        np.array(all_preds),
        np.array(all_probs)
    )

    loss = running_loss / len(loader)
    
    wandb.log({
        f"loss (epoch)/{split}": loss,
    }, step=epoch)
    if MULTILABEL_CORRESPONDANCE_STRATEGY == 'soft_ml':
        wandb.log({
            f"acc_top1_softml (epoch)/{split}": metrics[0],
        }, step=epoch)

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

        train_loss, train_metrics = run_epoch(
            train_loader,
            'train',
            training=True
        )

        val_loss, val_metrics = run_epoch(
            val_loader,
            'val',
            training=False
        )

        print("\nTRAIN")
        print("Loss:",train_loss)
        if MULTILABEL_CORRESPONDANCE_STRATEGY == 'soft_ml':
            print("Acc_top1_soft_multilabels:", train_metrics[0])
        else:
            print("Acc:",train_metrics[0])
            print("Prec:",train_metrics[1])
            print("Recall:",train_metrics[2])
            print("F1:",train_metrics[3])
            print("AUROC:",train_metrics[4])

        print("\nVAL")
        print("Loss:",val_loss)
        if MULTILABEL_CORRESPONDANCE_STRATEGY == 'soft_ml':
            print("Acc_top1_soft_multilabel", val_metrics[0])
            # Save logs
            with open(TRAIN_METRICS,"a") as f:
                f.write(
                    f"{epoch},{train_loss},{train_metrics[0]}\n"
                )
            with open(VAL_METRICS,"a") as f:
                f.write(
                    f"{epoch},{val_loss},{val_metrics[0]}\n"
                )
        else:
            print("Acc:",val_metrics[0])
            print("Prec:",val_metrics[1])
            print("Recall:",val_metrics[2])
            print("F1:",val_metrics[3])
            print("AUROC:",val_metrics[4])
            # Save logs
            with open(TRAIN_METRICS,"a") as f:
                f.write(
                    f"{epoch},{train_loss},{train_metrics[0]},{train_metrics[1]},{train_metrics[2]},{train_metrics[3]},{train_metrics[4]}\n"
                )
            with open(VAL_METRICS,"a") as f:
                f.write(
                    f"{epoch},{val_loss},{val_metrics[0]},{val_metrics[1]},{val_metrics[2]},{val_metrics[3]},{val_metrics[4]}\n"
                )

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
            "epoch": EPOCHS,
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

test_loss, test_metrics = run_epoch(
    test_loader,
    'test',
    training=False
)

print("\nTEST")
print("Loss:",test_loss)
if MULTILABEL_CORRESPONDANCE_STRATEGY == 'soft_ml':
    print("Acc_top1_soft_multilabel", test_metrics[0])
    
    # Save logs
    with open(TEST_METRICS,"w") as f:
        f.write("loss,Acc_top1_soft_multilabel\n")
        f.write(
            f"{test_loss},{test_metrics[0]}\n"
        )
else:
    print("Acc:",test_metrics[0])
    print("Prec:",test_metrics[1])
    print("Recall:",test_metrics[2])
    print("F1:",test_metrics[3])
    print("AUROC:",test_metrics[4])

    # Save logs
    with open(TEST_METRICS,"w") as f:
        f.write("loss,accuracy,precision,recall,f1,auroc\n")
        f.write(
            f"{test_loss},{test_metrics[0]},{test_metrics[1]},{test_metrics[2]},{test_metrics[3]},{test_metrics[4]}\n"
        )

print("\nTest evaluation saved.")
print(f'Elapsed time for test evaluation: {(time() - TIME_STAMP_TEST_EVAL):.2f}s')

wandb.finish()
# ----------------------------
# Export Test Predictions
# ----------------------------

print("\nExporting test predictions...")
TIME_STAMP_INFERENCE = time()

model.eval()

res = {}
all_preds = []
all_probs = []
all_labels_enc = []
all_labels_enc_softml_str = []

with torch.no_grad():
    for images, labels_enc, labels_enc_softml_str, labels, floraveg_ids in tqdm(test_loader):

        images = images.to(DEVICE)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1).values

        # all_preds.extend(preds.cpu().numpy())
        # all_probs.extend(probs.cpu().numpy())
        # all_labels_enc_softml_str.extend(labels_enc_softml_str) if MULTILABEL_CORRESPONDANCE_STRATEGY == 'soft_ml' else all_labels_enc.extend(labels_enc.cpu().numpy())
        res.update({
            fid.item(): {
                'all_preds': preds[i].cpu().to(torch.int32).item(),
                'all_preds_confidence': confidence[i].cpu().item(),
                'all_probs': probs[i].cpu().numpy(),
                'all_labels': labels[i],
                'all_labels_enc': labels_enc[i].cpu().numpy(),
                'all_labels_enc_softml_str': labels_enc_softml_str[i],
            }
            for i, fid in enumerate(floraveg_ids)
        })


# Decode labels
pred_df = df_test.reset_index(drop=True).copy()
if MULTILABEL_CORRESPONDANCE_STRATEGY == 'soft_ml':
    # Length of dataset = n_unique id_floraveg
    pred_df = pred_df.drop_duplicates(subset=['id_floraveg']).reset_index(drop=True)
    dataset_labels_table = {**train_loader.dataset.label_encoding_table, **test_loader.dataset.label_encoding_table}
    for i in range(len(pred_df)):
        floraveg_id = pred_df.loc[i, 'id_floraveg']
        pred_df.loc[i, 'pred_label'] = dataset_labels_table[res[floraveg_id]['all_preds']]
        pred_df.loc[i, 'pred_label_encoded'] = res[floraveg_id]['all_preds']
        pred_df.loc[i, 'pred_confidence'] = res[floraveg_id]['all_preds_confidence']
        pred_df.loc[i, 'habitats_code'] = res[floraveg_id]['all_labels']
        pred_df.loc[i, 'habitats_code_ID'] = res[floraveg_id]['all_labels_enc_softml_str']
        pred_df.loc[i, 'valid_prediction'] = str(int(pred_df.loc[i, 'pred_label_encoded'])) in pred_df.loc[i, 'habitats_code_ID']
    pred_df['pred_label_encoded'] = pred_df['pred_label_encoded'].astype(int)
    pred_df['label'] = pred_df['habitats_code_ID']
else:
    # Length of dataset = n_samples 'augmented' through 1-to-k soft labelling correspondance
    pred_labels = le.inverse_transform(all_preds)
    pred_df["pred_label_encoded"] = all_preds
    pred_df["pred_label"] = pred_labels
    pred_df["pred_confidence"] = np.max(np.array(all_probs), axis=1)  # Dependent on the use of CE

pred_df.to_csv(PREDICTIONS_PATH, index=False)

print("Predictions saved:", PREDICTIONS_PATH)
print(f'Elapsed time for inference export: {(time() - TIME_STAMP_INFERENCE):.2f}s')
print(f'Total Elapsed time: {(time() - TIME_STAMP_START):.2f}s')