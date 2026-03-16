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

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split, StratifiedKFold
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
INFERENCE = False
MULTILABEL_CORRESPONDANCE_STRATEGY = 'naive'  # One of ['naive', 'random sampling']

CSV_FILE_TRAIN_FREQ_SPLIT = "metadata_labels_merged_freq_split-10.33%_train.csv"
CSV_FILE_TRAIN_SPATIAL_SPLIT = "metadata_labels_merged_gps_only_train-0.54min.csv"
CSV_FILE_TEST_FREQ_SPLIT = "metadata_labels_merged_freq_split-10.33%_test.csv"
CSV_FILE_TEST_SPATIAL_SPLIT = "metadata_labels_merged_gps_only_test-0.54min.csv"

CSV_FILE = CSV_FILE_TRAIN_FREQ_SPLIT
CSV_FILE_TEST = CSV_FILE_TEST_FREQ_SPLIT
IMAGE_DIR = "Images"
OUTPUT_DIR = "baselines/B0/"

N_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_METRICS = os.path.join(OUTPUT_DIR, "train_metrics.csv")
VAL_METRICS = os.path.join(OUTPUT_DIR, "val_metrics.csv")
TEST_METRICS = os.path.join(OUTPUT_DIR, "inference/test_metrics.csv")

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_model.pth")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "inference/test_predictions.csv")

TIME_STAMP_START = time()
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

        img_path = os.path.join(
            self.image_dir,
            row["filename_photos"].strip()
        )

        image = Image.open(img_path).convert("RGB")

        # label = self.df[self.df['id_floraveg'] == row["id_floraveg"]]['label'].values.tolist()
        # print(f'Dataset Label: {label}')
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label# ' '.join(str(l) for l in label)
    
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

        img_path = os.path.join(
            self.image_dir,
            row["filename_photos"].strip()
        )

        image = Image.open(img_path).convert("RGB")

        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

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

        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------
# Load CSV
# ----------------------------

df = pd.read_csv(CSV_FILE)
df_test = pd.read_csv(CSV_FILE_TEST)

le = LabelEncoder()
le.fit(pd.concat([df, df_test])["habitats_code"])
df["label"] = le.transform(df["habitats_code"])

df_test = df_test[df_test["habitats_code"].isin(le.classes_)].copy()  # Should not change anything
df_test["label"] = le.transform(df_test["habitats_code"])

num_classes = len(le.classes_)
print("[INFO] Total number of classes:", num_classes)

habitats_counts = df['label'].value_counts()
habitats_single_occurrence = habitats_counts[habitats_counts == 1]
if len(habitats_single_occurrence) > 1:
    df_habitats_single_occurrence = df[df['label'].isin(habitats_single_occurrence.index)]
    df = df[~df['label'].isin(habitats_single_occurrence.index)]
    print(f'[WARNING]: The following habitat codes were excluded from validation and kept in train because they only have one occurrence: {habitats_single_occurrence.index.tolist()}')

# ----------------------------
# Transforms
# ----------------------------

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
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
    case 'random sampling':
        dataset = HabitatDatasetRandomlySampleDuplicateLabelsMatching

train_dataset = dataset(train_df, IMAGE_DIR, train_tf)
val_dataset = dataset(val_df, IMAGE_DIR, val_tf)
test_dataset = TestHabitatDataset(df_test, IMAGE_DIR, val_tf)

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

model = models.resnet18(weights="IMAGENET1K_V1")

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(y_true, y_pred, y_prob):

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

    return acc, precision, recall, f1, auroc


# ----------------------------
# Epoch runner
# ----------------------------

def run_epoch(loader, training=True):

    if training:
        model.train()
    else:
        model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    running_loss = 0

    for images, labels in tqdm(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        if training:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        # print(f'Labels: {labels}')
        # print(f'Preds: {preds}')
        # if preds in labels:
        #     preds = labels[0]

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

    loss = running_loss / len(loader)

    return loss, metrics


# ----------------------------
# Init logs
# ----------------------------

with open(TRAIN_METRICS,"w") as f:
    f.write("epoch,loss,accuracy,precision,recall,f1,auroc\n")

with open(VAL_METRICS,"w") as f:
    f.write("epoch,loss,accuracy,precision,recall,f1,auroc\n")


# ----------------------------
# Training
# ----------------------------
if not INFERENCE:
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        print("\nEpoch",epoch)

        train_loss, train_metrics = run_epoch(
            train_loader,
            training=True
        )

        val_loss, val_metrics = run_epoch(
            val_loader,
            training=False
        )

        print("\nTRAIN")
        print("Loss:",train_loss)
        print("Acc:",train_metrics[0])
        print("Prec:",train_metrics[1])
        print("Recall:",train_metrics[2])
        print("F1:",train_metrics[3])
        print("AUROC:",train_metrics[4])

        print("\nVAL")
        print("Loss:",val_loss)
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
    training=False
)

print("\nTEST")
print("Loss:",test_loss)
print("Acc:",test_metrics[0])
print("Prec:",test_metrics[1])
print("Recall:",test_metrics[2])
print("F1:",test_metrics[3])
print("AUROC:",test_metrics[4])

with open(TEST_METRICS,"w") as f:
    f.write("loss,accuracy,precision,recall,f1,auroc\n")
    f.write(
        f"{test_loss},{test_metrics[0]},{test_metrics[1]},{test_metrics[2]},{test_metrics[3]},{test_metrics[4]}\n"
    )

print("\nTest evaluation saved.")
print(f'Elapsed time for test evaluation: {(time() - TIME_STAMP_TEST_EVAL):.2f}s')

# ----------------------------
# Export Test Predictions
# ----------------------------

print("\nExporting test predictions...")
TIME_STAMP_INFERENCE = time()

model.eval()

all_preds = []
all_probs = []

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Decode labels
pred_labels = le.inverse_transform(all_preds)

# Build dataframe
pred_df = df_test.reset_index(drop=True).copy()

pred_df["pred_label_encoded"] = all_preds
pred_df["pred_label"] = pred_labels

# Optional: max probability
pred_df["pred_confidence"] = np.max(np.array(all_probs), axis=1)

pred_df.to_csv(PREDICTIONS_PATH, index=False)

print("Predictions saved:", PREDICTIONS_PATH)
print(f'Elapsed time for inference export: {(time() - TIME_STAMP_INFERENCE):.2f}s')
print(f'Total Elapsed time: {(time() - TIME_STAMP_START):.2f}s')