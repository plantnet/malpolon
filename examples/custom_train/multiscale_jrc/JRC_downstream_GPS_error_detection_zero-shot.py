"""This version trains a downstream task only over the CBN-Med region of GLC24 PA plots."""
# %%
import os
from collections import OrderedDict
from types import SimpleNamespace
from typing import List, Union, Optional, Callable, Any
from tqdm import tqdm
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_score,
    average_precision_score,
    PrecisionRecallDisplay,
    precision_recall_curve,
    RocCurveDisplay,
    roc_curve,
)
import wandb
import torchvision
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.io import read_image
from torchmetrics.functional import minkowski_distance, mean_squared_error
from torchvision.transforms import CenterCrop, Resize

from malpolon.data.datasets.jrc_multiscale import (
    LandscapeDatasetSimple, load_LUCAS_img,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_model import (
    ModelSimCLR, MultiLabelClassifier, ErrorDetectionClassifier,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_contrastive_losses import (
    KoLeoLoss, MCR
)
from transforms import (transforms_species, transforms_satellite)


# Parameters

SPECIES_INPUT_SIZE = 518
LANDSCAPE_INPUT_SIZE = 518
SATELLITE_INPUT_SIZE = 128

ROOT_PATH_LUCAS = 'dataset/scale_2_landscape/'
OUTPUT_DIR = 'outputs/Downstream_GPS_error_detection_LUCAS/'

DATA_PATHS = {'train': {
                  'landscape_dir': os.path.join(ROOT_PATH_LUCAS, 'LUCAS/'),
                },
              'val': {
                  'landscape_dir': os.path.join(ROOT_PATH_LUCAS, 'LUCAS/'),
                },
              'test': {
                  'landscape_dir': os.path.join(ROOT_PATH_LUCAS, 'LUCAS/'),
                }
             }
METADATA_PATHS = {'train':  os.path.join(ROOT_PATH_LUCAS, "lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_train-0.06min_noisy_100m.csv"),
                  'val':  os.path.join(ROOT_PATH_LUCAS, "lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_val-0.06min_noisy_100m.csv"),
                  'test':  os.path.join(ROOT_PATH_LUCAS, "glc24_pa_test_private_CBN-med_matching-LUCAS-500m_noisy_1000m.csv"),
                 }

## Hyper-params
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
# NUM_WORKERS = 8
# WARMUP_EPOCHS = 5

## Inference mode
SCORE_MODE = "cosine"  # "cosine" or "sigmoid"

args = {
        'arch': 'multi-loss',  # always paired with gps
        'OAR_job_id': os.getenv("OAR_JOB_ID", "no_jobid"),
        'batch_size': 8,
        'ckpt_path': 'wandb/archive/run-20251012_185226-u6tiioze/files/best.pth.tar',
        'resume_wandb_run': False,
        'device': "cuda",
        'disable_cuda': False,
        'dropout': 0.1,
        'epochs': 40,
        'fp16_precision': False,
        'freeze_gps_backbone': True,
        'freeze_modality_backbone': True,
        'gpu_index': 0,
        'learning_rate': 0.01, # 0.00025,
        'log_every_n_steps': 0.05,  # if float, percentage of the epoch (e.g. 0.25 would log 4 times per epoch). If int, number of steps.
        'max_iter': torch.inf,
        'name': "[Downstream: GPS error detection] (from u6tiioze)",
        'out_dim': 2048,
        'subset': None,  # nb of random samples for train & val. Either int or float (percentage of the dataset size).
        'subset_cls': None,  # nb of random samples per class for train & val. Either int or float (percentage of the dataset size).
        'wandb_project': 'GPS_error_detection',
        'weight_decay': 1e-3,
        'workers': os.cpu_count(),
        'warmup_epochs': 0,
        'log_images': True,  # If True, logs images to wandb
        'skip_modalities': ['satellite', 'species'], 
        'downstream_modalities_to_process': ['landscape_img', 'landscape_gps'],  # Will skip modalities during training
        'eval_type': 'linear_probing',  # Evaluation strategy: 'linear_probing', 'fine_tuning', 'knn'
        'num_labels': 11255,
        'loss_criterion': 'BCE',  # Takes values in ['cross_entropy', 'BCE']
        'predict': True,
        'wandb_mode': 'online',  # 'online', 'offline', 'disabled'
        'metrics': {'accuracy_type': 'precision',
                    'accuracy_average': 'micro',
                    'accuracy_topks': (1, 5, 20),
                   },
    }
args = SimpleNamespace(**args) if isinstance(args, dict) else args
writer = wandb.init(
    entity = "tlarcher-phd-jrc",
    id = getattr(args, 'ckpt_path', '').split('/')[-2].split('-')[2] if (getattr(args, 'ckpt_path', None) and getattr(args, 'resume_wandb_run', False)) else None,
    project = getattr(args, 'wandb_project', None),
    name = args.name,
    notes = f"",
    config = args,
    job_type = 'inference' if getattr(args, 'predict', False) else 'train',
    mode = getattr(args, 'wandb_mode', 'offline'),
)

## Dataloaders
class LandscapeGPSErrorDetection(LandscapeDatasetSimple):    
    def load_LUCAS_imgs_error_detection(
        self,
        sample,
        root_path: str = "dataset/scale_2_landscape/",
        return_gps: Optional[bool] = False,
        return_ids: Optional[bool] = False,
        return_fps: Optional[bool] = False,
        return_n_img_per_lid: Optional[bool] = False,
        gps_col: list = ['lon', 'lat'],
        transform: Callable = None,
    ):
        lucas_data = {}
        imgs = []
        for l_id, l_fp in zip(sample['lucas_matching_ids'].split(';'), sample['file_path'].split(';')):
            lucas_data[l_id] = {'fps': l_fp.split()}
            lucas_data[l_id]['n_imgs'] = 0

        # Iterate over every lucas_id
        for l_id, v in lucas_data.items():
            imgs = []
            # Iterate over every 6 views of each lucas_id
            for l_fp in v['fps']:
                try:
                    imgs.append(torchvision.io.read_image(str(Path(root_path) / Path(l_fp))))
                except:
                    print(f'[WARNING]: LUCAS image {l_fp} not found.')
                    continue
            lucas_data[l_id]['imgs'] = imgs
            lucas_data[l_id]['n_imgs'] = len(imgs)
            if sum(len(img) for img in lucas_data[l_id]['imgs']) == 0:
                lucas_data[l_id]['imgs'] = torch.zeros(1, 3, LANDSCAPE_INPUT_SIZE, LANDSCAPE_INPUT_SIZE) -1
            else:
                imgs = [transform(i) for i in lucas_data[l_id]['imgs']]
                lucas_data[l_id]['imgs'] = torch.stack(imgs, dim=0)
        
        imgs = torch.cat([lucas_data[idx]['imgs'] for idx in lucas_data.keys()], dim=0)
        gps = tuple(sample[gps_col].values.flatten())
        fps = sample['file_path']
        ids = sample['lucas_matching_ids']
        n_img_per_ids = [lucas_data[l_id]['n_imgs'] for l_id in lucas_data.keys()]
        res = [imgs]

        # Order: imgs, gps, ids, fps
        if return_gps:
            res.append(gps)
        if return_ids:
            res.append(ids)
        if return_fps:
            res.append(fps)
        if return_n_img_per_lid:
            res.append(n_img_per_ids)
        return tuple(res)

    def __getitem__(self, index) -> Any:
        """Return a sample of the dataset.

        Args:
            index (int): sample index

        Returns:
            tuple: image, coordinates, index, query id
        """
        imgs, gps = self.img, self.coords
        if not self.metadata.empty:
            sample = self.metadata.iloc[index]
            imgs, gps, lucas_ids, fps, n_img_per_ids = self.load_LUCAS_imgs_error_detection(sample, self.root_path, **self.dataset_kwargs,
                                                       return_gps=True,
                                                       return_ids=True,
                                                       return_fps=True,
                                                       return_n_img_per_lid=True,
                                                       transform=self.transform)
            imgs = imgs.to(torch.float32)
            if torch.equal(imgs, torch.zeros(1, 3, LANDSCAPE_INPUT_SIZE, LANDSCAPE_INPUT_SIZE) -1):
                gps = (1000, 1000)
                gps_noisy = (1000, 1000)
                gps_match = True
            else:
                gps = tuple(sample[['lon', 'lat']].values.flatten())
                gps_noisy = tuple(sample[['lon_noisy', 'lat_noisy']].values.flatten())
                gps_match = bool(sample['gps_match'])
            surveyId = int(sample[self.query_id])  # self.query_id inherited from LandscapeDatasetSimple. By default: 'id' and should be equal to surveyId if the CSV file is based off GLC24 PA
            lucas_ids = [int(l_id) for l_id in lucas_ids.split(';')]

        # Order: imgs, gps, gps_noisy, gps_match (binary label), plot ID, LUCAS IDs
        return imgs, torch.Tensor(gps), torch.Tensor(gps_noisy), torch.tensor(gps_match),  torch.tensor([surveyId]), np.array(lucas_ids), np.array(n_img_per_ids)


def transforms_landscape():
    def CenterCropToMaxDim(img):
        max_dim = max(img.shape[-2:])
        return CenterCrop((max_dim, max_dim))(img)

    ts = [lambda x: CenterCropToMaxDim(x),
          Resize((SPECIES_INPUT_SIZE, SPECIES_INPUT_SIZE))]  # bilinear by default

    return transforms.Compose(ts)

def collate_landscape(original_batch):
    imgs, gpss, gpss_noisy, gps_match, s_ids, l_ids, n_img_per_ids = zip(*original_batch)

    # Stackable quantities
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    gpss_noisy_batched = torch.stack(list(gpss_noisy), dim=0)
    gps_match_batched = torch.stack(list(gps_match), dim=0)
    s_ids_batched = torch.stack(list(s_ids), dim=0)
    
    # Quantities with variable amounts (impossible to stack)
    l_ids_batched = list(l_ids)
    n_img_per_ids_batched = list(n_img_per_ids)
    return img_batched, gps_batched, gpss_noisy_batched, gps_match_batched, s_ids_batched, l_ids_batched, n_img_per_ids_batched

custom_collate = collate_landscape
dataset_train = LandscapeGPSErrorDetection(
    root_path = DATA_PATHS['train']['landscape_dir'],
    fp_metadata = METADATA_PATHS['train'],
    transform = transforms_landscape(),
    subset = args.subset,
    cls_id=None,
)
dataset_val = LandscapeGPSErrorDetection(
    root_path = DATA_PATHS['val']['landscape_dir'],
    fp_metadata = METADATA_PATHS['val'],
    transform = transforms_landscape(),
    subset = args.subset,
    cls_id=None,
)
dataset_test = LandscapeGPSErrorDetection(
    root_path = DATA_PATHS['test']['landscape_dir'],
    fp_metadata = METADATA_PATHS['test'],
    transform = transforms_landscape(),
    subset = args.subset,
    cls_id=None,
)

train_loader = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
            collate_fn=custom_collate,
            sampler=None)
val_loader = DataLoader(
            dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True,
            collate_fn=custom_collate,
            sampler=None)
test_loader = DataLoader(
            dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
            collate_fn=custom_collate,
            sampler=None)

## Models
# model_species = ModelSimCLR(base_model='species', out_dim=args.out_dim, dropout=args.dropout,
#                             freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone)
model_landscape = ModelSimCLR(base_model='landscape', out_dim=args.out_dim, dropout=args.dropout,
                                freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone)
# model_satellite = ModelSimCLR(base_model='satellite', out_dim=args.out_dim, dropout=args.dropout,
#                                 gps_encoder=model_species.gps_encoder, gps_head=model_species.gps_contrastive_head,
#                                 freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone)
# model = torch.nn.ModuleList([model_species, model_landscape, model_satellite])
model = model_landscape.to(args.device)  # Must happen before instanciating he optimizer in case of loading a checkpoint
model = torch.nn.DataParallel(model, device_ids=[0])

# Transfer learning: linear probing / fine-tuning
def filter_state_dict_keys(state_dict, prefix="landscape"):
    return {
        k.split(prefix)[1]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }

def filter_state_dict_keys_data_parallel(state_dict, prefix="landscape"):
    return {
        f'module.{k.split(prefix)[1]}': v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }

if args.ckpt_path:
    checkpoint = torch.load(args.ckpt_path, map_location='cuda' if not args.disable_cuda else 'cpu')
    landscape_sd = filter_state_dict_keys_data_parallel(checkpoint['state_dict'], prefix='landscape.')
    model.load_state_dict(landscape_sd)
    print(f"Checkpoint loaded from {args.ckpt_path}")

optimizer = torch.optim.AdamW(model.module.parameters() if isinstance(model, torch.nn.parallel.DataParallel) else model.parameters(),
                                lr=args.learning_rate, weight_decay=args.weight_decay)
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.05,  # Starts from 10 * lr
    end_factor=1.0,     # Ends at 1.0 * lr = 1e-3
    total_iters=args.warmup_epochs,
)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")


## Pipeline

def inspect_first_batch(images, labels, epoch, phase, save_dir="first_batch_images", max_images=3):
    """
    images: Tensor [B, C, H, W]
    labels: Tensor [B, num_classes]
    """
    print(f"\n[{phase.upper()}] Epoch {epoch} — First batch inspection")

    # Image stats
    print(
        f"Images → min: {images.min().item():.4f}, "
        f"max: {images.max().item():.4f}, "
        f"mean: {images.mean().item():.4f}"
    )

    # Label stats
    print(
        f"Labels → min: {labels.min().item()}, "
        f"max: {labels.max().item()}, "
        f"Nb positives per batch: {labels.sum(dim=1)}"
    )

    # Show first few images
    os.makedirs(save_dir, exist_ok=True)
    images = images.detach().cpu()
    for i in range(min(max_images, images.size(0))):
        img = images[i]
        ## Min-Max normalization per channel
        amins = img.amin(dim=(1, 2))
        amaxs = img.amax(dim=(1, 2))
        img_norm = (img - amins[:, None, None]) / (amaxs - amins + 1e-6)[:, None, None]  # Simple min-max normalization per channel
        
        # If image is normalized, optionally denormalize here
        img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
        img_norm_np = img_norm.permute(1, 2, 0).numpy()
        data = {f"{phase}_epoch{epoch}_img{i}_rgb_raw.png":
                    {'img': img_np[:,:,:3],
                     'args': {'vmin': 0, 'vmax': 255, 'cmap': None}},
                f"{phase}_epoch{epoch}_img{i}_rgb_normalized.png":
                    {'img': img_norm_np[:,:,:3],
                     'args': {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}},
                f"{phase}_epoch{epoch}_img{i}_nir_raw.png":
                    {'img': img_np[:,:,3],
                     'args': {'vmin': 0, 'vmax': 255, 'cmap': None}},
                f"{phase}_epoch{epoch}_img{i}_nir_normalized.png":
                    {'img': img_norm_np[:,:,3],
                     'args': {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}}}
        for fp, imgargs in data.items():
            file_path = os.path.join(save_dir, fp)
            plt.imsave(file_path, imgargs['img'], **imgargs['args'])

def recall_at_k(y_true, y_scores, k):
    """
    y_true: np.ndarray [N, C] (0/1)
    y_scores: np.ndarray [N, C] (probabilities)
    """
    recalls = []

    for yt, ys in zip(y_true, y_scores):
        true_idx = np.where(yt == 1)[0]
        if len(true_idx) == 0:
            continue

        topk_idx = np.argsort(ys)[-k:]
        hits = len(set(true_idx) & set(topk_idx))
        recalls.append(hits / len(true_idx))

    return float(np.mean(recalls)) if recalls else 0.0

def get_regularizer(features, regularizer_name: str,
                    koleo_eps: float = 1e-6,
                    mcr_eps: float = 1e-6):
    """Retrieves the right regularizer.

    Possible values of regularizer_name: 'koleo', 'mcr'.
    """
    if regularizer_name == 'koleo':
        regularizer = KoLeoLoss(eps=koleo_eps)
        reg_term = regularizer(features, eps=koleo_eps)
    elif regularizer_name == 'mcr':
        regularizer = MCR(eps=mcr_eps)
        reg_term = regularizer(features)
    else:
        raise NotImplementedError(f"Regularizer {regularizer_name} not implemented.")
    return reg_term
    
def forward_accumulate(model: torch.nn.Module,
                       criterion: torch.nn.Module,
                       input: torch.tensor,
                       input_type: str,
                       loss: torch.tensor,
                       labels: torch.tensor,
                       scaler: GradScaler = None,
                       regularizer: str = '') -> tuple:
    # Calling backward multiple times is the same as accumulating the gradients (summing the loss) and calling backward once after.
    if isinstance(model, torch.nn.DataParallel):
        logits = model.module(input, input_type)
    else:
        logits = model(input, input_type)
    loss += criterion(logits, labels)
    if regularizer:
        loss += get_regularizer(logits, regularizer)
    return logits, loss

# TO ADAPT following dataloader modifications
def train_validate(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    num_classes: int,
    output_dir: str,
    f1_threshold: float = 0.3,
):
    os.makedirs(output_dir, exist_ok=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")

    metrics_path = os.path.join(output_dir, "metrics.csv")

    # Initialize CSV
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "phase", "loss",
            "f1_micro", "auc", "precision",
            "recall_100", "recall_20"
        ])

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            is_train = phase == "train"
            model.train() if is_train else model.eval()

            all_labels = []
            all_probs = []
            running_loss = 0.0

            for step, data in enumerate(tqdm(loader)):
                images, gps, gps_noisy, gps_match, index, query_id = data
                labels = gps_match

                images = images.to(device)
                labels = labels.to(device).float()
                gps_noisy = gps_noisy.to(device)

                # TO ADAPT
                # if step == 0:
                #     inspect_first_batch(images, labels, epoch, phase)

                with torch.set_grad_enabled(is_train):
                    data = {
                        "landscape_img": images,
                        "landscape_gps": gps
                    }

                    loss = 0.0
                    if isinstance(model, torch.nn.DataParallel):
                        logits_gps, logits_img = model.module(images, gps_noisy)
                    else:
                        logits_gps, logits_img = model(images, gps_noisy)
                    # Pass logits to model which will compare the distance between each modality's features to determine if they match or not
                    if isinstance(model, torch.nn.DataParallel):
                        logits = model.module(logits_gps, logits_img)
                    else:
                        logits = model(logits_gps, logits_img)
                    loss = criterion(logits.flatten(), labels)
                    # loss += get_regularizer(logits, regularizer)

                    if is_train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                probs = torch.sigmoid(logits)

                running_loss += loss.item() * images.size(0)
                all_labels.append(labels.detach().cpu())
                all_probs.append(probs.detach().cpu())

            # Stack results
            y_true = torch.cat(all_labels).numpy()
            y_prob = torch.cat(all_probs).numpy()
            y_pred = (y_prob >= f1_threshold).astype(int)

            # Metrics
            f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
            precision = precision_score(y_true, y_pred, average="micro", zero_division=0)

            try:
                auc = roc_auc_score(y_true, y_prob, average="micro")
            except ValueError:
                auc = float("nan")

            recall_100 = recall_at_k(y_true, y_prob, k=min(100, num_classes))
            recall_20 = recall_at_k(y_true, y_prob, k=min(20, num_classes))

            epoch_loss = running_loss / len(loader.dataset)

            print(
                f"[{phase.upper()}] Epoch {epoch} | "
                f"Loss: {epoch_loss:.4f} | "
                f"F1-micro: {f1_micro:.4f} | "
                f"AUC: {auc:.4f} | "
                f"Precision: {precision:.4f} | "
                f"Recall@100: {recall_100:.4f} | "
                f"Recall@20: {recall_20:.4f}"
            )

            # Append metrics to CSV
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, phase, epoch_loss,
                    f1_micro, auc, precision,
                    recall_100, recall_20
                ])

            # Save best model (validation only)
            if phase == "val" and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    os.path.join(output_dir, "best.pt")
                )

        # Save last epoch checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(output_dir, "last.pt")
        )


def run_inference(
    model,
    checkpoint_path,
    test_loader,
    device,
    output_dir: str = 'outputs/inference/',
    score_mode: str = "cosine",  # "cosine" or "sigmoid"
):
    assert score_mode in ["cosine", "sigmoid"], "score_mode must be 'cosine' or 'sigmoid'"

    os.makedirs(output_dir, exist_ok=True)

    model.to(device)
    model.eval()

    all_labels = []
    all_scores = []
    all_ids = []

    with torch.no_grad():
        for step, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, gps, gps_noisy, gps_match, surveyIds, lucas_ids, n_img_per_ids = data

            labels = gps_match.float().to(device)
            images = images.to(device)
            gps_noisy = gps_noisy.to(device)

            # Forward pass
            """
            Since there are 1 to k LUCAS_ID per geolocalized sample;
            and there are 1 to 6 image per LUCAS_ID,
            we compute the mean logit values for all LUCAS_IDs tied to a geo-tagged point
            """
            logits_img, logits_gps, start, stop = [], [], 0, 0
            for sample_idx in range(test_loader.batch_size):
                stop = start + n_img_per_ids[sample_idx].sum()  # Number of loaded images per LUCAS_ID, per surveyId (i.e. plot, i.e. sample)
                stop = stop + 1 if start == stop else stop  # If no images found for the current LUCAS_ID
                images_sample = images[start:stop]
                gps_sample = gps_noisy[sample_idx].repeat(images_sample.shape[0], 1)
                if isinstance(model, torch.nn.DataParallel):
                    logit_gps, logit_img = model.module(images_sample, gps_sample)
                else:
                    logit_gps, logit_img = model.module(images_sample, gps_sample)
                logits_img.append(logit_img.mean(dim=0))
                logits_gps.append(logit_gps[0])  # The same GPS values are passed in forward (se torch.repeat()) so all logits are equal. No need to call mean()
                start = stop
            logits_img = torch.stack(logits_img, dim=0)
            logits_gps = torch.stack(logits_gps, dim=0)

            # Score computation options
            if score_mode == "cosine":
                emb_gps = F.normalize(logits_gps, dim=-1)
                emb_img = F.normalize(logits_img, dim=-1)
                # scores = torch.sum(emb_gps * emb_img, dim=-1)  # Manual
                scores = F.cosine_similarity(emb_gps, emb_img, dim=1)  # Using torchmetrics implementation (handles edge cases)

            elif score_mode == "sigmoid":
                scores = torch.sigmoid((logits_gps - logits_img).sum(dim=-1))  # logits are from similar feature spaces

            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())
            all_ids.extend(surveyIds.cpu().ravel().tolist())

    # Stack
    y_true = torch.cat(all_labels).numpy()
    y_scores = torch.cat(all_scores).numpy()

    # Metrics
    ## AUCs
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    print(f"[{score_mode}] ROC-AUC: {roc_auc:.4f}")
    print(f"[{score_mode}] PR-AUC: {pr_auc:.4f}")
    
    ## Curves
    prec, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, estimator_name='GPS-Image Cosine Similarity Score')
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name='GPS-Image Cosine Similarity Score')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    plt.suptitle('Zero-shot inference of GPS-degraded GLC24 PA, from contrastive multi-scale pretraining.', fontsize=18)
    ax1.set_title('ROC curve')
    ax2.set_title('Precision-Recall curve')
    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"curves_{score_mode}.png"))
    plt.close()
    
    ## Wandb
    y_true_wb = y_true
    y_scores_wb = y_scores

    ### --- Scalars ---
    data = [
        ["roc_auc", roc_auc],
        ["pr_auc", pr_auc],
    ]
    table = wandb.Table(data=data, columns=["metric", "value"])
    wandb.log({
        "AUC_bar_plot": wandb.plot.bar(
            table,
            "metric",   # x-axis
            "value",    # y-axis
            title="AUC Metrics"
        )
    })

    ### --- Loging Matplotlib curves ---
    ### Note: wandb's built-in pr-recall and roc curves plotting functions do not handle binary classification with scores of shape (N,)
    wandb.log({
        f"{score_mode}/curves_plot": wandb.Image(
            os.path.join(output_dir, f"curves_{score_mode}.png")
        )
    })

    # Save per-sample results
    results_df = pd.DataFrame({
        "id": all_ids,
        "label": y_true,
        "score": y_scores
    })

    results_path = os.path.join(output_dir, f"scores_{score_mode}.csv")
    results_df.to_csv(results_path, index=False)

    # Save metrics
    metrics_df = pd.DataFrame([{
        "score_mode": score_mode,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }])

    metrics_path = os.path.join(output_dir, f"metrics_{score_mode}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved results to {output_dir}")


# Train / Infer !

if not args.predict:
    train_validate(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        args.epochs,
        args.num_labels,
        output_dir='outputs/Downstream satellite img+gps GLC24_CBN-Med/',
        f1_threshold=0.3,
    )
else:
    run_inference(
        model,
        args.ckpt_path,
        test_loader,
        device=device,
        output_dir = OUTPUT_DIR,
        score_mode=SCORE_MODE,
    )
    
