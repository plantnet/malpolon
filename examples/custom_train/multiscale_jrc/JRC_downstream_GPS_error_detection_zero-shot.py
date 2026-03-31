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

ROOT_PATH_LUCAS = 'dataset/scale_2_landscape'
OUTPUT_DIR = 'outputs/Downstream_GPS_error_detection_LUCAS_noise_mixture/'

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
METADATA_PATHS = {
    # 'train':  os.path.join(ROOT_PATH_LUCAS, "gps_noisy/lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_train-0.06min_noise_mixture.csv"),
    # 'val':  os.path.join(ROOT_PATH_LUCAS, "gps_noisy/lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_val-0.06min_noise_mixture.csv"),
    'test':  os.path.join(ROOT_PATH_LUCAS, "gps_noisy/glc24_pa_test_private_CBN-med_matching-LUCAS-500m_noise_mixture.csv"),
                 }
SEEDS = [2]# [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

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
        'freeze_gps_backbone': True,
        'freeze_modality_backbone': True,
        'name': "[Downstream: GPS error detection] (from u6tiioze)",
        'out_dim': 2048,
        'subset': None,  # nb of random samples for train & val. Either int or float (percentage of the dataset size).
        'subset_cls': None,  # nb of random samples per class for train & val. Either int or float (percentage of the dataset size).
        'wandb_project': 'GPS_error_detection',
        'workers': os.cpu_count(),
        'log_images': True,  # If True, logs images to wandb
        'skip_modalities': ['satellite', 'species'], 
        'downstream_modalities_to_process': ['landscape_img', 'landscape_gps'],  # Will skip modalities during training
        'eval_type': 'linear_probing',  # Evaluation strategy: 'linear_probing', 'fine_tuning', 'knn'
        'predict': True,
        'verbose': False,
        'wandb_mode': 'online',  # 'online', 'offline', 'disabled'
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
    def __init__(
        self,
        root_path: str = None,
        fp_metadata: str = None,
        transform: Callable = None,
        dataset_kwargs: dict = {},
        query_id: str = 'id',
        verbose:bool = False,
        **kwargs
    ) -> None:
        super().__init__(root_path, fp_metadata, transform, dataset_kwargs, query_id=query_id, **kwargs)
        self.verbose = verbose
   
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
        missing_imgs = []
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
                    if self.verbose:
                        print(f'[WARNING]: LUCAS image {l_fp} not found.')
                    missing_imgs.append(l_fp)
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
        res.append(missing_imgs)
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
            imgs, gps, lucas_ids, fps, n_img_per_ids, missing_imgs = self.load_LUCAS_imgs_error_detection(
                sample,
                self.root_path,
                **self.dataset_kwargs,
                return_gps=True,
                return_ids=True,
                return_fps=True,
                return_n_img_per_lid=True,
                transform=self.transform
            )
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
        return imgs, torch.Tensor(gps), torch.Tensor(gps_noisy), torch.tensor(gps_match),  torch.tensor([surveyId]), np.array(lucas_ids), np.array(n_img_per_ids), np.array(missing_imgs)


def transforms_landscape():
    def CenterCropToMaxDim(img):
        max_dim = max(img.shape[-2:])
        return CenterCrop((max_dim, max_dim))(img)

    ts = [lambda x: CenterCropToMaxDim(x),
          Resize((SPECIES_INPUT_SIZE, SPECIES_INPUT_SIZE))]  # bilinear by default

    return transforms.Compose(ts)

def collate_landscape(original_batch):
    imgs, gpss, gpss_noisy, gps_match, s_ids, l_ids, n_img_per_ids, missing_imgs = zip(*original_batch)

    # Stackable quantities
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    gpss_noisy_batched = torch.stack(list(gpss_noisy), dim=0)
    gps_match_batched = torch.stack(list(gps_match), dim=0)
    s_ids_batched = torch.stack(list(s_ids), dim=0)
    
    # Quantities with variable amounts (impossible to stack)
    l_ids_batched = list(l_ids)
    n_img_per_ids_batched = list(n_img_per_ids)

    # No stacking needed
    missing_imgs_batched = np.concatenate(list(missing_imgs), axis=0)
    return img_batched, gps_batched, gpss_noisy_batched, gps_match_batched, s_ids_batched, l_ids_batched, n_img_per_ids_batched, missing_imgs_batched


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

from pathlib import Path
from collections import Counter

def remap_keys(d, key_map):
    """
    d: dict original
    key_map: dict {old_key: new_key}
    """
    return {
        key_map.get(k, k): v
        for k, v in d.items()
    }

def count_unique_last_letters(file_paths, ignore_case=False):
    """
    Count unique last letters of file names (without extension).

    Parameters
    ----------
    file_paths : list of str or Path
    ignore_case : bool
        If True, normalize letters to lowercase

    Returns
    -------
    dict : {letter: count}
    """
    letter_to_category = {
        'C': 'Cover',
        'N': 'North',
        'S': 'South',
        'E': 'East',
        'W': 'West',
        'P': 'Point',
    }
    last_letters = []

    for fp in file_paths:
        name = Path(fp).stem  # filename without extension
        if len(name) == 0:
            continue

        last_char = name[-1]
    
        # Option: keep only alphabetic characters
        if not last_char.isalpha():
            continue

        if ignore_case:
            last_char = last_char.lower()

        last_letters.append(last_char)
    out_dict = remap_keys(dict(Counter(last_letters)), letter_to_category)
    out_dict['Total_missing_imgs'] = sum(out_dict.values())
    return out_dict

def log_results_and_metrics(y_true, y_scores, all_ids, output_dir, score_mode, seed_prefix):
    ## AUCs
    print('y_scores: ', y_scores)
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
    plt.suptitle(f"Zero-shot inference of GPS-degraded GLC24 PA, from contrastive multi-scale pretraining{seed_prefix}.", fontsize=18)
    ax1.set_title('ROC curve')
    ax2.set_title('Precision-Recall curve')
    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"curves_{score_mode}{seed_prefix}.png"))
    plt.close()

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
            os.path.join(output_dir, f"curves_{score_mode}{seed_prefix}.png")
        )
    })

    # Save per-sample results
    results_df = pd.DataFrame({
        "id": all_ids,
        "label": y_true,
        "score": y_scores
    })

    results_path = os.path.join(output_dir, f"scores_{score_mode}{seed_prefix}.csv")
    results_df.to_csv(results_path, index=False)

    # Save metrics
    metrics_df = pd.DataFrame([{
        "score_mode": score_mode,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }])

    metrics_path = os.path.join(output_dir, f"metrics_{score_mode}{seed_prefix}.csv")
    metrics_df.to_csv(metrics_path, index=False)

def run_inference(
    model,
    checkpoint_path,
    test_loader,
    device,
    output_dir: str = 'outputs/inference/',
    score_mode: str = "cosine",  # "cosine" or "sigmoid"
    seed: Optional[Union[int, str]] = '',
):
    assert score_mode in ["cosine", "sigmoid"], "score_mode must be 'cosine' or 'sigmoid'"
    seed_prefix = f"{'_seed'+str(seed) if seed else ''}"
    
    if seed:
        print(f"\n=== Running inference for seed {seed} ===")

    os.makedirs(output_dir, exist_ok=True)

    model.to(device)
    model.eval()

    all_labels = []
    all_scores = []
    all_ids = []

    with torch.no_grad():
        for step, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            images, gps, gps_noisy, gps_match, surveyIds, lucas_ids, n_img_per_ids, missing_imgs = data
            uc_missing_imgs = count_unique_last_letters(missing_imgs)
            if len(missing_imgs) > 0:
                print(f"[Batch {step}] Missing LUCAS images: {uc_missing_imgs}")

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
    log_results_and_metrics(y_true, y_scores, all_ids, output_dir, score_mode, seed_prefix)

    print(f"Saved results to {output_dir}")


# Infer !
for seed in tqdm(SEEDS, 'Test set seeds'):
    # Load data for each seed
    dataset_test = LandscapeGPSErrorDetection(
        root_path = DATA_PATHS['test']['landscape_dir'],
        fp_metadata = f"{METADATA_PATHS['test'].split('.csv')[0]}_seed{seed}.csv",
        transform = transforms_landscape(),
        subset = args.subset,
        cls_id=None,
        verbose=args.verbose,
    )

    test_loader = DataLoader(
                dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
                collate_fn=collate_landscape,
                sampler=None)
    
    run_inference(
        model,
        args.ckpt_path,
        test_loader,
        device=args.device,
        output_dir = OUTPUT_DIR,
        score_mode=SCORE_MODE,
        seed=seed,
    )
    


# dataset_train = LandscapeGPSErrorDetection(
#     root_path = DATA_PATHS['train']['landscape_dir'],
#     fp_metadata = f"{METADATA_PATHS['train'].split('.csv')[0]}_seed{seed}.csv",
#     transform = transforms_landscape(),
#     subset = args.subset,
#     cls_id=None,
# )
# dataset_val = LandscapeGPSErrorDetection(
#     root_path = DATA_PATHS['val']['landscape_dir'],
#     fp_metadata = f"{METADATA_PATHS['val'].split('.csv')[0]}_seed{seed}.csv",
#     transform = transforms_landscape(),
#     subset = args.subset,
#     cls_id=None,
# )

# train_loader = DataLoader(
#             dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
#             collate_fn=collate_landscape,
#             sampler=None)
# val_loader = DataLoader(
#             dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True,
#             collate_fn=collate_landscape,
#             sampler=None)