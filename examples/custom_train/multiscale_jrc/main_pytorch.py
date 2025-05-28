"""Main script to run training or inference on JRC multicale datasets.

Author: Theo Larcher <theo.larcher@inria.fr>
"""
from types import SimpleNamespace
from typing import Any, List
from math import sqrt

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import CenterCrop, Resize

from malpolon.data.datasets.jrc_multiscale import (
    LandscapeDatasetSimple,
    SatelliteDatasetSimple,
    SpeciesDatasetSimple,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_contrastive_system import (
    SimCLR,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_model import (
    ModelSimCLR,
)
from transforms import (MinMaxNormalize, QuantileNormalizeFromPreComputedDatasetPercentiles)

# To address inconsistent image sizes, two options:
# 1. Define transforms to resize images to a fixed size
def transforms_species():
    def CenterCropToMaxDim(img):
        max_dim = max(img.shape[-2:])
        return CenterCrop((max_dim, max_dim))(img)

    ts = [lambda x: CenterCropToMaxDim(x),
          Resize((518, 518))]  # bilinear by default

    return transforms.Compose(ts)

def transforms_satellite():
    def CenterCropToMaxDim(img):
        max_dim = max(img.shape[-2:])
        return CenterCrop((max_dim, max_dim))(img)

    ts = [
        # QuantileNormalizeFromPreComputedDatasetPercentiles(),
        # MinMaxNormalize(),
        # torch.Tensor,
        # transforms.Normalize(mean=(0.5,) * 4, std=(0.5,) * 4)
    ]

    return transforms.Compose(ts)

# 2. Custom collate function returning directly a list of dictionaries with {'img': img_tensor, 'gps': gps_tuple}. But this implies adding a loop over the multi-dimensional tensors which defeats the purpose of batching.
def collate_species(original_batch):
    imgs, gpss = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    return img_batched, gps_batched

def collate_landscape(original_batch):
    imgs, gpss = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)  # Reshape to stack the views along the batch dim. Output is: [imgA_view1, imgA_view2, ..., imgB_view1, imgB_view2...]
    gps_batched = torch.stack(list(gpss), dim=0)
    # In order to address the inconsistent number of views of LUCAS images, we must choose a strategy between the 2 following:
    
    # a) Reshaping imgs to stack the views on the channel dim. This requires to adapt the model to accept k channels with k>3 probably.
    # img_batched = img_batched.reshape(1, -1, img_batched.shape[2], img_batched.shape[3])[0] 
    
    # b) Repeating the gps embeddings to match the new expanded batch dim because of LUCAS views. This requires to add an if case in the contrastive loss computation as the shapes of the similarity matrix are based on the batch_size which is artificially expanded.
    repeats = torch.tensor([x.shape[0] for x in imgs])
    gps_batched = torch.repeat_interleave(gps_batched, repeats, dim=0)  # Output is: [gps_imgA, gps_imgA,..., gps_imgB, gps_imgB...]
    return img_batched, gps_batched

def collate_satellite(original_batch):
    imgs, gpss, inds, sids = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    inds_batched = torch.stack(list(inds), dim=0)
    sids_batched = torch.cat(sids, dim=0)
    return img_batched, gps_batched, inds_batched, sids_batched

def main(args):
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    custom_collate = None
    # dataset = ContrastiveLearningDataset(args.data)
    if args.arch == 'species':
        custom_collate = collate_species
        train_dataset = SpeciesDatasetSimple(
            root_path = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
            fp_metadata = 'dataset/scale_1_species/PN_gbif_France_2005-2025_illustrated_CBN-med_train-10.0min.csv',
            transform = transforms_species(),
        )
        val_dataset = SpeciesDatasetSimple(
            root_path = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
            fp_metadata = 'dataset/scale_1_species/PN_gbif_France_2005-2025_illustrated_CBN-med_val-10.0min.csv',
            transform = transforms_species(),
        )

    elif args.arch == 'landscape':
        custom_collate = collate_landscape
        train_dataset = LandscapeDatasetSimple(
            root_path = 'dataset/scale_2_landscape/LUCAS',
            fp_metadata = 'dataset/scale_2_landscape/lucas_harmo_cover_exif_train-10.0min_CBN-Med.csv',
            transform = transforms_species(),
        )
        val_dataset = LandscapeDatasetSimple(
            root_path = 'dataset/scale_2_landscape/LUCAS',
            fp_metadata = 'dataset/scale_2_landscape/lucas_harmo_cover_exif_val-10.0min_CBN-Med.csv',
            transform = transforms_species(),
        )
    
    elif args.arch == 'satellite':
        custom_collate = collate_satellite
        train_dataset = SatelliteDatasetSimple(
            root_path = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',
            fp_metadata = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_unique_surveyId_train-0.06min.csv',
            transform = transforms_satellite(),
        )
        val_dataset = SatelliteDatasetSimple(
            root_path = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',
            fp_metadata = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_unique_surveyId_val-0.06min.csv',
            transform = transforms_satellite(),
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)

    model = ModelSimCLR(base_model=args.arch, out_dim=args.out_dim, dropout=args.dropout)
    args.learning_rate = args.learning_rate * sqrt(args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.05,  # Starts from 10 * lr
        end_factor=1.0,     # Ends at 1.0 * lr = 1e-3
        total_iters=10      # Warmup for 10 epochs
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=25)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])


    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=cosine_scheduler, args=args)
        simclr.train(train_loader, val_loader, max_iter=args.max_iter)


if __name__ == "__main__":
    args_ns = {
        'arch': 'satellite',  # always paired with gps
        'epochs': 30,
        'out_dim': 512,
        'batch_size': 64,
        'n_views': 2,  # must be equal to the number of modalities passed to the contrastive loss
        'temperature': 0.07,
        'warmup_epochs': 10,
        # 'learning_rate': 0.0015625,  # SimCLRv2 recommends 0.1 for batch size 4096. Assuming linear correlation between lr and BS, BS of 64 gives: 0.1*(64/4096)
        'learning_rate': 0.00025,
        'dropout': 0.1,
        'weight_decay': 1e-3,
        'ema_decay': 0.999,  # Exponential moving average decay. Not currently used
        'log_every_n_steps': 15,
        'max_iter': torch.inf,
        'fp16_precision': False,
        'workers': 0,
        'gpu_index': 0,
        'disable_cuda': False,
    }
    args_ns = SimpleNamespace(**args_ns)
    main(args_ns)
