"""Main script to run training or inference on JRC multicale datasets.

Author: Theo Larcher <theo.larcher@inria.fr>
"""
import os
from types import SimpleNamespace
from typing import Any, List
from math import sqrt

import wandb
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
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
    imgs, gpss, inds, ids = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    inds_batched = torch.stack(list(inds), dim=0)
    ids_batched = torch.cat(ids, dim=0)
    return img_batched, gps_batched, inds_batched, ids_batched

def collate_landscape(original_batch):
    imgs, gpss, inds, ids = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    inds_batched = torch.stack(list(inds), dim=0)
    ids_batched = torch.cat(ids, dim=0)
    return img_batched, gps_batched, inds_batched, ids_batched

# Version multi-view per row
# def collate_landscape(original_batch):
#     imgs, gpss = zip(*original_batch)
#     img_batched = torch.cat(list(imgs), dim=0)  # Reshape to stack the views along the batch dim. Output is: [imgA_view1, imgA_view2, ..., imgB_view1, imgB_view2...]
#     gps_batched = torch.stack(list(gpss), dim=0)
#     # In order to address the inconsistent number of views of LUCAS images, we must choose a strategy between the 2 following:
    
#     # a) Reshaping imgs to stack the views on the channel dim. This requires to adapt the model to accept k channels with k>3 probably.
#     # img_batched = img_batched.reshape(1, -1, img_batched.shape[2], img_batched.shape[3])[0] 
    
#     # b) Repeating the gps embeddings to match the new expanded batch dim because of LUCAS views. This requires to add an if case in the contrastive loss computation as the shapes of the similarity matrix are based on the batch_size which is artificially expanded.
#     repeats = torch.tensor([x.shape[0] for x in imgs])
#     gps_batched = torch.repeat_interleave(gps_batched, repeats, dim=0)  # Output is: [gps_imgA, gps_imgA,..., gps_imgB, gps_imgB...]
#     return img_batched, gps_batched

def collate_satellite(original_batch):
    imgs, gpss, inds, sids = zip(*original_batch)
    img_batched = torch.cat(list(imgs), dim=0)
    gps_batched = torch.stack(list(gpss), dim=0)
    inds_batched = torch.stack(list(inds), dim=0)
    sids_batched = torch.cat(sids, dim=0)
    return img_batched, gps_batched, inds_batched, sids_batched

def main(args, writer):
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.update({'device': torch.device('cuda')}, allow_val_change=True)
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.update({'device': torch.device('cpu')}, allow_val_change=True)
        args.update({'gpu_index': -1}, allow_val_change=True)

    # Datasets
    custom_collate = None
    # dataset = ContrastiveLearningDataset(args.data)
    if args.arch == 'species':
        custom_collate = collate_species
        train_dataset = SpeciesDatasetSimple(
            root_path = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
            fp_metadata = 'dataset/scale_1_species/PN_gbif_France_2005-2025_illustrated_CBN-med_train-0.06min_no_3-duplicates.csv',
            transform = transforms_species(),
            subset = args.subset,
	    query_id = 'gbifID',
        )
        val_dataset = SpeciesDatasetSimple(
            root_path = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
            fp_metadata = 'dataset/scale_1_species/PN_gbif_France_2005-2025_illustrated_CBN-med_val-0.06min_no_3-duplicates.csv',
            transform = transforms_species(),
            subset = args.subset,
	    query_id = 'gbifID',	
        )

    elif args.arch == 'landscape':
        custom_collate = collate_landscape
        train_dataset = LandscapeDatasetSimple(
            root_path = 'dataset/scale_2_landscape/',
            fp_metadata = 'dataset/scale_2_landscape/lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_train-0.06min_abaca.csv',
            transform = transforms_species(),
            subset = args.subset,
        )
        val_dataset = LandscapeDatasetSimple(
            root_path = 'dataset/scale_2_landscape/',
            fp_metadata = 'dataset/scale_2_landscape/lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_val-0.06min_abaca.csv',
            transform = transforms_species(),
            subset = args.subset,
        )
    
    elif args.arch == 'satellite':
        custom_collate = collate_satellite
        train_dataset = SatelliteDatasetSimple(
            root_path = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',  # /mnt/data_disk/malpolon/jrc/
            fp_metadata = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_unique_surveyId_train-0.06min.csv',  # /mnt/data_disk/malpolon/jrc/
            transform = transforms_satellite(),
            subset = args.subset,
        )
        val_dataset = SatelliteDatasetSimple(
            root_path = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',  # /mnt/data_disk/malpolon/jrc/
            fp_metadata = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_unique_surveyId_val-0.06min.csv',  # /mnt/data_disk/malpolon/jrc/
            transform = transforms_satellite(),
            subset = args.subset,
        )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False, collate_fn=custom_collate)

    # Model
    model = ModelSimCLR(base_model=args.arch, out_dim=args.out_dim, dropout=args.dropout,
                        freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone,
                        sat_ckpt=None)
    model = model.to(args.device)  # Must happen before instanciating he optimizer in case of loading a checkpoint
    # model = torch.nn.DataParallel(model, device_ids=[0])

    ema_model = None
    if args.ema_decay >= 0 and args.ema_decay < 1:
        ema_model = ModelSimCLR(base_model=args.arch, out_dim=args.out_dim, dropout=args.dropout,
                                freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone,
                                sat_ckpt=None)
        ema_model = ema_model.to('cpu')
        ema_model.load_state_dict(model.state_dict())
    
    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.05,  # Starts from 10 * lr
        end_factor=1.0,     # Ends at 1.0 * lr = 1e-3
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    # Transfer learning / fine-tuning / resuming
    if args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path, map_location='cuda' if not args.disable_cuda else 'cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint loaded from {args.ckpt_path}")
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded from checkpoint")
        if 'epoch' in checkpoint:
            args.epochs += checkpoint['epoch']
            cosine_scheduler.T_max = args.epochs  # update T_max of the scheduler to match the new number of epochs
            args.last_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {checkpoint['epoch']}")
        # Pre-step the scheduler to "resume" it
        for _ in range(args.last_epoch):
            scheduler.step()

    if isinstance(args.log_every_n_steps, float):
        args.log_every_n_steps_train = max(int(args.log_every_n_steps * len(train_loader)), 1)
        args.log_every_n_steps_val = max(int(args.log_every_n_steps * len(val_loader)), 1)
    else:
        args.log_every_n_steps_train = args.log_every_n_steps
        args.log_every_n_steps_val = args.log_every_n_steps
    args.log_every_n_steps_train = min(args.log_every_n_steps_train, len(train_loader))
    args.log_every_n_steps_val = min(args.log_every_n_steps_val, len(val_loader))

    # Run
    ## It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args, writer=writer)
        simclr.train(train_loader, val_loader, max_iter=args.max_iter)

def init_wandb(args):
    args_ns = SimpleNamespace(**args) if isinstance(args, dict) else args
    name = getattr(args_ns, 'name', 'default-name')
    writer = wandb.init(
        entity="tlarcher-phd-jrc",
        id=getattr(args_ns, 'ckpt_path', '').split('/')[-2].split('-')[2] if (getattr(args_ns, 'ckpt_path', None) and getattr(args_ns, 'resume_wandb_run', False)) else None,
        project=getattr(args_ns, 'wandb_project', None),
        name=name['value'] if isinstance(name, dict) else name,  #'Unique surveyId spatial split 0.06min, dropout',
        notes=f"",
        config=args_ns,
        job_type='inference' if getattr(args_ns, 'predict', False) else 'train',
        mode=getattr(args_ns, 'wandb_mode', 'offline'),
    )
    args_ns.writer = writer
    return args_ns, writer

def init_sweep(args):
    sweep_cfg = OmegaConf.to_container(OmegaConf.load(f"wandb_sweep_{args['arch']}.yaml"), resolve=True)
    for k, v in sweep_cfg['parameters'].items():
        if k in args:
            args[k] = v
        else:
            print(f"Warning: Sweep parameter {k} not found in default args dictionary.")
    args_ns = SimpleNamespace(**args) if isinstance(args, dict) else args
    return args_ns


if __name__ == "__main__":
    # import os
    # os.system('wandb offline')
    args = {
        'arch': 'satellite',  # always paired with gps
        'OAR_job_id': os.getenv("OAR_JOB_ID", "no_jobid"),
        'batch_size': 32,
        'ckpt_path': None, # 'wandb/run-20250604_170638-3sn5y6f2/files/last.pth.tar',
        'resume_wandb_run': False,
        'device': "cuda",
        'disable_cuda': False,
        'dropout': 0.1,
        'ema_model': False,
        'ema_decay': 0.999,  # Exponential moving average decay. Not currently used
        'ema_update_step': 1,
        'epochs': 40,
        'shuffle_train': True,
        'fp16_precision': True,
        'freeze_gps_backbone': False,
        'freeze_modality_backbone': False,
        'gpu_index': 0,
        'learning_rate': 0.00025,
        'log_every_n_steps': 0.1,  # if float, percentage of the epoch (e.g. 0.25 would log 4 times per epoch). If int, number of steps.
        'max_iter': torch.inf,
        'name': "[Test-watch] CRISP: satellite from scratch + KoLeo (single_diag) train shuffle ON, temp=0.7, koleo_w=0.01",
        'n_views': 2,  # must be equal to the number of modalities passed to the contrastive loss
        'out_dim': 512,
        'subset': None,  # nb of random samples for train & val. Either int or float (percentage of the dataset size).
        'symmetric_loss': True,  # If True, the contrastive loss is computed symmetrically (i.e. matching IMG to GPS and also GPS to IMG, i.e. 2 half diagonals in the simMatrix)
        'temperature': 2.659,
        'wandb_project': 'Sandbox', # Takes values in ['Sandbox', 'Contrastive learning pairwise']
        'weight_decay': 1e-3,
        'workers': 24,# os.cpu_count(),
        'warmup_epochs': 0,
        'koleo_weight': 0.001,
        'koleo_eps': 1e-4,
        'mcr_eps': 0.05,
        'loss_criterion': 'cross_entropy',  # Takes values in ['cross_entropy', 'cosine_mcr', 'crisp', 'cosine_embedding', 'cosine_embedding_from_sim', 'cosine_similarity_mean', 'cosine_loss_pytorch_like', 'cosine_embedding_loss']. By Default: cross_entropy
        'wandb_mode': 'disabled',  # 'online' or 'disabled'
    }
    sweep_id = os.getenv("WANDB_SWEEP_ID")
    if sweep_id:
        args_ns = init_sweep(args)
        print('args_ns.name: ', args_ns.name)
        print(f"🚀 Running under a W&B sweep agent (sweep ID {sweep_id})\n")
        args_ns, writer = init_wandb(args_ns)
    else:
        print("🧑‍💻 Running standalone (manual run)\n")
        args_ns, writer = init_wandb(args)
    config_wandb = args_ns.writer.config
    config_wandb.update({'learning_rate': config_wandb.learning_rate * sqrt(config_wandb.learning_rate)}, allow_val_change=True)
    main(config_wandb, writer)
