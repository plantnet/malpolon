"""Main script to run training or inference on JRC multicale datasets.

Author: Theo Larcher <theo.larcher@inria.fr>
"""
from types import SimpleNamespace
from typing import Any, Callable, List
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
    MultiscaleDatasetSimple,
    MultiscaleDatasetJointWithLabels,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_contrastive_system_multiloss_downstream import (
    SimCLRToMultilabelClassification,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_model import (
    ModelSimCLR, MultiLabelClassifier,
)
from malpolon.data.datasets.geolifeclef2024_pre_extracted import \
    GLC24Datamodule
from malpolon.logging import Summary
from malpolon.models.custom_models.glc2024_pre_extracted_prediction_system import \
    ClassificationSystemGLC24
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

def collate_multiscale(original_batch):
    (imgs_species, imgs_landscape, imgs_satellite, 
     gpss_species, gpss_landscape, gpss_satellite,
     inds, inds_species, inds_landscape, inds_satellite,
     ids_species, ids_landscape, ids_satellite,
     labels_species, labels_landscape, labels_satellite) = zip(*original_batch)

    inds = torch.stack(list(inds), dim=0)

    img_batched_species = torch.cat(list(imgs_species), dim=0)
    label_batches_species = torch.stack(list(labels_species), dim=0)
    gps_batched_species = torch.stack(list(gpss_species), dim=0)
    inds_batched_species = torch.stack(list(inds_species), dim=0)
    ids_batched_species = torch.cat(ids_species, dim=0)

    img_batched_landscape = torch.cat(list(imgs_landscape), dim=0)
    label_batches_landscape = torch.stack(list(labels_landscape), dim=0)
    gps_batched_landscape = torch.stack(list(gpss_landscape), dim=0)
    inds_batched_landscape = torch.stack(list(inds_landscape), dim=0)
    ids_batched_landscape = torch.cat(ids_landscape, dim=0)

    img_batched_satellite = torch.cat(list(imgs_satellite), dim=0)
    label_batched_satellite = torch.stack(list(labels_satellite), dim=0)
    gps_batched_satellite = torch.stack(list(gpss_satellite), dim=0)
    inds_batched_satellite = torch.stack(list(inds_satellite), dim=0)
    ids_batched_satellite = torch.cat(ids_satellite, dim=0)

    return {
        'indices': inds,
        'species': (img_batched_species, gps_batched_species, inds_batched_species, ids_batched_species, label_batches_species),
        'landscape': (img_batched_landscape, gps_batched_landscape, inds_batched_landscape, ids_batched_landscape, label_batches_landscape),
        'satellite': (img_batched_satellite, gps_batched_satellite, inds_batched_satellite, ids_batched_satellite, label_batched_satellite),
    }

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

    # Datasets
    custom_collate = None
    if args.arch == 'species':
        custom_collate = collate_species
        test_dataset = SpeciesDatasetSimple(
            root_path = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
            fp_metadata = 'dataset/scale_1_species/glc24_pa_test_private_CBN-med_matching-LUCAS-500m.csv',
            transform = transforms_species(),
            subset = args.subset,
        )
    elif args.arch == 'landscape':
        custom_collate = collate_landscape
        test_dataset = LandscapeDatasetSimple(
            root_path = 'dataset/scale_2_landscape/',
            fp_metadata = 'dataset/scale_2_landscape/glc24_pa_test_private_CBN-med_matching-LUCAS-500m.csv',
            transform = transforms_species(),
            subset = args.subset,
        )
    elif args.arch == 'satellite':
        custom_collate = collate_satellite
        test_dataset = SatelliteDatasetSimple(
            root_path = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',
            fp_metadata = 'dataset/scale_3_satellite/glc24_pa_test_private_CBN-med_matching-LUCAS-500m.csv',
            transform = transforms_satellite(),
            subset = args.subset,
        )
    elif args.arch == 'multi-loss':
        custom_collate = collate_multiscale
        train_dataset = MultiscaleDatasetJointWithLabels(
            root_path_species = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
            fp_metadata_species = 'dataset/scale_1_species/PN_gbif_France_2005-2025_illustrated_CBN-med_train-0.06min_no_3-duplicates.csv',
            root_path_landscape = 'dataset/scale_2_landscape/',
            fp_metadata_landscape = 'dataset/scale_2_landscape/lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_train-0.06min.csv',
            root_path_satellite = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',
            # fp_metadata_satellite = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_unique_surveyId_train-0.06min.csv',
            fp_metadata_satellite = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_surveyId_split-10.0%_train.csv',
            transform_species = transforms_species(),
            transform_landscape = transforms_species(),
            transform_satellite = transforms_satellite(),
            subset = args.subset,
            skip_modalities = args.skip_modalities,
            task = 'multilabel_classification',
            num_classes=args.num_labels,
        )
        val_dataset = MultiscaleDatasetJointWithLabels(
            root_path_species = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
            fp_metadata_species = 'dataset/scale_1_species/PN_gbif_France_2005-2025_illustrated_CBN-med_val-0.06min_no_3-duplicates.csv',
            root_path_landscape = 'dataset/scale_2_landscape/',
            fp_metadata_landscape = 'dataset/scale_2_landscape/lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_val-0.06min.csv',
            root_path_satellite = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',
            # fp_metadata_satellite = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_unique_surveyId_val-0.06min.csv',
            fp_metadata_satellite = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_surveyId_split-10.0%_val.csv',
            transform_species = transforms_species(),
            transform_landscape = transforms_species(),
            transform_satellite = transforms_satellite(),
            subset = args.subset,
            skip_modalities = args.skip_modalities,
            task = 'multilabel_classification',
            num_classes=args.num_labels,
        )
        test_dataset = MultiscaleDatasetJointWithLabels(
            root_path_species = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
            fp_metadata_species = 'dataset/scale_3_satellite/glc24_pa_test_private_CBN-med_matching-LUCAS-500m_exploded_merged.csv',
            root_path_landscape = 'dataset/scale_2_landscape/',
            fp_metadata_landscape = 'dataset/scale_3_satellite/glc24_pa_test_private_CBN-med_matching-LUCAS-500m_exploded_merged.csv',
            root_path_satellite = 'dataset/scale_3_satellite/PA_Test_SatellitePatches/',
            fp_metadata_satellite = 'dataset/scale_3_satellite/glc24_pa_test_private_CBN-med_matching-LUCAS-500m_exploded_merged.csv',
            transform_species = transforms_species(),
            transform_landscape = transforms_species(),
            transform_satellite = transforms_satellite(),
            subset = args.subset,
            skip_modalities = args.skip_modalities,
            task = 'multilabel_classification',
            num_classes=args.num_labels,
        )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=custom_collate)
    test_loader.dataset[0]
    # Model
    model_species = ModelSimCLR(base_model='species', out_dim=args.out_dim, dropout=args.dropout,
                                freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone)
    model_landscape = ModelSimCLR(base_model='landscape', out_dim=args.out_dim, dropout=args.dropout,
                                  gps_encoder=model_species.gps_encoder, gps_head=model_species.gps_contrastive_head,
                                  freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone)
    model_satellite = ModelSimCLR(base_model='satellite', out_dim=args.out_dim, dropout=args.dropout,
                                  gps_encoder=model_species.gps_encoder, gps_head=model_species.gps_contrastive_head,
                                  freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone)
    # model = torch.nn.ModuleDict({'species': model_species, 'landscape': model_landscape, 'satellite': model_satellite})
    model = torch.nn.ModuleList([model_species, model_landscape, model_satellite])
    model = model.to(args.device)  # Must happen before instanciating he optimizer in case of loading a checkpoint


    # Transfer learning: linear probing / fine-tuning
    if args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path, map_location='cuda' if not args.disable_cuda else 'cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint loaded from {args.ckpt_path}")
    
    # Evaluation strategy
    if args.eval_type == 'knn':
        raise NotImplementedError("KNN evaluation is not implemented in this script. Please implement it if needed.")
    classifier = MultiLabelClassifier(model[0].gps_encoder, model[0].modality_encoder, model[1].modality_encoder, model[2].modality_encoder,
                                      classifier_type=args.eval_type, num_labels=args.num_labels, skip_modalities=args.skip_modalities)

    # Optimization
    args.learning_rate = args.learning_rate * sqrt(args.batch_size)
    optimizer = torch.optim.AdamW(classifier.parameters(),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.05,  # Starts from 10 * lr
        end_factor=1.0,     # Ends at 1.0 * lr = 1e-3
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    if isinstance(args.log_every_n_steps, float):
        args.log_every_n_steps_train = max(int(args.log_every_n_steps * len(train_loader)), 1)
        args.log_every_n_steps_test = max(int(args.log_every_n_steps * len(test_loader)), 1)
    else:
        args.log_every_n_steps_train = args.log_every_n_steps
        args.log_every_n_steps_test = args.log_every_n_steps
    args.log_every_n_steps_train = min(args.log_every_n_steps_train, len(train_loader))
    args.log_every_n_steps_test = min(args.log_every_n_steps_test, len(test_loader))

    # Run
    ## It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    if args.predict:
        with torch.cuda.device(args.gpu_index):
            downstream_pipeline = SimCLRToMultilabelClassification(model=classifier, optimizer=optimizer, scheduler=cosine_scheduler, args=args)
            downstream_pipeline.predict(test_loader)
    else:
        with torch.cuda.device(args.gpu_index):
            downstream_pipeline = SimCLRToMultilabelClassification(model=classifier, optimizer=optimizer, scheduler=cosine_scheduler, args=args)
            downstream_pipeline.train(train_loader, val_loader, max_iter=args.max_iter, verbose=args.verbose)


if __name__ == "__main__":
    args = {
        'arch': 'multi-loss',  # always paired with gps
        'batch_size': 32,
        'ckpt_path': 'wandb/archive/run-20250724_181933-tr7gs4v2/files/last.pth.tar',
        'resume_wandb_run': False,
        'device': "cuda",
        'disable_cuda': False,
        'dropout': 0.1,
        'ema_decay': 0.999,  # Exponential moving average decay. Not currently used
        'epochs': 40,
        'fp16_precision': False,
        'freeze_gps_backbone': True,
        'freeze_modality_backbone': True,
        'gpu_index': 0,
        'learning_rate': 0.00025,
        'log_every_n_steps': 0.05,  # if float, percentage of the epoch (e.g. 0.25 would log 4 times per epoch). If int, number of steps.
        'max_iter': torch.inf,
        'name': "Downstream task > GLC24 train/val, multi-loss model (sat only) frozen bb, linear-probing (3 hidd layers), f1 threshold computed on val",
        'n_views': 2,  # must be equal to the number of modalities passed to the contrastive loss
        'out_dim': 512,
        'subset': None,  # nb of random samples for train & val. Either int or float (percentage of the dataset size).
        'symmetric_loss': True,  # If True, the contrastive loss is computed symmetrically (i.e. matching IMG to GPS and also GPS to IMG, i.e. 2 half diagonals in the simMatrix)
        'temperature': 0.07,
        'wandb_project': 'Sandbox', # Takes values in 'Sandbox', 'Contrastive learning pairwise'
        'weight_decay': 1e-3,
        'workers': 0,
        'warmup_epochs': 0,
        'log_images': False,  # If True, logs images to wandb
        'skip_modalities': ['species', 'landscape'],  # Will skip modalities during training
        'eval_type': 'linear_probing',  # Evaluation strategy: 'linear_probing', 'fine_tuning', 'knn'
        'num_labels': 11255,
        'predict': False,
        'verbose': True,
    }
    # import os
    # os.system('wandb offline')
    args_ns = SimpleNamespace(**args)
    main(args_ns)
