"""Main script to run training or inference on JRC multicale datasets.

Author: Theo Larcher <theo.larcher@inria.fr>
"""
import os
import json
from types import SimpleNamespace
from math import sqrt
import wandb
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from malpolon.data.datasets.jrc_multiscale import (
    LandscapeDatasetSimple,
    SatelliteDatasetSimple,
    SpeciesDatasetSimple,

    MultiscaleDatasetJointWithLabels,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_contrastive_system_multiloss_downstream_new import (
    SimCLR_downstream,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_model import (
    ModelSimCLR, MultiLabelClassifier,
)
from transforms import (transforms_species, transforms_satellite)
from custom_dataloader_utils import (collate_species, collate_landscape, collate_satellite,
                                     collate_multiscale, DataFrameMultiIdSampler)



def main(args, writer):
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
    if args.arch == 'species':
        custom_collate = collate_species
        test_dataset = SpeciesDatasetSimple(
            root_path = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
            fp_metadata = 'dataset/scale_1_species/glc24_pa_test_private_CBN-med_matching-LUCAS-500m.csv',
            transform = transforms_species(),
            subset = args.subset,
            subset_cls = args.subset_cls,
        )
    elif args.arch == 'landscape':
        custom_collate = collate_landscape
        test_dataset = LandscapeDatasetSimple(
            root_path = 'dataset/scale_2_landscape/',
            fp_metadata = 'dataset/scale_2_landscape/glc24_pa_test_private_CBN-med_matching-LUCAS-500m.csv',
            transform = transforms_species(),
            subset = args.subset,
            subset_cls = args.subset_cls,
        )
    elif args.arch == 'satellite':
        custom_collate = collate_satellite
        test_dataset = SatelliteDatasetSimple(
            root_path = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',
            fp_metadata = 'dataset/scale_3_satellite/glc24_pa_test_private_CBN-med_matching-LUCAS-500m.csv',
            transform = transforms_satellite(),
            subset = args.subset,
            subset_cls = args.subset_cls,
        )
    elif args.arch == 'multi-loss':
        custom_collate = collate_multiscale
        if not args.predict:
            train_dataset = MultiscaleDatasetJointWithLabels(
                root_path_species = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
                fp_metadata_species = 'dataset/scale_1_species/PN_gbif_France_2005-2025_illustrated_CBN-med_train-0.06min_no_3-duplicates.csv',
                root_path_landscape = 'dataset/scale_2_landscape/',
                fp_metadata_landscape = 'dataset/scale_2_landscape/lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_train-0.06min_abaca.csv',
                root_path_satellite = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',
                fp_metadata_satellite = 'dataset/scale_3_satellite/geolifeclef-2024/GLC24_PA_metadata_train_train-10.0min.csv',
                # fp_metadata_satellite = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_unique_surveyId_train-0.06min.csv',
                # fp_metadata_satellite = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_surveyId_split-10.0%_train.csv',
                transform_species = transforms_species(),
                transform_landscape = transforms_species(),
                transform_satellite = transforms_satellite(),
                subset = args.subset,
                subset_cls = args.subset_cls,
                skip_modalities = args.skip_modalities,
                task = 'multilabel_classification',
                num_classes=args.num_labels,
                query_ids = {'species': 'gbifID', 'landscape': 'id', 'satellite': 'surveyId'},
                keep_id_duplicates = True,
            )
            val_dataset = MultiscaleDatasetJointWithLabels(
                root_path_species = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
                fp_metadata_species = 'dataset/scale_1_species/PN_gbif_France_2005-2025_illustrated_CBN-med_val-0.06min_no_3-duplicates.csv',
                root_path_landscape = 'dataset/scale_2_landscape/',
                fp_metadata_landscape = 'dataset/scale_2_landscape/lucas_harmo_cover_exif_nona_fixed_gps_CBN-Med_expanded_essentials_exists_val-0.06min_abaca.csv',
                root_path_satellite = 'dataset/scale_3_satellite/PA_Train_SatellitePatches/',
                fp_metadata_satellite = 'dataset/scale_3_satellite/geolifeclef-2024/GLC24_PA_metadata_train_val-10.0min.csv',
                # fp_metadata_satellite = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_unique_surveyId_val-0.06min.csv',
                # fp_metadata_satellite = 'dataset/scale_3_satellite/glc24_pa_train_CBN-med_surveyId_split-10.0%_val.csv',
                transform_species = transforms_species(),
                transform_landscape = transforms_species(),
                transform_satellite = transforms_satellite(),
                subset = args.subset,
                subset_cls = args.subset_cls,
                skip_modalities = args.skip_modalities,
                task = 'multilabel_classification',
                num_classes=args.num_labels,
                query_ids = {'species': 'gbifID', 'landscape': 'id', 'satellite': 'surveyId'},
                keep_id_duplicates = True,
            )
        else:
            test_dataset = MultiscaleDatasetJointWithLabels(
                root_path_species = 'dataset/scale_1_species/Gbif_Illustrations_PO_gbif_glc24_PN-only_CBN-med_matching-LUCAS-500',
                fp_metadata_species = 'dataset/scale_3_satellite/glc24_pa_test_private_CBN-med_matching-LUCAS-500m_exploded_merged_with_species_grouped.csv',
                root_path_landscape = 'dataset/scale_2_landscape/',
                fp_metadata_landscape = 'dataset/scale_3_satellite/glc24_pa_test_private_CBN-med_matching-LUCAS-500m_exploded_merged_with_species_grouped.csv',
                root_path_satellite = 'dataset/scale_3_satellite/PA_Test_SatellitePatches/',
                fp_metadata_satellite = 'dataset/scale_3_satellite/geolifeclef-2024/GLC24_PA_metadata_test.csv',  # glc24_pa_test_private_CBN-med_matching-LUCAS-500m_exploded_merged_with_species_SAT_ONLY.csv',
                transform_species = transforms_species(),
                transform_landscape = transforms_species(),
                transform_satellite = transforms_satellite(),
                subset = args.subset,
                subset_cls = args.subset_cls,
                skip_modalities = args.skip_modalities,
                task = 'multilabel_classification',
                num_classes=args.num_labels,
                query_ids = {'species': 'gbifID', 'landscape': 'id', 'satellite': 'surveyId'},
                keep_id_duplicates = True,
            )
            # import pandas as pd
            # from malpolon.data.datasets.geolifeclef2024_pre_extracted import TrainDataset
            # custom_collate = None
            
            # train_dataset = TrainDataset(pd.read_csv('dataset/scale_3_satellite/geolifeclef-2024/GLC24_PA_metadata_train_train-10.0min.csv'),
            #                              num_classes = args.num_labels,
            #                              bioclim_data_dir = "dataset/scale_3_satellite/geolifeclef-2024/TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-train-bioclimatic_monthly/",
            #                              landsat_data_dir = "dataset/scale_3_satellite/geolifeclef-2024/TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-train-landsat_time_series/",
            #                              sentinel_data_dir = "dataset/scale_3_satellite/geolifeclef-2024/PA_Train_SatellitePatches_RGB/pa_train_patches_rgb/",
            #                              task = 'classification_multilabel',)
            # val_dataset = TrainDataset(pd.read_csv('dataset/scale_3_satellite/geolifeclef-2024/GLC24_PA_metadata_train_val-10.0min.csv'),
            #                              num_classes = args.num_labels,
            #                              bioclim_data_dir = "dataset/scale_3_satellite/geolifeclef-2024/TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-test-bioclimatic_monthly/",
            #                              landsat_data_dir = "dataset/scale_3_satellite/geolifeclef-2024/TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-test-landsat_time_series/",
            #                              sentinel_data_dir = "dataset/scale_3_satellite/geolifeclef-2024/PA_test_SatellitePatches_RGB/pa_test_patches_rgb/",
            #                              task = 'classification_multilabel',)
            

    # Dataloaders
    if args.predict:
        test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
        sampler=None, collate_fn=custom_collate)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True,
            sampler=None, collate_fn=custom_collate)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True,
            sampler=None, collate_fn=custom_collate)
    # Model
    model_species = ModelSimCLR(base_model='species', out_dim=args.out_dim, dropout=args.dropout,
                                freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone)
    model_landscape = ModelSimCLR(base_model='landscape', out_dim=args.out_dim, dropout=args.dropout,
                                  gps_encoder=model_species.gps_encoder, gps_head=model_species.gps_contrastive_head,
                                  freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone)
    model_satellite = ModelSimCLR(base_model='satellite', out_dim=args.out_dim, dropout=args.dropout,
                                  gps_encoder=model_species.gps_encoder, gps_head=model_species.gps_contrastive_head,
                                  freeze_modality_backbone=args.freeze_modality_backbone, freeze_gps_backbone=args.freeze_gps_backbone)
    model = torch.nn.ModuleDict({'species': model_species, 'landscape': model_landscape, 'satellite': model_satellite})
    # model = torch.nn.ModuleList([model_species, model_landscape, model_satellite])
    model = model.to(args.device)  # Must happen before instanciating he optimizer in case of loading a checkpoint

    # Transfer learning: linear probing / fine-tuning
    if args.ckpt_path and args.predict == "False":
        checkpoint = torch.load(args.ckpt_path, map_location='cuda' if not args.disable_cuda else 'cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint loaded from {args.ckpt_path}")

    # Evaluation strategy
    if args.eval_type == 'knn':
        raise NotImplementedError("KNN evaluation is not implemented in this script. Please implement it if needed.")
    classifier = MultiLabelClassifier(model['species'].gps_encoder, model['species'].gps_contrastive_head,
                                      model['species'].modality_encoder, model['species'].modality_contrastive_head,
                                      model['landscape'].modality_encoder, model['species'].modality_contrastive_head,
                                      model['satellite'].modality_encoder, model['satellite'].modality_contrastive_head,
                                      classifier_type=args.eval_type, contrastive_head_out_dim=args.out_dim,
                                      num_labels=args.num_labels, skip_modalities=args.skip_modalities)
    classifier = torch.nn.DataParallel(classifier, device_ids=[0])

    # Inference
    if args.ckpt_path and args.predict == "True":
        checkpoint = torch.load(args.ckpt_path, map_location='cuda' if not args.disable_cuda else 'cpu')
        classifier.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint loaded from {args.ckpt_path}")

    # DEBUG: REPLACING SATELLITE ENCODER WITH THAT OF MME
    # from torch import nn
    # from torchvision import models
    # from mme_model_lukas import MultiModalEnsembleC
    # MME = MultiModalEnsembleC()
    # state_dict = torch.load('mme_model_lukas_weights.bin')
    # MME.from_pretrained(state_dict)
    
    
    # for encoder in [classifier.gps_encoder, classifier.gps_contrastive_head, classifier.species_encoder, classifier.species_contrastive_head, classifier.landscape_encoder, classifier.landscape_contrastive_head, classifier.satellite_encoder, classifier.satellite_contrastive_head]:
    #     for param in encoder.parameters():
    #         param.requires_grad = False
    # END DEBUG

    # Optimization
    optimizer = torch.optim.AdamW(classifier.module.classifier.parameters() if isinstance(classifier, torch.nn.parallel.DataParallel) else classifier.classifier.parameters(),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.05,  # Starts from 10 * lr
        end_factor=1.0,     # Ends at 1.0 * lr = 1e-3
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

    if not args.predict:
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
    if args.predict:
        with torch.cuda.device(args.gpu_index):
            downstream_pipeline = SimCLR_downstream(model=classifier, optimizer=optimizer, scheduler=scheduler, writer=writer, args=args)
            downstream_pipeline.predict(test_loader)
    else:
        with torch.cuda.device(args.gpu_index):
            downstream_pipeline = SimCLR_downstream(model=classifier, optimizer=optimizer, scheduler=scheduler, writer=writer, args=args)
            downstream_pipeline.train(train_loader, val_loader, max_iter=args.max_iter)

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
    file_path = os.path.join(writer.dir, "run_args.yaml")
    with open(file_path, "w") as f:
        try:
            f.write(json.dumps(args, indent=2))
        except TypeError:
            f.write(str(args))
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
    args = {
        'arch': 'multi-loss',  # always paired with gps
        'OAR_job_id': os.getenv("OAR_JOB_ID", "no_jobid"),
        'batch_size': 64,
        'ckpt_path':  'wandb/archive/run-20251012_185226-u6tiioze/files/best.pth.tar', # 'wandb/archive/run-20251012_185226-u6tiioze/files/best.pth.tar',
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
        'name': "TEST TO DELETE  (from u6tiioze)",
        'out_dim': 2048,
        'subset': None,  # nb of random samples for train & val. Either int or float (percentage of the dataset size).
        'subset_cls': 0.1,  # nb of random samples per class for train & val. Either int or float (percentage of the dataset size).
        'wandb_project': 'Sandbox', # Takes values in 'Sandbox', 'Contrastive learning pairwise'
        'weight_decay': 1e-3,
        'workers': os.cpu_count(),
        'warmup_epochs': 0,
        'log_images': True,  # If True, logs images to wandb
        'skip_modalities': ['landscape', 'species'], 
        'downstream_modalities_to_process': ['satellite_img', 'satellite_gps'],  # Will skip modalities during training
        'eval_type': 'linear_probing',  # Evaluation strategy: 'linear_probing', 'fine_tuning', 'knn'
        'num_labels': 11255,
        'loss_criterion': 'BCE',  # Takes values in ['cross_entropy', 'BCE']
        'predict': False,
        'wandb_mode': 'disabled',  # 'online', 'offline', 'disabled'
        'metrics': {'accuracy_type': 'precision',
                    'accuracy_average': 'micro',
                    'accuracy_topks': (1, 5, 20),
                   },
    }
    ### sklearn LabelEncoder
    if args.get('subset_cls', False):
        args['num_labels'] = int(args['num_labels'] * args['subset_cls']) if isinstance(args['subset_cls'], float) else args['subset_cls']
    ###
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