"""This version trains a downstream task only over the CBN-Med region of GLC24 PA plots."""
# %%
import os
from types import SimpleNamespace
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import csv
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from typing import List, Union, Optional, Callable, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score
)

from malpolon.models.custom_models.jrc_multiscale.jrc_multiscale_geo_encoder_model import (
    ModelSimCLR, MultiLabelClassifier,
)
from malpolon.models.custom_models.jrc_multiscale.jrc_contrastive_losses import (
    KoLeoLoss, MCR
)
from transforms import (transforms_species, transforms_satellite)
from torchvision import transforms
from torchvision.io import read_image
from matplotlib import pyplot as plt


# %% [markdown]
# ## Parameters

# %%
## Data params

SPECIES_INPUT_SIZE = 518
LANDSCAPE_INPUT_SIZE = 518
SATELLITE_INPUT_SIZE = 128

ROOT_PATH_GLC24 = 'dataset/scale_3_satellite/geolifeclef-2024/'
SATELLITE_RGB_TRAIN_FP = 'PA_Train_SatellitePatches_RGB/'
SATELLITE_RGB_TEST_FP = 'PA_Test_SatellitePatches_RGB/'
SATELLITE_NIR_TRAIN_FP = 'PA_Train_SatellitePatches_NIR/'
SATELLITE_NIR_TEST_FP = 'PA_Test_SatellitePatches_NIR/'

DATA_PATHS = {'train': {
                  'landsat_data_dir': os.path.join(ROOT_PATH_GLC24, 'TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-train-landsat_time_series/'),
                  'bioclim_data_dir': os.path.join(ROOT_PATH_GLC24, 'TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-train-bioclimatic_monthly/'),
                  'sentinel_data_dir': os.path.join(ROOT_PATH_GLC24, 'PA_Train_SatellitePatches_RGB/pa_train_patches_rgb/')
                },
              'test': {
                  'landsat_data_dir': os.path.join(ROOT_PATH_GLC24, 'TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-test-landsat_time_series/'),
                  'bioclim_data_dir': os.path.join(ROOT_PATH_GLC24, 'TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-test-bioclimatic_monthly/'),
                  'sentinel_data_dir': os.path.join(ROOT_PATH_GLC24, 'PA_Test_SatellitePatches_RGB/pa_test_patches_rgb/')
                }
             }
METADATA_PATHS = {'train':  os.path.join('/'.join(ROOT_PATH_GLC24.split('/')[:-2]), "GLC24_PA_metadata_train_split-10.0%_train_CBN-Med.csv"),
                  'val':  os.path.join('/'.join(ROOT_PATH_GLC24.split('/')[:-2]), "GLC24_PA_metadata_train_split-10.0%_val_CBN-Med.csv"),
                  'test':  os.path.join('/'.join(ROOT_PATH_GLC24.split('/')[:-2]), "GLC24_PA_metadata_test_CBN-Med.csv")}

## Models
CKPT_PATH = 'wandb/archive/run-20251012_185226-u6tiioze/files/best.pth.tar'  # Pretext task multi-modal

## Hyper-params
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6

# NUM_WORKERS = 8
# WARMUP_EPOCHS = 5


# %%
args = {
        'arch': 'multi-loss',  # always paired with gps
        'OAR_job_id': os.getenv("OAR_JOB_ID", "no_jobid"),
        'batch_size': 16,
        'ckpt_path':  'wandb/archive/run-20251012_185226-u6tiioze/files/best.pth.tar',  # 'outputs/Downstream satellite img+gps GLC24_CBN-Med | subset_cls=0.01/last.pt'
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
        'name': "TEST TO DELETE INFERENCE [Downstream] GLC24 train/val, multi-loss model frozen bb, linear-probing (3 hidd layers), f1 threshold computed on val, landscape+gps -> landscape+gps (from u6tiioze)",
        'out_dim': 2048,
        'subset': None,  # nb of random samples for train & val. Either int or float (percentage of the dataset size).
        'subset_cls': None,  # nb of random samples per class for train & val. Either int or float (percentage of the dataset size).
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
        'wandb_mode': 'online',  # 'online', 'offline', 'disabled'
        'metrics': {'accuracy_type': 'precision',
                    'accuracy_average': 'micro',
                    'accuracy_topks': (1, 5, 20),
                   },
    }
args = SimpleNamespace(**args) if isinstance(args, dict) else args

# %% [markdown]
# ## Datasets

# %%

def construct_patch_path(data_path, survey_id):
    """Construct the patch file path.

    File path is reconstructed based on plot_id as './CD/AB/XXXXABCD.jpeg'.

    Parameters
    ----------
    data_path : str
        root path
    survey_id : int
        observation id

    Returns
    -------
    (str)
        patch path
    """
    path = data_path
    for pid in (str(survey_id)[-2:], str(survey_id)[-4:-2]):
        path = os.path.join(path, pid)
    path = os.path.join(path, f"{survey_id}.jpeg")
    return path


def load_landsat(path, transform=None):
    """Load Landsat pre-extracted time series data.

    Loads pre-extracted time series data from Landsat satellite
    time series, stored as torch tensors.

    Parameters
    ----------
    path : str
        path to data cube
    transform : callable, optional
        data transform, by default None

    Returns
    -------
    (array)
        numpy array of loaded transformed data
    """
    landsat_sample = torch.nan_to_num(torch.load(path))
    if isinstance(landsat_sample, torch.Tensor):
        # landsat_sample = landsat_sample.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
        landsat_sample = landsat_sample.numpy()  # Convert tensor to numpy array
    if transform:
        landsat_sample = transform(landsat_sample)
    return landsat_sample


def load_bioclim(path, transform=None):
    """Load Bioclim pre-extracted time series data.

    Loads pre-extracted time series data from bioclim environmental
    time series, stored as torch tensors.

    Parameters
    ----------
    path : str
        path to data cube
    transform : callable, optional
        data transform, by default None

    Returns
    -------
    (array)
        numpy array of loaded transformed data
    """
    bioclim_sample = torch.nan_to_num(torch.load(path))
    if isinstance(bioclim_sample, torch.Tensor):
        # bioclim_sample = bioclim_sample.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
        bioclim_sample = bioclim_sample.numpy()  # Convert tensor to numpy array
    if transform:
        bioclim_sample = transform(bioclim_sample)
    return bioclim_sample


def load_sentinel(path, survey_id, transform=None):
    """Load Sentinel-2A pre-extracted patch data.

    Loads pre-extracted data from Sentinel-2A satellite image patches,
    stored as image patches.

    Parameters
    ----------
    path : str
        path to data cube
    survey_id: str
        observation id which identifies the patch to load
    transform : callable, optional
        data transform, by default None

    Returns
    -------
    (array)
        numpy array of loaded transformed data
    """
    rgb_sample = read_image(construct_patch_path(path, survey_id)).numpy()
    nir_sample = read_image(construct_patch_path(path.replace("rgb", "nir").replace("RGB", "NIR"), survey_id)).numpy()
    sentinel_sample = np.concatenate((rgb_sample, nir_sample), axis=0).astype(np.float32)
    # sentinel_sample = np.transpose(sentinel_sample, (1, 2, 0))
    if transform:
        # sentinel_sample = transform(torch.tensor(sentinel_sample.astype(np.float32)))
        sentinel_sample = transform(sentinel_sample)
    return sentinel_sample


class TrainDataset(Dataset):
    """Train dataset with training transform functions.

    Inherits Dataset.

    Returns
    -------
    (tuple)
        tuple of data samples (landsat, bioclim, sentinel), label tensor (speciesId) and surveyId
    """
    num_classes = 11255

    def __init__(
        self,
        metadata: pd.DataFrame,
        num_classes: int = 11255,
        bioclim_data_dir: str = None,
        landsat_data_dir: str = None,
        sentinel_data_dir: str = None,
        transform: Callable = None,
        task: str = 'classification_multilabel',
        subset_cls: float = None,
        **kwargs,
    ):
        """Class constructor.

        Parameters
        ----------
        metadata : pd.DataFrame
            observation dataframe.
        num_classes : int, optional
            number of unique labels in the dataset, by default 11255
        bioclim_data_dir : str, optional
            path to the bioclim dataset directory, by default None
        landsat_data_dir : str, optional
            path to the landsat dataset directory, by default None
        sentinel_data_dir : str, optional
            path to the sentinel dataset directory, by default None
        transform : Callable, optional
            transform function to apply to the data, by default None
        task : str, optional
            deep learning task to perform, by default 'classification_multilabel'
        """
        self.transform = transform if transform else {'landsat': None, 'bioclim': None, 'sentinel': None}
        self.sentinel_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
        ])
        # self.sentinel_transform = None
        self.task = task
        self.num_classes = num_classes
        self.landsat_data_dir = landsat_data_dir
        self.bioclim_data_dir = bioclim_data_dir
        self.sentinel_data_dir = sentinel_data_dir
        self.metadata = metadata
        if 'speciesId' in self.metadata.columns:
            self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(self.metadata['speciesId']):
                self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
        else:
            self.metadata['speciesId'] = [None] * len(self.metadata)
        # self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)  # Kills the muiltilabel aspect
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        self.unique_survey_ids = list(self.label_dict.keys())
        if subset_cls:
            self.unique_survey_ids = self.unique_survey_ids[:int(len(self.unique_survey_ids) * subset_cls)]
            self.metadata = self.metadata[self.metadata['surveyId'].isin(self.unique_survey_ids)].reset_index(drop=True)
            self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()

    def __len__(self):
        return len(self.unique_survey_ids)  # Nb of unique surveyIds

    def __getitem__(self, idx):
        survey_id = self.unique_survey_ids[idx]
        slice = self.metadata[self.metadata['surveyId'] == survey_id]
        slice_sample = slice.sample(n=1).iloc[0]  # Randomly sample one row for the survey ID
        label, lon, lat = slice_sample['speciesId'], slice_sample['lon'], slice_sample['lat']
        data_samples = []

        # Landsat data (pre-extracted time series)
        if self.landsat_data_dir is not None:
            landsat_sample = load_landsat(os.path.join(self.landsat_data_dir, f"GLC24-PA-train-landsat-time-series_{survey_id}_cube.pt"),
                                          transform=self.transform['landsat'])
            data_samples.append(torch.tensor(np.array(landsat_sample), dtype=torch.float32))
        # Bioclim data (pre-extractions time series)
        if self.bioclim_data_dir is not None:
            bioclim_sample = load_bioclim(os.path.join(self.bioclim_data_dir, f"GLC24-PA-train-bioclimatic_monthly_{survey_id}_cube.pt"),
                                          transform=self.transform['bioclim'])
            data_samples.append(torch.tensor(np.array(bioclim_sample), dtype=torch.float32))
        # Sentinel data (patches)
        if self.sentinel_data_dir is not None:
            sentinel_sample = load_sentinel(self.sentinel_data_dir, survey_id,
                                            transform=self.transform['sentinel'])
            data_samples.append(torch.tensor(np.array(sentinel_sample), dtype=torch.float32))

        # Multilabel classification
        species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        label = torch.zeros(self.num_classes)  # Initialize label tensor
        for species_id in species_ids:
            label[species_id] = 1  # Set the corresponding class index to 1 for each species

        return {'data': tuple(data_samples),
                'label': label,
                'gps': torch.tensor([lon, lat], dtype=torch.float32),
                'survey_id': survey_id}


class TestDataset(TrainDataset):
    """Test dataset with test transform functions.

    Inherits TrainDataset.

    Parameters
    ----------
    TrainDataset : Dataset
        inherits TrainDataset attributes and __len__() method
    """
    __test__ = False

    def __init__(
        self,
        metadata: pd.DataFrame,
        num_classes: int = 11255,
        bioclim_data_dir: str = None,
        landsat_data_dir: str = None,
        sentinel_data_dir: str = None,
        transform: Callable = None,
        task: str = 'classification_multilabel',
        subset_cls: float = None,
    ):
        """Class constructor.

        Parameters
        ----------
        See TrainDataset description.
        """
        self.transform = transform if transform else {'landsat': None, 'bioclim': None, 'sentinel': None}
        self.sentinel_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
        ])
        super().__init__(metadata, bioclim_data_dir=bioclim_data_dir, landsat_data_dir=landsat_data_dir, sentinel_data_dir=sentinel_data_dir, transform=transform)
        self.observation_ids = metadata['surveyId']
        self.label_dict = {k: v[0].split() for k, v in self.label_dict.items()}
        self.label_dict = {k: list(np.array(v).astype(int)) for k, v in self.label_dict.items()}
        # self.num_classes = num_classes
        # self.task = task

    def __getitem__(self, idx):
        survey_id = self.unique_survey_ids[idx]
        slice = self.metadata[self.metadata['surveyId'] == survey_id]
        slice_sample = slice.sample(n=1).iloc[0]  # Randomly sample one row for the survey ID
        lon, lat = slice_sample['lon'], slice_sample['lat']
        data_samples = []

        # Landsat data (pre-extracted time series)
        if self.landsat_data_dir is not None:
            landsat_sample = load_landsat(os.path.join(self.landsat_data_dir, f"GLC24-PA-test-landsat_time_series_{survey_id}_cube.pt"),
                                          transform=self.transform['landsat'])
            data_samples.append(torch.tensor(np.array(landsat_sample), dtype=torch.float32))
        # Bioclim data (pre-extractions time series)
        if self.bioclim_data_dir is not None:
            bioclim_sample = load_bioclim(os.path.join(self.bioclim_data_dir, f"GLC24-PA-test-bioclimatic_monthly_{survey_id}_cube.pt"),
                                          transform=self.transform['bioclim'])
            data_samples.append(torch.tensor(np.array(bioclim_sample), dtype=torch.float32))
        # Sentinel data (patches)
        if self.sentinel_data_dir is not None:
            sentinel_sample = load_sentinel(self.sentinel_data_dir, survey_id,
                                            transform=self.transform['sentinel'])
            data_samples.append(torch.tensor(np.array(sentinel_sample), dtype=torch.float32))

        # Multilabel classification
        species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        label = torch.zeros(self.num_classes)  # Initialize label tensor
        for species_id in species_ids:
            label[species_id] = 1  # Set the corresponding class index to 1 for each species

        return {'data': tuple(data_samples),
                'label': label,
                'gps': torch.tensor([lon, lat], dtype=torch.float32),
                'survey_id': survey_id}

# %% [markdown]
# ## Dataloaders

# %%
class GLC24DataModuleBasic():
    def __init__(
        self,
        data_paths: dict[str, dict[str, str]],
        metadata_paths: dict[str, str],
        task: str = 'classification_multilabel',
        num_classes: int = 11255,
        dataset_kwargs: Optional[dict[str, Any]] = None,
        subset_cls: Optional[float] = None,
    ) -> None:
         self.data_paths = data_paths
         self.metadata_paths = metadata_paths
         self.task = task
         self.dataset_kwargs = dataset_kwargs
         self.num_classes = num_classes
         self.subset_cls = subset_cls

    def get_dataset(
            self,
            split: str,
            transform: Callable = None,
            **kwargs
        ):
            """Dataset getter.

            Parameters
            ----------
            split : str
                dataset split to get, can take values in ['train', 'val', 'test']
            transform : Callable
                transformfunctions to apply to the data

            Returns
            -------
            Union[TrainDataset, TestDataset]
                dataset class to return
            """
            match split:
                case 'train':
                    train_metadata = pd.read_csv(self.metadata_paths['train'])
                    dataset = TrainDataset(train_metadata, self.num_classes, **self.data_paths['train'], subset_cls=self.subset_cls, transform=transform, task=self.task)
                    self.dataset_train = dataset
                case 'val':
                    val_metadata = pd.read_csv(self.metadata_paths['val'])
                    dataset = TrainDataset(val_metadata, **self.data_paths['train'], transform=transform, subset_cls=self.subset_cls, task=self.task)
                    self.dataset_val = dataset
                case 'test':
                    test_metadata = pd.read_csv(self.metadata_paths['test'])
                    dataset = TestDataset(test_metadata, **self.data_paths['test'], transform=transform, subset_cls=self.subset_cls, task=self.task)
                    self.dataset_test = dataset
            return dataset
    
    @property
    def train_transform(self):
        """Return the training transform functions for each data modality.

        The normalization values are computed from the training dataset
        (pre-extracted values) for each modality.

        Returns
        -------
        (dict)
            dictionary of transform functions for each data modality.
        """
        all_transforms = [torch.tensor]
        landsat_transforms = [transforms.Normalize(mean=[30.071] * 6,
                                                   std=[24.860] * 6)]
        bioclim_transforms = [transforms.Normalize(mean=[3884.726] * 4,
                                                   std=[2939.538] * 4)]
        sentinel_transforms = [transforms.Normalize(mean=[78.761, 82.859, 71.288] + [146.082],
                                                    std=[26.074, 24.484, 23.275] + [39.518])]

        return {'landsat': transforms.Compose(all_transforms + landsat_transforms),
                'bioclim': transforms.Compose(all_transforms + bioclim_transforms),
                'sentinel': transforms.Compose(all_transforms + sentinel_transforms)}

    @property
    def test_transform(self):
        """Return the test transform functions for each data modality.

        The normalization values are computed from the test dataset
        (pre-extracted values) for each modality.

        Returns
        -------
        (dict)
            dictionary of transform functions for each data modality.
        """
        all_transforms = [torch.tensor]
        landsat_transforms = [transforms.Normalize(mean=[30.923] * 6,
                                                   std=[25.722] * 6)]
        bioclim_transforms = [transforms.Normalize(mean=[4004.812] * 4,
                                                   std=[3437.992] * 4)]
        sentinel_transforms = [transforms.Normalize(mean=[78.761, 82.859, 71.288] + [143.796],
                                                    std=[26.074, 24.484, 23.275] + [43.626])]
        return {'landsat': transforms.Compose(all_transforms + landsat_transforms),
                'bioclim': transforms.Compose(all_transforms + bioclim_transforms),
                'sentinel': transforms.Compose(all_transforms + sentinel_transforms)}

# %%
glc24datamodule = GLC24DataModuleBasic(DATA_PATHS, METADATA_PATHS, task='classification_multilabel', subset_cls=args.subset_cls)
dataset_train = glc24datamodule.get_dataset(split='train')
dataset_val = glc24datamodule.get_dataset(split='val',)
dataset_test = glc24datamodule.get_dataset(split='test')

train_loader = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True,
            sampler=None)
val_loader = DataLoader(
            dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True,
            sampler=None)
test_loader = DataLoader(
            dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
            sampler=None)

# %% [markdown]
# ## Models

# %%
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

# %%
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

device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")

# %% [markdown]
# ## Pipeline

# %%
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

# %%
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

            for step, data_dict in enumerate(tqdm(loader)):
                images_landsat, images_bioclim, images_sentinel = data_dict["data"]
                images = images_sentinel.clone()
                labels = data_dict["label"]
                gps = data_dict["gps"]

                images = images.to(device)
                labels = labels.to(device).float()
                gps = torch.tensor(gps, dtype=torch.float32).to(device)

                if step == 0:
                    inspect_first_batch(images, labels, epoch, phase)

                with torch.set_grad_enabled(is_train):
                    data = {
                        "satellite_img": images,
                        "satellite_gps": gps
                    }

                    loss = 0.0
                    for k in args.downstream_modalities_to_process:
                        logits, loss = forward_accumulate(
                            model, criterion, data[k], k,
                            loss=loss, labels=labels
                        )

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
    num_classes: str = 11255,
    output_dir: str = 'outputs/inference/',
    threshold=0.3,
):
    criterion = torch.nn.BCEWithLogitsLoss()
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            images_landsat, images_bioclim, images_sentinel = data_dict["data"]
            images = images_sentinel.clone()
            labels = data_dict["label"]
            gps = data_dict["gps"]
            survey_ids = data_dict["survey_id"]

            images = images.to(device)
            labels = labels.to(device).float()
            gps = gps.to(device).float()

            data = {
                "satellite_img": images,
                "satellite_gps": gps
            }

            logits = None
            for k, v in data.items():
                logits, _ = forward_accumulate(
                    model, criterion, v, k, loss=0.0, labels=labels
                )

            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_ids.extend(survey_ids)

    # Stack
    ids = torch.tensor(all_ids).numpy()
    y_true = torch.cat(all_labels).numpy()
    multi_label_indices = [' '.join(np.where(row > 0)[0].astype(str).tolist()) for row in y_true]
    y_prob = torch.cat(all_probs).numpy()
    preds = torch.argsort(torch.tensor(y_prob), dim=1, descending=True).numpy()

    # Save predictions
    print(f"Saving top-25 predictions to {output_dir}...")
    preds_path = os.path.join(output_dir, "predictions_top25.csv")
    df = pd.DataFrame({'surveyId': ids.tolist(),
                       'probas': [' '.join(y_prob[i, :25].astype(str).tolist()) for i in range(y_prob.shape[0])],
                       'predictions': [' '.join(preds[i, :25].astype(str).tolist()) for i in range(preds.shape[0])],
                       'target_species_ids': multi_label_indices})
    df.to_csv(preds_path, index=False)
    print('Done.')
    
    print(f"Saving all predictions to {output_dir}...")
    preds_path = os.path.join(output_dir, "predictions_all.csv")
    df = pd.DataFrame({'surveyId': ids.tolist(),
                       'probas': [' '.join(y_prob[i].astype(str).tolist()) for i in range(y_prob.shape[0])],
                       'predictions': [' '.join(preds[i].astype(str).tolist()) for i in range(preds.shape[0])],
                       'target_species_ids': multi_label_indices})
    df.to_csv(preds_path, index=False)
    print('Done.')

    # # Metrics
    # y_pred = (y_prob >= threshold).astype(int)
    # f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    # precision = precision_score(y_true, y_pred, average="micro", zero_division=0)

    # try:
    #     auc = roc_auc_score(y_true, y_prob, average="micro")
    # except ValueError:
    #     auc = float("nan")

    # recall_100 = recall_at_k(y_true, y_prob, k=min(100, num_classes))
    # recall_20 = recall_at_k(y_true, y_prob, k=min(20, num_classes))

    # metrics_path = os.path.join(output_dir, "metrics.csv")
    # with open(metrics_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([
    #         "f1_micro", "auc", "precision", "recall_100", "recall_20"
    #     ])
    #     writer.writerow([
    #         f1_micro, auc, precision, recall_100, recall_20
    #     ])

    # print(
    #     f"[TEST] "
    #     f"F1-micro: {f1_micro:.4f} | "
    #     f"AUC: {auc:.4f} | "
    #     f"Precision: {precision:.4f} | "
    #     f"Recall@100: {recall_100:.4f} | "
    #     f"Recall@20: {recall_20:.4f}"
    # )


# %% [markdown]
# ## Train !

# %%
train_validate(
    classifier,
    train_loader,
    val_loader,
    optimizer,
    device,
    args.epochs,
    args.num_labels,
    output_dir='outputs/Downstream satellite img+gps GLC24_CBN-Med | subset_cls=0.01/',
    f1_threshold=0.3,
)


# run_inference(
#     classifier,
#     'outputs/Downstream satellite img+gps GLC24_CBN-Med | subset_cls=0.01/last.pt',
#     test_loader,
#     device=device,
#     num_classes = args.num_labels,
#     output_dir = 'outputs/inference/Downstream satellite img+gps GLC24_CBN-Med | subset_cls=0.01/',
#     threshold=0.3
# )
