"""This module provides Datasets and Datamodule for GeoLifeCLEF2024 data.

Author: Lukas Picek <lukas.picek@inria.fr>
        Theo Larcher <theo.larcher@inria.fr>

License: GPLv3
Python version: 3.10.6
"""
import os
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torchvision.io import read_image

from malpolon.data.data_module import BaseDataModule
from malpolon.data.utils import split_obs_spatially


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

    def __init__(self, metadata, num_classes=11255, bioclim_data_dir=None, landsat_data_dir=None, sentinel_data_dir=None, transform=None):
        self.transform = transform
        self.sentinel_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
        ])
        # self.sentinel_transform = None
        self.num_classes = num_classes
        self.landsat_data_dir = landsat_data_dir
        self.bioclim_data_dir = bioclim_data_dir
        self.sentinel_data_dir = sentinel_data_dir
        self.metadata = metadata
        if 'speciesId' in self.metadata.columns:
            self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
            self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
        else:
            self.metadata['speciesId'] = [None] * len(self.metadata)
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId[idx]
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

        species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        label = torch.zeros(self.num_classes)  # Initialize label tensor
        for species_id in species_ids:
            label_id = species_id
            label[label_id] = 1  # Set the corresponding class index to 1 for each species

        return tuple(data_samples) + (label, survey_id)


class TestDataset(TrainDataset):
    """Test dataset with test transform functions.

    Inherits TrainDataset.

    Parameters
    ----------
    TrainDataset : Dataset
        inherits TrainDataset attributes and __len__() method
    """
    def __init__(self, metadata, bioclim_data_dir=None, landsat_data_dir=None, sentinel_data_dir=None, transform=None):
        self.transform = transform
        self.sentinel_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
        ])
        super().__init__(metadata, bioclim_data_dir=bioclim_data_dir, landsat_data_dir=landsat_data_dir, sentinel_data_dir=sentinel_data_dir, transform=transform)
        self.targets = np.array([0] * len(metadata))
        self.observation_ids = metadata['surveyId']

    def __getitem__(self, idx):
        survey_id = self.metadata.surveyId[idx]
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

        species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        label = torch.zeros(self.num_classes)  # Initialize label tensor
        for species_id in species_ids:
            label_id = species_id
            label[label_id] = 1  # Set the corresponding class index to 1 for each species
        return tuple(data_samples) + (label, survey_id,)


class GLC24Datamodule(BaseDataModule):
    """Data module for GeoLifeCLEF 2024 dataset."""
    def __init__(
        self,
        data_paths: dict,
        metadata_paths: dict,
        num_classes: int,
        train_batch_size: int = 64,
        inference_batch_size: int = 16,
        num_workers: int = 16,
        sampler: Callable = None,
        dataset_kwargs: dict = {},
        download_data: bool = False,
        **kwargs,
    ):
        """Class constructor.

        Parameters
        ----------
        data_paths : dict
            a 2-level dictionary containing data paths. 1st level keys:
            "train" and "test", each containing another dictionary with
            keys: "landsat_data_dir", "bioclim_data_dir",
            "sentinel_data_dir" and values: the corresponding data paths
            as strings.
        metadata_paths : dict
            a dictionary containing the paths to the observations (or
            "metadata") as values for keys "train", "test", "val"
        num_classes : int
            number of classes to train on.
        train_batch_size : int, optional
            training batch size, by default 64
        inference_batch_size : int, optional
            inference batch size, by default 16
        num_workers : int, optional
            number of PyTorch workers, by default 16
        sampler : Callable, optional
            dataloader sampler to use, by default None (standard
            iteration)
        download_data : bool, optional
            if true, will offer to download the pre-extracted data from
            Seafile, by default False
        """
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.data_paths = data_paths
        self.metadata_paths = metadata_paths
        self.sampler = sampler
        self.dataset_kwargs = dataset_kwargs
        self.num_classes = num_classes
        self.root = "data/"
        self.__dict__.update(kwargs)
        self.root = Path(self.root)
        if download_data:
            self.download()

    def get_dataset(self, split, transform, **kwargs):
        match split:
            case 'train':
                train_metadata = pd.read_csv(self.metadata_paths['train'])
                dataset = TrainDataset(train_metadata, self.num_classes, **self.data_paths['train'], transform=transform, **self.dataset_kwargs)
                self.dataset_train = dataset
            case 'val':
                val_metadata = pd.read_csv(self.metadata_paths['val'])
                dataset = TrainDataset(val_metadata, **self.data_paths['train'], transform=transform, **self.dataset_kwargs)
                self.dataset_val = dataset
            case 'test':
                test_metadata = pd.read_csv(self.metadata_paths['test'])
                dataset = TestDataset(test_metadata, **self.data_paths['test'], transform=transform, **self.dataset_kwargs)
                self.dataset_test = dataset
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_val,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_predict,
            # sampler=self.sampler(self.dataset_predict, size=self.size, units=self.units),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    def _check_integrity(self):
        downloaded = (self.root / "GLC24_P0_metadata_train.csv").exists()
        split = (self.root / "GLC24_PA_metadata_train_train-10.0min.csv").exists()
        if downloaded and not split:
            print('Data already downloaded but not split. Splitting data spatially into train (90%) & val (10%) sets.')
            split_obs_spatially(str(self.root / "GLC24_PA_metadata_train.csv"), val_size=0.10)
            split = True
        return downloaded and split

    def download(self):
        """Download the GeolifeClef2024 dataset."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        try:
            import kaggle  # pylint: disable=C0415,W0611 # noqa: F401
        except OSError as error:
            raise OSError("Have you properly set up your Kaggle API token ? For more information, please refer to section 'Authentication' of the kaggle documentation : https://www.kaggle.com/docs/api") from error

        answer = input("You are about to download the GeoLifeClef2024 dataset which weighs ~3 GB. Do you want to continue ? [y/n]")
        if answer.lower() in ["y", "yes"]:
            if 'geolifeclef-2024' in self.root.parts:
                self.root = self.root.parent
            subprocess.call(f"kaggle competitions download -c geolifeclef-2024 -p {self.root}", shell=True)
            print(f"Extracting geolifeclef-2024 to {self.root}")
            extract_archive(os.path.join(self.root, "geolifeclef-2024.zip"), os.path.join(self.root, "geolifeclef-2024/"), remove_finished=True)
            if self.root.parts[-1] != "geolifeclef-2024":
                self.root = self.root / "geolifeclef-2024"

            # Split the dataset spatially
            print('Splitting data spatially into train (90%) & val (10%) sets.')
            split_obs_spatially(str(self.root / "GLC24_PA_metadata_train.csv"), val_size=0.10)
        else:
            print("Aborting download")
            return

    @property
    def train_transform(self):
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
