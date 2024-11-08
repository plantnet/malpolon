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
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import (download_and_extract_archive,
                                        extract_archive)
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

    def __init__(
        self,
        metadata: pd.DataFrame,
        num_classes: int = 11255,
        bioclim_data_dir: str = None,
        landsat_data_dir: str = None,
        sentinel_data_dir: str = None,
        transform: Callable = None,
        task: str = 'classification_multilabel',
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
            self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
        else:
            self.metadata['speciesId'] = [None] * len(self.metadata)
        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)  # Should we ?
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        survey_id = self.metadata.surveyId.iloc[idx]
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

        if 'multiclass' in self.task:
            label = self.metadata.speciesId.iloc[idx]
        else:
            species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
            label = torch.zeros(self.num_classes)  # Initialize label tensor
            for species_id in species_ids:
                label[species_id] = 1  # Set the corresponding class index to 1 for each species

        return tuple(data_samples) + (label, survey_id,)


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
        task: str = 'classification_multilabel'
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
        self.targets = np.array([0] * len(self.metadata))
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

        if 'multiclass' in self.task:
            label = self.metadata.speciesId[idx]
        else:
            species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
            label = torch.zeros(self.num_classes)  # Initialize label tensor
            for species_id in species_ids:
                label[species_id] = 1  # Set the corresponding class index to 1 for each species

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
        task: str = 'classification_multilabel',
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
        dataset_kwargs : dict, optional
            additional keyword arguments to pass to the dataset, by default {}
        download_data : bool, optional
            if true, will offer to download the pre-extracted data from
            Seafile, by default False
        task : str, optional
            Task to perform. Can take values in ['classification_multiclass',
            'classification_multilabel'], by default 'classification_multilabel'
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
        self.task = task

    def get_dataset(
        self,
        split: str,
        transform: Callable,
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
                dataset = TrainDataset(train_metadata, self.num_classes, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_train = dataset
            case 'val':
                val_metadata = pd.read_csv(self.metadata_paths['val'])
                dataset = TrainDataset(val_metadata, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_val = dataset
            case 'test':
                test_metadata = pd.read_csv(self.metadata_paths['test'])
                dataset = TestDataset(test_metadata, **self.data_paths['test'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_test = dataset
        return dataset

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_val,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return dataloader

    def _check_integrity(self):
        """Check if the dataset is already downloaded and split into train and val sets."

        Returns
        -------
        (bool)
            True if the dataset is already downloaded and split, False otherwise.
        """
        paths = ['EnvironmentalRasters', 'PA-test-landsat_time_series',
                 'PA_Test_SatellitePatches_NIR', 'PA_Test_SatellitePatches_RGB',
                 'PA-train-landsat_time_series', 'PA_Train_SatellitePatches_NIR',
                 'PA_Train_SatellitePatches_RGB', 'TimeSeries-Cubes',
                 'GLC24_P0_metadata_train.csv', 'GLC24_PA_metadata_train.csv',
                 'GLC24_PA_metadata_test.csv', 'GLC24_SAMPLE_SUBMISSION.csv']
        downloaded = all(map(lambda x: (self.root / x).exists(), paths))

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


class TrainDatasetHabitat(TrainDataset):
    """GLC24 pre-extracted train dataset for habitat classification.

    Parameters
    ----------
    Inherits TrainDataset.
    """
    def __init__(self, metadata, classes, bioclim_data_dir=None, landsat_data_dir=None, sentinel_data_dir=None, transform=None, task='classification_multilabel'):
        metadata = metadata[metadata['habitatId'].notna()]
        metadata = metadata[metadata['habitatId'] != 'Unknown']
        self.label_encoder = LabelEncoder().fit(classes)
        metadata['habitatId_encoded'] = self.label_encoder.transform(metadata['habitatId'])
        metadata.rename({'PlotObservationID': 'surveyId'}, axis=1, inplace=True)
        metadata.rename({'habitatId_encoded': 'speciesId'}, axis=1, inplace=True)

        super().__init__(metadata, num_classes=len(classes), bioclim_data_dir=bioclim_data_dir, landsat_data_dir=landsat_data_dir, sentinel_data_dir=sentinel_data_dir, transform=transform, task=task)

        self.metadata = metadata
        if 'speciesId' in self.metadata.columns:
            self.metadata = self.metadata.dropna(subset='speciesId').reset_index(drop=True)
            self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
        else:
            self.metadata['speciesId'] = [None] * len(self.metadata)
        self.metadata = self.metadata.drop_duplicates(subset=["surveyId", "habitatId"]).reset_index(drop=True)  # Should we ?
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        self.targets = self.metadata['speciesId'].values
        self.observation_ids = self.metadata['surveyId']


class TestDatasetHabitat(TestDataset):
    """GLC24 pre-extracted test dataset for habitat classification.

    Parameters
    ----------
    Inherits TestDataset.
    """
    def __init__(self, metadata, classes, bioclim_data_dir=None, landsat_data_dir=None, sentinel_data_dir=None, transform=None, task='classification_multilabel'):
        metadata = metadata[metadata['habitatId'].notna()]
        metadata = metadata[metadata['habitatId'] != 'Unknown']
        self.label_encoder = LabelEncoder().fit(classes)
        metadata['habitatId_encoded'] = self.label_encoder.transform(metadata['habitatId'])
        metadata.rename({'PlotObservationID': 'surveyId'}, axis=1, inplace=True)
        metadata.rename({'habitatId_encoded': 'speciesId'}, axis=1, inplace=True)

        super().__init__(metadata, bioclim_data_dir=bioclim_data_dir, landsat_data_dir=landsat_data_dir, sentinel_data_dir=sentinel_data_dir, transform=transform, task=task)
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        self.targets = self.metadata['speciesId'].values
        self.observation_ids = self.metadata['surveyId']


class GLC24DatamoduleHabitats(GLC24Datamodule):
    """GLC24 pre-extracted datamodule for habitat classification.

    Parameters
    ----------
    Inherits GLC24Datamodule.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Union of train and test cls
        self.classes = ['MA221', 'MA222', 'MA223', 'MA224', 'MA225', 'MA241', 'MA251',
                        'MA252', 'MA253', 'MAa', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16',
                        'N18', 'N19', 'N1A', 'N1B', 'N1D', 'N1G', 'N1H', 'N1J', 'N21',
                        'N22', 'N31', 'N32', 'N35', 'Q11', 'Q21', 'Q22', 'Q24', 'Q25',
                        'Q41', 'Q42', 'Q43', 'Q44', 'Q51', 'Q52', 'Q53', 'Q54', 'Q61',
                        'Q62', 'Q63', 'R11', 'R12', 'R13', 'R14', 'R16', 'R18', 'R19',
                        'R1A', 'R1B', 'R1D', 'R1E', 'R1F', 'R1H', 'R1M', 'R1P', 'R1Q',
                        'R1R', 'R1S', 'R21', 'R22', 'R23', 'R24', 'R31', 'R32', 'R34',
                        'R35', 'R36', 'R37', 'R41', 'R43', 'R44', 'R45', 'R51', 'R52',
                        'R54', 'R55', 'R56', 'R57', 'R61', 'R62', 'R63', 'S21', 'S22',
                        'S23', 'S24', 'S25', 'S26', 'S31', 'S32', 'S33', 'S34', 'S35',
                        'S36', 'S37', 'S38', 'S41', 'S42', 'S51', 'S52', 'S53', 'S54',
                        'S61', 'S62', 'S63', 'S91', 'S92', 'S93', 'T11', 'T12', 'T13',
                        'T15', 'T16', 'T17', 'T18', 'T19', 'T1A', 'T1B', 'T1C', 'T1D',
                        'T1E', 'T1F', 'T1H', 'T21', 'T22', 'T24', 'T27', 'T29', 'T31',
                        'T32', 'T33', 'T34', 'T35', 'T36', 'T37', 'T39', 'T3A', 'T3C',
                        'T3D', 'T3F', 'T3J', 'T3K', 'T3M', 'U22', 'U24', 'U26', 'U27',
                        'U28', 'U29', 'U32', 'U33', 'U34', 'U36', 'U37', 'U38', 'U3A',
                        'U3D', 'U71', 'V11', 'V12', 'V13', 'V14', 'V15', 'V32', 'V33',
                        'V34', 'V35', 'V37', 'V38', 'V39']

    def get_dataset(self, split, transform, **kwargs):
        match split:
            case 'train':
                train_metadata = pd.read_csv(self.metadata_paths['train'])
                dataset = TrainDatasetHabitat(train_metadata, self.classes, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_train = dataset
            case 'val':
                val_metadata = pd.read_csv(self.metadata_paths['val'])
                dataset = TrainDatasetHabitat(val_metadata, self.classes, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_val = dataset
            case 'test':
                test_metadata = pd.read_csv(self.metadata_paths['test'])
                dataset = TestDatasetHabitat(test_metadata, self.classes, **self.data_paths['test'], transform=transform, task=self.task, **self.dataset_kwargs)
                self.dataset_test = dataset
        return dataset

    def _check_integrity_habitat(self):
        paths = {'predictors': ['EnvironmentalRasters', 'PA-test-landsat_time_series',
                                'PA_Test_SatellitePatches_NIR', 'PA_Test_SatellitePatches_RGB',
                                'PA-train-landsat_time_series', 'PA_Train_SatellitePatches_NIR',
                                'PA_Train_SatellitePatches_RGB', 'TimeSeries-Cubes'],
                 'metadata': ['GLC24_PA_metadata_habitats-lvl3_test.csv',
                              'GLC24_PA_metadata_habitats-lvl3_train.csv',
                              'GLC24_PA_metadata_habitats-lvl3_train_split-10.0%_all.csv',
                              'GLC24_PA_metadata_habitats-lvl3_train_split-10.0%_train.csv',
                              'GLC24_PA_metadata_habitats-lvl3_train_split-10.0%_val.csv']}
        downloaded_p = all(map(lambda x: Path(self.root / x).exists(), paths['predictors']))
        downloaded_m = all(map(lambda x: Path(self.root / x).exists(), paths['metadata']))
        return downloaded_p, downloaded_m

    def download(self):
        downloaded_p, downloaded_m = self._check_integrity_habitat()

        # Metadata
        if not downloaded_m:
            print('Downloading observations ("metadata")...')
            download_and_extract_archive(
                "https://lab.plantnet.org/seafile/f/583b1878f0694eeca163/?dl=1",
                self.root,
                filename='GLC24_PA_metadata_habitats-lvl3.zip',
                md5='24dc7e126f2bac79a63fdacb4f210f19',
                remove_finished=True
            )
        else:
            print('Observations ("metadata") already downloaded.')

        # Predictors
        if not downloaded_p:
            print('Downloading data ("predictors")...')
            self.root = self.root.parent / "geolifeclef-2024"
            super().download()
            self.root = self.root.parent / "geolifeclef-2024_habitats"
            links = {"../geolifeclef-2024/TimeSeries-Cubes/": "TimeSeries-Cubes",
                     "../geolifeclef-2024/PA_Train_SatellitePatches_RGB/": "PA_Train_SatellitePatches_RGB",
                     "../geolifeclef-2024/PA_Train_SatellitePatches_NIR/": "PA_Train_SatellitePatches_NIR",
                     "../geolifeclef-2024/PA-train-landsat_time_series/": "PA-train-landsat_time_series",
                     "../geolifeclef-2024/PA_Test_SatellitePatches_RGB/": "PA_Test_SatellitePatches_RGB",
                     "../geolifeclef-2024/PA_Test_SatellitePatches_NIR/": "PA_Test_SatellitePatches_NIR",
                     "../geolifeclef-2024/PA-test-landsat_time_series/": "PA-test-landsat_time_series",
                     "../geolifeclef-2024/EnvironmentalRasters/": "EnvironmentalRasters"}
            for k, v in links.items():
                os.system(f'ln -sf {k} {str(self.root / v)}')
        else:
            print('Data ("predictors") already downloaded.')
