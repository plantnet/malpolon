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
import rasterio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import extract_archive

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
    for sub_path in (str(survey_id)[-2:], str(survey_id)[-4:-2]):
        path = os.path.join(path, sub_path)
    path = os.path.join(path, f"{survey_id}.tiff")

    return path


def quantile_normalize(band):
    """Perform normalization on an array.

    Args:
        band (_type_): _description_

    Returns:
        _type_: _description_
    """
    band = np.array(band, dtype=np.float32)
    min_val = np.nanmin(band)  # Use nanmin to ignore NaNs
    max_val = np.nanmax(band)  # Use nanmax to ignore NaNs

    if max_val == min_val:
        return np.zeros_like(band)  # If max and min are the same, return an array of zeros

    return ((band - min_val) / (max_val - min_val)).astype(np.float32)


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
    landsat_sample = torch.nan_to_num(torch.load(path, weights_only=True))
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
    bioclim_sample = torch.nan_to_num(torch.load(path, weights_only=True))
    if isinstance(bioclim_sample, torch.Tensor):
        # bioclim_sample = bioclim_sample.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
        bioclim_sample = bioclim_sample.numpy()  # Convert tensor to numpy array
    if transform:
        bioclim_sample = transform(bioclim_sample)
    return bioclim_sample


def load_sentinel(path, transform=None):
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
    with rasterio.open(path) as dataset:
        image = dataset.read(out_dtype=np.float32)  # Read all bands
        # image = np.array([quantile_normalize(band) for band in image])  # Apply quantile normalization
        # image = np.transpose(image, (1, 2, 0))  # Convert to HWC format
    if transform:
        image = transform(image)
    return image


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
        subset: str = 'train',  # train or val
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
        self.subset = subset
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
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        if 'multiclass' in self.task:
            self.metadata = self.metadata.drop_duplicates(subset="surveyId")
        self.metadata = self.metadata.reset_index(drop=True)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get a dataset sample.

        Args:
            idx (int): n-th sample

        Returns:
            (tuple): tuple of data samples (landsat, bioclim, sentinel), label tensor (speciesId) and surveyId
        """
        survey_id = self.metadata.surveyId.iloc[idx]
        data_samples = []

        # Landsat data (pre-extracted time series)
        if self.landsat_data_dir is not None:
            landsat_sample = load_landsat(os.path.join(self.landsat_data_dir, f"GLC25-PA-{self.subset}-landsat-time-series_{survey_id}_cube.pt"),
                                          transform=self.transform['landsat'])
            data_samples.append(torch.tensor(np.array(landsat_sample), dtype=torch.float32))
        # Bioclim data (pre-extractions time series)
        if self.bioclim_data_dir is not None:
            bioclim_sample = load_bioclim(os.path.join(self.bioclim_data_dir, f"GLC25-PA-{self.subset}-bioclimatic_monthly_{survey_id}_cube.pt"),
                                          transform=self.transform['bioclim'])
            data_samples.append(torch.tensor(np.array(bioclim_sample), dtype=torch.float32))
        # Sentinel data (patches)
        if self.sentinel_data_dir is not None:
            sentinel_sample = load_sentinel(construct_patch_path(self.sentinel_data_dir, survey_id),
                                            transform=self.transform['sentinel'])
            data_samples.append(torch.tensor(np.array(sentinel_sample), dtype=torch.float32))

        # Labels
        if 'multiclass' in self.task:
            label = self.metadata.speciesId.iloc[idx]
        else:
            species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
            label = torch.zeros(self.num_classes)  # Initialize label tensor
            label[species_ids] = 1  # Set the corresponding class index to 1 for each species

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
        subset: str = 'test',  # test or val
        task: str = 'classification_multilabel'
    ):
        """Class constructor.

        Parameters
        ----------
        See TrainDataset description.
        """
        self.transform = transform if transform else {'landsat': None, 'bioclim': None, 'sentinel': None}
        super().__init__(metadata, num_classes=num_classes, bioclim_data_dir=bioclim_data_dir, landsat_data_dir=landsat_data_dir, sentinel_data_dir=sentinel_data_dir, transform=transform, subset=subset, task=task)
        self.targets = np.array([0] * len(self.metadata))
        self.observation_ids = self.metadata['surveyId']
        self.coordinates = self.metadata[['lon', 'lat']].values
        self.subset = 'test'

    def __getitem__(self, idx):
        """Get a dataset sample.

        Args:
            idx (int): n-th sample

        Returns:
            (tuple): tuple of data samples (landsat, bioclim, sentinel), label tensor (speciesId) and surveyId
        """
        survey_id = self.metadata.surveyId[idx]
        data_samples = []

        # Landsat data (pre-extracted time series)
        if self.landsat_data_dir is not None:
            landsat_sample = load_landsat(os.path.join(self.landsat_data_dir, f"GLC25-PA-{self.subset}-landsat_time_series_{survey_id}_cube.pt"),
                                          transform=self.transform['landsat'])
            data_samples.append(torch.tensor(np.array(landsat_sample), dtype=torch.float32))
        # Bioclim data (pre-extractions time series)
        if self.bioclim_data_dir is not None:
            bioclim_sample = load_bioclim(os.path.join(self.bioclim_data_dir, f"GLC25-PA-{self.subset}-bioclimatic_monthly_{survey_id}_cube.pt"),
                                          transform=self.transform['bioclim'])
            data_samples.append(torch.tensor(np.array(bioclim_sample), dtype=torch.float32))
        # Sentinel data (patches)
        if self.sentinel_data_dir is not None:
            sentinel_sample = load_sentinel(construct_patch_path(self.sentinel_data_dir, survey_id),
                                            transform=self.transform['sentinel'])
            data_samples.append(torch.tensor(np.array(sentinel_sample), dtype=torch.float32))

        if 'multiclass' in self.task:
            label = self.metadata.speciesId[idx]
        else:
            species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
            label = torch.zeros(self.num_classes)  # Initialize label tensor
            label[species_ids] = 1  # Set the corresponding class index to 1 for each species

        return tuple(data_samples) + (label, survey_id,)


class GLC25Datamodule(BaseDataModule):
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
        self.root = "dataset/"
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
                dataset = TrainDataset(train_metadata, self.num_classes, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs, subset='train')
                self.dataset_train = dataset
            case 'val':
                val_metadata = pd.read_csv(self.metadata_paths['val'])
                dataset = TrainDataset(val_metadata, **self.data_paths['train'], transform=transform, task=self.task, **self.dataset_kwargs, subset='train')
                self.dataset_val = dataset
            case 'test':
                test_metadata = pd.read_csv(self.metadata_paths['test'])
                dataset = TestDataset(test_metadata, **self.data_paths['test'], transform=transform, task=self.task, **self.dataset_kwargs, subset='test')
                self.dataset_test = dataset
        return dataset

    def _check_integrity(self):
        """Check if the dataset is already downloaded and split into train and val sets.

        Returns
        -------
        (bool)
            True if the dataset is already downloaded and split, False otherwise.
        """
        paths = ['EnvironmentalValues', 'SateliteTimeSeries-Landsat',
                 'SatelitePatches', 'EnvironmentalValues', 'BioclimTimeSeries',
                 'GLC25_P0_metadata_train.csv', 'GLC25_PA_metadata_train.csv',
                 'GLC25_PA_metadata_test.csv', 'GLC25_SAMPLE_SUBMISSION.csv']
        downloaded = all(map(lambda x: (self.root / x).exists(), paths))

        split = (self.root / "GLC25_PA_metadata_train_train-0.6min.csv").exists()
        if downloaded and not split:
            print('Data already downloaded but not split. Splitting data spatially into train (90%) & val (10%) sets.')
            split_obs_spatially(str(self.root / "GLC25_PA_metadata_train.csv"), val_size=0.10, spacing=0.01)
            split = True
        return downloaded and split

    def download(self):
        """Download the GeolifeClef2025 dataset."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        try:
            import kaggle  # pylint: disable=C0415,W0611 # noqa: F401
        except OSError as error:
            raise OSError("Have you properly set up your Kaggle API token ? For more information, please refer to section 'Authentication' of the kaggle documentation : https://www.kaggle.com/docs/api") from error

        answer = input("You are about to download the GeoLifeClef2025 dataset which weighs ~3 GB. Do you want to continue ? [y/n]")
        if answer.lower() in ["y", "yes"]:
            if 'geolifeclef-2024' in self.root.parts:
                self.root = self.root.parent
            subprocess.call(f"kaggle competitions download -c geolifeclef-2025 -p {self.root}", shell=True)
            print(f"Extracting geolifeclef-2024 to {self.root}")
            extract_archive(os.path.join(self.root, "geolifeclef-2025.zip"), os.path.join(self.root, "geolifeclef-2025/"), remove_finished=True)
            if self.root.parts[-1] != "geolifeclef-2025":
                self.root = self.root / "geolifeclef-2025"

            # Split the dataset spatially
            print('Splitting data spatially into train (90%) & val (10%) sets.')
            split_obs_spatially(str(self.root / "GLC25_PA_metadata_train.csv"), val_size=0.10, spacing=0.01)
        else:
            print("Aborting download")
            return

    def get_val_dataset(self) -> Dataset:
        """Call self.get_dataset to return the validation dataset.

        Returns
        -------
        Dataset
            validation dataset
        """
        dataset = self.get_dataset(
            split="val",
            transform=self.val_transform,
        )
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
        landsat_transforms = [transforms.Normalize(mean=[30.654] * 6,
                                                   std=[25.702] * 6)]
        bioclim_transforms = [transforms.Normalize(mean=[3914.847] * 4,
                                                   std=[3080.644] * 4)]
        sentinel_transforms = [transforms.Normalize(mean=[629.624, 691.815, 460.605] + [2959.370],
                                                    std=[435.995, 371.396, 342.897] + [925.369])]

        return {'landsat': transforms.Compose(all_transforms + landsat_transforms),
                'bioclim': transforms.Compose(all_transforms + bioclim_transforms),
                'sentinel': transforms.Compose(all_transforms + sentinel_transforms)}

    @property
    def val_transform(self):
        """Return the training transform functions for each data modality.

        The normalization values are computed from the training dataset
        (pre-extracted values) for each modality.

        Returns
        -------
        (dict)
            dictionary of transform functions for each data modality.
        """
        all_transforms = [torch.tensor]
        landsat_transforms = [transforms.Normalize(mean=[30.269] * 6,
                                                   std=[25.212] * 6)]
        bioclim_transforms = [transforms.Normalize(mean=[3955.529] * 4,
                                                   std=[3234.002] * 4)]
        sentinel_transforms = [transforms.Normalize(mean=[633.110, 692.764, 462.189] + [2950.603],
                                                    std=[465.046, 398.975, 370.759] + [927.021])]

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
        landsat_transforms = [transforms.Normalize(mean=[26.188] * 6,
                                                   std=[29.624] * 6)]
        bioclim_transforms = [transforms.Normalize(mean=[3932.149] * 4,
                                                   std=[3490.368] * 4)]
        sentinel_transforms = [transforms.Normalize(mean=[517.786, 565.655, 376.777] + [2289.862],
                                                    std=[530.537, 497.530, 427.435] + [1510.104])]
        return {'landsat': transforms.Compose(all_transforms + landsat_transforms),
                'bioclim': transforms.Compose(all_transforms + bioclim_transforms),
                'sentinel': transforms.Compose(all_transforms + sentinel_transforms)}
