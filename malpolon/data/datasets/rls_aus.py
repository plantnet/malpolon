"""This module provides Datasets and Datamodule for GeoLifeCLEF2024 data.

Author: Lukas Picek <lukas.picek@inria.fr>
        Theo Larcher <theo.larcher@inria.fr>
Adapted by: Gaetan Morand <gaetan.morand@umontpellier.fr>

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


class TrainDataset(Dataset):
    """Train dataset with training transform functions.

    Inherits Dataset.

    Returns
    -------
    (tuple)
        tuple of data samples (landsat, bioclim, sentinel), label tensor (speciesId) and surveyId
    """

    def __init__(self,
                 dataset: pd.DataFrame,
                 num_classes: int,
                 envdata_dir: str = None,
                 transform: Callable = None,
                 task: str = 'regression_multilabel',
                 **kwargs,
    ):
        """Class constructor.

        Parameters
        ----------
        dataset : pd.DataFrame
            observation dataframe.
        num_classes : int, optional
            number of unique labels in the dataset, by default 11255
        envdata_dir : str, optional
            path to the landsat dataset directory, by default None
        transform : Callable, optional
            transform function to apply to the data, by default None
        task : str, optional
            deep learning task to perform, by default 'classification_multilabel'
        """
        self.transform = transform if transform else None

        self.task = task
        self.num_classes = num_classes
        self.envdata_dir = envdata_dir
        self.dataset = dataset[dataset['subset'] == 'train']

        first_species_index = list(dataset.columns).index('geom') + 1
        self.labels = dataset.columns[first_species_index:-1]
        assert len(self.labels) == self.num_classes
        self.targets = self.dataset.loc[:, self.labels]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        survey_id = self.dataset.index[idx]
        data_samples = []

        # Environmental data
        if self.envdata_dir is not None:
            envdata_sample = np.load(Path(self.envdata_dir/ f"{survey_id}.npy"))
            data_samples.append(torch.tensor(envdata_sample, dtype=torch.float32))

        target = torch.tensor(self.targets.iloc[idx], dtype=torch.float32)

        return tuple(data_samples) + (target, survey_id,)


class TestDataset(TrainDataset):
    """Test dataset with test transform functions.

    Inherits TrainDataset.

    Parameters
    ----------
    TrainDataset : Dataset
        inherits TrainDataset attributes and __len__() method
    """
    __test__ = False

    def __init__(self,
                 dataset: pd.DataFrame,
                 num_classes: int,
                 envdata_dir: str = None,
                 transform: Callable = None,
                 task: str = 'classification_multilabel'
    ):
        """Class constructor.

        Parameters
        ----------
        See TrainDataset description.
        """
        self.transform = transform if transform else {'landsat': None, 'bioclim': None, 'sentinel': None}

        super().__init__(dataset, envdata_dir=envdata_dir, transform=transform)
        self.targets = np.zeros([num_classes,len(self.dataset)])
        self.observation_ids = dataset.index

    def __getitem__(self, idx):
        survey_id = self.dataset.index[idx]
        data_samples = []

        # Landsat data (pre-extracted time series)
        if self.envdata_dir is not None:
            envdata_sample = np.load(Path(self.envdata_dir/ f"{survey_id}.npy"))
            data_samples.append(torch.tensor(envdata_sample, dtype=torch.float32))

        target = torch.tensor(self.targets.iloc[idx], dtype=torch.float32)

        return tuple(data_samples) + (target, survey_id,)


class RLSDatamodule(BaseDataModule):
    """Data module for GeoLifeCLEF 2024 dataset."""
    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        num_classes: int,
        train_batch_size: int = 64,
        inference_batch_size: int = 16,
        num_workers: int = 16,
        sampler: Callable = None,
        dataset_kwargs: dict = {},
        task: str = 'regression_multilabel',
        **kwargs,
    ):
        """Class constructor.

        Parameters
        ----------
        data_paths : str
        metadata_paths : str
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
        task : str, optional
            Task to perform. Can take values in ['classification_multiclass',
            'classification_multilabel'], by default 'classification_multilabel'
        """
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.sampler = sampler
        self.dataset_kwargs = dataset_kwargs
        self.num_classes = num_classes
        self.__dict__.update(kwargs)
        self.root = Path(data_path)
        self.task = task

    def get_dataset(self,
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

        metadata = pd.read_csv(self.metadata_path)
        sub_dataset = metadata[metadata['subset'] == split].drop(columns='subset')

        if split == 'test':
            dataset = TestDataset(sub_dataset, self.num_classes, self.data_path, transform=transform, task=self.task, **self.dataset_kwargs)
            self.dataset_test = dataset
        else:
            dataset = TrainDataset(sub_dataset, self.num_classes, self.data_path, transform=transform, task=self.task, **self.dataset_kwargs)
            self.dataset_train = dataset

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


