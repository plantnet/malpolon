"""This module provides Sentinel-2 related classes based on torchgeo.

Sentinel-2 data is queried from Microsoft Planetary Computer (MPC).

Author: Theo Larcher <theo.larcher@inria.fr>
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.torchgeo_datasets import RasterTorchGeoDataset

if TYPE_CHECKING:
    import numpy.typing as npt
    Patches = npt.NDArray
    Targets = npt.NDArray[np.int64]


class ConcatPatchRasterDataset(Dataset):
    def __init__(
        self,
        datasets: list[dict[Dataset]],  # list of dictionaries with keys 'callable' and 'kwargs'
        split: str,
        transform: Callable,
        task: str
    ) -> None:
        super().__init__()
        self.datasets = []
        for ds in deepcopy(datasets):
            dataset = eval(ds['callable'])
            if 'providers' in ds['kwargs'].keys():
                providers = []
                for provider in ds['kwargs']['providers']:
                    providers.append(eval(provider['callable'])(**provider['kwargs']))
                ds['kwargs']['providers'] = providers
            self.datasets.append(dataset(**ds['kwargs'], split=split, transform=transform, task=task))

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Patches, Targets]:
        data, labels = ([], [])
        for ds in self.datasets:
            if isinstance(ds, RasterTorchGeoDataset):
                item = ds[ds._default_sample_to_getitem(idx)]
            else:
                item = ds[idx]
            data.append(item[0])
            labels.append(item[1])
        try:
            data_tensor = torch.cat(tuple(data))
        except ValueError:
            print('The data returned by your concat_datasets must be tensors of the same shape. Please check your datasets arguments.')
        labels = np.array(labels[0])  # Takes the labels returned by the first dataset
        return data_tensor, labels

    def __len__(self) -> int:
        return min(len(ds) for ds in self.datasets)


class ConcatTorchGeoDataModule(BaseDataModule):
    def __init__(
        self,
        concat_datasets: list[dict[Dataset, Any]],
        dataset_path: str = 'dataset/',
        labels_name: str = 'labels.csv',
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        binary_positive_classes: list = [],
        task: str = 'classification_multiclass',  # ['classification_binary', 'classification_multiclass', 'classification_multilabel']
    ):
        """Class constructor.

        Parameters
        ----------
        dataset_path : str
            path to the directory containing the data
        labels_name : str, optional
            labels file name, by default 'labels.csv'
        train_batch_size : int, optional
            train batch size, by default 32
        inference_batch_size : int, optional
            inference batch size, by default 256
        num_workers : int, optional
            how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the
            main process, by default 8
        size : int, optional
            size of the 2D extracted patches. Patches can either be
            square (int/float value) or rectangular (tuple of int/float).
            Defaults to a square of size 200, by default 200
        units : Units, optional
             The queries' unit system, must have a value in
             ['pixel', 'crs', 'm', 'meter', 'metre]. This sets the unit you want
             your query to be performed in, even if it doesn't match
             the dataset's unit system, by default Units.CRS
        crs : int, optional
            The queries' `coordinate reference system (CRS)`. This
            argument sets the CRS of the dataset's queries. The value
            should be equal to the CRS of your observations. It takes
            any EPSG integer code, by default 4326
        binary_positive_classes : list, optional
            labels' classes to consider valid in the case of binary
            classification with multi-class labels (defaults to all 0),
            by default []
        task : str, optional
            machine learning task (used to format labels accordingly),
            by default 'classification_multiclass'
        """
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.concat_datasets = OmegaConf.to_container(concat_datasets)
        self.dataset_path = dataset_path  # Gets overwritten by concat_datasets specific dataset_path
        self.labels_name = labels_name
        self.task = task
        self.binary_positive_classes = binary_positive_classes

    def get_dataset(self, split, transform=None, **kwargs) -> Dataset:
        dataset = ConcatPatchRasterDataset(self.concat_datasets,
                                           split,
                                           transform,
                                           self.task)
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
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    @property
    def train_transform(self):
        pass

    @property
    def test_transform(self):
        pass
