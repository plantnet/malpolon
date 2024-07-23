"""This module provides Sentinel-2 related classes based on torchgeo.

Sentinel-2 data is queried from Microsoft Planetary Computer (MPC).

NOTE: "unused" imports are necessary because they are evaluated in the
eval() function. These classes are passed by the user in the
config file along with their arguments.

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
from malpolon.data.datasets.geolifeclef2024 import (JpegPatchProvider,
                                                    PatchesDataset,
                                                    PatchesDatasetMultiLabel)
from malpolon.data.datasets.torchgeo_datasets import (RasterBioclim,
                                                      RasterTorchGeoDataset)
from malpolon.data.datasets.torchgeo_sentinel2 import (RasterSentinel2,
                                                       Sentinel2GeoSampler)

if TYPE_CHECKING:
    import numpy.typing as npt
    Patches = npt.NDArray
    Targets = npt.NDArray[np.int64]


class ConcatPatchRasterDataset(Dataset):
    """Concatenation dataset.

    This class concatenates multiple datasets into a single one with a
    single sampler. It is useful when you want to train a model on
    multiple datasets at the same time _(e.g.: to train both rasters
    and pre-extracted jpeg patches)_.
    In the case of RasterTorchgeDataset, the __getitem__ method calls
    a private method _default_sample_to_getitem to convert the iterating
    index to the correct index for the dataset. This is necessary because
    the RasterTorchGeoDataset class uses a custom dict-based sampler but
    the other classes don't.

    The minimum required class arguments _(i.e. observation_ids, targets,
    coordinates)_ are taken from the first dataset in the list.

    Target labels are taken from the first dataset in the list.

    All datasets must return tensors of the same shape.
    """
    def __init__(
        self,
        datasets: list[dict[Dataset]],  # list of dictionaries with keys 'callable' and 'kwargs'
        split: str,
        transform: Callable,
        task: str
    ) -> None:
        """Class constructor.

        Parameters
        ----------
        datasets : list[dict[Dataset]]
            list of dictionaries with keys 'callable' and 'kwargs' on
            which to call the datasets.
        transform : Callable
            data transform callable function.
        task : str
            deep learning task.
        """
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
        self.observation_ids = self.datasets[0].observation_ids
        self.targets = self.datasets[0].targets
        if hasattr(self.datasets[0], 'coordinates'):
            self.coordinates = self.datasets[0].coordinates

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Patches, Targets]:
        """Query an item from the dataset.

        Parameters
        ----------
        idx : int
            item index (standard int).

        Returns
        -------
        Tuple[Patches, Targets]
            concatenated data and corresponding label(s).
        """
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
    """Data module to handle concatenation dataset.

    Inherits BaseDataModule
    """
    def __init__(
        self,
        dataset_kwargs: list[dict[Dataset, Any]],
        dataset_path: str = 'dataset/',
        labels_name: str = 'labels.csv',
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        binary_positive_classes: list = [],
        task: str = 'classification_multiclass',  # ['classification_binary', 'classification_multiclass', 'classification_multilabel']
        **kwargs,
    ):
        """Class constructor.

        Parameters
        ----------
        concat_datasets : list[dict[Dataset, Any]]
            list of dictionaries with keys 'callable' and 'kwargs' on
            which to call the datasets.
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
        binary_positive_classes : list, optional
            labels' classes to consider valid in the case of binary
            classification with multi-class labels (defaults to all 0),
            by default []
        task : str, optional
            machine learning task (used to format labels accordingly),
            by default 'classification_multiclass'
        """
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_kwargs = OmegaConf.to_container(dataset_kwargs, resolve=True)
        self.dataset_path = dataset_path  # Gets overwritten by concat_datasets specific dataset_path
        self.labels_name = labels_name
        self.task = task
        self.binary_positive_classes = binary_positive_classes

    def get_dataset(self, split, transform=None, **kwargs) -> Dataset:
        dataset = ConcatPatchRasterDataset(self.dataset_kwargs,
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
