"""This module provides Sentinel-2 related classes based on torchgeo.

Sentinel-2 data is queried from Microsoft Planetary Computer (MPC).

Author: Theo Larcher <theo.larcher@inria.fr>
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import planetary_computer
import pystac
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import GeoSampler
from torchvision import transforms

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.torchgeo_datasets import RasterTorchGeoDataset

if TYPE_CHECKING:
    import numpy.typing as npt

    Patches = npt.NDArray
    Targets = npt.NDArray[np.int64]


class Sentinel2TorchGeoDataModule(BaseDataModule):
    """Data module for Sentinel-2A dataset."""
    def __init__(
        self,
        dataset_path: str,
        labels_name: str = 'labels.csv',
        train_batch_size: int = 32,
        inference_batch_size: int = 16,
        num_workers: int = 8,
        size: int = 200,
        units: str = 'pixel',
        crs: int = 4326,
        binary_positive_classes: list = [],
        task: str = 'classification_multiclass',  # ['classification_binary', 'classification_multiclass', 'classification_multilabel']
        dataset_kwargs: dict = {},
        download_data_sample: bool = False,
        **kwargs,
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
        dataset_kwargs : dict, optional
            additional keyword arguments for the dataset, by default {}
        download_data_sample: bool, optional
            whether to download a sample of Sentinel-2 data, by default False
        """
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.labels_name = labels_name
        self.size = size
        self.units = units
        self.crs = crs
        self.sampler = Sentinel2GeoSampler
        self.task = task
        self.binary_positive_classes = binary_positive_classes
        self.dataset_kwargs = dataset_kwargs
        if download_data_sample:
            self.download_data_sample()

    def download_data_sample(self):
        """Download 4 Sentinel-2A tiles from MPC.

        This method is useful to quickly download a sample of
        Sentinel-2A tiles via Microsoft Planetary Computer (MPC).
        The referenced of the tile downloaded are specified by the
        `tile_id` and `timestamp` variables. Tiles are not downloaded
        if they already have been and are found locally.
        """
        tile_id = 'T31TEJ'
        timestamp = '20190801T104031'
        item_urls = [f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2A_MSIL2A_{timestamp}_R008_{tile_id}_20201004T190635"]
        for item_url in item_urls:
            item = pystac.Item.from_file(item_url)
            signed_item = planetary_computer.sign(item)
            for band in ["B08", "B03", "B02", "B04"]:
                asset_href = signed_item.assets[band].href
                filename = urlparse(asset_href).path.split("/")[-1]
                download_url(asset_href, self.dataset_path, filename)

    def get_dataset(self, split, transform, **kwargs):
        dataset = RasterSentinel2(
            self.dataset_path,
            labels_name=self.labels_name,
            split=split,
            task=self.task,
            binary_positive_classes=self.binary_positive_classes,
            transform=transform,
            **self.dataset_kwargs
        )
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_train,
            sampler=self.sampler(self.dataset_train, size=self.size, units=self.units, crs=self.crs),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_val,
            sampler=self.sampler(self.dataset_val, size=self.size, units=self.units, crs=self.crs),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_test,
            sampler=self.sampler(self.dataset_test, size=self.size, units=self.units, crs=self.crs),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_predict,
            sampler=self.sampler(self.dataset_predict, size=self.size, units=self.units),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.2],
                    std=[0.229, 0.224, 0.225, 0.2]
                ),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                transforms.CenterCrop(size=128),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.2],
                    std=[0.229, 0.224, 0.225, 0.2]
                ),
            ]
        )


class RasterSentinel2(RasterTorchGeoDataset):
    """Raster dataset adapted for Sentinel-2 data.

    Inherits RasterTorchGeoDataset.
    """
    filename_glob = "T*_B0*_10m.tif"
    filename_regex = r"T31TEJ_20190801T104031_(?P<band>B0[\d])"
    date_format = "%Y%m%dT%H%M%S"
    is_image = True
    separate_files = True
    all_bands = ["B02", "B03", "B04", "B08"]
    plot_bands = ["B04", "B03", "B02"]

    def plot(
        self,
        sample: Patches
    ) -> Figure:
        """Plot a 3-bands dataset patch (sample).

        Plots a dataset sample by selecting the 3 bands indicated in
        the `plot_bands` variable (in the same order).
        By default, the method plots the RGB bands.

        Parameters
        ----------
        sample : Patches
            dataset's patch to plot

        Returns
        -------
        Figure
            matplotlib figure containing the plot
        """
        # Find the correct band index order
        plot_indices = []
        for band in self.plot_bands:
            plot_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample[plot_indices].permute(1, 2, 0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig


class RasterSentinel2GLC23(RasterSentinel2):
    """Adaptation of RasterSentinel2 for new GLC23 observations."""
    filename_glob = "*.tif"
    filename_regex = r"(?P<band>red|green|blue|nir)_2021"
    all_bands = ["red", "green", "blue", "nir"]
    plot_bands = ["red", "green", "blue"]

    def _load_observation_data(
        self,
        root: Path = None,
        obs_fn: str = None,
        subsets: str = ['train', 'test', 'val'],
    ) -> pd.DataFrame:
        if any([root is None, obs_fn is None]):
            return pd.DataFrame(columns=['longitude', 'latitude', 'species_id', 'subset'])
        labels_fp = obs_fn if len(obs_fn.split('.csv')) >= 2 else f'{obs_fn}.csv'
        labels_fp = root / labels_fp
        df = pd.read_csv(
            labels_fp,
            sep=";",
        )
        df.rename(columns={'lon': 'longitude',
                           'lat': 'latitude',
                           'speciesId': 'species_id',
                           'glcID': 'observation_id'}, inplace=True)
        # df['subset'] = df['test'].apply(lambda x: 'test' if x else 'train')

        # train_inds = np.argwhere(df['subset']=='train')
        # np.random.shuffle(train_inds)
        # val_inds = train_inds[:ceil(len(train_inds) * 0.2)]
        # df.loc[np.ravel(val_inds), 'subset'] = 'val'
        self.unique_labels = np.sort(np.unique(df['species_id']))
        try:
            subsets = [subsets] if isinstance(subsets, str) else subsets
            ind = df.index[df["subset"].isin(subsets)]
            df = df.loc[ind]
        except ValueError as e:
            print('Unrecognized subset name.\n'
                  'Please use one or several amongst: ["train", "test", "val"], as a string or list of strings.\n',
                  {e})
        return df


class Sentinel2GeoSampler(GeoSampler):
    """Custom sampler for RasterSentinel2.

    This custom sampler is used by RasterSentinel2 to query the dataset
    with the fully constructed dictionary. The sampler is passed to and
    used by PyTorch dataloaders in the training/inference workflow.

    Inherits GeoSampler.

    NOTE: this sampler is compatible with any class inheriting
          RasterTorchGeoDataset's `__getitem__` method so the name of
          this sampler may become irrelevant when more dataset-specific
          classes inheriting RasterTorchGeoDataset are created.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: Optional[int] = None,
        roi: Optional[BoundingBox] = None,
        units: str = 'pixel',
        crs: str = 'crs',
    ) -> None:
        super().__init__(dataset, roi)
        self.units = units
        self.crs = crs
        self.size = (size, size) if isinstance(size, (int, float)) else size
        self.coordinates = dataset.coordinates
        self.length = length if length is not None else len(dataset.observation_ids)
        self.observation_ids = dataset.observation_ids.values

    def __iter__(self) -> Iterator[BoundingBox]:
        """Yield a dict to iterate over a RasterTorchGeoDataset dataset.

        Yields
        ------
        Iterator[BoundingBox]
            dataset input query
        """
        for _ in range(len(self)):
            coords = tuple(self.coordinates[_])
            obs_id = self.observation_ids[_]
            yield {'lon': coords[0], 'lat': coords[1],
                   'crs': self.crs,
                   'size': self.size,
                   'units': self.units,
                   'obs_id': obs_id}

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length
