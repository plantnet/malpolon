"""This module provides Sentinel-2 related classes based on torchgeo.

Sentinel-2 data is queried from Microsoft Planetary Computer (MPC).

Author: Theo Larcher <theo.larcher@inria.fr>
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import planetary_computer
import pystac
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torchgeo.datasets.utils import download_url
from torchvision import transforms

from malpolon.data.datasets.torchgeo_datasets import (RasterGeoDataModule,
                                                      RasterTorchGeoDataset)

if TYPE_CHECKING:
    import numpy.typing as npt

    Patches = npt.NDArray
    Targets = npt.NDArray[np.int64]


class Sentinel2TorchGeoDataModule(RasterGeoDataModule):
    """Data module for Sentinel-2A dataset."""
    def __init__(
        self,
        dataset_path: str,
        download_data_sample: bool = False,
        **kwargs,
    ):
        """Class constructor.

        Parameters
        ----------
        dataset_path : str
            path to the directory containing the data
        download_data_sample: bool, optional
            whether to download a sample of Sentinel-2 data, by default False
        """
        super().__init__(dataset_path, **kwargs)
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
