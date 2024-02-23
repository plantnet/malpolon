"""This module provides Datasets and Providers for GeoLifeCLEF2023 data.

Author: Benjamin Deneu <benjamin.deneu@inria.fr>
        Theo Larcher <theo.larcher@inria.fr>

License: GPLv3
Python version: 3.10.6
"""

import itertools
import logging
import math
import os
from abc import abstractmethod
from pathlib import Path
from random import random
from typing import Callable, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from malpolon.data.get_jpeg_patches_stats import standardize as jpeg_stand


class PatchesDataset(Dataset):
    """Patches dataset class.

    This class provides a PyTorch-friendly dataset to handle patch data
    and labels. The data can be .jpeg or .tif files of variable depth (i.e.
    multi-spectral data). Each __getitem__ call returns a patch extracted from
    the dataset.

    Args:
        Dataset (Dataset): PyTorch Dataset class.
    """

    def __init__(
        self,
        occurrences: str,
        providers: Iterable,
        transform: Callable = None,
        target_transform: Callable = None,
        id_name: str = "glcID",
        label_name: str = "speciesId",
        item_columns: Iterable = ['lat', 'lon', 'patchID'],
    ):
        """Class constructor.

        Parameters
        ----------
        occurrences : str
            path to the occurrences (observations) file
        providers : Iterable
            list of providers to extract patches from
        transform : Callable, optional
            data transform function passed as callable, by default None
        target_transform : Callable, optional
            labels transform function passed as callable, by default
            None
        id_name : str, optional
            observation id name, by default "glcID"
        label_name : str, optional
            name of the species label in the observation file,
            by default "speciesId"
        item_columns : Iterable, optional
            columns to keep (by names) for further usage,
            by default ['lat', 'lon', 'patchID']
        """
        self.occurences = Path(occurrences)
        self.base_providers = providers
        self.transform = transform
        self.target_transform = target_transform
        self.provider = MetaPatchProvider(self.base_providers, self.transform)

        df = pd.read_csv(self.occurences, sep=";",
                         header='infer', low_memory=False)

        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values

    def __len__(self):
        """Return the size of the dataset.

        Returns:
            (int): number of occurrences.
        """
        return len(self.observation_ids)

    def __getitem__(self, index):
        """Return a dataset element.

        Returns an element from a dataset id (0 to n) with its label.

        Args:
            index (int): dataset id.

        Returns:
            (tuple): tuple of data patch (tensor) and label (int).
        """
        item = self.items.iloc[index].to_dict()

        patch = self.provider[item]
        if not torch.is_tensor(patch):
            patch = torch.from_numpy(patch).float()

        target = self.targets[index]
        if self.target_transform:
            target = self.target_transform(target)

        return patch, target

    def plot_patch(self, index):
        """Plot a patch based on a dataset id.

        Calls the plot method of the parent provider class through
        MetaPatchProvider, with a dataset int index.

        Args:
            index (int): dataset index.
        """
        item = self.items.iloc[index].to_dict()
        self.provider.plot_patch(item)


class PatchesDatasetMultiLabel(PatchesDataset):
    """Multilabel patches dataset.

    Like PatchesDataset but provides one-hot encoded labels.

    Args:
        PatchesDataset (PatchesDataset): pytorch friendly dataset class.
    """

    def __init__(
        self,
        occurrences: str,
        providers: Iterable,
        n_classes: Union[int, str] = 'max',
        id_getitem: str = 'patchId',
        **kwargs
    ):
        """Class constructor.

        Parameters
        ----------
        occurrences : str
            path to the occurrences (observations) file
        providers : Iterable
            list of providers to extract patches from
        n_classes : str, optional
            Number of classes. 3 options are available:
            - 'max' : infered from maximum value of the species id column
            - 'length' : infered from the number of unique species id
            - int : defined by user input
            By default 'max'.
        id_getitem : str, optional
            column id to query the multiple observations by,
            by default 'patchID'

        Raises
        ------
        ValueError
            raises a ValueError if n_classes is not 1 of the 3 options
        """
        super().__init__(occurrences, providers, **kwargs)
        self.id_getitem = id_getitem
        self.targets_sorted = []
        self.observation_ids = np.unique(self.observation_ids)
        match n_classes:
            case 'max':
                self.n_classes = np.max(self.targets) + 1
            case 'length':
                self.n_classes = len(np.unique(self.targets))
            case _ if isinstance(n_classes, int):
                self.n_classes = n_classes
            case _:
                raise ValueError('n_classes must be "max", "length" or an integer')

    def __getitem__(self, index):
        """Return a dataset element.

        Returns an element from a dataset id (0 to n) with the labels in
        one-hot encoding.

        Args:
            index (int): dataset id.

        Returns:
            (tuple): tuple of data patch (tensor) and labels (list).
        """
        item = self.items.iloc[index].to_dict()
        pid_rows_i = self.items[self.items[self.id_getitem] == item[self.id_getitem]].index
        self.targets_sorted = np.unique(self.targets)

        patch = self.provider[item]
        if not torch.is_tensor(patch):
            patch = torch.from_numpy(patch).float()

        targets = np.zeros(self.n_classes)
        for idx in pid_rows_i:
            target = self.targets[idx]
            targets[np.where(self.targets_sorted == target)] = 1
        targets = torch.from_numpy(targets)

        return patch, targets


class TimeSeriesDataset(Dataset):
    """Time series dataset.

    Like PatchesDataset but adapted to time series data which is formatted as
    .csv files where each row is an occurrence and each column is a timestamp.
    Timeseries present one additional dimension compared to visual patches.
    Timeseries do not all have the same size and have a parametrable
    no_data_value argument.

    Args:
        Dataset (Dataset): pytorch Dataset class.
    """

    def __init__(
        self,
        occurrences: str,
        providers: Iterable,
        transform: Callable = None,
        target_transform: Callable = None,
        id_name: str = "glcID",
        label_name: str = "speciesId",
        item_columns: Iterable = ['timeSerieID'],
    ):
        """Class constructor.

        Parameters
        ----------
        occurrences : str
            path to the occurrences (observations) file
        providers : Iterable
            list of providers to extract patches from
        transform : Callable, optional
            data transform function passed as callable, by default None
        target_transform : Callable, optional
            labels transform function passed as callable, by default
            None
        id_name : str, optional
            observation id name, by default "glcID"
        label_name : str, optional
            name of the species label in the observation file,
            by default "speciesId"
        item_columns : Iterable, optional
            columns to keep (by names) for further usage,
            by default ['timeSerieID']
        """
        self.occurences = Path(occurrences)
        self.base_providers = providers
        self.transform = transform
        self.target_transform = target_transform
        self.provider = MetaTimeSeriesProvider(self.base_providers,
                                               self.transform)

        df = pd.read_csv(self.occurences, sep=";",
                         header='infer', low_memory=False)

        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values

    def __len__(self):
        """Return the size of the dataset.

        Returns:
            (int): number of occurrences.
        """
        return len(self.observation_ids)

    def __getitem__(self, index):
        """Return a time series.

        Returns a tuple of time series data and its label.
        The data in in the following shape :
        [1, n_bands, max_length_ts]

        Args:
            index (int): dataset id.

        Returns:
            (tuple): tuple of time series data (tensor) and labels (list).
        """
        item = self.items.iloc[index].to_dict()

        patch = self.provider[item]
        if not torch.is_tensor(patch):
            patch = torch.from_numpy(patch).float()

        target = self.targets[index]
        if self.target_transform:
            target = self.target_transform(target)

        return patch, target

    def plot_ts(self, index):
        """Plot a time series occurrence.

        Calls the plot method of the time series provider through
        MetaPatchProvider, with a dataset int index.

        Args:
            index (int): data index.
        """
        item = self.items.iloc[index].to_dict()
        self.provider.plot_ts(item)


class PatchProvider():
    """Parent class for all GLC23 patch data providers.

    This class implements common implemented & abstract methods used by
    all patch providers. Particularly, the plot method is designed to
    accommodate all cases of the GLC23 patch datasets.
    """
    def __init__(
        self,
        size: int = 128,
        normalize: bool = False
    ) -> None:
        """Class constructor.

        Parameters
        ----------
        size : int
            patch extraction size, by default 128
        normalize : bool, optional
            If True, performs (mean, std) normalization over each raster
            individually or over all pre-extracted patches.
            By default False.
        """
        self.patch_size = size
        self.normalize = normalize
        self.nb_layers = 0
        self.bands_names = [''] * self.nb_layers

    @abstractmethod
    def __getitem__(self, item):
        """Return a patch (parent class)."""

    def __repr__(self):
        """Represent the class.

        Returns:
            (str): class representation.
        """
        return self.__str__()

    @abstractmethod
    def __str__(self):
        """Return descriptive information about the clss.

        Returns:
            (str): class description.
        """

    def __len__(self):
        """Return the size of the provider.

        Returns:
            (int): number of layers (depth).
        """
        return self.nb_layers

    def plot_patch(self, item):
        """Plot all layers of a given patch.

        A patch is selected based on a key matching the associated
        provider's __get__() method.

        Args:
            item (dict): provider's get index.
        """
        patch = self[item]
        if self.nb_layers == 1:
            plt.figure(figsize=(10, 10))
            plt.imshow(patch[0])
        else:
            # calculate the number of rows and columns for the subplots grid
            rows = int(math.ceil(math.sqrt(self.nb_layers)))
            cols = int(math.ceil(self.nb_layers / rows))

            # create a figure with a grid of subplots
            fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

            # flatten the subplots array to easily access the subplots
            axs = axs.flatten()

            # loop through the layers of patch data
            for i in range(self.nb_layers):
                # display the layer on the corresponding subplot
                axs[i].imshow(patch[i])
                axs[i].set_title(f'layer_{i}: {self.bands_names[i]}')
                axs[i].axis('off')

            # remove empty subplots
            for i in range(self.nb_layers, rows * cols):
                fig.delaxes(axs[i])

        plt.suptitle('Tensor for item: ' + str(item), fontsize=16)

        # show the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


class MetaPatchProvider(PatchProvider):
    """Parent class for patch providers.

    This class interfaces patch providers with patch datasets.

    Args:
        (PatchProvider): inherits PatchProvider.
    """
    def __init__(
        self,
        providers: Iterable[Callable],
        transform: Callable = None
    ):
        """Class constructor.

        Parameters
        ----------
        providers : Iterable[Callable]
            list of providers to extract patches from
        transform : Callable, optional
            transform function to apply on the patches, by default None
        """
        super().__init__(0, None)
        self.providers = providers
        self.nb_layers = np.sum([len(provider) for provider in self.providers])
        self.bands_names = list(itertools.chain.from_iterable([provider.bands_names for provider in self.providers]))
        self.transform = transform

    def __getitem__(self, item):
        """Return a patch.

        This getter is used by a patch dataset class and calls each provider's
        getter method to return concatenated patches.

        Args:
            item (dict): provider index.

        Returns:
            (array): concatenaned patch from all providers.
        """
        patch = np.concatenate([provider[item] for provider in self.providers])

        if self.transform:
            patch = self.transform(torch.from_numpy(patch).float())

        return patch

    def __str__(self):
        """Return descriptive information about the clss.

        Returns:
            (str): class description.
        """
        result = 'Providers:\n'
        for provider in self.providers:
            result += str(provider)
            result += '\n'
        return result


class RasterPatchProvider(PatchProvider):
    """Patch provider for .tif raster files.

    This class handles rasters stored as .tif files and returns
    patches from them based on a dict key containing (lon, lat)
    coordinates.

    Args:
        (PatchProvider): inherits PatchProvider.
    """
    def __init__(
        self,
        raster_path: str,
        size: int = 128,
        spatial_noise: int = 0,
        normalize: bool = True,
        fill_zero_if_error: bool = False,
        nan_value: Union[int, float] = 0
    ):
        """Class constructor.

        Parameters
        ----------
        raster_path : str
            path to the .tif raster file
        size : int, optional
            size of the patches to extract, by default 128
        spatial_noise : int, optional
            Data augmentation technique that shifts the patch center
            (observation (lon, lat)) by a random value between 0 and 1.
            The noise is computed as follows :
            `x * 2 * spatial_noise - spatial_noise` where `x` is a random
            value between 0 and 1.
            The noise is then added to the patch center coordinates.
            By default 0
        normalize : bool, optional
            if True performs (mean, std) normalization raster-wise,
            by default True
        fill_zero_if_error : bool, optional
            if the output patch tensor is not of the expected size
            (nb_layers, size, size), fills the patch with zeros instead,
            by default False
        nan_value : Union[int, float], optional
            value to replace NaN values with, by default 0
        """
        super().__init__(size, normalize)
        self.spatial_noise = spatial_noise
        self.fill_zero_if_error = fill_zero_if_error
        self.transformer = None
        self.name = os.path.basename(os.path.splitext(raster_path)[0])
        self.normalize = normalize

        # open the tif file with rasterio
        with rasterio.open(raster_path) as src:
            # read the metadata of the file
            meta = src.meta
            # update the count of the meta to match the number of layers
            meta.update(count=src.count)

            # read the data from the raster
            self.data = src.read()

            # get the NoData value from the raster
            self.nodata_value = src.nodatavals

            # iterate through all the layers
            for i in range(src.count):
                # replace the NoData values with np.nan
                self.data = self.data.astype(float)
                self.data[i] = np.where(self.data[i] == self.nodata_value[i],
                                        np.nan,
                                        self.data[i])
                if self.normalize:
                    self.data[i] = (self.data[i] - np.nanmean(self.data[i])) / np.nanstd(self.data[i])
                self.data[i] = np.where(np.isnan(self.data[i]),
                                        nan_value,
                                        self.data[i])

            self.nb_layers = src.count

            self.x_min = src.bounds.left
            self.y_min = src.bounds.bottom
            self.x_resolution = src.res[0]
            self.y_resolution = src.res[1]
            self.n_rows = src.height
            self.n_cols = src.width
            self.crs = src.crs
        if self.nb_layers > 1:
            self.bands_names = [self.name + '_' + str(i + 1) for i in range(self.nb_layers)]
        else:
            self.bands_names = [self.name]

        self.epsg = self.crs.to_epsg()
        if self.epsg != 4326:
            # create a pyproj transformer object to convert lat, lon to EPSG:32738
            self.transformer = pyproj.Transformer.from_crs("epsg:4326",
                                                           self.epsg,
                                                           always_xy=True)

    def __getitem__(self, item):
        """Return a patch from a .tif raster.

        This getter returns a patch of size self.size from a .tif raster
        loaded in self.data using GPS coordinates projected in EPSG:4326.

        Args:
            item (dict): dictionary containing at least latitude and longitude
                         keys ({'lat': lat, 'lon':lon})

        Returns:
            (array): the environmental vector (size>1 or size=1).
        """
        # convert the lat, lon coordinates to EPSG:32738
        if self.transformer:
            lon, lat = self.transformer.transform(item['lon'], item['lat'])  # pylint: disable=unpacking-non-sequence
        else:
            (lon, lat) = (item['lon'], item['lat'])

        # add noise as data augmentation
        if self.spatial_noise > 0:
            lon = lon + ((random() * 2 * self.spatial_noise) - self.spatial_noise)
            lat = lat + ((random() * 2 * self.spatial_noise) - self.spatial_noise)

        # calculate the x, y coordinate of the point of interest
        x = int(self.n_rows - (lat - self.y_min) / self.y_resolution)
        y = int((lon - self.x_min) / self.x_resolution)

        # read the data of the patch from all layers
        if self.patch_size == 1:
            patch_data = [self.data[i, x, y] for i in range(self.nb_layers)]
        else:
            patch_data = [self.data[i, x - (self.patch_size // 2): x + (self.patch_size // 2), y - (self.patch_size // 2): y + (self.patch_size // 2)] for i in range(self.nb_layers)]

        tensor = np.concatenate([patch[np.newaxis] for patch in patch_data])
        if self.fill_zero_if_error and tensor.shape != (self.nb_layers, self.patch_size, self.patch_size):
            tensor = np.zeros((self.nb_layers, self.patch_size, self.patch_size))
        return tensor

    def __str__(self):
        """Return descriptive information about the clss.

        Returns:
            (str): class description.
        """
        result = '-' * 50 + '\n'
        result += 'n_layers: ' + str(self.nb_layers) + '\n'
        result += 'x_min: ' + str(self.x_min) + '\n'
        result += 'y_min: ' + str(self.y_min) + '\n'
        result += 'x_resolution: ' + str(self.x_resolution) + '\n'
        result += 'y_resolution: ' + str(self.y_resolution) + '\n'
        result += 'n_rows: ' + str(self.n_rows) + '\n'
        result += 'n_cols: ' + str(self.n_cols) + '\n'
        result += '-' * 50
        return result


class MultipleRasterPatchProvider(PatchProvider):
    """Patch provider for multiple PatchProvider.

    This provider is useful when having to load several patch modalities
    through RasterPatchProvider by selecting to desired data in the 'select'
    argument of the constructor.

    Args:
        PatchProvider (_type_): _description_
    """
    def __init__(
        self,
        rasters_folder: str,
        select: Iterable[str] = None,
        **kwargs,
    ):
        """Class constructor.

        Parameters
        ----------
        rasters_folder : str
            path to the folder containing the rasters
        select : Iterable[str], optional
            List of rasters prefix names. Only the rasters containing
            those names will be loaded.
            By default None
        """
        super().__init__(**kwargs)
        files = os.listdir(rasters_folder)
        if select:
            rasters_paths = [r + '.tif' for r in select]
        else:
            rasters_paths = [f for f in files if f.endswith('.tif')]
        self.rasters_providers = []

        for path in tqdm(rasters_paths):
            self.rasters_providers.append(RasterPatchProvider(rasters_folder + path, **kwargs))
        self.nb_layers = np.sum([len(raster) for raster in self.rasters_providers])
        self.bands_names = list(itertools.chain.from_iterable([raster.bands_names for raster in self.rasters_providers]))

    def __getitem__(self, item):
        """Return multiple patches.

        Returns multiple patches from multiple raster providers in a
        3-dimensional numpy array.

        Args:
            item (dict): providers index.

        Returns:
            (array): array of patches.
        """
        return np.concatenate([raster[item] for raster in self.rasters_providers])

    def __str__(self):
        """Return descriptive information about the clss.

        Returns:
            (str): class description.
        """
        result = 'Rasters in folder:\n'
        for raster in self.rasters_providers:
            result += str(raster)
            result += '\n'
        return result


class JpegPatchProvider(PatchProvider):
    """JPEG patches provider for GLC23.

    Provides tensors of multi-modal patches from JPEG patch files
    of rasters of the GLC23 challenge.

    Attributes:
        (PatchProvider): inherits PatchProvider.
    """

    def __init__(
        self,
        root_path: str,
        select: Iterable[str] = None,
        normalize: bool = False,
        patch_transform: Callable = None,
        size: int = 128,
        dataset_stats: str = 'jpeg_patches_stats.csv',
        id_getitem: str = 'patchID'
    ):
        """Class constructor.

        Parameters
        ----------
        root_path : str
            root path to the directory containg all patches modalities
        select : Iterable[str], optional
            List of rasters prefix names. Only the rasters containing
            those names will be loaded.
            by default None
        normalize : bool, optional
            normalize patches, by default False
        patch_transform : Iterable[Callable], optional
            custom transformation function, by default None
        size : int, optional
            size of the patches to extract, by default 128
        dataset_stats : str, optional
            Path to a csv file containing mean and std values of the
            patches dataset which can be used to normalize the data
            (depending on argument `normalize`). If the file does not
            exists, the method will attempt to compute it: WARNING,
            THIS CAN TAKE A VERY BIG AMOUNT OF TIME.
            By default 'jpeg_patches_stats.csv'
        id_getitem : str, optional
            Labels column id on which are based the organization of
            patches in folders and sub-folders.
            Patches should be organized in the following way:
            root_path/YZ/WX/patchID.jpeg with patchId being the value
            ABCDWXYZ.
            By default 'patchID'
        """
        super().__init__(size, normalize)
        self.patch_transform = patch_transform
        self.root_path = root_path
        self.ext = '.jpeg'
        self.n_rows = 0
        self.n_cols = 0
        self.dataset_stats = os.path.join(self.root_path, dataset_stats)
        self.id_getitem = id_getitem

        self.channel_folder = {'red': 'rgb', 'green': 'rgb', 'blue': 'rgb',
                               'swir1': 'swir1',
                               'swir2': 'swir2',
                               'nir': 'nir'}
        if not select:
            sub_dirs = next(os.walk(root_path))[1]
            select = [k for k, v in self.channel_folder.items() if v in sub_dirs]

        self.channels = [c.lower() for c in select]
        self.nb_layers = len(self.channels)
        self.bands_names = self.channels

    def __getitem__(self, item):
        """Return a tensor composed of every channels of a jpeg patch.

        Looks for every spectral bands listed in self.channels and returns
        a 3-dimensionnal patch concatenated in 1 tensor. The index used to
        query the right patch is a dictionnary with at least one key/value
        pair : {'patchID', <patchID_value>}.

        Args:
            item (dict): dictionnary containing the patchID necessary to
                         identify the jpeg patch to return.

        Raises:
            KeyError: the 'patchID' key is missing from item
            Exception: item is not a dictionnary as expected

        Returns:
            (tensor): multi-channel patch tensor.
        """
        try:
            id_ = str(int(item[self.id_getitem]))
        except KeyError as e:
            raise KeyError(f'The {self.id_getitem} key does not exists.') from e

        # folders that contain patches
        sub_folder_1 = id_[-2:]
        sub_folder_2 = id_[-4:-2]
        list_tensor = {'order': [], 'tensors': []}

        for channel in self.channels:
            if channel not in list_tensor['order']:
                path = os.path.join(self.root_path,
                                    self.channel_folder[channel],
                                    sub_folder_1, sub_folder_2, id_ + self.ext)
                try:
                    img = np.asarray(Image.open(path))
                    if set(['red', 'green', 'blue']).issubset(self.channels) and channel in ['red', 'green', 'blue']:
                        img = img.transpose((2, 0, 1))
                        list_tensor['order'].extend(['red', 'green', 'blue'])
                    else:
                        if channel in ['red', 'green', 'blue']:
                            img = img[:, :, 'rgb'.find(channel[0])]
                        img = np.expand_dims(img, axis=0)
                        list_tensor['order'].append(channel)
                except ImportError as e:
                    logging.critical('%s\nCould not open %s properly.'
                                     ' Setting array to 0.', e.msg, path)
                    img = np.zeros((1, self.patch_size, self.patch_size))
                    list_tensor['order'].append(channel)
                if self.normalize:
                    if os.path.isfile(self.dataset_stats):
                        df = pd.read_csv(self.dataset_stats, sep=';')
                        mean, std = df.loc[0, 'mean'], df.loc[0, 'std']
                    else:
                        mean, std = jpeg_stand(self.root_path,
                                               [self.ext],
                                               output=self.dataset_stats)
                    img = (img - mean) / std
                for depth in img:
                    list_tensor['tensors'].append(np.expand_dims(depth, axis=0))
        tensor = np.concatenate(list_tensor['tensors'])
        if self.patch_transform:
            for transform in self.patch_transform:
                tensor = transform(tensor)
        self.channels = list_tensor['order']
        self.n_rows = img.shape[1]
        self.n_cols = img.shape[2]
        return tensor

    def __str__(self):
        """Return descriptive information about the clss.

        Returns:
            (str): class description.
        """
        result = '-' * 50 + '\n'
        result += 'n_layers: ' + str(self.nb_layers) + '\n'
        result += 'n_rows: ' + str(self.n_rows) + '\n'
        result += 'n_cols: ' + str(self.n_cols) + '\n'
        result += '-' * 50
        return result


class TimeSeriesProvider():
    """Provide time series data.

    This provider is the parent class of time series providers.
    It handles time series data stored as .csv files where each file
    has values for a single spectral band (red, green, infra-red etc...).
    """
    def __init__(
        self,
        root_path: str,
        eos_replace_value: Union[int, float] = -1,
        transform: Iterable[Callable] = None
    ) -> None:
        """Class constructor.

        Parameters
        ----------
        root_path : str
            path to the directory containing the time series data
        transform : Iterable[Callable], optional
            list of transformation functions to apply on the time series,
            by default None
        eos_replace_value : Union[int, float], optional
            _description_, by default -1
        """
        self.root_path = root_path
        self.nb_layers = 0
        self.min_sequence = 0
        self.max_sequence = 0
        self.bands_names = ['']
        self.layers_length = 0
        self.eos_replace_value = eos_replace_value
        self.features_col = []
        self.transform = transform

    @abstractmethod
    def __getitem__(self, item):
        """Return a time series (parent class)."""

    def __repr__(self):
        """Represent the class.

        Returns:
            (str): class representation.
        """
        return self.__str__()

    @abstractmethod
    def __str__(self):
        """Return descriptive information about the clss.

        Returns:
            (str): class description.
        """
        result = f"{'-' * 50} \n"
        result += f'nb_layers: {self.nb_layers}\n'
        result += f'min_sequence: {self.min_sequence}\n'
        result += f'max_sequence: {self.max_sequence}\n'
        result += f'bands_names: {self.bands_names}\n'
        result += '-' * 50
        return result

    def __len__(self):
        """Return the size of the provider.

        Returns:
            (int): number of layers (depth).
        """
        return self.nb_layers

    def plot_ts(self, item):
        """Plot one or more time series in a graphs.

        This method plots the times series data for each modality (spectral
        band) in graphs, for a given occurrence.
        The abciss values are timestamps; ordinate values are the data of the
        occurrence for the corresponding modality.

        An occurrence is identified by a unique timeSerieID value constructed
        from a geographical location (lon, lat) and a quarterly time stamp.
        The occurrence can be linked to the label file via timeSerieID.

        Args:
            item (dict): time series index.
        """
        tss = self[item]
        if self.nb_layers == 1:
            plt.figure(figsize=(20, 20))
            ts_ = tss
            eos_start_ind = np.where(ts_[0, 0] == self.eos_replace_value)[0]
            eos_start_ind = eos_start_ind[0] if eos_start_ind != [] else ts_.shape[2]
            plt.plot(range(eos_start_ind), ts_[0, 0, :eos_start_ind],
                     '-.', c='blue', marker='+')
            plt.plot(range(eos_start_ind, ts_.shape[2]), ts_[0, 0, eos_start_ind:],
                     '', c='red', marker='+')
            plt.title(f'layer: {self.bands_names}\n{item}')
            plt.xticks(list(range(ts_.shape[2]))[::4] + [ts_.shape[2] - 1],
                       self.features_col[::4] + [self.features_col[-1]],
                       rotation='vertical')
            plt.xlabel('Time (quarterly composites)')
            plt.ylabel('Band composite value (uint8)')
            plt.grid(True)
        else:
            # calculate the number of rows and columns for the subplots grid
            rows = int(math.ceil(math.sqrt(self.nb_layers)))
            cols = int(math.ceil(self.nb_layers / rows))

            # create a figure with a grid of subplots
            fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

            # flatten the subplots array to easily access the subplots
            axs = axs.flatten()

            lli = np.cumsum([0] + self.layers_length)

            # Case of calling plot_ts directly on MultipleTimeSeriesProvider instead of MetaTimeSeriesProvider
            if not isinstance(self.eos_replace_value, list):
                self.eos_replace_value = [self.eos_replace_value] * self.nb_layers
            features_col = self.features_col
            if len(np.array(self.features_col).shape) <= 1:
                features_col = [features_col] * self.nb_layers

            # loop through the layers of tss data
            for i in range(self.nb_layers):
                ts_ = tss[0, i]

                k_provider = np.argwhere(i + 1 > lli)
                k_provider = 0 if k_provider.shape[0] == 0 else k_provider[-1][0]

                eos_start_ind = np.where(ts_ == self.eos_replace_value[i])[0]
                eos_start_ind = eos_start_ind[0] if len(eos_start_ind) > 0 else ts_.shape[0] - 1
                # display the layer on the corresponding subplot
                axs[i].plot(range(eos_start_ind), ts_[:eos_start_ind],
                            '-.', c='blue', marker='+')
                axs[i].plot(range(eos_start_ind, ts_.shape[0]), ts_[eos_start_ind:],
                            '', c='red', marker='+')
                axs[i].set_title(f'layer: {self.bands_names[i]}\n{item}')
                axs[i].set_xticks(list(range(ts_.shape[0]))[::4] + [ts_.shape[0] - 1],
                                  features_col[k_provider][::4] + [features_col[k_provider][-1]],
                                  rotation='vertical')
                axs[i].set_xlabel('Time (quarterly composites)')
                axs[i].set_ylabel('Band composite value (uint8)')
                axs[i].grid(True)

            # remove empty subplots
            for i in range(self.nb_layers, rows * cols):
                fig.delaxes(axs[i])

        # show the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


class MetaTimeSeriesProvider(TimeSeriesProvider):
    """Time Series provider called by TimeSeriesDataset to handle TS providers.

    This TS provider handles all TS providers passed in TimeSeriesDataset to
    provide multi-modal time series objects.

    Args:
        (TimeSeriesProvider) : inherits TimeSeriesProvider.
    """
    def __init__(
        self,
        providers: Iterable[Callable],
        transform: Iterable[Callable] = None
    ):
        """Class constructor.

        Parameters
        ----------
        providers : Iterable[Callable]
            list of providers to extract patches from
        transform : Iterable[Callable], optional
            list of transform functions to apply on the time series,
            by default None
        """
        super().__init__('', True, transform)
        self.providers = providers
        self.layers_length = [provider.nb_layers for provider in self.providers]
        self.nb_layers = sum(self.layers_length)
        self.bands_names = list(itertools.chain.from_iterable([provider.bands_names for provider in self.providers]))
        self.features_col = [provider.features_col for provider in self.providers]
        self.eos_replace_value = []
        for provider, ll_ in zip(self.providers, self.layers_length):
            self.eos_replace_value.extend([provider.eos_replace_value] * ll_)

    def __getitem__(self, item):
        """Return the time series from all TS providers.

        This getter is called by a TimeSeriesDataset and returns a 3-dimensional
        array containing time series from all providers. The time series is
        select from the TS index `item` which is a dictionary containing at
        least 1 key/value pair : {'timeSerieID': <timeSerieID_value>}.

        Args:
            item (dict): time series index.

        Returns:
            (array): array containing the time series from all TS providers.
        """
        patch = np.concatenate([provider[item] for provider in self.providers], axis=1)
        if self.transform:
            patch = self.transform(torch.from_numpy(patch).float())

        return patch

    def __str__(self):
        """Return descriptive information about the clss.

        Returns:
            (str): class description.
        """
        result = 'Providers:\n'
        for provider in self.providers:
            result += str(provider)
            result += '\n'
        return result


class CSVTimeSeriesProvider(TimeSeriesProvider):
    """Implement TimeSeriesProvider for .csv time series.

    Only loads time series from a single .csv file.
    If the time series of an occurrence is smaller than the longest one in
    the .csv file, the remaining columns are filled with the 'eos' string.

    Args:
        (TimeSeriesProvider) : inherits TimeSeriesProvider.
    """
    def __init__(
        self,
        ts_data_path: str,
        normalize: bool = False,
        ts_id: str = 'timeSerieID',
        features_col: list = [],
        eos_replace_value: Union[int, float] = -1,
        transform: Iterable[Callable] = None
    ) -> None:
        """Class constructor.

        Parameters
        ----------
        ts_data_path : str
            path to the .csv file containing the time series data
        normalize : bool, optional
            normalize the time series data, by default False
        ts_id : str, optional
            time series id, by default 'timeSerieID'
        features_col : list, optional
            time series columns to keep (if empty list then keeps all),
            by default []
        eos_replace_value : Iterable[int, float], optional
            value to replace 'eos' string in the time series data,
            by default -1
        transform : Iterable[Callable], optional
            list of transformation functions to apply on the time series,
            by default None

        Raises
        ------
        KeyError
            raises a key error if some values in `features_col`
            do not match the `ts_data` column names.
        """
        super().__init__(ts_data_path, normalize, transform)
        self.ts_id = ts_id
        self.ts_data_path = ts_data_path
        self.ts_data = pd.read_csv(ts_data_path, sep=';').set_index(self.ts_id, drop=False)
        self.ts_data = self.ts_data.replace('eos', eos_replace_value).astype(np.int16)
        self.eos_replace_value = eos_replace_value
        self.max_sequence, self.min_sequence = self.get_min_max_sequence(eos_replace_value)
        self.nb_layers = 1
        if not features_col:
            self.features_col = list(self.ts_data.columns[1:])
        elif len(set(features_col).intersection(self.ts_data.columns)) == len(features_col):
            self.features_col = features_col
            self.ts_data = self.ts_data[features_col]
        else:
            raise KeyError('Some values in `features_col` do not match the `ts_data` column names.')
        self.bands_names = [os.path.basename(os.path.splitext(ts_data_path)[0])]

    def get_min_max_sequence(self, eos_replace_value):
        """Determine the size of smallest and longest of all time series.

        Args:
            eos_replace_value (float): replaces all 'eos' string values in the
                                       time series with the given value.

        Returns:
            (tuple): min and max lengths of time series.
        """
        min_seq = len(self.ts_data.columns)
        max_row = self.ts_data.loc[(self.ts_data == eos_replace_value).sum(axis=1).idxmax()]
        max_seq = (max_row != eos_replace_value).sum()
        return min_seq, max_seq

    def __getitem__(self, item):
        """Return a time series.

        This getter returns a time series occurrence in tensor fashionned array,
        based on the `item` index which is a dictionnary containing at least
        1 key/value pair : {'timeSeriesID': <timeSeriesID_value>}.

        Args:
            item (dict): time series index.

        Returns:
            (array): time series occurrence.
        """
        # coordinates must be GPS-like, in the 4326 CRS
        tensor = np.array([self.ts_data.loc[item[self.ts_id], self.features_col]])
        tensor = np.expand_dims(tensor, axis=0)
        if self.transform:
            for transform in self.transform:
                tensor = transform(tensor)
        return tensor

    def __len__(self):
        """Return the size of the time series.

        Returns:
            (int): length of biggest sequence.
        """
        return self.max_sequence

    def __str__(self):
        """Return descriptive information about the clss.

        Returns:
            (str): class description.
        """
        result = '-' * 50 + '\n'
        result += 'provider: MultipleCSVTimeSeriesProvider\n'
        result += 'root_path: ' + str(self.root_path) + '\n'
        result += 'n_layers: ' + str(self.nb_layers) + '\n'
        result += 'ts_id: ' + str(self.ts_id) + '\n'
        result += 'eos_replace_value: ' + str(self.eos_replace_value) + '\n'
        result += 'max_sequence: ' + str(self.max_sequence) + '\n'
        result += 'min_sequence: ' + str(self.min_sequence) + '\n'
        result += '-' * 50
        return result


class MultipleCSVTimeSeriesProvider(TimeSeriesProvider):
    """Like CSVTimeSeriesProvider but with several .csv files.

    Args:
        (TimeSeriesProvider) : inherits TimeSeriesProvider
    """
    # Be careful not to place the label .csv with the data .csv and leaving
    # select=None as the provider would then list all .csv files as data including the label file.
    def __init__(
        self,
        root_path: str,
        select: list = [],  # ['red', 'green', 'blue', 'ir', 'swir1', 'swir2']
        normalize: bool = False,
        ts_id: str = 'timeSerieID',
        features_col: list = [],
        eos_replace_value: Union[int, float] = -1,
        transform: Iterable[Callable] = None
    ) -> None:
        """Class constructor.

        Parameters
        ----------
        root_path : str
            path to the directory containing the time series data
        select : list, optional
            list of time series files to load,
            by default []
        ts_id : str, optional
            time series id,
            by default 'timeSerieID'
        features_col : list, optional
            time series columns to keep (if empty list then keeps all),
            by default []
        eos_replace_value : Union[int, float], optional
            value to replace 'eos' string in the time series data,
            by default -1
        transform : Iterable[Callable], optional
            list of transformation functions to apply on the time series,
            by default None
        """
        super().__init__(root_path, normalize, transform)
        self.root_path = root_path
        self.ts_id = ts_id
        self.eos_replace_value = eos_replace_value
        self.select = [c.lower() for c in select]

        files = os.listdir(root_path)
        ts_paths = [f for f in files if f.endswith('.csv')]
        if select:
            select = [f'time_series_{r}.csv' for r in select]
            ts_paths = [r for r in files if r in select]
            if len(ts_paths) != len(select):
                logging.warning('Could not find all files based on `select`.'
                                ' Loading only the ones which names match...'
                                '(see the `ts_paths` attribute for the'
                                ' complete list)')
        self.ts_paths = ts_paths
        self.ts_providers = [CSVTimeSeriesProvider(root_path + path,
                                                   normalize=normalize,
                                                   ts_id=ts_id,
                                                   features_col=features_col,
                                                   eos_replace_value=eos_replace_value,
                                                   transform=transform) for path in ts_paths]
        self.nb_layers = len(self.ts_providers)
        self.layers_length = [provider.nb_layers for provider in self.ts_providers]
        self.bands_names = list(itertools.chain.from_iterable([provider.bands_names for provider in self.ts_providers]))
        self.min_sequence = min(list(itertools.chain.from_iterable([[ts_.min_sequence] for ts_ in self.ts_providers])))
        self.max_sequence = max(list(itertools.chain.from_iterable([[ts_.max_sequence] for ts_ in self.ts_providers])))
        self.features_col = self.ts_providers[0].features_col

    def __getitem__(self, item):
        """Return multiple time series.

        Like CSVTimeSeriesProvider but concatenantes the time series in a
        tensor fashioned 3-dimensional array to provide a multi-modal time
        series array. This array is of the shape :
        (1, n_modalities, max_sequence_length)

        Args:
            item (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.concatenate([ts_[item] for ts_ in self.ts_providers], axis=1)

    def __str__(self):
        """Return descriptive information about the clss.

        Returns:
            (str): class description.
        """
        result = '-' * 50 + '\n'
        result += 'provider: MultipleCSVTimeSeriesProvider\n'
        result += 'root_path: ' + str(self.root_path) + '\n'
        result += 'n_layers: ' + str(self.nb_layers) + '\n'
        result += 'ts_id: ' + str(self.ts_id) + '\n'
        result += 'eos_replace_value: ' + str(self.eos_replace_value) + '\n'
        result += 'modalities: ' + str(self.select) + '\n'
        result += '-' * 50
        return result
