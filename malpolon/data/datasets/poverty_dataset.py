import os
import json
from typing import Callable

import numpy as np
import rasterio
import pandas as pd
from matplotlib import pyplot

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from malpolon.data.data_module import BaseDataModule

import datetime


class PovertyDataModule(BaseDataModule):
    def __init__(
            self,
            tif_dir: str = 'landsat_tiles/',
            dataset_path: str = 'examples/poverty/dataset/',
            labels_name: str = 'observations_2013+.csv',
            train_batch_size: int = 32,
            inference_batch_size: int = 16,
            num_workers: int = 8,
            **kwargs
    ):

        """DataModule for the Poverty dataset.
        Args:
            tif_dir (str): directory containing the tif files
            dataset_path (str): path to the dataset
            labels_name (str): name of the csv file containing the labels
            train_batch_size (int): batch size for training
            inference_batch_size (int): batch size for inference
            num_workers (int): number of workers for the DataLoader
            fold (int): fold to use for training
            transform (torchvision.transforms): transform to apply to the data"""

        super().__init__()
        dataframe = pd.read_csv(dataset_path + labels_name)
        self.dataframe = dataframe
        self.dataframe_train = dataframe[dataframe['subset'] == 'train']
        self.dataframe_val = dataframe[dataframe['subset'] == 'val']
        self.dataframe_test = dataframe[dataframe['subset'] == 'test']
        self.tif_dir = dataset_path + tif_dir
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.dict_normalize = json.load(open('examples/poverty/mean_std_normalize.json', 'r'))
        self.num_workers = num_workers
        self.task = 'regression'

    def train_transform(self) -> Callable:
        return transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=self.dict_normalize['mean'], std=self.dict_normalize['std']),
        ])

    def test_transform(self) -> Callable:
        return transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Normalize(mean=self.dict_normalize['mean'], std=self.dict_normalize['std']),
        ])

    def get_dataset(self, split: str, **kwargs) -> Dataset:
        if split == 'train':
            dataset = MSDataset(self.dataframe_train, self.tif_dir, transform=self.train_transform())
        elif split == 'val':
            dataset = MSDataset(self.dataframe_val, self.tif_dir, transform=self.train_transform())
        elif split == 'test':
            dataset = MSDataset(self.dataframe_test, self.tif_dir, transform=self.test_transform())
        return dataset

    def train_dataloader(self):
        return DataLoader(self.get_train_dataset(), batch_size=self.train_batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.get_val_dataset(), batch_size=self.inference_batch_size, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.get_test_dataset(), batch_size=self.inference_batch_size, num_workers=self.num_workers,
                          persistent_workers=True)


class MSDataset(Dataset):
    """ Dataset returning the LANDSAT tiles and wealth index corresponding to the DHS cluster.
        Rasters were previously downloaded from Earth Engine and stored in the 'landsat_tiles' directory.
        Images contain 8 bands, one of them being a nightlight image. Only the first 7 bands are selected."""

    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            root_dir (string): Directory with all the images.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.targets = dataframe['wealthpooled'].values
        self.observation_ids = dataframe['cluster'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.dataframe.iloc[idx]

        value = row.wealthpooled.astype('float')

        tile_name = os.path.join(self.root_dir,
                                 str(row.country) + "_" + str(row.year),
                                 str(row.cluster) + ".tif"
                                 )

        tile = np.empty([7, 255, 255])

        with rasterio.open(tile_name) as src:

            for band in src.indexes[0:-1]:
                tile[band - 1, :, :] = src.read(band)

        tile = np.nan_to_num(tile)
        tile = self.transform(torch.tensor(tile, dtype=torch.float32))
        value = torch.tensor(value, dtype=torch.float32).unsqueeze(-1)
        return tile, value

    def plot(self, idx, rgb=False, save=True):
        """Plot the tile at the given index.
           Args:
                idx (int): index of the tile to plot
                rgb (bool): if True, plot the RGB image, otherwise plot the 7 bands
                save (bool): if True, save the plot in the 'examples/poverty' directory"""

        tile, value = self.__getitem__(idx)

        tile = tile.numpy()

        if rgb:
            fig, ax = pyplot.subplots(1, 1, figsize=(6, 6))
            img_rgb = tile[0:3, ...][::-1, ...].transpose(1, 2, 0)
            ax.imshow(img_rgb)  #
            ax.axis('off')
            # ax.set_title(f"Value: {value}, RGB")
        else:

            fig, axs = pyplot.subplots(2, 4, figsize=(12, 6))

            for i, ax in enumerate(axs.flat[0:-1]):
                ax.imshow(tile[i, ...], cmap='pink')

                ax.set_title(f"Band: {i}")

        # fig.suptitle(f"Value: {value}")
        if save:
            fig.savefig(f'examples/poverty/plot_{idx}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png')

        pyplot.tight_layout()
        pyplot.show()
        return tile
