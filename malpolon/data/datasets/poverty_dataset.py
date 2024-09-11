import os
import json
import sys
from typing import Callable, Any, Union
from pathlib import Path

import numpy as np
import rasterio
import pandas as pd
from matplotlib import pyplot

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# Force work with the malpolon GitHub package localized at the root of the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from malpolon.data.data_module import BaseDataModule

import datetime


# Folds for the Poverty dataset
FOLD = {1: (['A', 'B', 'C'], ['D'], ['E']), 2: (['B', 'C', 'D'], ['E'], ['A']), 3: (['C', 'D', 'E'], ['A'], ['B']),
        4: (['D', 'E', 'A'], ['B'], ['C']), 5: (['E', 'A', 'B'], ['C'], ['D'])}


class JitterCustom:
    """Custom Jitter class to apply the same transformation to all bands of the image."""
    def __init__(self, brightness=0.1, contrast=0.1):
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, img):
        for i in range(7):
            img[i] = self.jitter(img[i].unsqueeze(0)).squeeze(0)

        return img


class PovertyDataModule(BaseDataModule):
    def __init__(
            self,
            tif_dir: str = 'landsat_tiles/',
            dataset_path: str = 'examples/poverty/dataset/',
            labels_name: str = 'observation_2013+.csv',
            train_batch_size: int = 32,
            inference_batch_size: int = 16,
            num_workers: int = 8,
            fold: int = 1,
            cach_data: bool = True,
            transform=None,
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
        self.dataframe_train = dataframe[dataframe['fold'].isin(FOLD[fold][0])]
        self.dataframe_val = dataframe[dataframe['fold'].isin(FOLD[fold][1])]
        self.dataframe_test = dataframe[dataframe['fold'].isin(FOLD[fold][2])]
        self.tif_dir = dataset_path + tif_dir
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.dict_normalize = json.load(open('examples/poverty/mean_std_normalize.json', 'r'))
        self.num_workers = num_workers
        self.task = 'regression'

    def train_transform(self) -> Callable:
        return torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            # JitterCustom(),
            torchvision.transforms.Normalize(mean=self.dict_normalize['mean'], std=self.dict_normalize['std']),
        ])

    def test_transform(self) -> Callable:
        return torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(mean=self.dict_normalize['mean'], std=self.dict_normalize['std']),
        ])

    def get_dataset(self, split: str, transform: Callable, **kwargs) -> Dataset:
        if split == 'train':
            dataset = MSDataset(self.dataframe_train, self.tif_dir, transform=transform)
        elif split == 'val':
            dataset = MSDataset(self.dataframe_val, self.tif_dir, transform=transform)
        elif split == 'test':
            dataset = MSDataset(self.dataframe_test, self.tif_dir, transform=transform)
        return dataset

    def get_train_dataset(self) -> Dataset:
        dataset = self.get_dataset(
            split="train",
            transform=self.train_transform(),
        )
        return dataset

    def get_val_dataset(self) -> Dataset:
        dataset = self.get_dataset(
            split="val",
            transform=self.test_transform(),
        )
        return dataset

    def get_test_dataset(self) -> Dataset:
        dataset = self.get_dataset(
            split="test",
            transform=self.test_transform(),
        )
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

    def export_predict_csv(self,
                           predictions: Union[Tensor, np.ndarray],
                           probas: Union[Tensor, np.ndarray] = None,
                           single_point_query: dict = None,
                           out_name: str = "predictions",
                           out_dir: str = './',
                           return_csv: bool = False,
                           top_k: int = None,
                           **kwargs: Any) -> Any:

        """Exports the predictions and probabilities to a csv file.
        The key names are tailored for the DHS dataset."""

        out_name = out_name + ".csv" if not out_name.endswith(".csv") else out_name
        fp = Path(out_dir) / Path(out_name)
        top_k = top_k if top_k is not None else predictions.shape[1]
        if single_point_query:
            df = pd.DataFrame({'observation_id': [
                single_point_query['observation_id'] if 'observation_id' in single_point_query else None],
                               'lon': [single_point_query['lon']],
                               'lat': [single_point_query['lat']],
                               'crs': [single_point_query['crs']],
                               'target_species_id': tuple(np.array(single_point_query['species_id']).astype(
                                   str) if 'species_id' in single_point_query else None),
                               'predictions': tuple(predictions[:, :top_k].astype(str)),
                               'probas': tuple(probas[:, :top_k].astype(str))})
        else:
            test_ds = self.get_test_dataset()
            targets = test_ds.targets
            df = pd.DataFrame({'cluster': test_ds.observation_ids,
                               'lon': [None] * len(test_ds) if not hasattr(test_ds,
                                                                           'coordinates') else test_ds.coordinates[:,
                                                                                               0],
                               'lat': [None] * len(test_ds) if not hasattr(test_ds,
                                                                           'coordinates') else test_ds.coordinates[:,
                                                                                               1],
                               'wealthpooled': tuple(np.array(targets).astype(str)),
                               'predicted_wealth': tuple(predictions[:, :top_k].numpy().astype(str)),
                               'probas': [None] * len(predictions)})

        if probas is not None:
            df['probas'] = tuple(probas[:, :top_k].astype(str))
        for key in ['probas', 'predicted_wealth', 'wealthpooled']:
            if df.loc[0, key] is not None and not isinstance(df.loc[0, key], str) and len(df.loc[0, key]) >= 1:
                df[key] = df[key].apply(' '.join)
        print(df)
        df.to_csv(fp, index=False, sep=';', **kwargs)
        if return_csv:
            return df
        return None


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
            ax.set_title(f"Value: {value}, RGB")
        else:

            fig, axs = pyplot.subplots(2, 4, figsize=(12, 6))

            for i, ax in enumerate(axs.flat[0:-1]):
                ax.imshow(tile[i, ...], cmap='pink')

                ax.set_title(f"Band: {i}")

        fig.suptitle(f"Value: {value}")
        if save:
            fig.savefig(f'examples/poverty/plot_{idx}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png')

        pyplot.tight_layout()
        pyplot.show()
        return tile
