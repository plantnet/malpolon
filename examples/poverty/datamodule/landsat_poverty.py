import torch
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import pickle

import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import rasterio

# TODO : Add JITTER and NORMALIZER to transfomer LightningDataModule, top remove ``preprocess_landsat`` step


NORMALIZER = 'dataset/normalizer.pkl'
BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1', 'NIGHTLIGHTS']
DESCRIPTOR = {
    'cluster': "float",
    'lat': "float",
    "lon": "float",
    'wealthpooled': "float",
    'BLUE': "float",
    'GREEN': "float",
    'RED': "float",
    'NIR': "float",
    'SWIR1': "float",
    'SWIR2': "float",
    'TEMP1': "float",
    'NIGHTLIGHTS': "float"
}

JITTER =transforms.ColorJitter(brightness=0.1, contrast=0.1)


def preprocess_landsat(raster, normalizer, jitter=None):
    for i in range(7):

        # Color Jittering transform
        tmp_shape = raster[i].shape
        if jitter:
            raster[i] = torch.reshape(
                jitter(raster[i][None, :, :]),
                tmp_shape
            )

        # Dataset normalization
        raster[i] = (raster[i] - normalizer[0][i]) / (normalizer[1][i])

    return raster



class PovertyDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, tif_dir, batch_size=32, transform=None, val_split=0.2):
        super().__init__()
        self.dataframe = pd.read_csv(csv_file)
        self.tif_dir = tif_dir
        self.batch_size = batch_size
        self.transform = torch.nn.Sequential(
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip()
        )
        self.val_split = val_split

    def setup(self, stage=None):
        full_dataset = MSDataset(self.dataframe, self.tif_dir)
        
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class MSDataset(Dataset):

    def __init__(self, dataframe, root_dir, normalizer=NORMALIZER):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            root_dir (string): Directory with all the images.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        with open(normalizer, 'rb') as f:
            self.normalizer = pickle.load(f)

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
                tile[band-1, :, :] = src.read(band)

        tile = np.nan_to_num(tile)

        # tile = preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], JITTER)

        return torch.tensor(tile, dtype=torch.float32), torch.tensor(value, dtype=torch.float32)
    

if __name__ == "__main__":

    dm = PovertyDataModule("dataset/observation_2013+.csv", "dataset/landsat_tiles")
    dm.setup()
    dl = dm.train_dataloader()
    for i, (x, y) in enumerate(dl):
        print(x.shape, y.shape)
        if i > 3:
            break