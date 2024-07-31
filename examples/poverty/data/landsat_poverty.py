import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchgeo.datasets.utils import BoundingBox
import torchvision
import pickle

from malpolon.data.data_module import BaseDataModule
from torchgeo.datasets import  RasterDataset

import rasterio

# TODO : Change gdal to read .tiff files


NORMALIZER = 'datasets/normalizer.pkl'
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

JITTER = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1)


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




class MSDataset(Dataset):

    def __init__(self, dataframe, root_dir, normalizer=NORMALIZER, test_flag=False):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            root_dir (string): Directory with all the images.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        with open(normalizer, 'rb') as f:
            self.normalizer = pickle.load(f)
        self.test_flag = test_flag

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

        tile = torch.from_numpy(np.nan_to_num(tile))

        
        transforms = torch.nn.Sequential(
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip()
        )

        tile = transforms(tile)

        tile = preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], JITTER)
        return tile, value