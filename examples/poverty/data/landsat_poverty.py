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
        tile_name = os.path.join(self.root_dir,
                                 str(row.country) + "_" + str(row.year),
                                 str(row.cluster) + ".tif"
                                 )

        raster = gdal.Open(tile_name)
        tile = np.empty([8, 255, 255])
        for band in range(raster.RasterCount):
            tile[band, :, :] = raster.GetRasterBand(band + 1).ReadAsArray()
        value = row.wealthpooled.astype('float')
        tile = torch.from_numpy(np.nan_to_num(tile))

        # We only select MS bands
        tile = tile[:7, :, :]

        # Close Raster (Safety Measure)
        raster = None
        if self.test_flag:
            transforms = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(224),
            )

            tile = transforms(tile)

            tile = preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], jitter=None)
            return idx, tile, value

        transforms = torch.nn.Sequential(
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip()
        )

        tile = transforms(tile)

        tile = preprocess_landsat(tile, self.normalizer['landsat_+_nightlights'], JITTER)
        return tile, value