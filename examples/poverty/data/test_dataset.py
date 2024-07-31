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


class CustomTiffDataset(RasterDataset):
    def __init__(self, root: str, transforms=None):
        super().__init__(transforms=transforms)
        self.root = root
        self.files = list(self.root.glob("*.tiff"))  # Assuming all .tiff files are in the root directory

    def __getitem__(self, index: int):
        filepath = self.files[index]
        with rasterio.open(filepath) as src:
            image = src.read()  # Reads the image as a NumPy array
            image = torch.from_numpy(image).float()  # Convert to torch tensor
            if self.transforms:
                image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.files)
    
    
    
class PovertyTiffDataset(RasterDataset):

    def __init__():

        pass

class PovertyDataModule(BaseDataModule):
    def __init__():
        pass