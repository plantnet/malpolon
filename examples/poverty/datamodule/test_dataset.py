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
from matplotlib import pyplot
import random

from landsat_poverty import MSDataset,PovertyDataModule

BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']

if __name__ == "__main__":

    
    root="dataset/landsat_tiles"

    datamodule = PovertyDataModule()
    datamodule.setup()
    train_dataset = datamodule.get_dataset()
    train_dataset.plot(idx=[0])

    
    