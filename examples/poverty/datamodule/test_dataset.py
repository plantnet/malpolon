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


class CustomTiffDataset(RasterDataset):
    filename_glob = "*.tif"
    # filename_regex = r"^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])"
    # date_format = "%Y%m%dT%H%M%S"
    is_image = True
    separate_files = True
    all_bands = BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
    rgb_bands = ['BLUE', 'GREEN', 'RED']
    
    
    
class PovertyTiffDataset(RasterDataset):

    def __init__():

        pass

class PovertyDataModule(BaseDataModule):
    def __init__():
        pass


if __name__ == "__main__":

    
    root="examples/poverty/data/landsat_7_less/angola_2015/"
    tile = np.empty([8, 255, 255])
    

    files = os.listdir(root)
    
    tif_files = [file for file in files if file.endswith('.tif')]
    file = random.choice(tif_files)
    with rasterio.open(root+file) as src:
        
        for band in src.indexes:
            tile[band-1, :, :] = src.read(band)
        

    
    fig, axs = pyplot.subplots(2, 4, figsize=(12, 6))

    for i, ax in enumerate(axs.flat):
        ax.imshow(tile[i, ...], cmap='pink')
        ax.set_title(f"Band {i+1}")

    pyplot.tight_layout()
    pyplot.show()
    