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

BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']

if __name__ == "__main__":

    
    root="dataset/landsat_tiles/angola_2015/"
    tile = np.empty([8, 255, 255])
    

    files = os.listdir(root)
    
    tif_files = [file for file in files if file.endswith('.tif')]
    file = random.choice(tif_files)
    with rasterio.open(root+file) as src:
        
        for band in src.indexes:
            tile[band-1, :, :] = src.read(band)
        

    
    fig, axs = pyplot.subplots(2, 4, figsize=(12, 6))

    for i, ax in enumerate(axs.flat):
        ax.imshow(tile[0:3, ...].transpose(1,2,0))#, cmap='pink'
        ax.set_title(f"Band: {i}")

    pyplot.tight_layout()
    pyplot.show()
    