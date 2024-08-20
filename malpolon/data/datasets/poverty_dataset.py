import os
import random
import json

import numpy as np
import rasterio
import pandas as pd
from matplotlib import pyplot

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

import datetime



# TODO : CHECK JITTER WORKS
# TODO : REPRODUICE Mathieu 2-mpa Results


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
FOLD = {1 : (['A','B','C','D'], ['E']), 2 : (['A','B','C','E'], ['D']), 3 : (['A','B','D','E'], ['C']), 4 : (['A','C','D','E'], ['B']), 5 : (['B','C','D','E'], ['A'])}

class JitterCustom:

    def __init__(self, brightness=0.1, contrast=0.1):
        
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)
    
    def __call__(self, img):

        for i in range(7):

            img[i] = self.jitter(img[i].unsqueeze(0)).squeeze(0)

        return img
        


class PovertyDataModule(pl.LightningDataModule):
    def __init__(
            self,  
            tif_dir : str = 'landsat_tiles/', 
            dataset_path: str = 'examples/poverty/dataset/',
            labels_name: str = 'observation_2013+.csv',
            train_batch_size: int = 32,
            inference_batch_size: int = 16,
            num_workers: int = 8,
            fold: int = 1,
            cach_data: bool = True,
            val_split : float = 0.2,
            transform=None,
            **kwargs
        ):
        super().__init__()
        dataframe = pd.read_csv(dataset_path+labels_name)
        self.dataframe = dataframe
        self.dataframe_train = dataframe[dataframe['fold'].isin(FOLD[fold][0])]
        self.dataframe_val = dataframe[dataframe['fold'].isin(FOLD[fold][1])]
        self.tif_dir = dataset_path+tif_dir
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.dict_normalize = json.load(open('examples/poverty/mean_std_normalize.json', 'r'))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            JitterCustom(),
            torchvision.transforms.Normalize(mean=self.dict_normalize['mean'], std=self.dict_normalize['std']),
        ]
        ) if transform is None else transform
        self.val_split = val_split
        self.num_workers = num_workers
        
    def get_dataset(self):
        dataset = MSDataset(self.dataframe, self.tif_dir,transform=self.transform)
        return dataset

    def get_train_dataset(self):
        dataset = MSDataset(self.dataframe_train, self.tif_dir,transform=self.transform)
        return dataset

    def get_val_dataset(self):
        dataset = MSDataset(self.dataframe_val, self.tif_dir,transform=self.transform)
        return dataset

    def setup(self, stage=None):
        full_dataset = MSDataset(self.dataframe, self.tif_dir,transform=self.transform)
        
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.inference_batch_size, num_workers=self.num_workers,persistent_workers=True)


class MSDataset(Dataset):

    def __init__(self, dataframe, root_dir,transform=None):
        """
        Args:
            dataframe (Pandas DataFrame): Pandas DataFrame containing image file names and labels.
            root_dir (string): Directory with all the images.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform=transform

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
        tile = self.transform(torch.tensor(tile, dtype=torch.float32))
        value = torch.tensor(value, dtype=torch.float32).unsqueeze(-1)
        return tile, value
    
    def plot(self, idx, rgb=False, save=True):

        tile, value = self.__getitem__(idx)
        
        tile=tile.numpy()
        
        if rgb:
            fig, ax = pyplot.subplots(1, 1, figsize=(6, 6))
            img_rgb=tile[0:3, ...][::-1, ... ].transpose(1,2,0)
            ax.imshow(img_rgb) #
            ax.set_title(f"Value: {value}, RGB")
        else :

            fig, axs = pyplot.subplots(2, 4, figsize=(12, 6))

            for i, ax in enumerate(axs.flat[0:-1]):

                ax.imshow(tile[i, ...], cmap='pink')

                ax.set_title(f"Band: {i}")

        fig.suptitle(f"Value: {value}")
        if save :
            fig.savefig(f'examples/poverty/plot_{idx}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png')
        
        pyplot.tight_layout()
        pyplot.show()
        return tile