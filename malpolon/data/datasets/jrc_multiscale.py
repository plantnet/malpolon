import os
import numpy as np
import geopandas as gpd
import pandas as pd
import torch
import torchvision
# import rasterio
import timm

from copy import deepcopy
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Optional, Callable, Any
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

from malpolon.data.datasets.geolifeclef2024 import JpegPatchProvider, PatchesDataset


def load_GPS_data(
    fp_gps: str = "dataset/geoloc/PN_gbif_France_2005-2025_illustrated_CBN-med.csv",
    cols: Union[str, list] = 'all',
):
    df_gps = pd.read_csv(fp_gps)
    df_gps = df_gps[cols] if cols != 'all' else df_gps
    assert df_gps['in_polygon'].all()

    gs = gpd.GeoSeries.from_wkt(df_gps['geometry'])
    gdf_gps = gpd.GeoDataFrame(df_gps, geometry=gs, crs="EPSG:4326")
    return gdf_gps

def load_LUCAS_img(
    id: Union[str, int],
    metadata: pd.DataFrame,
    root_path: str = "dataset/scale_2_landscape/",
    views: list = ['cover', 'north', 'south',  'east', 'west', 'point'],  # Takes values in ['cover', 'north', 'south',  'east', 'west', 'point']
    return_img_path: Optional[bool] = False,
    return_img_gps: Optional[bool] = False,
    id_col: str = 'id',
    transform: Callable = None,
):
    img, fps = [], []
    metadata = metadata[metadata[id_col] == id].copy()
    gps = tuple(metadata[['gps_long', 'gps_lat']].values.flatten())
    for v_i, v in enumerate(views):
        if metadata[f'file_path_gisco_{v}'].values[0] is None or (isinstance(metadata[f'file_path_gisco_{v}'].values[0], str) and len(metadata[f'file_path_gisco_{v}'].values[0]) <= 0):
            continue
        try:
            fp = '/'.join(metadata[f'file_path_gisco_{v}'].values[0].split('/')[-5:])
            img.append(torchvision.io.read_image(str(Path(root_path) / Path(fp))))
            fps.append(fp)
        except:
            print(f"Image {fp} not found.")
    if len(img) == 0:
        img = torch.zeros(1, 3, 518, 518) -1
    else:
        img = [transform(i) for i in img]
        img = torch.stack(img, dim=0)
    if return_img_gps and return_img_path:
        return img, gps, fps
    if return_img_gps:
        return img, gps
    if return_img_path:
        return img, fps
    return img

def load_species_img(
    id: Union[str, int],
    img_dir: Optional[str] = 'dataset/images/scale_1_species/',
    img_formats: Optional[List[str]] = ['jpeg', 'jpg', 'png'],
    return_img_path: Optional[bool] = False,
):
    for img_format in img_formats:
        img_path = Path(os.path.join(img_dir, f"{id}.{img_format}"))
        if os.path.exists(img_path):
            img = torchvision.io.read_image(str(img_path))
            if return_img_path:
                return img, [img_path.name]
            return img
    raise FileNotFoundError(f"No matching image found for id: {id}")

class DatasetSimple(Dataset):
    def __init__(
        self,
        root_path: str = None,
        fp_metadata: str = None,
        transform: Callable = None,
        dataset_kwargs: dict = {},
        subset: Union[int, float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root_path = root_path
        self.metadata = pd.read_csv(f'{Path(fp_metadata)}') if fp_metadata is not None else pd.DataFrame()
        if subset:
            subset_length = int(len(self.metadata) * subset) if isinstance(subset, float) else subset
            self.metadata = self.metadata.sample(n=min(subset_length, len(self.metadata)), random_state=42).reset_index(drop=True)
        self.transform = lambda x: x if transform is None else transform(x)
        self.dataset_kwargs = dataset_kwargs
        self.img, self.coords = torch.empty(0), (-np.inf, -np.inf)
        
    def __len__(self):
        return len(self.metadata)
    
    @abstractmethod
    def __getitem__(self, index) -> Any:
        """Returns a sample of the dataset."""


class SpeciesDatasetSimple(DatasetSimple):
    def __init__(
        self,
        root_path: str = None,
        fp_metadata: str = None,
        transform: Callable = None,
        dataset_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__(root_path, fp_metadata, transform, dataset_kwargs, **kwargs)

    def __getitem__(self, index) -> Any:
        img, coords = self.img, self.coords
        
        if not self.metadata.empty:
            sample = self.metadata.iloc[index]
            img = load_species_img(sample['gbifID'], self.root_path, **self.dataset_kwargs)
            img = img.unsqueeze(0)  # Adds a batch dimension
            img = img.to(torch.float32)
            img = self.transform(img)
            coords = tuple(sample[['decimalLongitude', 'decimalLatitude']].values.flatten())
        
        # return {'img': img, 'gps': coords}
        return img, torch.Tensor(coords)


class LandscapeDatasetSimple(DatasetSimple):
    def __init__(
        self,
        root_path: str = None,
        fp_metadata: str = None,
        transform: Callable = None,
        dataset_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__(root_path, fp_metadata, transform, dataset_kwargs, **kwargs)
        self.fp_columns = ['file_path_gisco_north', 'file_path_gisco_south', 'file_path_gisco_east', 'file_path_gisco_west', 'file_path_gisco_point', 'file_path_gisco_cover']
        if 'file_path_gisco_cover' not in self.metadata.columns:
            self.metadata['file_path_gisco_cover'] = self.metadata['file_path_gisco_north'].copy()
            self.metadata[self.fp_columns] = self.metadata[self.fp_columns].fillna('')
            self.metadata['file_path_gisco_cover'] = self.metadata['file_path_gisco_cover'].apply(lambda x: str(Path(x).parent / Path(Path(x).stem[:-1] + 'C' + Path(x).suffix)))
        # import time
        # time.time()
        # print('Starting to filter metadata...')
        # self._filter_metadata_on_existing_data()
        # print(f'Elapsed time for filtering metadata: {time.time() - start_time:.2f} seconds')

    def _filter_metadata_on_existing_data(self):
        fp_to_none = 0
        dropped_rows = 0
        from tqdm import tqdm
        for rowi, row in tqdm(self.metadata.iterrows()):
            n_missing_files = 0
            for c in self.fp_columns:
                if not os.path.exists(os.path.join(self.root_path, '/'.join(row[c].split('/')[-5:]))):
                    n_missing_files += 1
                    self.metadata.loc[rowi, c] = None
                    fp_to_none += 1
            if n_missing_files >= 6:
                self.metadata.drop(rowi, inplace=True)
                dropped_rows += 1
        print(f"Filtered {fp_to_none} file paths to None and dropped {dropped_rows} rows from metadata.")

    def __getitem__(self, index) -> Any:
        img, coords = self.img, self.coords
        if not self.metadata.empty:
            sample = self.metadata.iloc[index]
            img = load_LUCAS_img(sample['id'], self.metadata, self.root_path, **self.dataset_kwargs, transform=self.transform)
            img = img.to(torch.float32)
            # img = self.transform(img)
            if torch.equal(img, torch.zeros(1, 3, 518, 518) -1):
                coords = (1000, 1000)
            else:
                coords = tuple(sample[['gps_long', 'gps_lat']].values.flatten())

        # return {'img': img, 'gps': coords}
        return img, torch.Tensor(coords)
    

class SatelliteDatasetSimple(DatasetSimple):
    def __init__(
        self,
        root_path: str = None,
        fp_metadata: str = None,
        transform: Callable = None,
        kwargs_sat_provider: dict = {'select': ['red','green','blue','nir'],
                                     'size': 128},
        kwargs_sat_dataset: dict = {'item_columns': ['lat', 'lon', 'surveyId'],
                                    'labels_name': ['lat', 'lon']},
        **kwargs,
    ) -> None:
        super().__init__(root_path, fp_metadata, transform, **kwargs)
        # Remove the duplicate GPS-img pairs corresponding to the multiple entries of the same surveyId because of multiple occurrences on the same place
        self.metadata = self.metadata.drop_duplicates(subset=['surveyId'], keep='first')
        if not self.metadata.empty:
            self.sat_provider = JpegPatchProvider(
                self.root_path,  # 'dataset/scale_3_satellite/data_subset/PA_Train_SatellitePatches/',
                **kwargs_sat_provider, # default value, that of he pre-extracted patches
            )
            self.sat_dataset = PatchesDataset(
                occurrences=fp_metadata,  # 'dataset/scale_3_satellite/GLC24-PA-data_subset.csv',
                providers=[self.sat_provider],
                **kwargs_sat_dataset,
            )
        # self.metadata = self.metadata.sample(n=min(10000, len(self.metadata)))

    def __getitem__(self, index) -> Any:
        img, coords = self.img, self.coords
        if not self.metadata.empty:
            img, (sat_lat, sat_lon) = self.sat_dataset[index]  # Same as: sat_provider[{'surveyId': 80000}]
            img = torch.unsqueeze(img, dim=0)  # Adds a batch dimension
            img = img.to(torch.float32)
            img = self.transform(img)
            coords = (sat_lon, sat_lat)

        # return {'img': img, 'gps': coords}
        return img, torch.Tensor(coords), torch.tensor([index]), torch.tensor([self.sat_dataset.items.iloc[index]['surveyId']])


class MultiscaleDatasetSimple(Dataset):
    def __init__(
        self,
        root_path_species: str = None,
        fp_metadata_species: str = None,
        root_path_landscape: str = None,
        fp_metadata_landscape: str = None,
        root_path_satellite: str = None,
        fp_metadata_satellite: str = None,
        transform_species: Callable = None,
        transform_landscape: Callable = None,
        transform_satellite: Callable = None,
        kwargs_sat_provider: dict = {'select': ['red','green','blue','nir'],
                                     'size': 128},
        kwargs_sat_dataset: dict = {'item_columns': ['lat', 'lon', 'surveyId'],
                                    'labels_name': ['lat', 'lon']},
        **kwargs,
    ) -> None:
        super().__init__()
        self.root_path_species = root_path_species
        self.root_path_landscape = root_path_landscape
        self.root_path_satellite = root_path_satellite
        self.metadata_species = pd.read_csv(f'{Path(fp_metadata_species)}') if fp_metadata_species is not None else pd.DataFrame()
        self.metadata_landscape = pd.read_csv(f'{Path(fp_metadata_landscape)}') if fp_metadata_landscape is not None else pd.DataFrame()
        self.metadata_satellite = pd.read_csv(f'{Path(fp_metadata_satellite)}') if fp_metadata_satellite is not None else pd.DataFrame()
        self.transforms = {'species': lambda x: x if transform_species is None else transform_species,
                           'landscape': lambda x: x if transform_landscape is None else transform_landscape,
                           'satellite': lambda x: x if transform_satellite is None else transform_satellite}
        
        self.sat_provider = JpegPatchProvider(
            self.root_path_satellite,  # 'dataset/scale_3_satellite/data_subset/PA_Train_SatellitePatches/',
            **kwargs_sat_provider, # default value, that of he pre-extracted patches
        )
        self.sat_dataset = PatchesDataset(
            occurrences=fp_metadata_satellite,  # 'dataset/scale_3_satellite/GLC24-PA-data_subset.csv',
            providers=[self.sat_provider],
            **kwargs_sat_dataset,
        )
        
    def __len__(self):
        return max(len(self.metadata_species), len(self.metadata_landscape), len(self.metadata_satellite))
    
    def __getitem__(self, index) -> Any:
        # Species
        species_img, species_coords = None, None
        if not self.metadata_species.empty:
            species_sample = self.metadata_species.iloc[index]
            species_img = load_species_img(species_sample['gbifID'], self.root_path_species)
            species_img = self.transforms['species'](species_img)
            species_coords = tuple(species_sample[['decimalLongitude', 'decimalLatitude']].values.flatten())
        
        # Landscape
        landscape_img, landscape_coords = None, None
        if not self.metadata_landscape.empty:
            landscape_sample = self.metadata_landscape.iloc[index]
            landscape_img = load_LUCAS_img(landscape_sample['id'], self.metadata_landscape, self.root_path_landscape)
            landscape_img = self.transforms['landscape'](landscape_img)
            landscape_coords = tuple(landscape_sample[['gps_long', 'gps_lat']].values.flatten())
        
        # Satellite
        satellite_img, satellite_coords = None, None
        if not self.metadata_satellite.empty:
            satellite_img, (sat_lat, sat_lon) = self.sat_dataset[index]  # Same as: sat_provider[{'surveyId': 80000}]
            satellite_img = torch.unsqueeze(satellite_img, dim=0)  # Adds a batch dimension
            satellite_img = self.transforms['satellite'](satellite_img)
            satellite_coords = (sat_lon, sat_lat)

        sample = {'species':
                    {'img': species_img,
                     'gps': species_coords},
                  'landscape':
                    {'img': landscape_img,
                     'gps': landscape_coords},
                  'satellite':
                    {'img': satellite_img,
                     'gps': satellite_coords}}
        return sample