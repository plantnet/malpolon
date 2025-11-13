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
from collections.abc import Iterable

from malpolon.data.datasets.geolifeclef2024 import JpegPatchProvider, PatchesDataset


SPECIES_INPUT_SIZE = 518
LANDSCAPE_INPUT_SIZE = 518
SATELLITE_INPUT_SIZE = 128

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

# Version multi-view per row
# def load_LUCAS_img(
#     id: Union[str, int],
#     metadata: pd.DataFrame,
#     root_path: str = "dataset/scale_2_landscape/",
#     views: list = ['cover', 'north', 'south',  'east', 'west', 'point'],  # Takes values in ['cover', 'north', 'south',  'east', 'west', 'point']
#     return_img_path: Optional[bool] = False,
#     return_img_gps: Optional[bool] = False,
#     id_col: str = 'id',
#     transform: Callable = None,
# ):
#     img, fps = [], []
#     metadata = metadata[metadata[id_col] == id].copy()
#     gps = tuple(metadata[['gps_long', 'gps_lat']].values.flatten())
#     for v_i, v in enumerate(views):
#         if metadata[f'file_path_gisco_{v}'].values[0] is None or (isinstance(metadata[f'file_path_gisco_{v}'].values[0], str) and len(metadata[f'file_path_gisco_{v}'].values[0]) <= 0):
#             continue
#         try:
#             fp = '/'.join(metadata[f'file_path_gisco_{v}'].values[0].split('/')[-5:])
#             img.append(torchvision.io.read_image(str(Path(root_path) / Path(fp))))
#             fps.append(fp)
#         except:
#             print(f"Image {fp} not found.")
#     if len(img) == 0:
#         img = torch.zeros(1, 3, LANDSCAPE_INPUT_SIZE, LANDSCAPE_INPUT_SIZE) -1
#     else:
#         img = [transform(i) for i in img]
#         img = torch.stack(img, dim=0)
#     if return_img_gps and return_img_path:
#         return img, gps, fps
#     if return_img_gps:
#         return img, gps
#     if return_img_path:
#         return img, fps
#     return img

# Version expanded and exists
def load_LUCAS_img(
    id: Union[str, int],
    metadata: pd.DataFrame,
    root_path: str = "dataset/scale_2_landscape/",
    return_img_path: Optional[bool] = False,
    return_img_gps: Optional[bool] = False,
    gps_col: list = ['lon', 'lat'],
    fp_cols: str = ['full_path', 'full_path_missing', 'full_path_2022', 'full_path_cover'],
    transform: Callable = None,
):
    img, fps = [], []
    metadata = metadata.iloc[id]
    gps = tuple(metadata[gps_col].values.flatten())
    for fp_col in fp_cols:
        try:
            fp = metadata[fp_col]
            img.append(torchvision.io.read_image(str(Path(root_path) / Path(fp))))
            fps.append(fp)
            break
        except:
            continue
    if len(img) == 0:
        img = torch.zeros(1, 3, LANDSCAPE_INPUT_SIZE, LANDSCAPE_INPUT_SIZE) -1
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
    print(f"WARNING: No matching species image found for id: {id} Returning zero tensor")
    return torch.zeros(3, SPECIES_INPUT_SIZE, SPECIES_INPUT_SIZE) if not return_img_path else (torch.zeros(3, SPECIES_INPUT_SIZE, SPECIES_INPUT_SIZE), [])

class DatasetSimple(Dataset):
    def __init__(
        self,
        root_path: str = None,
        fp_metadata: str = None,
        transform: Callable = None,
        dataset_kwargs: dict = {},
        subset: Union[int, float] = None,
        query_id: str = 'surveyId',
        **kwargs,
    ) -> None:
        super().__init__()
        self.root_path = root_path
        self.metadata = pd.read_csv(f'{Path(fp_metadata)}') if fp_metadata is not None else pd.DataFrame()
        self.subset = subset
        if subset:
            subset_length = int(len(self.metadata) * subset) if isinstance(subset, float) else subset
            self.metadata = self.metadata.sample(n=min(subset_length, len(self.metadata)), random_state=42).reset_index(names='index_no_subset')
        if self.__len__() == 0:
            raise ValueError(f"The dataset metadata is empty after applying the subset: {subset}. Please check the metadata file or increase the subset value.")
        self.transform = lambda x: x if transform is None else transform(x)
        self.dataset_kwargs = dataset_kwargs
        self.img, self.coords = torch.empty(0), (-np.inf, -np.inf)
        self.query_id = query_id
        
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
        query_id: str = 'surveyId',
        **kwargs,
    ) -> None:
        super().__init__(root_path, fp_metadata, transform, dataset_kwargs, query_id=query_id, **kwargs)

    def __getitem__(self, index) -> Any:
        img, coords = self.img, self.coords
        
        if not self.metadata.empty:
            sample = self.metadata.iloc[index]
            img = load_species_img(int(sample[self.query_id]), self.root_path, **self.dataset_kwargs)
            img = img.unsqueeze(0)  # Adds a batch dimension
            img = img.to(torch.float32)
            img = self.transform(img)
            coords = tuple(sample[['lon', 'lat']].values.flatten())
            id = int(sample[self.query_id])
        
        # return {'img': img, 'gps': coords}
        return img, torch.Tensor(coords), torch.tensor([index]), torch.tensor([id])


class LandscapeDatasetSimple(DatasetSimple):
    def __init__(
        self,
        root_path: str = None,
        fp_metadata: str = None,
        transform: Callable = None,
        dataset_kwargs: dict = {},
        query_id: str = 'id',
        **kwargs,
    ) -> None:
        super().__init__(root_path, fp_metadata, transform, dataset_kwargs, query_id=query_id, **kwargs)

    def __getitem__(self, index) -> Any:
        img, coords = self.img, self.coords
        if not self.metadata.empty:
            sample = self.metadata.iloc[index]
            img = load_LUCAS_img(index, self.metadata, self.root_path, **self.dataset_kwargs, transform=self.transform)
            img = img.to(torch.float32)
            # img = self.transform(img)
            if torch.equal(img, torch.zeros(1, 3, LANDSCAPE_INPUT_SIZE, LANDSCAPE_INPUT_SIZE) -1):
                coords = (1000, 1000)
            else:
                coords = tuple(sample[['lon', 'lat']].values.flatten())
            id = int(sample[self.query_id])

        # return {'img': img, 'gps': coords}
        return img, torch.Tensor(coords), torch.tensor([index]), torch.tensor([id])

class SatelliteDatasetSimple(DatasetSimple):
    def __init__(
        self,
        root_path: str = None,
        fp_metadata: str = None,
        transform: Callable = None,
        kwargs_sat_provider: dict = {'select': ['red','green','blue','nir'],
                                     'size': SATELLITE_INPUT_SIZE},
        kwargs_sat_dataset: dict = {'item_columns': ['lat', 'lon', 'surveyId', 'speciesId'],
                                    'labels_name': ['lat', 'lon']},
        query_id: str = 'surveyId',
        **kwargs,
    ) -> None:
        super().__init__(root_path, fp_metadata, transform, query_id=query_id, **kwargs)
        # Remove the duplicate GPS-img pairs corresponding to the multiple entries of the same surveyId because of multiple occurrences on the same place
        if not getattr(kwargs, 'keep_id_duplicates', True):
            self.metadata = self.metadata.drop_duplicates(subset=[self.query_id], keep='first')
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
        if self.subset:
            self.sat_dataset.items = self.sat_dataset.items.loc[self.metadata.index_no_subset].reset_index(names='index_no_subset')
            self.sat_dataset.observation_ids = self.sat_dataset.observation_ids[self.metadata.index_no_subset]  # np.array, assuming self.metadata indexing starts from 0 to N
            self.sat_dataset.targets = self.sat_dataset.targets[self.metadata.index_no_subset]  # np.array, assuming self.metadata indexing starts from 0 to N

    def __getitem__(self, index) -> Any:
        img, coords = self.img, self.coords
        if not self.metadata.empty:
            img, (sat_lat, sat_lon) = self.sat_dataset[index]  # Same as: sat_provider[{'surveyId': 80000}]
            img = torch.unsqueeze(img, dim=0)  # Adds a batch dimension
            img = img.to(torch.float32)
            img = self.transform(img)
            coords = (sat_lon, sat_lat)
            sat_id = self.sat_dataset.items.iloc[index][self.query_id]

        # return {'img': img, 'gps': coords}
        return img, torch.Tensor(coords), torch.tensor([index]), torch.tensor([sat_id])


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
                                     'size': SATELLITE_INPUT_SIZE},
        kwargs_sat_dataset: dict = {'item_columns': ['lat', 'lon', 'surveyId'],
                                    'labels_name': ['lat', 'lon']},
        skip_modalities: List[str] = [],
        query_ids: dict[str] = {'species': 'surveyId', 'landscape': 'id', 'satellite': 'surveyId'},
        **kwargs,
    ) -> None:
        super().__init__()
        self.skip_modalities = skip_modalities
        if 'species' in self.skip_modalities:
            self.species_dataset = None
            print('[INFO]: species modality skipped, will return dummy labels -1 and images 0')
        else:
            self.species_dataset = SpeciesDatasetSimple(
                root_path = root_path_species,
                fp_metadata = fp_metadata_species,
                transform = transform_species,
                query_id = query_ids['species'],
                **kwargs,
            )
        if 'landscape' in self.skip_modalities:
            self.landscape_dataset = None
            print('[INFO]: landscape modality skipped, will return dummy labels -1 and images 0')
        else:
            self.landscape_dataset = LandscapeDatasetSimple(
                root_path = root_path_landscape,
                fp_metadata = fp_metadata_landscape,
                transform = transform_landscape, 
                query_id = query_ids['landscape'],
                **kwargs,
            )
        if 'satellite' in self.skip_modalities:
            self.satellite_dataset = None
            print('[INFO]: satellite modality skipped, will return dummy labels -1 and images 0')
        else:
            self.satellite_dataset = SatelliteDatasetSimple(
                root_path = root_path_satellite,
                fp_metadata = fp_metadata_satellite,
                transform = transform_satellite,
                kwargs_sat_provider = kwargs_sat_provider,
                kwargs_sat_dataset = kwargs_sat_dataset,
                query_id = query_ids['satellite'],
                **kwargs,
            )
        
    def __len__(self):
        len_species = len(self.species_dataset) if self.species_dataset is not None else 0
        len_landscape = len(self.landscape_dataset) if self.landscape_dataset is not None else 0
        len_satellite = len(self.satellite_dataset) if self.satellite_dataset is not None else 0
        return max(len_species, len_landscape, len_satellite)
    
    def __getitem__(self, index) -> Any:
        # Species
        if 'species' in self.skip_modalities:
            # species_img, species_coords, species_idx, species_id = torch.zeros(1, 3, SPECIES_INPUT_SIZE, SPECIES_INPUT_SIZE), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            species_img, species_coords, species_idx, species_id = torch.zeros(1, 3, 1, 1), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
        else:
            if index > len(self.species_dataset) - 1:
                idx = torch.randint(0, len(self.species_dataset), (1,)).item()
            else:
                idx = index
            species_img, species_coords, species_idx, species_id = self.species_dataset[idx]
        
        # Landscape
        if 'landscape' in self.skip_modalities:
            # landscape_img, landscape_coords, landscape_idx, landscape_id = torch.zeros(1, 3, LANDSCAPE_INPUT_SIZE, LANDSCAPE_INPUT_SIZE), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            landscape_img, landscape_coords, landscape_idx, landscape_id = torch.zeros(1, 3, 1, 1), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
        else:
            if index > len(self.landscape_dataset) - 1:
                idx = torch.randint(0, len(self.landscape_dataset), (1,)).item()
            else:
                idx = index
            landscape_img, landscape_coords, landscape_idx, landscape_id = self.landscape_dataset[idx]
        
        # Satellite
        if 'satellite' in self.skip_modalities:
            # satellite_img, satellite_coords, satellite_idx, satellite_id = torch.zeros(1, 3, SATELLITE_INPUT_SIZE, SATELLITE_INPUT_SIZE), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            satellite_img, satellite_coords, satellite_idx, satellite_id = torch.zeros(1, 3, 1, 1), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
        else:
            if index > len(self.satellite_dataset) - 1:
                idx = torch.randint(0, len(self.satellite_dataset), (1,)).item()
            else:
                idx = index
            satellite_img, satellite_coords, satellite_idx, satellite_id = self.satellite_dataset[idx]

        sample = (species_img,  # 1
                  landscape_img,  # 2
                  satellite_img,  # 3
                  species_coords,  # 4
                  landscape_coords,  # 5
                  satellite_coords,  # 6
                  torch.tensor([index]),  # 7
                  species_idx,  # 8
                  landscape_idx,  # 9
                  satellite_idx,  # 10
                  species_id,  # 11
                  landscape_id,  # 12
                  satellite_id)  # 13
        
        # sample = {'species':
        #             {'img': species_img,
        #              'gps': species_coords,
        #              'index': torch.tensor([index]),
        #              'id': species_id},
        #           'landscape':
        #             {'img': landscape_img,
        #              'gps': landscape_coords,
        #              'index': torch.tensor([index]),
        #              'id': landscape_id},
        #           'satellite':
        #             {'img': satellite_img,
        #              'gps': satellite_coords,
        #              'index': torch.tensor([index]),
        #              'id': satellite_id}}
        
        return sample

class MultiscaleDatasetJoint(MultiscaleDatasetSimple):
    """Dataset intended for downstream task evaluation.

    All modalities will be loaded using a common metadata file (or at least separate metadata files
    which share identical indexing and lengths)

    Inherits MultiscaleDatasetSimple
    """
    def __init__(self, root_path_species: str = None, fp_metadata_species: str = None, root_path_landscape: str = None, fp_metadata_landscape: str = None, root_path_satellite: str = None, fp_metadata_satellite: str = None, transform_species: Callable[..., Any] = None, transform_landscape: Callable[..., Any] = None, transform_satellite: Callable[..., Any] = None, kwargs_sat_provider: dict = { 'select': ['red', 'green', 'blue', 'nir'],'size': SATELLITE_INPUT_SIZE }, kwargs_sat_dataset: dict = { 'item_columns': ['lat', 'lon', 'surveyId'],'labels_name': ['lat', 'lon'] }, skip_modalities: List[str] = [], **kwargs) -> None:
        super().__init__(root_path_species, fp_metadata_species, root_path_landscape, fp_metadata_landscape, root_path_satellite, fp_metadata_satellite, transform_species, transform_landscape, transform_satellite, kwargs_sat_provider, kwargs_sat_dataset, skip_modalities, **kwargs)
        
    def __getitem__(self, index) -> Any:
        # Species
        if 'species' in self.skip_modalities:
            # species_img, species_coords, species_idx, species_id = torch.zeros(1, 3, SPECIES_INPUT_SIZE, SPECIES_INPUT_SIZE), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            species_img, species_coords, species_idx, species_id = torch.zeros(1, 3, 1, 1), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
        else:
            species_img, species_coords, species_idx, species_id = self.species_dataset[index]

        # Landscape
        if 'landscape' in self.skip_modalities:
            # landscape_img, landscape_coords, landscape_idx, landscape_id = torch.zeros(1, 3, LANDSCAPE_INPUT_SIZE, LANDSCAPE_INPUT_SIZE), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            landscape_img, landscape_coords, landscape_idx, landscape_id = torch.zeros(1, 3, 1, 1), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
        else:
            landscape_img, landscape_coords, landscape_idx, landscape_id = self.landscape_dataset[index]

        # Satellite
        if 'satellite' in self.skip_modalities:
            # satellite_img, satellite_coords, satellite_idx, satellite_id = torch.zeros(1, 3, SATELLITE_INPUT_SIZE, SATELLITE_INPUT_SIZE), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            satellite_img, satellite_coords, satellite_idx, satellite_id = torch.zeros(1, 3, 1, 1), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
        else:
            satellite_img, satellite_coords, satellite_idx, satellite_id = self.satellite_dataset[index]

        sample = (species_img,
                  landscape_img,
                  satellite_img,
                  species_coords,
                  landscape_coords,
                  satellite_coords,
                  torch.tensor([index]),
                  species_idx,
                  landscape_idx,
                  satellite_idx,
                  species_id,
                  landscape_id,
                  satellite_id)

        return sample


class MultiscaleDatasetJointWithLabels(MultiscaleDatasetSimple):
    """Dataset intended for downstream task evaluation.

    All modalities will be loaded using a common metadata file (or at least separate metadata files
    which share identical indexing and lengths)

    Inherits MultiscaleDatasetSimple
    """
    def __init__(self, root_path_species: str = None, fp_metadata_species: str = None, root_path_landscape: str = None, fp_metadata_landscape: str = None, root_path_satellite: str = None, fp_metadata_satellite: str = None, transform_species: Callable[..., Any] = None, transform_landscape: Callable[..., Any] = None, transform_satellite: Callable[..., Any] = None, kwargs_sat_provider: dict = { 'select': ['red', 'green', 'blue', 'nir'],'size': SATELLITE_INPUT_SIZE }, kwargs_sat_dataset: dict = { 'item_columns': ['lat', 'lon', 'surveyId'],'labels_name': ['lat', 'lon'] }, skip_modalities: List[str] = [], **kwargs) -> None:
        super().__init__(root_path_species, fp_metadata_species, root_path_landscape, fp_metadata_landscape, root_path_satellite, fp_metadata_satellite, transform_species, transform_landscape, transform_satellite, kwargs_sat_provider, kwargs_sat_dataset, skip_modalities, **kwargs)
        self.task = kwargs['task'] if 'task' in kwargs else 'multilabel'
        self.num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 1

    def _labels_to_onehot(self, labels: Union[int, List[int]], num_classes: int) -> torch.Tensor:
        """Convert labels to one-hot encoding."""
        if not isinstance(labels, Iterable):
            labels = [labels]
        one_hot = torch.zeros(num_classes, dtype=torch.float32)
        for label in labels:
            one_hot[int(label)] = 1.0
        return one_hot
    
    def _find_other_speciesid_from_surveyid(self, df, id: int, col_sid='speciesId', col_queryid = 'id') -> List[int]:
        """Find other speciesId associated with the same surveyId."""
        all_species_ids = df[df[col_queryid] == id][col_sid].unique().tolist()
        
        return all_species_ids

    def __getitem__(self, index) -> Any:
        # Species
        if 'species' in self.skip_modalities:
            # species_img, species_coords, species_idx, species_id = torch.zeros(1, 3, SPECIES_INPUT_SIZE, SPECIES_INPUT_SIZE), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            species_img, species_coords, species_idx, species_id = torch.zeros(1, 3, 1, 1), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            species_label = torch.tensor([-1])
        else:
            species_img, species_coords, species_idx, species_id = self.species_dataset[index]
            species_label = self.species_dataset.metadata.iloc[index]['speciesId'] if 'speciesId' in self.species_dataset.metadata.columns else -1
            if 'multilabel' in self.task:
                species_label = self._find_other_speciesid_from_surveyid(self.species_dataset.metadata, species_id.item(), col_queryid=self.species_dataset.query_id)
                species_label = self._labels_to_onehot(species_label, self.num_classes)

        # Landscape
        if 'landscape' in self.skip_modalities:
            # landscape_img, landscape_coords, landscape_idx, landscape_id = torch.zeros(1, 3, LANDSCAPE_INPUT_SIZE, LANDSCAPE_INPUT_SIZE), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            landscape_img, landscape_coords, landscape_idx, landscape_id = torch.zeros(1, 3, 1, 1), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            landscape_label = torch.tensor([-1])
        else:
            landscape_img, landscape_coords, landscape_idx, landscape_id = self.landscape_dataset[index]
            landscape_label = self.landscape_dataset.metadata.iloc[index]['speciesId'] if 'speciesId' in self.landscape_dataset.metadata.columns else -1
            if 'multilabel' in self.task:
                landscape_label = self._find_other_speciesid_from_surveyid(self.landscape_dataset.metadata, landscape_id.item(), col_queryid=self.landscape_dataset.query_id)
                landscape_label = self._labels_to_onehot(landscape_label, self.num_classes)

        # Satellite
        if 'satellite' in self.skip_modalities:
            # satellite_img, satellite_coords, satellite_idx, satellite_id = torch.zeros(1, 3, SATELLITE_INPUT_SIZE, SATELLITE_INPUT_SIZE), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            satellite_img, satellite_coords, satellite_idx, satellite_id = torch.zeros(1, 3, 1, 1), torch.tensor([-1000, -1000]), torch.tensor([index]), torch.tensor([-1])
            satellite_label = torch.tensor([-1])
        else:
            satellite_img, satellite_coords, satellite_idx, satellite_id = self.satellite_dataset[index]
            satellite_label = self.satellite_dataset.metadata.iloc[index]['speciesId'] if 'speciesId' in self.satellite_dataset.metadata.columns else -1
            if 'multilabel' in self.task:
                satellite_label = self._find_other_speciesid_from_surveyid(self.satellite_dataset.metadata, satellite_id.item(), col_queryid=self.satellite_dataset.query_id)
                satellite_label = self._labels_to_onehot(satellite_label, self.num_classes)

        sample = (species_img,  #0
                  landscape_img,  #1
                  satellite_img,  #2
                  species_coords,  #3
                  landscape_coords,  #4
                  satellite_coords,  #5
                  torch.tensor([index]),  #6
                  species_idx,  #7
                  landscape_idx,  #8
                  satellite_idx,  #9
                  species_id,  #10
                  landscape_id,  #11
                  satellite_id,  #12
                  species_label,  #13
                  landscape_label,  #14
                  satellite_label)  #15
        # sample = {'species':
        #             {'img': species_img,
        #              'gps': species_coords,
        #              'index': torch.tensor([index]),
        #              'id': species_id},
        #           'landscape':
        #             {'img': landscape_img,
        #              'gps': landscape_coords,
        #              'index': torch.tensor([index]),
        #              'id': landscape_id},
        #           'satellite':
        #             {'img': satellite_img,
        #              'gps': satellite_coords,
        #              'index': torch.tensor([index]),
        #              'id': satellite_id}}

        return sample