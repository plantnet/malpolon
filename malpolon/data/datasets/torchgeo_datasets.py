"""This module provides raster related classes based on torchgeo.

Author: Theo Larcher <theo.larcher@inria.fr>
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence, Union

import numpy as np
import pandas as pd
import pyproj
from matplotlib import pyplot as plt
from pyproj import CRS, Transformer
from torchgeo.datasets import BoundingBox, RasterDataset
from torchgeo.samplers import Units

from malpolon.data.utils import is_point_in_bbox, to_one_hot_encoding

if TYPE_CHECKING:
    import numpy.typing as npt
    Patches = npt.NDArray
    Targets = npt.NDArray[np.int64]

ALL_NORTHERN_EPSG_CODES = list(range(32601, 32662))
EUROPE_EPSG_CODE = [3035]


class RasterTorchGeoDataset(RasterDataset):
    """Generic torchgeo based raster datasets.

    Datasets based on this class return patches from raster files and
    can be queried by either a torchgeo BoundingBox object, a tuple of
    coordinates in the dataset's CRS or a dictionary specifying
    coordinates and the wanted CRS. Additionally one can specify the
    desired size and units of the wanted patch even if they don't match the
    dataset's.

    RasterTorchGeoDataset inherits torchgeo's RasterDataset class.
    """
    def __init__(
        self,
        root: str = "data",
        labels_name: str = None,
        split: str = None,  # 'train', 'test', 'val', 'all'
        crs: Any = None,
        res: float = None,
        bands: Sequence[str] = None,
        transform: Callable = None,
        transform_target: Callable = None,
        patch_size: Union[int, float, tuple] = 256,
        query_units: str = 'pixel',
        query_crs: Union[int, str, CRS] = 'self',
        obs_data_columns: dict = {'x': 'lon',
                                  'y': 'lat',
                                  'index': 'surveyId',
                                  'species_id': 'speciesId',
                                  'split': 'subset'},
        task: str = 'multiclass',  # ['binary', 'multiclass', 'multilabel']
        binary_positive_classes: list = [],
        cache: bool = True,
    ) -> None:
        """Class constructor.

        Parameters
        ----------
        root : str, optional
            path to the directory containing the data and labels, by default "data"
        labels_name : str, optional
            labels file name, by default None
        split : str, optional
            dataset subset desired for labels selection, by default None
        crs : Any | None, optional
            `coordinate reference system (CRS)` to warp to
            (defaults to the CRS of the first file found), by default None
        res : float | None, optional
            resolution of the dataset in units of CRS
            (defaults to the resolution of the first file found), by default None
        bands : Sequence[str] | None, optional
            bands to return (defaults to all bands), by default None
        transform : Callable | None, optional
            a callable function that takes an input sample and returns
            a transformed version, by default None
        transform_target : Callable | None, optional
            a callable function that takes an input target and returns
            a transformed version, by default None
        patch_size : int, optional
            size of the 2D extracted patches. Patches can either be
            square (int/float value) or rectangular (tuple of int/float).
            Defaults to a square of size 256, by default 256
        query_units : str, optional
            unit system of the dataset's queries, by default 'pixel'
        query_crs: Union[int, str, CRS], optional
            CRS of the dataset's queries, by default 'self' (same as dataset's CRS)
        obs_data_columns: dict, optional
            this dictionary allows users to match the dataset attributes
            with custom column names of their obs data file,
            by default::

              {'x': 'lon',
               'y': 'lat',
               'index': 'surveyId',
               'species_id': 'speciesId',
               'split': 'subset'}

            Here's a description of the keys:

            - 'x', 'y': coordinates of the obs points (by default 'lon', 'lat' as per the WGS84 system)
            - 'index': obs ID over which to iterate during the training loop
            - 'species_id': species ID (label) associated with the obs points
            - 'split': dataset split column name

        task : str, optional
            machine learning task (used to format labels accordingly), by default 'multiclass'
        binary_positive_classes : list, optional
            labels' classes to consider valid in the case of binary
            classification with multi-class labels (defaults to all 0),
            by default []
        cache : bool, optional
            if True, cache file handle to speed up repeated sampling, by default True
        """
        super().__init__(root, crs, res, bands, None, cache)
        self.patch_size = patch_size
        self.crs_pyproj = CRS(self.crs.data['init']) if self.crs.is_epsg_code else self.crs
        self.units = self.crs_pyproj.axis_info[0].unit_name if self.crs.is_epsg_code else self.crs.data['units']
        self.training = split != "test"
        self.task = task
        self.binary_positive_classes = set(binary_positive_classes)
        self.transform = transform
        self.transform_target = transform_target
        self._query_units = query_units
        self._query_crs = query_crs
        self._load_observation_data(Path(root), labels_name, split, obs_data_columns)
        # df = self._load_observation_data(Path(root), labels_name, split)
        # self.observation_ids = df.index
        # self.coordinates = df[["lon", "lat"]].values
        # self.targets = df["speciesId"].values
        # self._query_units = query_units
        # self._query_crs = query_crs

    def __len__(self) -> int:
        return len(self.observation_ids)

    def _load_observation_data(
        self,
        root: Path = None,
        obs_fn: str = None,
        subsets: str = ['train', 'test', 'val'],
        keys: dict = {'x': 'lon',
                      'y': 'lat',
                      'index': 'surveyId',
                      'species_id': 'speciesId',
                      'split': 'subset'}
    ) -> pd.DataFrame:
        """Load observation data from a CSV file.

        Reads values from a CSV file containing lon/lat coordinates,
        species id (labels) and dataset subset info (train/test/val).
        The associated columns must have the following values:
        ['longitude', 'latitude', 'speciesId', 'subset']

        If no value is given to root or obs_fn, the method returns an
        empty labels DataFrame.

        Parameters
        ----------
        root : Path
            directory containing the observation (labels) file, by default None.
        obs_fn : str
            observations file name, by default None.
        subsets : str
            desired data subset amongst ["train", "test", "val"], by default
            ["train", "test", "val"] (no restriction).

        Returns
        -------
        pd.DataFrame
            labels DataFrame
        """
        x_key, y_key = keys['x'], keys['y']
        index_key = keys['index']
        species_id_key = keys['species_id']
        split_key = keys['split']

        if any([root is None, obs_fn is None]):
            df = pd.DataFrame(columns=[x_key, y_key, species_id_key, split_key])
            self.observation_ids = df.index
            self.coordinates = df[["lon", "lat"]].values
            self.targets = df["speciesId"].values
            return df
        labels_fp = obs_fn if len(obs_fn.split('.csv')) >= 2 else f'{obs_fn}.csv'
        labels_fp = root / labels_fp
        df = pd.read_csv(
            labels_fp,
            sep=",",
            index_col=index_key,
        )
        self.unique_labels = np.sort(np.unique(df[species_id_key]))
        try:
            subsets = [subsets] if isinstance(subsets, str) else subsets
            ind = np.unique(df.index[df[split_key].isin(subsets)])
            df = df.loc[ind]
        except ValueError as e:
            print('Unrecognized subset name.\n'
                  'Please use one or several amongst: ["train", "test", "val"], as a string or list of strings.\n',
                  {e})
        self.observation_ids = df.index
        self.coordinates = df[[x_key, y_key]].values
        self.targets = df[species_id_key].values
        return df

    def coords_transform(
        self,
        lon: Union[int, float],
        lat: Union[int, float],
        input_crs: Union[str, int, CRS] = "4326",
        output_crs: Union[str, int, CRS] = "self",
    ) -> tuple[float, float]:
        """Transform coordinates from one CRS to another.

        Parameters
        ----------
        lon : Union[int, float]
            longitude
        lat : Union[int, float]
            latitude
        input_crs : Union[str, int, CRS], optional
            Input CRS, by default "4326"
        output_crs : Union[str, int, CRS], optional
            Output CRS, by default "self"

        Returns
        -------
        tuple
            Transformed coordinates.
        """
        if not isinstance(input_crs, CRS):
            input_crs = self.crs if input_crs == "self" else pyproj.CRS.from_epsg(input_crs)
        if not isinstance(output_crs, CRS):
            output_crs = self.crs if output_crs == "self" else pyproj.CRS.from_epsg(output_crs)
        if input_crs == output_crs:
            return lon, lat
        transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
        return transformer.transform(lon, lat)

    def point_to_bbox(
        self,
        lon: Union[int, float],
        lat: Union[int, float],
        size: Union[tuple, int] = None,
        units: str = 'crs',
        crs: Union[int, str] = 'self',
    ) -> BoundingBox:
        """Convert a geographical point to a torchgeo BoundingBox.

        This method converts a 2D point into a 2D torchgeo bounding box (bbox).
        If 'size' is in the CRS' unit system, the bbox is computed directly
        from the point's coordinates.
        If 'size' is in pixels, 'size' is multiplied by the resolution of the
        dataset.
        If 'size' is in meters and the dataset's unit system isn't, the point is
        projected into the nearest meter-based CRS (from a list defined as
        constant at the begining of this file), the bbox vertices' min and max
        are computed in thise reference system, then they are projected back
        into the input CRS 'crs'. If the dataset's CRS doesn't match en EPSG
        code but is instead built from a WKT, the nearest meter-based CRS
        will always be EPSG:3035.

        By default, 'size' is set to the dataset's 'patch_size' value via None.

        Parameters
        ----------
        lon : Union[int, float]
            longitude
        lat : Union[int, float]
            latitude
        size : Union[tuple, int], optional
            Patch size, by default None. If passed as an int, the patch will be
            square. If passed as a tuple (width, height), can be rectangular.
            By default None.
        units : str, optional
            The coordinates' unit system, must have a value in ['pixel', 'crs'].
            The size of the bbox will adapt to the unit. If 'pixel' is
            selected, the bbox size will be multiplied by the dataset
            resolution. Selecting 'crs' will not modify the bbox size. In that
            case the returned bbox will be of size:
            (size[0], size[1]) <metric_of_the_dataset (usually meters)>.
            Defaults to 'crs'.
        crs : Union[int, str]
            CRS of the point's lon/lat coordinates, by default 'self'.

        Returns
        -------
        BoundingBox
            Corresponding torchgeo BoundingBox.
        """
        crs = self.crs_pyproj if crs == 'self' else crs
        units = {'pixel': Units.PIXELS, 'crs': Units.CRS, 'm': 'm', 'meter': 'm', 'metre': 'm'}[units]
        size = self.patch_size if size is None else size  # size = size or self.patch_size
        size = (size, size) if isinstance(size, (int, float)) else size
        if units == Units.PIXELS:
            size = (size[0] * self.res, size[1] * self.res)

        # Compute the new value of size if the query is in meters but the dataset's unit isn't.
        if units == 'm' and not self.crs_pyproj.axis_info[0].unit_name in ['metre', 'meter', 'm']:
            # Find closest meter EPSG
            best_crs = {'code': '',
                        'center_distance': np.inf}
            if self.crs_pyproj.to_epsg() is not None:
                lon_geodetic, lat_geodetic = self.coords_transform(lon, lat, input_crs=crs, output_crs=self.crs_pyproj.geodetic_crs)
            else:
                lon_geodetic, lat_geodetic = self.coords_transform(lon, lat, input_crs=crs, output_crs=CRS(EUROPE_EPSG_CODE[0]))
            for code in ALL_NORTHERN_EPSG_CODES:
                epsg_aou = CRS.from_epsg(code).area_of_use
                epsg_lon_center, epsg_lat_center = (epsg_aou.west + epsg_aou.east) / 2, (epsg_aou.south + epsg_aou.north) / 2
                center_distance = np.linalg.norm(np.array([lon_geodetic, lat_geodetic]) - np.array([epsg_lon_center, epsg_lat_center]))
                if center_distance <= best_crs['center_distance']:
                    best_crs['code'] = code
                    best_crs['center_distance'] = center_distance
            best_crs = CRS.from_epsg(best_crs['code'])

            # Project lon, lat to best meter EPSG, compute bbox and project back to dataset's crs
            transformer = Transformer.from_crs(crs, best_crs, always_xy=True)
            lon_proj, lat_proj = transformer.transform(lon, lat)
            bounds_proj = (lon_proj - size[0] / 2, lon_proj + size[0] / 2), (lat_proj - size[1] / 2, lat_proj + size[1] / 2)  # (xmin, xmax, ymin, ymax)
            bounds = transformer.transform(*bounds_proj, direction="INVERSE")  # (xmin, xmax, ymin, ymax)
            size = (bounds[0][1] - bounds[0][0]), (bounds[1][1] - bounds[1][0])

        minx = lon - size[0] / 2
        maxx = lon + size[0] / 2
        miny = lat - size[1] / 2
        maxy = lat + size[1] / 2
        return BoundingBox(minx=minx, maxx=maxx, miny=miny, maxy=maxy, mint=0, maxt=0)

    def _valid_query_point(self,
                           query: dict
                           ) -> bool:
        """Check that the query coordinates land in the dataset bounds.

        Parameters
        ----------
        query : dict
            input query containing coordinates

        Returns
        -------
        bool
            True if the coordinates land in the dataset bounds,
            False otherwise.
        """
        epsg4326 = pyproj.CRS.from_epsg(4326)
        coords_4326 = query['lon'], query['lat']
        bounds_4326 = self.bounds
        if query['crs'] != epsg4326:
            transformer = Transformer.from_crs(query['crs'], epsg4326)
            coords_4326 = transformer.transform(query['lon'], query['lat'])
            coords_4326 = coords_4326[1], coords_4326[0]
        if self.crs_pyproj != epsg4326:
            transformer = Transformer.from_crs(self.crs_pyproj, epsg4326)
            bounds_4326 = transformer.transform_bounds(self.bounds.minx, self.bounds.miny, self.bounds.maxx, self.bounds.maxy)
            bounds_4326 = (bounds_4326[1], bounds_4326[3], bounds_4326[0], bounds_4326[2])
        return is_point_in_bbox(coords_4326, bounds_4326)

    def _format_label_to_task(
        self,
        label: np.ndarray
    ) -> Union[np.ndarray, int]:
        """Format label(s) returned to match the task type.

        Depending on the classification task (binary, multiclass or
        multilabel), labels have to be formatted accordingly.

        - **Binary**: label is an `int` equal to 1 if potential labels,
        i.e. input label values, contain an elligible value (i.e. in
        self.binary_positive_classes); 0 otherwise.
        - **Multiclass**: label is an `int` which value can be any
        class index (i.e. 'speciesId'). If several label match a
        coordinate set, the 1st one of the lsit is choosen.
        - **Multilabel**: label is a list of class index, containing all
        speciesId observed to a coordinate set.

        Parameters
        ----------
        label : np.ndarray
            Potential labels matching a coordinate set.

        Returns
        -------
        Union[int, list]
            Formated label(s).
        """
        if self.task == 'classification_binary':
            label = 1 if set(label) & set(self.binary_positive_classes) else 0
            label = np.array([label])
            return label
        if self.task == 'classification_multiclass':
            return label[0]
        if self.task == 'classification_multilabel':
            return to_one_hot_encoding(label, self.unique_labels)
        return label

    def get_label(
        self,
        df: pd.DataFrame,
        query_lon: float,
        query_lat: float,
        obs_id: int = None
    ) -> Union[np.ndarray, int]:
        """Return the label(s) matching the query coordinates.

        This method takes into account the fact that several labels can
        match a single coordinate set. For that reason, the labels
        are chosen according to the value of the 'obs_id' parameter
        (matching the observation_id column of the labels DataFrame).
        If no value is given to 'obs_id', all matching labels are returned.

        Parameters
        ----------
        df : pd.DataFrame
            dataset DataFrame composed of columns:
            ['lon', 'lat', 'observation_id']
        query_lon : float
            longitude value of the query point.
        query_lat : float
            latitude value of the query point.
        obs_id : int, optional
            observation ID tied to the query point, by default None

        Returns
        -------
        Union[np.ndarray, int]
            target label(s).
        """
        if obs_id is None:
            return self.targets[df.index[(df['lon'] == query_lon) & (df['lat'] == query_lat)].values]
        return self.targets[df.index[df['observation_id'] == obs_id].values]

    def _default_sample_to_getitem(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        coords = tuple(self.coordinates[idx])
        obs_id = self.observation_ids[idx]
        return {'lon': coords[0], 'lat': coords[1],
                'crs': self._query_crs,
                'size': self.patch_size,
                'units': self._query_units,
                'obs_id': obs_id}

    def __getitem__(
        self,
        query: Union[int, dict, tuple, list, set, BoundingBox]
    ) -> Dict[str, Any]:
        """Query an item from the dataset.

        Supports querying the dataset with coordinates in the dataset's CRS
        or in another CRS.
        The dataset is always queried with a torchgeo BoundingBox because it is
        itself a torchgeo dataset, but the query in this getter method can be
        passed as a tuple, list, set, dict or BoundingBox.

        - Use case 1: query is a [list, tuple, set] of 2 elements : lon, lat.
          Here the CRS and Units system are by default those of the dataset's.
        - Use case 2: query is a torchgeo BoundingBox.
          Here the CRS and Units system are by default those of the dataset's.
        - Use case 3: query is a dict containing the following necessary keys: {'lon', 'lat'},
          and optional keys: {'crs', 'units', 'size'} which values default to those of
          the dataset's.

        In Use case 3, if the 'crs' key is registered and it is different from
        the dataset's CRS, the coordinates of the point are projected into the
        dataset's CRS and the value of the key is overwritten by said CRS.

        Use cases 1 and 3 give the possibility to easily query the dataset using
        only a point and a bounding box (bbox) size, using the desired input CRS.

        The unit of measurement of the bbox can be set to ['m', 'meters', 'metres']
        even if the dataset's unit is different as the points will be projected
        in the nearest meter-based CRS (see `self.point_to_bbox()`). Note that
        depending on your dataset's CRS, querying a meter-based bbox may result
        in rectangular patches because of deformations.

        Parameters
        ----------
        query : Union[dict, tuple, BoundingBox]
            item query containing geographical coordinates. It can be of
            different types for different use.
            One can query a patch by providing a BoundingBox using
            `torchgeo.datasets.BoundingBox` constructor; or by given a center
            and a size.

            --- BoundingBox strategy ---

            Must follow : BoundingBox(minx, maxx, miny, maxy, mint, maxt)

            --- Point strategy ---

            If tuple, must follow : (lon, lat) and the CRS of the coordinates
            will be assumed to be the dataset's.
            If dict, must follow : {'lon': lon, 'lat': lat, <'crs': crs>} and
            the coordinates CRS can be specified. If not, it will be assumed
            that it is equal to the dataset's.
            In both cases, a BoundingBox is generated to pursue the query.

        Returns
        -------
        Dict[str, Any]
            dataset patch.
        """
        if not isinstance(query, BoundingBox):
            # Use case 1
            if isinstance(query, (tuple, list, set)):
                query = {'lon': query[0], 'lat': query[1], 'crs': self.crs_pyproj, 'size': None}
            query_lon, query_lat = query['lon'], query['lat']

            if not self._valid_query_point(query):
                raise ValueError("Your chosen point lands outside of your dataset CRS after projection.")

            # Use Case 3
            if 'crs' in query.keys() and query['crs'] != self.crs_pyproj:
                transformer = Transformer.from_crs(query['crs'], self.crs_pyproj, always_xy=True)
                query['lon'], query['lat'] = transformer.transform(query['lon'], query['lat'])

            if 'size' not in query.keys():
                query['size'] = self.patch_size
            if 'units' not in query.keys():
                query['units'] = 'pixel'
            if 'crs' not in query.keys() or query['crs'] != self.crs_pyproj:
                query['crs'] = self.crs_pyproj
            if 'obs_id' not in query.keys():
                query['obs_id'] = None

            query_obs_id = query['obs_id']
            query = self.point_to_bbox(query['lon'], query['lat'], query['size'], query['units'], query['crs'])

            # Use Case 2
            patch = super().__getitem__(query)
            df = pd.DataFrame(self.coordinates, columns=['lon', 'lat'])
            df['observation_id'] = self.observation_ids

            label = self.get_label(df, query_lon, query_lat, query_obs_id)
            label = self._format_label_to_task(label)
            sample = patch['image']
            if self.transform:
                sample = self.transform(sample)
            if self.transform_target:
                label = self.transform_target(label)
            return sample, label
        sample = super().__getitem__(query)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class RasterBioclim(RasterTorchGeoDataset):
    """Raster dataset adapted for CHELSA Bioclimatic data.

    Inherits RasterTorchGeoDataset.
    """
    filename_glob = "bio_*.tif"
    filename_regex = r"(?P<band>bio_[\d])"
    date_format = "%Y%m%dT%H%M%S"
    is_image = True
    separate_files = True
    all_bands = ["bio_1", "bio_2", "bio_3", "bio_4"]
    plot_bands = 'all_bands'

    def __init__(self, root: str = "data", labels_name: str = None, split: str = None, crs: Any = None, res: float = None, bands: Sequence[str] = None, transform: Callable[..., Any] = None, transform_target: Callable[..., Any] = None, patch_size: int | float | tuple = 256, query_units: str = 'pixel', query_crs: int | str | CRS = 'self', obs_data_columns: Dict = {'x': 'lon', 'y': 'lat', 'index': 'surveyId', 'species_id': 'speciesId', 'split': 'subset'}, task: str = 'multiclass', binary_positive_classes: list = [], cache: TYPE_CHECKING = True, **kwargs) -> None:
        super().__init__(root, labels_name, split, crs, res, bands, transform, transform_target, patch_size, query_units, query_crs, obs_data_columns, task, binary_positive_classes, cache)
        self.__dict__.update(kwargs)
        if self.plot_bands == 'plot_bands':
            self.plot_bands = self.all_bands

    def plot(self, sample: Patches):
        """Plot all layers of a given patch.

        A patch is selected based on a key matching the associated
        provider's __get__() method.

        Args:
            item (dict): provider's get index.
        """
        nb_layers = len(self.plot_bands)
        patch = self[sample]
        if nb_layers == 1:
            plt.figure(figsize=(10, 10))
            plt.imshow(patch[0])
        else:
            # calculate the number of rows and columns for the subplots grid
            rows = int(math.ceil(math.sqrt(nb_layers)))
            cols = int(math.ceil(nb_layers / rows))

            # create a figure with a grid of subplots
            fig, axs = plt.subplots(rows, cols, figsize=(10, 10))

            # flatten the subplots array to easily access the subplots
            axs = axs.flatten()

            # loop through the layers of patch data
            for i, band_name in enumerate(self.plot_bands):
                # display the layer on the corresponding subplot
                axs[i].imshow(patch[i])
                axs[i].set_title(f'layer_{i}: {band_name}')
                axs[i].axis('off')

            # remove empty subplots
            for i in range(nb_layers, rows * cols):
                fig.delaxes(axs[i])

        plt.suptitle('Tensor for sample: ' + str(sample), fontsize=16)

        # show the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
