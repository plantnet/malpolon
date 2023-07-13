from __future__ import annotations

from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator,
                    Optional, Sequence, Tuple, Union)

import numpy as np
import pandas as pd
import pyproj
import torch
from matplotlib import pyplot as plt
from pyproj import CRS, Transformer
from torchgeo.datasets import (BoundingBox, GeoDataset, RasterDataset, stack_samples, unbind_samples)

from torchgeo.samplers import (GeoSampler, RandomGeoSampler, Units)

from malpolon.data.utils import is_point_in_bbox, to_one_hot_encoding

if TYPE_CHECKING:
    from collections.abc import Collection

    import numpy.typing as npt

    Patches = npt.NDArray
    Targets = npt.NDArray[np.int64]

ALL_NORTHERN_EPSG_CODES = list(range(32601, 32662))
EUROPE_EPSG_CODE = [3035]


class RasterTorchGeoDataset(RasterDataset):
    def __init__(self,
                 root: str = "data",
                 split: str = None,  # 'train', 'test', 'val', 'all'
                 labels_name: str = None,
                 crs: Any | None = None,
                 res: float | None = None,
                 bands: Sequence[str] | None = None,
                 transforms: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
                 cache: bool = True,
                 patch_size: int = 256,
                 task: str = 'multiclass',  # ['binary', 'multiclass', 'multilabel']
                 binary_positive_classes: list = []
                 ) -> None:
        super().__init__(root, crs, res, bands, transforms, cache)
        self.patch_size = patch_size
        self.crs_pyproj = CRS(self.crs.data['init'])
        self.units = self.crs_pyproj.axis_info[0].unit_name
        self.training = split != "test"
        self.labels_name = labels_name
        self.task = task
        self.binary_positive_classes = set(binary_positive_classes)

        df = self._load_observation_data(Path(root), split)
        self.observation_ids = df.index
        self.coordinates = df[["longitude", "latitude"]].values
        # if self.training:
        #     self.targets = df["species_id"].values
        # else:
        #     self.targets = None
        self.targets = df["species_id"].values

    def _load_observation_data(
        self,
        root: Path,
        subset: str,
    ) -> pd.DataFrame:
        if self.labels_name is None:
            return pd.DataFrame(columns=['longitude', 'latitude', 'species_id', 'subset'])
        labels_fp = self.labels_name if len(self.labels_name.split('.csv')) >= 2 else f'{self.labels_name}.csv'
        labels_fp = root / labels_fp
        df = pd.read_csv(
            labels_fp,
            sep=",",
            index_col="observation_id",
        )
        self.unique_labels = np.sort(np.unique(df['species_id']))
        try:
            subset = [subset] if isinstance(subset, str) else subset
            ind = df.index[df["subset"].isin(subset)]
            df = df.loc[ind]
        except ValueError as e:
            print('Unrecognized subset name.\n'
                  'Please use one or several amongst: ["train", "test", "val"], as a string or list of strings.\n',
                  {e})

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

    def point_to_bbox(self,
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
        into the input CRS 'crs'.

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
        units : str, optional
            The coordinates' unit system, must have a value in ['pixel', 'crs'].
            The size of the bbox will adapt to the unit. If 'pixel' is
            selected, the bbox size will be multiplied by the dataset
            resolution. Selecting 'crs' will not modify the bbox size. In that
            case the returned bbox will be of size:
            (size[0], size[1]) <metric_of_the_dataset (usually meters)>.
            Defaults to 'crs'.
        crs : Union[int, str]
            CRS of the point's lon/lat coordinates, by default None.

        Returns
        -------
        BoundingBox
            Corresponding torchgeo BoundingBox.
        """
        crs = self.crs_pyproj if crs == 'self' else crs
        units = {'pixel': Units.PIXELS, 'crs': Units.CRS, 'm': 'm', 'meter': 'm', 'metre': 'm'}[units]
        size = self.patch_size if size is None else size
        size = (size, size) if isinstance(size, (int, float)) else size
        if units == Units.PIXELS:
            size = (size[0] * self.res, size[1] * self.res)

        # Compute the new value of size if the query is in meters but the dataset's unit isn't.
        if units == 'm' and not self.crs_pyproj.axis_info[0].unit_name in ['metre', 'meter', 'm']:  # isinstance(self.crs, pyproj.CRS.from_epsg(4326)):
            # Find closest meter EPSG
            best_crs = {'code': '',
                        'center_distance': np.inf}
            lon_geodetic, lat_geodetic = self.coords_transform(lon, lat, input_crs=crs, output_crs=self.crs_pyproj.geodetic_crs)
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

    def _valid_query_point(self, query):
        epsg4326 = pyproj.CRS.from_epsg(4326)
        coords_4326 = query['lat'], query['lon']
        bounds_4326 = self.bounds
        if query['crs'] != epsg4326:
            transformer = Transformer.from_crs(query['crs'], epsg4326)
            coords_4326 = transformer.transform(query['lat'], query['lon'])
        if self.crs_pyproj != epsg4326:
            transformer = Transformer.from_crs(self.crs_pyproj, epsg4326)
            bounds_4326 = transformer.transform_bounds(self.bounds.minx, self.bounds.miny, self.bounds.maxx, self.bounds.maxy)
        return is_point_in_bbox(coords_4326, bounds_4326)

    def _format_label_to_task(self, label):
        """Format label(s) returned to match the task type.

        Depending on the classification task (binary, multiclass or
        multilabel), labels have to be formatted accordingly.
        - **Binary**: label is an `int` equal to 1 if potential labels,
        i.e. input label values, contain an elligible value (i.e. in
        self.binary_positive_classes); 0 otherwise.
        - **Multiclass**: label is an `int` which value can be any
        class index (i.e. 'species_id'). If several label match a
        coordinate set, the 1st one of the lsit is choosen.
        - **Multilabel**: label is a list of class index, containing all
        species_id observed to a coordinate set.

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
            return [1] if set(label) & set(self.binary_positive_classes) else [0]
        if self.task == 'classification_multiclass':
            return label[0]
        if self.task == 'classification_multilabel':
            return to_one_hot_encoding(label, self.unique_labels)
        return label

    def __getitem__(self, query: Union[dict, tuple, list, set, BoundingBox]) -> Dict[str, Any]:
        """Query an item from the dataset.

        Supports querying the dataset with coordinates in the dataset's CRS
        or in another CRS.
        The dataset is always queried with a torchgeo BoundingBox because it is
        itself a torchgeo dataset, but the query in this getter method can be
        passed as a tuple, list, set, dict or BoundingBox.
        Use case 1:
            query is a [list, tuple, set] of 2 elements : lon, lat.
            Here the CRS and Units system are by default those of the dataset's.
        Use case 2:
            query is a torchgeo BoundingBox.
            Here the CRS and Units system are by default those of the dataset's.
        Use case 3:
            query is a dict containing the following necessary keys: {'lon', 'lat'},
            and optional keys: {'crs', 'units', 'size'} which values default to those of
            the dataset's.

        In Use case 3, if the 'crs' key is registered and it is different from
        the dataset's CRS, the coordinates of the point are projected into the
        dataset's CRS and the value of the key is overwritten by said CRS.

        Use cases 1 and 3 give the possibility to easily query the dataset using
        only a point and a bounding box (bbox) size, using the desired input CRS.

        The unit of measurement of the bbox can be set to ['m', 'meters', 'metres']
        even if the dataset's unit is different as the points will be projected
        in the nearest meter-based CRS (see self.point_to_bbox()). Note that
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
            taht it is equal to the dataset's.
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
                raise Exception("Your chosen point lands outside of your dataset CRS after projection.")

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

            query = self.point_to_bbox(query['lon'], query['lat'], query['size'], query['units'], query['crs'])

            # Use Case 2
            patch = super().__getitem__(query)
            df = pd.DataFrame(self.coordinates, columns=['lon', 'lat'])
            label = self.targets[df.index[(df['lon'] == query_lon) & (df['lat'] == query_lat)].values]
            label = self._format_label_to_task(label)
            return patch['image'], label
        return super().__getitem__(query)

    # NOTE: Fix me to not be rgb specifics
    def plot(self, sample) -> npt.NDArray:
        """Plot a dataset's sample and return the figure.

        Parameters
        ----------
        sample : dict
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)
    
        return fig


class RasterSentinel2(RasterTorchGeoDataset):
    filename_glob = "T*_B0*_10m.tif"
    filename_regex = r"T31TEJ_20190801T104031_(?P<band>B0[\d])"
    date_format = "%Y%m%dT%H%M%S"
    is_image = True
    separate_files = True
    all_bands = ["B02", "B03", "B04", "B08"]
    rgb_bands = ["B04", "B03", "B02"]

    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample[rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig


class Sentinel2GeoSampler(GeoSampler):
    def __init__(self,
                 dataset: GeoDataset,
                 size: Union[Tuple[float, float], float],
                 length: Optional[int] = None,
                 roi: Optional[BoundingBox] = None,
                 units: Units = 'pixel',
                 crs: str = 'crs',
    ) -> None:
        super().__init__(dataset, roi)
        self.units = units
        self.crs = crs
        self.size = (size, size) if isinstance(size, (int, float)) else size
        self.coordinates = dataset.coordinates
        self.length = length if length is not None else len(dataset.observation_ids)

    def __iter__(self) -> Iterator[BoundingBox]:
        for _ in range(len(self)):
            coords = tuple(self.coordinates[_])
            # if is_point_in_bbox(coords, self.bounds):
            yield {'lon': coords[0], 'lat': coords[1],
                   'crs': self.crs,
                   'size': self.size,
                   'units': self.units}

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length
