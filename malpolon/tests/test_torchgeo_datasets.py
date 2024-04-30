"""This script tests the torchgeo datasets module.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import torch

from malpolon.data.datasets.torchgeo_datasets import RasterTorchGeoDataset

DATA_PATH = Path("malpolon/tests/data/")


def test_patch_query_torchgeo():
    class Sentinel2(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_sentinel2_test_sample.tif"
        is_image = True
        separate_files = True
        all_bands = ["B08"]
        rgb_bands = ["B08"]

    class MicroLifeClef(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_mlc_test_sample.tif"
        is_image = True
        separate_files = True
        all_bands = ["bio_1"]
        rgb_bands = ["bio_1"]
    dataset_s2, dataset_mlc = Sentinel2('./'), MicroLifeClef('./')
    patch1_s2 = dataset_s2[{'lon': 3.87075, 'lat': 43.61135, 'crs': dataset_s2.crs_pyproj.geodetic_crs, 'units': 'pixel', 'size': (100, 100)}][0][0]
    patch2_s2 = dataset_s2[{'lon': 570265.8337376957, 'lat': 4829076.115471331, 'crs': dataset_s2.crs_pyproj, 'units': 'm', 'size': 1000}][0][0]
    mlc_center = (dataset_mlc.bounds[1] + dataset_mlc.bounds[0])/2, (dataset_mlc.bounds[3] + dataset_mlc.bounds[2])/2
    patch_mlc = dataset_mlc[{'lon': mlc_center[0], 'lat': mlc_center[1], 'crs': dataset_mlc.crs_pyproj, 'units': 'pixel', 'size': 200}][0][0]
    expected_patch_s2 = torch.load(DATA_PATH / 'torchgeo_sentinel2_expected.raw')
    expected_patch_mlc = torch.load(DATA_PATH / 'torchgeo_mlc_expected.raw')
    assert tuple(patch1_s2.shape) == expected_patch_s2.shape
    assert tuple(patch2_s2.shape) == expected_patch_s2.shape
    assert tuple(patch_mlc.shape) == expected_patch_mlc.shape
    np.testing.assert_allclose(patch1_s2, expected_patch_s2)
    np.testing.assert_allclose(patch2_s2, expected_patch_s2)
    np.testing.assert_allclose(patch_mlc, expected_patch_mlc)

def test_load_observation_data() -> None:
    keys = {'x': 'longitude',
            'y': 'latitude',
            'index': 'observation_id',
            'species_id': 'species_id',
            'split': 'subset'}
    class Sentinel2(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_sentinel2_test_sample.tif"
        is_image = True
        separate_files = True
        all_bands = ["B08"]
        rgb_bands = ["B08"]
    dataset_s2 = Sentinel2('./')
    df = dataset_s2._load_observation_data(root=DATA_PATH,
                                           obs_fn='sentinel2_raster_torchgeo.csv',
                                           keys=keys)
    assert type(df) is pd.DataFrame

def test_coords_transform() -> None:
    class Sentinel2(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_sentinel2_test_sample.tif"
        is_image = True
        separate_files = True
        all_bands = ["B08"]
        rgb_bands = ["B08"]
    dataset_s2 = Sentinel2('./')
    querys = {'lon': [3.87075, 3.87075, 570265.8337376957],
              'lat': [43.61135, 43.61135, 4829076.115471331],
              'input_crs': [pyproj.CRS.from_epsg(4326), "4326", "self"],  # Sentinel-2 CRS is 32631
              'output_crs': [pyproj.CRS.from_epsg(4326), "4326", "self"]}
    expected = {0: (3.87075, 43.61135),
                1: (3.87075, 43.61135),
                2: (570265.8337376957, 4829076.115471331)}
    for o in range(len(querys['output_crs'])):
        for i in range(len(querys['input_crs'])):
            coords = dataset_s2.coords_transform(querys['lon'][i], querys['lat'][i],
                                                 querys['input_crs'][i],
                                                 querys['output_crs'][o])
            np.testing.assert_allclose(round(coords[0], 5),
                                       round(expected[o][0], 5),
                                       rtol=1e-5)

def test_point_to_bbox() -> None:
    class MicroLifeClef(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_mlc_test_sample.tif"
        is_image = True
        separate_files = True
        all_bands = ["bio_1"]
        rgb_bands = ["bio_1"]
    dataset_mlc = MicroLifeClef('./')
    # Meters
    bbox = dataset_mlc.point_to_bbox(3.87075,  # 570265.8337376957
                                     43.61135,  # 4829076.115471331
                                     size=200,
                                     units='m',
                                     crs='4326')
    bbox = np.array([bbox.minx, bbox.maxx, bbox.miny, bbox.maxy])
    expected_bbox = np.array([3.8694875, 3.8719917, 43.6104583, 43.61224])
    np.testing.assert_allclose(bbox.round(5),
                               expected_bbox.round(5),
                               rtol=1e-4)
    # Pixels
    bbox = dataset_mlc.point_to_bbox(3.87075,
                                     43.61135,
                                     size=50,
                                     units='pixel',
                                     crs='4326')
    bbox = np.array([bbox.minx, bbox.maxx, bbox.miny, bbox.maxy])
    expected_bbox = np.array([3.6624166666666667, 4.079083333333333, 43.403016666666666, 43.81968333333334])
    np.testing.assert_allclose(bbox.round(5),
                               expected_bbox.round(5),
                               rtol=1e-4)
    # CRS
    bbox = dataset_mlc.point_to_bbox(3.87075,
                                     43.61135,
                                     size=0.02,
                                     units='crs',
                                     crs='4326')
    bbox = np.array([bbox.minx, bbox.maxx, bbox.miny, bbox.maxy])
    expected_bbox = np.array([3.86075, 3.88075, 43.60135, 43.62135])
    np.testing.assert_allclose(bbox.round(5),
                               expected_bbox.round(5),
                               rtol=1e-4)

def test_valid_query_point() -> None:
    class Sentinel2(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_sentinel2_test_sample.tif"
        is_image = True
        separate_files = True
        all_bands = ["B08"]
        rgb_bands = ["B08"]
    class MicroLifeClef(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_mlc_test_sample.tif"
        is_image = True
        separate_files = True
        all_bands = ["bio_1"]
        rgb_bands = ["bio_1"]
    dataset_s2, dataset_mlc = Sentinel2('./'), MicroLifeClef('./')
    query1 = {'lon': 3.87075, 'lat': 43.61135, 'crs': pyproj.CRS.from_epsg(4326), 'units': 'pixel', 'size': (100, 100)}
    query2 = {'lon': 570265.8337376957, 'lat': 4829076.115471331, 'crs': pyproj.CRS.from_epsg(32631), 'units': 'pixel', 'size': (100, 100)}
    assert dataset_s2._valid_query_point(query1)
    assert dataset_s2._valid_query_point(query2)
    assert dataset_mlc._valid_query_point(query1)
    assert dataset_mlc._valid_query_point(query2)

def test_format_label_to_task() -> None:
    class Sentinel2(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_sentinel2_test_sample.tif"
        is_image = True
        separate_files = True
        all_bands = ["B08"]
        rgb_bands = ["B08"]
    dataset_s2 = Sentinel2('./')

    dataset_s2.task = 'classification_binary'
    dataset_s2.binary_positive_classes = [1, 2]
    labels_raw = [2, 3]
    labels_formatted = dataset_s2._format_label_to_task(labels_raw)
    assert labels_formatted == 1

    dataset_s2.task = 'classification_multiclass'
    labels_raw = [3]
    labels_formatted = dataset_s2._format_label_to_task(labels_raw)
    assert labels_formatted == 3

    dataset_s2.task = 'classification_multilabel'
    dataset_s2.unique_labels = [1, 2, 3, 4, 5]
    labels_raw = [1, 4]
    labels_expected = np.zeros(5, dtype=np.float32)
    labels_expected[0] = 1
    labels_expected[-2] = 1
    labels_formatted = dataset_s2._format_label_to_task(labels_raw)
    assert all(labels_formatted == labels_expected)

if __name__ == "__main__":
    test_patch_query_torchgeo()