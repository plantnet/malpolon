from pathlib import Path

import numpy as np
import pytest
import torch

from malpolon.data.datasets.torchgeo_datasets import RasterTorchGeoDataset

DATA_PATH = Path("./")  # malpolon/tests/")


def test_patch_query_torchgeo():
    class Sentinel2(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_sentinel2_test.tif"
        is_image = True
        separate_files = True
        all_bands = ["B08"]
        rgb_bands = ["B08"]

    class MicroLifeClef(RasterTorchGeoDataset):
        filename_glob = DATA_PATH / "torchgeo_mlc_test.tif"
        is_image = True
        separate_files = True
        all_bands = ["bio_1"]
        rgb_bands = ["bio_1"]
    dataset_s2, dataset_mlc = Sentinel2('./'), MicroLifeClef('./')
    patch1_s2 = dataset_s2[{'lon': 3.87075, 'lat': 43.61135, 'crs': dataset_s2.crs_pyproj.geodetic_crs, 'units': 'pixel', 'size': (100, 100)}][0][0]
    patch2_s2 = dataset_s2[{'lon': 570265.8337376957, 'lat': 4829076.115471331, 'crs': dataset_s2.crs_pyproj, 'units': 'm', 'size': 1000}][0][0]
    patch_mlc = dataset_mlc[{'lon': 3.87075, 'lat': 43.61135, 'crs': dataset_mlc.crs_pyproj, 'units': 'm', 'size': (930 * 0.7142857142857143 * 100, 930 * 100)}][0][0]
    expected_patch_s2 = torch.load(DATA_PATH / 'torchgeo_sentinel2_expected.raw')
    expected_patch_mlc = torch.load(DATA_PATH / 'torchgeo_mlc_expected.raw')
    assert tuple(patch1_s2.shape) == expected_patch_s2.shape
    assert tuple(patch2_s2.shape) == expected_patch_s2.shape
    assert tuple(patch_mlc.shape) == expected_patch_mlc.shape
    np.testing.assert_allclose(patch1_s2, expected_patch_s2)
    np.testing.assert_allclose(patch2_s2, expected_patch_s2)
    np.testing.assert_allclose(patch_mlc, expected_patch_mlc)


test_patch_query_torchgeo()
