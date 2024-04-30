"""This script tests the GeoLifeCLEF2022 dataset module.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
"""

from pathlib import Path

import numpy as np
import pytest

from malpolon.data.datasets.geolifeclef2022 import (
    GeoLifeCLEF2022Dataset, load_patch, visualize_observation_patch)
from malpolon.data.environmental_raster import PatchExtractor

DATA_PATH = Path("malpolon/tests/data/glc22")


SUBSET_SIZE = {
    "train": 1956,
    "val": 44,
    "train+val": 2000,
    "test": 46,
}

@pytest.mark.skip(reason="Slow and no guarantee of having the data available.")
@pytest.mark.parametrize("observation_id", (10561985, 10823364))
def test_load_patch(observation_id):
    patches = load_patch(observation_id, DATA_PATH, return_arrays=True)

    assert len(patches) == 4

    rgb_patch, near_ir_patch, altitude_patch, landcover_patch = patches.values()

    assert rgb_patch.shape == (256, 256, 3)
    assert rgb_patch.dtype == np.uint8

    assert near_ir_patch.shape == (256, 256)
    assert near_ir_patch.dtype == np.uint8

    assert altitude_patch.shape == (256, 256)
    assert altitude_patch.dtype == np.int16

    assert landcover_patch.shape == (256, 256)
    assert landcover_patch.dtype == np.uint8

@pytest.mark.skip(reason="Slow and no guarantee of having the data available.")
@pytest.mark.parametrize("subset", SUBSET_SIZE.keys())
def test_dataset_load_only_patches(subset):
    dataset = GeoLifeCLEF2022Dataset(DATA_PATH, subset, use_rasters=False)

    assert len(dataset) == SUBSET_SIZE[subset]

    result = dataset[0]

    if subset == "test":
        assert len(result) > 2
        data = result
    else:
        assert len(result) == 2
        data, target = result
        assert type(target) == np.int64

    assert len(data) == 4

@pytest.mark.skip(reason="Slow and no guarantee of having the data available.")
@pytest.mark.parametrize("subset", SUBSET_SIZE.keys())
def test_dataset_load_localisation(subset):
    dataset = GeoLifeCLEF2022Dataset(DATA_PATH, subset, use_rasters=False, use_localisation=True)

    assert len(dataset) == SUBSET_SIZE[subset]

    result = dataset[0]

    if subset == "test":
        assert len(result) > 2
        data = result
    else:
        assert len(result) == 2
        data, target = result
        assert type(target) == np.int64

    assert len(data) == 5
    assert len(data["localisation"]) == 2

@pytest.mark.skip(reason="Slow and no guarantee of having the data available.")
@pytest.mark.parametrize("subset", SUBSET_SIZE.keys())
def test_dataset_load_one_raster(subset):
    patch_extractor = PatchExtractor(DATA_PATH / "rasters", size=256)
    patch_extractor.append("bio_1")
    dataset = GeoLifeCLEF2022Dataset(
        DATA_PATH, subset, use_rasters=True, patch_extractor=patch_extractor
    )

    assert len(dataset) == SUBSET_SIZE[subset]

    result = dataset[0]

    if subset == "test":
        assert len(result) > 2
        data = result
    else:
        assert len(result) == 2
        data, target = result
        assert type(target) == np.int64

    assert len(data) == 5
    assert len(data["environmental_patches"]) == 1

@pytest.mark.skip(reason="Slow and no guarantee of having the data available.")
def test_dataset_load_all():
    subset = "train"
    dataset = GeoLifeCLEF2022Dataset(DATA_PATH, subset, use_rasters=True)

    assert len(dataset) == SUBSET_SIZE[subset]

    result = dataset[0]

    if subset == "test":
        assert len(result) > 2
        data = result
    else:
        assert len(result) == 2
        data, target = result
        assert type(target) == np.int64

    assert len(data) == 5
    assert len(data["environmental_patches"]) == 27

@pytest.mark.skip(reason="Slow and no guarantee of having the data available.")
@pytest.mark.parametrize("observation_id", (10561900, 22068100))
def test_patch_plotting(observation_id):
    patch = load_patch(observation_id, DATA_PATH)
    visualize_observation_patch(patch)