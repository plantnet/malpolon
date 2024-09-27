"""This script tests the GeoLifeCLEF2024 pre-extracted dataset module.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from malpolon.data.datasets.geolifeclef2024_pre_extracted import (
    TestDataset, TrainDataset, construct_patch_path, load_bioclim,
    load_landsat, load_sentinel)

ROOT_PATH = Path("malpolon/tests/data/glc24_pre_extracted/")

DATA_PATHS = {
    'train': {
        'surveyId': 1322301,
        'landsat_data_dir': str(ROOT_PATH / "TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-train-landsat_time_series/"),
        'bioclim_data_dir': str(ROOT_PATH / "TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-train-bioclimatic_monthly/"),
        'sentinel_data_dir': str(ROOT_PATH / "PA_Train_SatellitePatches_RGB/pa_train_patches_rgb/")
    },
    'test': {
        'surveyId': 1621686,
        'landsat_data_dir': str(ROOT_PATH / "TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-test-landsat_time_series/"),
        'bioclim_data_dir': str(ROOT_PATH / "TimeSeries-Cubes/TimeSeries-Cubes/GLC24-PA-test-bioclimatic_monthly/"),
        'sentinel_data_dir': str(ROOT_PATH / "PA_Test_SatellitePatches_RGB/pa_test_patches_rgb/")
    }
}


def test_construct_patch_path():
    surveyId = DATA_PATHS['train']['surveyId']
    path = Path(DATA_PATHS['train']['sentinel_data_dir'])
    patch_path = construct_patch_path(path, surveyId)

    assert str(patch_path) == str(path / "01/23/" / f"{surveyId}.jpeg")


def test_load_landsat():
    surveyId = DATA_PATHS['train']['surveyId']
    path = Path(DATA_PATHS['train']['landsat_data_dir']) / f'GLC24-PA-train-landsat-time-series_{surveyId}_cube.pt'
    x = load_landsat(path)

    assert list(x.shape) == [6, 4, 21]
    assert x.dtype == np.float32


def test_load_bioclim():
    surveyId = DATA_PATHS['train']['surveyId']
    path = Path(DATA_PATHS['train']['bioclim_data_dir']) / f'GLC24-PA-train-bioclimatic_monthly_{surveyId}_cube.pt'
    x = load_bioclim(str(path))

    assert list(x.shape) == [4, 19, 12]
    assert x.dtype == np.float32


def test_load_sentinel():
    surveyId = DATA_PATHS['train']['surveyId']
    path = DATA_PATHS['train']['sentinel_data_dir']
    x = load_sentinel(str(path), surveyId)

    assert list(x.shape) == [4, 128, 128]
    assert x.dtype == np.float32


def test_train_dataset():
    DATA_PATHS2 = deepcopy(DATA_PATHS)
    DATA_PATHS2['train'].pop('surveyId')
    DATA_PATHS2['test'].pop('surveyId')

    path = ROOT_PATH / 'metadata.csv'
    df_train = pd.read_csv(path)
    n_classes = 11255
    df_test = df_train[df_train['subset'] == 'test']
    df_train = df_train[df_train['subset'] == 'train']

    # Multilabel classification
    ds_train = TrainDataset(df_train, n_classes, **DATA_PATHS2['train'], transform=None, task='classification_multilabel')
    ds_test = TestDataset(df_test, n_classes, **DATA_PATHS2['test'], transform=None, task='classification_multilabel')

    ## Train
    res_train = ds_train[10]
    x_landsat, x_bioclim, x_sentinel, y, survey_id = res_train
    y_multilabel = torch.zeros(n_classes)
    y_multilabel[8748] = 1

    assert len(ds_train) == 48
    assert len(res_train) == 5
    assert survey_id == 1322301
    assert torch.equal(y, y_multilabel)
    assert isinstance(x_landsat, torch.Tensor)
    assert isinstance(x_bioclim, torch.Tensor)
    assert isinstance(x_sentinel, torch.Tensor)

    ## Test
    res_test = ds_test[10]
    x_landsat, x_bioclim, x_sentinel, y, survey_id = res_test
    y_multilabel = torch.zeros(n_classes)
    y_multilabel[4854] = 1

    assert len(ds_test) == 51
    assert len(res_test) == 5
    assert survey_id == 1621686
    assert torch.equal(y, y_multilabel)
    assert isinstance(x_landsat, torch.Tensor)
    assert isinstance(x_bioclim, torch.Tensor)
    assert isinstance(x_sentinel, torch.Tensor)
    assert hasattr(ds_test, 'targets') and len(ds_test.targets) == len(ds_test)
    assert hasattr(ds_test, 'observation_ids') and len(ds_test.observation_ids) == len(ds_test)

    # Multiclass classification
    ds_train = TrainDataset(df_train, n_classes, **DATA_PATHS2['train'], transform=None, task='classification_multiclass')
    ds_test = TestDataset(df_test, n_classes, **DATA_PATHS2['test'], transform=None, task='classification_multiclass')
    _, _, _, y, _ = ds_train[10]
    assert y == 8748
    assert hasattr(ds_test, 'targets') and len(ds_test.targets) == len(ds_test)
    assert hasattr(ds_test, 'observation_ids') and len(ds_test.observation_ids) == len(ds_test)
