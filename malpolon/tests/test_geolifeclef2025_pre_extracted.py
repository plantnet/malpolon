"""This script tests the GeoLifeCLEF2024 pre-extracted dataset module.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchvision import transforms
from malpolon.data.datasets.geolifeclef2025_pre_extracted import (
    TestDataset, TrainDataset, construct_patch_path, load_bioclim,
    load_landsat, load_sentinel)

ROOT_PATH = Path("malpolon/tests/data/glc25_pre_extracted/")
TRANSFORMS = transforms.Compose([torch.tensor])
DATA_PATHS = {
    'train': {
        'surveyId': 1027998,
        'landsat_data_dir': str(ROOT_PATH / "SateliteTimeSeries-Landsat/cubes/PA-train/"),
        'bioclim_data_dir': str(ROOT_PATH / "BioclimTimeSeries/cubes/PA-train/"),
        'sentinel_data_dir': str(ROOT_PATH / "SatelitePatches/PA-train")
    },
    'test': {
        'surveyId': 5000108,
        'landsat_data_dir': str(ROOT_PATH / "SateliteTimeSeries-Landsat/cubes/PA-test/"),
        'bioclim_data_dir': str(ROOT_PATH / "BioclimTimeSeries/cubes/PA-test/"),
        'sentinel_data_dir': str(ROOT_PATH / "SatelitePatches/PA-test")
    }
}


def test_construct_patch_path():
    surveyId = DATA_PATHS['train']['surveyId']
    path = Path(DATA_PATHS['train']['sentinel_data_dir'])
    patch_path = construct_patch_path(path, surveyId)

    assert str(patch_path) == str(path / "98/79/" / f"{surveyId}.tiff")


def test_load_landsat():
    surveyId = DATA_PATHS['train']['surveyId']
    path = Path(DATA_PATHS['train']['landsat_data_dir']) / f'GLC25-PA-train-landsat-time-series_{surveyId}_cube.pt'
    x = load_landsat(str(path))

    assert list(x.shape) == [6, 4, 21]
    assert x.dtype == np.float32

    x = load_landsat(str(path), transform=TRANSFORMS)
    assert x.dtype == torch.float32


def test_load_bioclim():
    surveyId = DATA_PATHS['train']['surveyId']
    path = Path(DATA_PATHS['train']['bioclim_data_dir']) / f'GLC25-PA-train-bioclimatic_monthly_{surveyId}_cube.pt'
    x = load_bioclim(str(path))

    assert list(x.shape) == [4, 19, 12]
    assert x.dtype == np.float32

    x = load_bioclim(str(path), transform=TRANSFORMS)
    assert x.dtype == torch.float32


def test_load_sentinel():
    surveyId = DATA_PATHS['train']['surveyId']
    path = construct_patch_path(DATA_PATHS['train']['sentinel_data_dir'], surveyId)
    x = load_sentinel(str(path))

    assert list(x.shape) == [4, 64, 64]
    assert x.dtype == np.float32

    x = load_sentinel(str(path), transform=TRANSFORMS)
    assert x.dtype == torch.float32


def test_train_dataset():
    DATA_PATHS2 = deepcopy(DATA_PATHS)
    DATA_PATHS2['train'].pop('surveyId')
    DATA_PATHS2['test'].pop('surveyId')

    path = ROOT_PATH / 'metadata.csv'
    df_train = pd.read_csv(path)
    n_classes = 11255
    df_test = df_train[df_train['subset'] == 'test']
    df_train = df_train[df_train['subset'] == 'train']

    # Train
    ds_train_multiclass = TrainDataset(df_train, n_classes, **DATA_PATHS2['train'], transform=None, task='classification_multiclass')
    ds_train_multilabel = TrainDataset(df_train, n_classes, **DATA_PATHS2['train'], transform=None, task='classification_multilabel')

    ## Multiclass classification
    res_train_mc = ds_train_multiclass[0]
    x_landsat_mc, x_bioclim_mc, x_sentinel_mc, y_mc, survey_id_mc = res_train_mc  # pylint: disable=W0632

    assert len(ds_train_multiclass) == 48
    assert len(res_train_mc) == 5
    assert y_mc == 10113
    assert survey_id_mc == 1027998
    assert isinstance(x_landsat_mc, torch.Tensor)
    assert isinstance(x_bioclim_mc, torch.Tensor)
    assert isinstance(x_sentinel_mc, torch.Tensor)

    ## Multilabel classification
    res_train_ml = ds_train_multilabel[0]
    x_landsat_ml, x_bioclim_ml, x_sentinel_ml, y_ml, survey_id_ml = res_train_ml  # pylint: disable=W0632
    y_multilabel = torch.zeros(n_classes)
    y_multilabel[[10111, 10112, 10113]] = 1

    assert len(ds_train_multilabel) == 50
    assert len(res_train_ml) == 5
    assert torch.equal(y_ml, y_multilabel)
    assert survey_id_ml == 1027998
    assert isinstance(x_landsat_ml, torch.Tensor)
    assert isinstance(x_bioclim_ml, torch.Tensor)
    assert isinstance(x_sentinel_ml, torch.Tensor)


    # Test
    ds_test_multiclass = TestDataset(df_test, n_classes, **DATA_PATHS2['test'], transform=None, task='classification_multiclass')
    ds_test_multilabel = TestDataset(df_test, n_classes, **DATA_PATHS2['test'], transform=None, task='classification_multilabel')

    # Multiclass classification
    res_test_mc = ds_test_multiclass[0]
    x_landsat_mc, x_bioclim_mc, x_sentinel_mc, y_mc, survey_id_mc = res_test_mc  # pylint: disable=W0632
    assert len(ds_test_multiclass) == 49
    assert len(res_test_mc) == 5
    assert survey_id_mc == 5000108
    assert y_mc == 1
    assert isinstance(x_landsat_mc, torch.Tensor)
    assert isinstance(x_bioclim_mc, torch.Tensor)
    assert isinstance(x_sentinel_mc, torch.Tensor)
    assert hasattr(ds_test_multiclass, 'targets') and len(ds_test_multiclass.targets) == len(ds_test_multiclass)
    assert hasattr(ds_test_multiclass, 'observation_ids') and len(ds_test_multiclass.observation_ids) == len(ds_test_multiclass)

    ## Multilabel classification
    res_test_ml = ds_test_multilabel[0]
    x_landsat_ml, x_bioclim_ml, x_sentinel_ml, y_ml, survey_id_ml = res_test_ml  # pylint: disable=W0632
    y_multilabel = torch.zeros(n_classes)
    y_multilabel[[1, 2]] = 1

    assert len(ds_test_multilabel) == 50
    assert len(res_test_ml) == 5
    assert survey_id_ml == 5000108
    assert torch.equal(y_ml, y_multilabel)
    assert isinstance(x_landsat_ml, torch.Tensor)
    assert isinstance(x_bioclim_ml, torch.Tensor)
    assert isinstance(x_sentinel_ml, torch.Tensor)
    assert hasattr(ds_test_multilabel, 'targets') and len(ds_test_multilabel.targets) == len(ds_test_multilabel)
    assert hasattr(ds_test_multilabel, 'observation_ids') and len(ds_test_multilabel.observation_ids) == len(ds_test_multilabel)
