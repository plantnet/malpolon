
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from malpolon.plot.map import (plot_map, plot_observation_dataset,
                               plot_observation_map)

DATA_PATH = Path("malpolon/tests/data/")

def test_plot_map() -> None:
    try:
        ax = plot_map(region="fr")
        ax = plot_map(region="us")
        df = pd.read_csv(f'{DATA_PATH}/sentinel2_raster_torchgeo.csv')
        extent = [min(df['longitude']) - 1, max(df['longitude']) + 1,
                  min(df['latitude']) - 1, max(df['latitude']) + 1]
        ax = plot_map(extent=extent)
    except Exception as e:  # pylint: disable=W0718
        print(e)
        assert False
    try:
        e = False
        ax = plot_map()
    except ValueError:
        e = True
    assert e

def test_plot_observation_map() -> None:
    try:
        df = pd.read_csv(f'{DATA_PATH}/sentinel2_raster_torchgeo.csv')
        ax = plot_map(extent=[min(df['longitude']) - 1, max(df['longitude']) + 1,
                              min(df['latitude']) - 1, max(df['latitude']) + 1])
        ax = plot_observation_map(longitudes=df['longitude'].values, latitudes=df['latitude'].values,
                                  ax=ax, c='b', label='train')
    except Exception as e:  # pylint: disable=W0718
        print(e)
        assert False

def test_plot_observation_dataset() -> None:
    try:
        df = pd.read_csv(f'{DATA_PATH}/sentinel2_raster_torchgeo.csv')
        obs_data_columns = {'x': 'longitude', 'y': 'latitude', 'index': 'surveyId', 'split': 'subset'}
        plot_observation_dataset(df=df, obs_data_columns=obs_data_columns, show_map=False)
    except Exception as e:  # pylint: disable=W0718
        print(e)
        assert False
