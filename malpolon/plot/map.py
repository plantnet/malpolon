"""Utilities for plotting maps.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>

"""

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt


def plot_map(
    *,
    region: Optional[str] = None,
    extent: Optional[npt.ArrayLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a map on which to show the observations.

    Parameters
    ----------
    region: string, either "fr" or "us"
        Region to show, France or US.
    extent: array-like of form [longitude min, longitude max, latitude min, latitude max]
        Explicit extent of the area to show, e.g., for zooming.
    ax: plt.Axes
        Provide an Axes to use instead of creating one.

    Returns
    -------
    plt.Axes:
        Returns the used Axes.
    """
    if region == "fr":
        extent = [-5.5, 10, 41, 52]
    elif region == "us":
        extent = [-126, -66, 24, 50]
    elif region is None and extent is None:
        raise ValueError("Either region or extent must be set")

    # Import outside toplevel to ease package management, especially
    # when working on a computing cluster because cartopy requires
    # binaries to be installed.
    import cartopy.crs as ccrs  # pylint: disable=C0415
    import cartopy.feature as cfeature  # pylint: disable=C0415

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_0_countries",
        scale="10m",
        facecolor="none",
    )

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(states_provinces, edgecolor="gray")

    ax.gridlines(
        draw_labels=True, dms=True, x_inline=False, y_inline=False, linestyle="--"
    )
    ax.set_aspect(1.25)

    return ax


def plot_observation_map(
    *,
    longitudes: npt.ArrayLike,
    latitudes: npt.ArrayLike,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot observations on a map.

    Parameters
    ----------
    longitude: array-like
        Longitudes of the observations.
    latitude: array-like
        Latitudes of the observations.
    ax: plt.Axes
        Provide an Axes to use instead of creating one.
    kwargs:
        Additional arguments to pass to plt.scatter.

    Returns
    -------
    plt.Axes:
        Returns the used Axes.
    """
    # Import outside toplevel to ease package management, especially
    # when working on a computing cluster because cartopy requires
    # binaries to be installed.
    import cartopy.crs as ccrs  # pylint: disable=C0415

    if ax is None:
        ax = plot_map()

    ax.scatter(longitudes, latitudes, transform=ccrs.Geodetic(), **kwargs)
    ax.legend()

    return ax


def plot_observation_dataset(
    *,
    df: pd.DataFrame,
    obs_data_columns: dict = {'x': 'lon',
                              'y': 'lat',
                              'index': 'surveyId',
                              'species_id': 'speciesId',
                              'split': 'subset'},
    show_map: bool = False
) -> plt.Axes:
    """Plot observations on a map from an observation dataset.

    This method expects a pandas DataFrame with columns containing
    coordinates, species ids, and dataset split information ('train',
    'test' or 'val').
    Users can specify the names of the columns containing these informations
    if they do not match the default names.

    Parameters
    ----------
    df : pd.DataFrame
        observation dataset
    obs_data_columns : _type_, optional
        dictionary matching custom dataframe keys with necessary keys,
        by default {'x': 'lon', 'y': 'lat', 'index': 'surveyId', 'species_id': 'speciesId', 'split': 'subset'}
    show_map : bool, optional
        if True, displays the map, by default False

    Returns
    -------
    plt.Axes
        map's ax object
    """
    obs_data_columns = dict(zip(obs_data_columns.values(), obs_data_columns.keys()))
    df = df.rename(columns=obs_data_columns)

    ax = plot_map(extent=[min(df['x']) - 1, max(df['x']) + 1,
                          min(df['y']) - 1, max(df['y']) + 1])
    colors = cycle('rbgcmykw')
    for split, group in df.groupby('split'):
        ax = plot_observation_map(longitudes=group['x'].values, latitudes=group['y'].values,
                                  ax=ax, c=next(colors), label=split)

    if show_map:
        plt.show()
    return ax
