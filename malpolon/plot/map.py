from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import numpy.typing as npt


def plot_map(
    *,
    region: Optional[str] = None,
    extent: Optional[npt.ArrayLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plots a map to show the observations on

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

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

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
