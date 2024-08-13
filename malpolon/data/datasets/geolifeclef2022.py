"""This module provides Datasets and Providers for GeoLifeCLEF2022 data.

This module has since been updated for GeoLifeCLEF2023

Author: Benjamin Deneu <benjamin.deneu@inria.fr>
        Titouan Lorieul <titouan.lorieul@gmail.com>

License: GPLv3
Python version: 3.8
"""

from __future__ import annotations

import os
import subprocess
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib import colormaps
from matplotlib.patches import Patch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision.datasets.utils import (download_and_extract_archive,
                                        extract_archive)

from malpolon.data.environmental_raster import PatchExtractor

from ...plot.map import plot_map
from ._base import DATA_MODULE

if TYPE_CHECKING:
    from collections.abc import Collection

    import numpy.typing as npt

    Patches = npt.NDArray
    Targets = npt.NDArray[np.int64]


def load_patch(
    observation_id: Union[int, str],
    patches_path: Union[str, Path],
    *,
    data: Union[str, list[str]] = "all",
    landcover_mapping: Optional[npt.NDArray] = None,
    return_arrays: bool = True,
    subfolder_strategy: bool = True,
) -> dict[str, Patches]:
    """Load the patch data associated to an observation id.

    Parameters
    ----------
    observation_id : integer / string
        Identifier of the observation.
    patches_path : string / pathlib.Path
        Path to the folder containing all the patches.
    data : string or list of string
        Specifies what data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    landcover_mapping : 1d array-like
        Facultative mapping of landcover codes, useful to align France and US codes.
    return_arrays : boolean
        If True, returns all the patches as Numpy arrays (no PIL.Image returned).
    subfolder_strategy : boolean
        If True, includes subfoldering strategy in the filename path, based on observation_id.

    Returns
    -------
    patches : dict containing 2d array-like objects
        Returns a dict containing the requested patches.
    """
    observation_id = str(observation_id)
    region, subfolder1, subfolder2 = "", "", ""

    if subfolder_strategy:
        region_id = observation_id[:1]
        if region_id == "1":
            region = "patches-fr"
        elif region_id == "2":
            region = "patches-us"
        else:
            raise ValueError(f"Incorrect 'observation_id' {observation_id},"
                             f" can not extract region id from it")

        subfolder1 = observation_id[-2:]
        subfolder2 = observation_id[-4:-2]

    filename = Path(patches_path) / region / subfolder1 / subfolder2 / observation_id

    patches = {}

    if data == "all":
        data = ["rgb", "near_ir", "landcover", "altitude"]

    if "rgb" in data:
        rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
        rgb_patch = Image.open(rgb_filename)
        if return_arrays:
            rgb_patch = np.array(rgb_patch)
        patches["rgb"] = rgb_patch

    if "near_ir" in data:
        near_ir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
        near_ir_patch = Image.open(near_ir_filename)
        if return_arrays:
            near_ir_patch = np.array(near_ir_patch)
        patches["near_ir"] = near_ir_patch

    if "altitude" in data:
        altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
        altitude_patch = tifffile.imread(altitude_filename)
        patches["altitude"] = altitude_patch

    if "landcover" in data:
        landcover_filename = filename.with_name(filename.stem + "_landcover.tif")
        landcover_patch = tifffile.imread(landcover_filename)
        if landcover_mapping is not None:
            landcover_patch = landcover_mapping[landcover_patch]
        patches["landcover"] = landcover_patch

    return patches


def visualize_observation_patch(
    patch: dict[str, Patches],
    *,
    observation_data: Optional[pd.Series] = None,
    landcover_labels: Optional[Collection] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """Plot patch data.

    Parameters
    ----------
    patch : dict containing 2d array-like objects
        Patch data as returned by `load_patch`.
    observation_data : pandas Series
        Row of the dataframe containing the data of the observation.
    landcover_labels : list
        Labels corresponding to the landcover codes.
    return_fig : boolean
        If True, returns the created plt.Figure object

    Returns
    -------
    fig : plt.Figure
        If return_fig is True, the used plt.Figure object    Returns
    """
    if landcover_labels is None:
        n_labels = np.max(patch["landcover"]) + 1
        landcover_labels = np.arange(n_labels)

    cmap = colormaps["viridis"].resampled(len(landcover_labels))

    legend_elements = []
    for landcover_label, color in zip(landcover_labels, cmap.colors):
        legend_elements.append(Patch(color=color, label=landcover_label))

    if observation_data is not None:
        localisation = observation_data[["latitude", "longitude"]]
        species_id = getattr(observation_data, "species_id", None)
        species_name = getattr(observation_data, "GBIF_species_name", None)
        kingdom_name = getattr(observation_data, "GBIF_kingdom_name", None)

        fig = plt.figure(figsize=(15, 10))

        g_s = fig.add_gridspec(1, 2, width_ratios=[1, 2])

        gs1 = g_s[0].subgridspec(1, 1)
        axe = fig.add_subplot(gs1[0], projection=ccrs.PlateCarree())
        region = "fr" if localisation[1] > -6 else "us"
        plot_map(region=region, ax=axe)
        axe.scatter(
            localisation[1],
            localisation[0],
            marker="o",
            s=100,
            transform=ccrs.PlateCarree(),
        )
        axe.set_title("Observation localisation")

        txt = f"Observation id: {observation_data.name}"
        txt += f"\nLocalisation: {localisation[0]:.3f}, {localisation[1]:.3f}"
        if species_id:
            txt += f"\nSpecies id: {species_id}"
        if species_name:
            txt += f"\nSpecies name: {species_name}"
        if kingdom_name:
            txt += f"\nKingdom: {kingdom_name}"
        pos = axe.get_position()
        fig.text(
            pos.x1 - pos.x0, pos.y0 - 0.2 * (pos.y1 - pos.y0), txt,
            ha="center", va="top"
        )

        gs2 = g_s[1].subgridspec(2, 2)
        axes = np.asarray([fig.add_subplot(gs) for gs in gs2])
    else:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))

    axes = axes.ravel()
    axes_iter = iter(axes)

    axe = next(axes_iter)
    axe.imshow(patch["rgb"])
    axe.set_title("RGB image")

    axe = next(axes_iter)
    axe.imshow(patch["near_ir"], cmap="gray")
    axe.set_title("Near-IR image")

    axe = next(axes_iter)
    vmin = round(patch["altitude"].min(), -1)
    vmax = round(patch["altitude"].max(), -1) + 10
    axe.imshow(patch["altitude"])
    cs2 = axe.contour(
        patch["altitude"],
        levels=np.arange(vmin, vmax, step=10),
        colors="w",
    )
    axe.clabel(cs2, inline=True, fontsize=10)
    axe.set_aspect("equal")
    axe.set_title("Altitude (in meters)")

    axe = next(axes_iter)
    axe.imshow(
        patch["landcover"],
        interpolation="none",
        cmap=cmap,
        vmin=0,
        vmax=len(legend_elements),
    )
    axe.set_title("Land cover")
    visible_landcover_categories = np.unique(patch["landcover"])
    legend = [legend_elements[i] for i in visible_landcover_categories]
    axe.legend(
        handles=legend, handlelength=0.75, bbox_to_anchor=(1, 0.5), loc="center left"
    )

    for axe in axes:
        axe.axis("off")

    if observation_data is None:
        fig.tight_layout()

    if return_fig:
        return fig
    return None


class GeoLifeCLEF2022Dataset(Dataset):
    """Pytorch dataset handler for GeoLifeCLEF 2022 dataset.

    Parameters
    ----------
    root : string or pathlib.Path
        Root directory of dataset.
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    region : string, either "both", "fr" or "us"
        Load the observations of both France and US or only a single region.
    patch_data : string or list of string
        Specifies what type of patch data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    use_rasters : boolean (optional)
        If True, extracts patches from environmental rasters.
    patch_extractor : PatchExtractor object (optional)
        Patch extractor to use if rasters are used.
    use_localisation : boolean
        If True, returns also the localisation as a tuple (latitude, longitude).
    transform : callable (optional)
        A function/transform that takes a list of arrays and returns a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    """
    def __init__(
        self,
        root: Union[str, Path],
        subset: str,
        *,
        region: str = "both",
        patch_data: str = "all",
        use_rasters: bool = True,
        patch_extractor: Optional[PatchExtractor] = None,
        use_localisation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ):
        root = Path(root)

        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError(f"Possible values for 'subset' are:"
                             f" {possible_subsets} (given {subset})")

        possible_regions = ["both", "fr", "us"]
        if region not in possible_regions:
            raise ValueError(f"Possible values for 'region' are:"
                             f" {possible_regions} (given {region})")

        self.root = root
        self.subset = subset
        self.region = region
        self.patch_data = patch_data
        self.use_localisation = use_localisation
        self.transform = transform
        self.target_transform = target_transform
        self.training = subset != "test"
        self.n_classes = 17037

        if download:
            self.download()

        df = self._load_observation_data(self.root,
                                         self.region,
                                         self.subset)

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values

        if self.training:
            self.targets = df["species_id"].values
        else:
            self.targets = None

        self.patch_extractor: Optional[PatchExtractor] = None

        if use_rasters:
            if patch_extractor is None:
                patch_extractor = PatchExtractor(self.root / "rasters",
                                                 size=256)
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor

    def download(self):
        """Download the GeolifeClef2023 dataset."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        try:
            import kaggle  # pylint: disable=C0415,W0611 # noqa: F401
        except OSError as error:
            raise OSError("Have you properly set up your Kaggle API token ? For more information, please refer to section 'Authentication' of the kaggle documentation : https://www.kaggle.com/docs/api") from error

        answer = input("You are about to download the GeoLifeClef2022 dataset which weighs ~62 GB. Do you want to continue ? [y/n]")
        if answer.lower() in ["y", "yes"]:
            if 'geolifeclef-2022-lifeclef-2022-fgvc9' in self.root.parts:
                self.root = self.root.parent
            subprocess.call(f"kaggle competitions download -c geolifeclef-2022-lifeclef-2022-fgvc9 -p {self.root}", shell=True)
            print(f"Extracting geolifeclef-2022-lifeclef-2022-fgvc9 to {self.root}")
            extract_archive(os.path.join(self.root, "geolifeclef-2022-lifeclef-2022-fgvc9.zip"), os.path.join(self.root, "geolifeclef-2022-lifeclef-2022-fgvc9/"))
            if self.root.parts[-1] != "geolifeclef-2022-lifeclef-2022-fgvc9":
                self.root = self.root / "geolifeclef-2022-lifeclef-2022-fgvc9"
        else:
            print("Aborting download")
            return

    def _check_integrity(self):
        return (self.root / "observations/observations_fr_train.csv").exists()

    def _load_observation_data(
        self,
        root: Path,
        region: str,
        subset: str,
    ) -> pd.DataFrame:
        if subset == "test":
            subset_file_suffix = "test"
        else:
            subset_file_suffix = "train"

        df_fr = pd.read_csv(
            root / "observations" / f"observations_fr_{subset_file_suffix}.csv",
            sep=";",
            index_col="observation_id",
        )
        df_us = pd.read_csv(
            root / "observations" / f"observations_us_{subset_file_suffix}.csv",
            sep=";",
            index_col="observation_id",
        )

        if region == "both":
            df = pd.concat((df_fr, df_us))
        elif region == "fr":
            df = df_fr
        elif region == "us":
            df = df_us

        if subset not in ["train+val", "test"]:
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        return df

    def __len__(self) -> int:
        """Return the number of observations in the dataset."""
        return len(self.observation_ids)

    def __getitem__(
        self,
        index: int,
    ) -> Union[dict[str, Patches], tuple[dict[str, Patches], Targets]]:
        """Return a dataset item.

        Args:
            index (int): dataset id.

        Returns:
            Union[dict[str, Patches], tuple[dict[str, Patches], Targets]]:
                data and labels corresponding to the dataset id.
        """
        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]

        patches = load_patch(observation_id, self.root, data=self.patch_data)

        # Extracting patch from rasters
        if self.patch_extractor is not None:
            environmental_patches = self.patch_extractor[(latitude, longitude)]
            patches["environmental_patches"] = environmental_patches

        if self.use_localisation:
            patches["localisation"] = np.asarray(
                [latitude, longitude], dtype=np.float32
            )

        if self.transform:
            patches = self.transform(patches)

        if self.training:
            target = self.targets[index]

            if self.target_transform:
                target = self.target_transform(target)

            return patches, target
        return patches, -1


class MiniGeoLifeCLEF2022Dataset(GeoLifeCLEF2022Dataset):
    """Pytorch dataset handler for a subset of GeoLifeCLEF 2022 dataset.

    It consists in a restriction to France and to the 100 most present
    plant species.

    Parameters
    ----------
    root : string or pathlib.Path
        Root directory of dataset.
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    patch_data : string or list of string
        Specifies what type of patch data to load, possible values: 'all',
        'rgb', 'near_ir', 'landcover' or 'altitude'.
    use_rasters : boolean (optional)
        If True, extracts patches from environmental rasters.
    patch_extractor : PatchExtractor object (optional)
        Patch extractor to use if rasters are used.
    use_localisation : boolean
        If True, returns also the localisation as a tuple (latitude, longitude).
    transform : callable (optional)
        A function/transform that takes a list of arrays and returns
        a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: str,
        *,
        patch_data: str = "all",
        use_rasters: bool = True,
        patch_extractor: Optional[PatchExtractor] = None,
        use_localisation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs,
    ):
        super().__init__(
            root,
            subset,
            region="fr",
            patch_data=patch_data,
            use_rasters=use_rasters,
            patch_extractor=patch_extractor,
            use_localisation=use_localisation,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.n_classes = 100

    def _load_observation_data(
        self,
        root: Path,
        region: str,
        subset: str,
    ) -> pd.DataFrame:
        if subset == "test":
            subset_file_suffix = "test"
        else:
            subset_file_suffix = "train"

        df = pd.read_csv(
            root / "observations" / f"observations_fr_{subset_file_suffix}.csv",
            sep=";",
            index_col="observation_id",
        )[:600]

        if subset == 'test':
            df = df.iloc[np.random.randint(0, len(df), 100)]
            df['species_id'] = [None] * len(df)
        else:
            file_name = "minigeolifeclef2022_species_details.csv"
            with resources.path(DATA_MODULE, file_name) as species_file_path:
                df_species = pd.read_csv(
                    species_file_path,
                    sep=";",
                    index_col="species_id",
                )[:600]

            df = df[np.isin(df["species_id"], df_species.index)]
            value_counts = df.species_id.value_counts()
            species_id = value_counts.iloc[:100].index
            df_species = df_species.loc[species_id]
            df = df[np.isin(df["species_id"], df_species.index)]

            label_encoder = LabelEncoder().fit(df_species.index)
            df["species_id"] = label_encoder.transform(df["species_id"])
            df_species.index = label_encoder.transform(df_species.index)

        if subset not in ["train+val", "test"]:
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        return df


class MicroGeoLifeCLEF2022Dataset(Dataset):
    """Pytorch dataset handler for a subset of GeoLifeCLEF 2022 dataset.

    It consists in a restriction to France and to the 100 most present plant species.

    Parameters
    ----------
    root : string or pathlib.Path
        Root directory of dataset.
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    patch_data : string or list of string
        Specifies what type of patch data to load, possible values:
        'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    use_rasters : boolean (optional)
        If True, extracts patches from environmental rasters.
    patch_extractor : PatchExtractor object (optional)
        Patch extractor to use if rasters are used.
    use_localisation : boolean
        If True, returns also the localisation as a tuple (latitude, longitude).
    transform : callable (optional)
        A function/transform that takes a list of arrays and returns
        a transformed version.
    target_transform : callable (optional)
        A function/transform that takes in the target and transforms it.
    download : boolean (optional)
        If True, downloads the dataset from the internet and puts it in root
        directory. If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(
        self,
        root,
        subset,
        *,
        patch_data="all",
        use_rasters=True,
        patch_extractor=None,
        use_localisation=False,
        transform=None,
        target_transform=None,
        download=False,
        **kwargs,
    ):
        root = Path(root)

        self.root = root
        self.subset = subset
        self.patch_data = patch_data
        self.use_localisation = use_localisation
        self.transform = transform
        self.target_transform = target_transform
        self.n_classes = 10

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted."
                               " You can use download=True to download it")

        df = pd.read_csv(
            root / "micro_geolifeclef_observations.csv",
            sep=";",
            index_col="observation_id",
        )

        if subset != "train+val":
            ind = df.index[df["subset"] == subset]
        else:
            ind = df.index[np.isin(df["subset"], ["train", "val"])]
        df = df.loc[ind]

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values
        self.targets = df["species_id"].values

        if use_rasters:
            if patch_extractor is None:
                patch_extractor = PatchExtractor(self.root / "rasters",
                                                 size=256)
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor
        else:
            self.patch_extractor = None

    def _check_integrity(self):
        return (self.root / "micro_geolifeclef_observations.csv").exists()

    def download(self):
        """Download the MicroGeolifeClef dataset."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://lab.plantnet.org/seafile/f/b07039ce11f44072a548/?dl=1",
            self.root,
            filename="micro_geolifeclef.zip",
            md5="ff27b08b624c91b1989306afe97f2c6d",
            remove_finished=True,
        )

    def __len__(self):
        """Return the number of observations in the dataset."""
        return len(self.observation_ids)

    def __getitem__(self, index):
        """Return a MicroGeolifeClef dataset item.

        Args:
            index (int): dataset id.

        Returns:
            (tuple): data and labels.
        """
        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]

        patches = load_patch(
            observation_id, self.root / "patches",
            data=self.patch_data,
            subfolder_strategy=False,
        )

        # Extracting patch from rasters
        if self.patch_extractor is not None:
            environmental_patches = self.patch_extractor[(latitude, longitude)]
            patches["environmental_patches"] = environmental_patches

        if self.use_localisation:
            patches["localisation"] = np.asarray([latitude, longitude],
                                                 dtype=np.float32)

        if self.transform:
            patches = self.transform(patches)

        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return patches, target
