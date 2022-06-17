from __future__ import annotations
from importlib import resources
from pathlib import Path
from typing import Callable, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from torch.utils.data import Dataset

from ..environmental_raster import PatchExtractor
from ._base import DATA_MODULE

if TYPE_CHECKING:
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
) -> list[Patches]:
    """Loads the patch data associated to an observation id.

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
    return_localisation : boolean
        If True, returns also the localisation as a tuple (latitude, longitude)
    return_arrays : boolean
        If True, returns all the patches as Numpy arrays (no PIL.Image returned).

    Returns
    -------
    patches : list of size 4 containing 2d array-like objects
        Returns a list containing all the patches in the following order: RGB, Near-IR, altitude and landcover.
    """
    observation_id = str(observation_id)

    region_id = observation_id[0]
    if region_id == "1":
        region = "patches-fr"
    elif region_id == "2":
        region = "patches-us"
    else:
        raise ValueError(
            "Incorrect 'observation_id' {}, can not extract region id from it".format(
                observation_id
            )
        )

    subfolder1 = observation_id[-2:]
    subfolder2 = observation_id[-4:-2]

    filename = Path(patches_path) / region / subfolder1 / subfolder2 / observation_id

    patches = []

    if data == "all":
        data = ["rgb", "near_ir", "landcover", "altitude"]

    if "rgb" in data:
        rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
        rgb_patch = Image.open(rgb_filename)
        if return_arrays:
            rgb_patch = np.asarray(rgb_patch)
        patches.append(rgb_patch)

    if "near_ir" in data:
        near_ir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
        near_ir_patch = Image.open(near_ir_filename)
        if return_arrays:
            near_ir_patch = np.asarray(near_ir_patch)
        patches.append(near_ir_patch)

    if "altitude" in data:
        altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
        altitude_patch = tifffile.imread(altitude_filename)
        patches.append(altitude_patch)

    if "landcover" in data:
        landcover_filename = filename.with_name(filename.stem + "_landcover.tif")
        landcover_patch = tifffile.imread(landcover_filename)
        if landcover_mapping is not None:
            landcover_patch = landcover_mapping[landcover_patch]
        patches.append(landcover_patch)

    return patches


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
    ):
        root = Path(root)

        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )

        possible_regions = ["both", "fr", "us"]
        if region not in possible_regions:
            raise ValueError(
                "Possible values for 'region' are: {} (given {})".format(
                    possible_regions, region
                )
            )

        self.root = root
        self.subset = subset
        self.region = region
        self.patch_data = patch_data
        self.use_localisation = use_localisation
        self.transform = transform
        self.target_transform = target_transform
        self.training = (subset != "test")
        self.n_classes = 17037

        df = self._load_observation_data(root, region, subset)

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values

        if self.training:
            self.targets = df["species_id"].values
        else:
            self.targets = None

        if use_rasters:
            if patch_extractor is None:
                patch_extractor = PatchExtractor(self.root / "rasters", size=256)
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor
        else:
            self.patch_extractor = None

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
            root
            / "observations"
            / "observations_fr_{}.csv".format(subset_file_suffix),
            sep=";",
            index_col="observation_id",
        )
        df_us = pd.read_csv(
            root
            / "observations"
            / "observations_us_{}.csv".format(subset_file_suffix),
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
        """Returns the number of observations in the dataset."""
        return len(self.observation_ids)

    def __getitem__(
        self,
        index: int,
    ) -> Union[
        Union[Patches, list[Patches]], tuple[Union[Patches, list[Patches]], Targets]
    ]:
        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]

        patches = load_patch(
            observation_id, self.root, data=self.patch_data
        )

        # Extracting patch from rasters
        if self.patch_extractor is not None:
            environmental_patches = self.patch_extractor[(latitude, longitude)]
            patches = patches + [environmental_patches]

        if self.use_localisation:
            patches.append([latitude, longitude])

        if len(patches) == 1:
            patches = patches[0]

        if self.transform:
            patches = self.transform(patches)

        if self.training:
            target = self.targets[index]

            if self.target_transform:
                target = self.target_transform(target)

            return patches, target
        else:
            return patches


class MiniGeoLifeCLEF2022Dataset(GeoLifeCLEF2022Dataset):
    """Pytorch dataset handler for a subset of GeoLifeCLEF 2022 dataset.
    It consists in a restriction to France and to the 100 most present plant species.

    Parameters
    ----------
    root : string or pathlib.Path
        Root directory of dataset.
    subset : string, either "train", "val", "train+val" or "test"
        Use the given subset ("train+val" is the complete training data).
    patch_data : string or list of string
        Specifies what type of patch data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    use_rasters : boolean (optional)
        If True, extracts patches from environmental rasters.
    patch_extractor : PatchExtractor object (optional)
        Patch extractor to use if rasters are used.
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
        patch_data: str = "all",
        use_rasters: bool = True,
        patch_extractor: Optional[PatchExtractor] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            subset,
            region="fr",
            patch_data=patch_data,
            use_rasters=use_rasters,
            patch_extractor=patch_extractor,
            transform=transform,
            target_transform=target_transform,
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
            root
            / "observations"
            / "observations_fr_{}.csv".format(subset_file_suffix),
            sep=";",
            index_col="observation_id",
        )

        file_name = "minigeolifeclef2022_species_details.csv"
        with resources.path(DATA_MODULE, file_name) as species_file_path:
            df_species = pd.read_csv(
                species_file_path,
                sep=";",
                index_col="species_id",
            )

        df = df[np.isin(df["species_id"], df_species.index)]
        value_counts = df.species_id.value_counts()
        species_id = value_counts.iloc[:100].index
        df_species = df_species.loc[species_id]
        df = df[np.isin(df["species_id"], df_species.index)]

        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder().fit(df_species.index)
        df["species_id"] = label_encoder.transform(df["species_id"])
        df_species.index = label_encoder.transform(df_species.index)

        if subset not in ["train+val", "test"]:
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        return df
