from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from torch.utils.data import Dataset

from ._base import DATA_MODULE


def load_patch(
    observation_id,
    patches_path,
    *,
    data="all",
    landcover_mapping=None,
    return_arrays=True
):
    """Loads the patch data associated to an observation id

    Parameters
    ----------
    observation_id : integer
        Identifier of the observation.
    patches_path : string / pathlib.Path
        Path to the folder containing all the patches.
    data : string or list of string
        Specifies what data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    landcover_mapping : 1d array-like
        Facultative mapping of landcover codes, useful to align France and US codes.
    return_arrays : boolean
        If True, returns all the patches as Numpy arrays (no PIL.Image returned).

    Returns
    -------
    patches : tuple of size 4 containing 2d array-like objects
        Returns a tuple containing all the patches in the following order: RGB, Near-IR, altitude and landcover.
    """
    observation_id = str(observation_id)

    region_id = observation_id[0]
    if region_id == "1":
        region = "fr"
    elif region_id == "2":
        region = "us"
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


class MiniGeoLifeCLEF2022Dataset(Dataset):
    """Pytorch dataset handler for GeoLifeCLEF 2022 dataset.

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
        root,
        subset,
        *,
        patch_data="all",
        use_rasters=True,
        patch_extractor=None,
        transform=None,
        target_transform=None
    ):
        self.root = Path(root)
        self.subset = subset
        self.patch_data = patch_data
        self.transform = transform
        self.target_transform = target_transform

        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )

        if subset == "test":
            subset_file_suffix = "test"
            self.training_data = False
        else:
            subset_file_suffix = "train"
            self.training_data = True

        df = pd.read_csv(
            self.root
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

        if self.training_data and subset != "train+val":
            ind = df.index[df["subset"] == subset]
            df = df.loc[ind]

        self.observation_ids = df.index
        self.coordinates = df[["latitude", "longitude"]].values
        self.species = df_species
        self.n_classes = len(self.species)

        if self.training_data:
            self.targets = df["species_id"].values
        else:
            self.targets = None

        if use_rasters:
            if patch_extractor is None:
                from .environmental_raster import PatchExtractor

                patch_extractor = PatchExtractor(self.root / "rasters", size=256)
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor
        else:
            self.patch_extractor = None

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]

        patches = load_patch(
            observation_id, self.root / "patches", data=self.patch_data
        )

        # Extracting patch from rasters
        if self.patch_extractor is not None:
            environmental_patches = self.patch_extractor[(latitude, longitude)]
            patches = patches + tuple(environmental_patches)

        # Concatenate all patches into a single tensor
        if len(patches) == 1:
            patches = patches[0]

        if self.transform:
            patches = self.transform(patches)

        if self.training_data:
            target = self.targets[index]

            if self.target_transform:
                target = self.target_transform(target)

            return patches, target
        else:
            return patches
