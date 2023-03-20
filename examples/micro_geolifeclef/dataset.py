from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from malpolon.data.environmental_raster import PatchExtractor


def load_patch(
    observation_id,
    patches_path,
    *,
    data="all",
    return_arrays=True,
):
    """Loads the patch data associated to an observation id.

    Parameters
    ----------
    observation_id : integer / string
        Identifier of the observation.
    patches_path : string / pathlib.Path
        Path to the folder containing all the patches.
    data : string or list of string
        Specifies what data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
     return_localisation : boolean
        If True, returns also the localisation as a tuple (latitude, longitude)
    return_arrays : boolean
        If True, returns all the patches as Numpy arrays (no PIL.Image returned).

    Returns
    -------
    patches : dict containing 2d array-like objects
        Returns a dict containing the requested patches.
    """
    filename = Path(patches_path) / str(observation_id)

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
        patches["landcover"] = landcover_patch

    return patches