from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
import rasterio                                                                

from torch.utils.data import Dataset

from malpolon.data.environmental_raster import PatchExtractor


def load_patch(
    observation_id,
    patches_path,
    *,
    data,   # c'est "patch_data" dans la fonction "get_dataset" du scripte principale   
    patch_data_ext,                                                               
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

    Returns
    -------
    patches : dict containing 2d array-like objects
        Returns a dict containing the requested patches.
    """
    patches = {}
    for n in range(0,len(data)) :
        filename_base = str(patches_path) + '_' + data[n] +'/'+str(observation_id)
        filename=filename_base + '_' + data[n] + patch_data_ext[n]
        ext = Path(filename).suffix
        if ext == '.tif':
            var_patch = tifffile.imread(filename)                                                             
            patches[data[n]] = var_patch    
        elif ext == '.jp2':
            with rasterio.open(filename) as src:                                                
                var_patch = src.read()                                                              
                var_patch = np.transpose(var_patch, (1, 2, 0))                                      
            patches[data[n]] = var_patch
        else :
            raise ValueError(f"L'extention {ext} n'est pas prise en compte par cet exemple malpolon.\n Vous devez convertir vos patches au format .tif ou en .jp2 ou modifier/compléter 'def load_patch' dans 'dataset.py'.")


    return patches # element important : ce n'est pas visible dans le code mais patches["xxx"] donne data["xxx"]





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
    download : boolean (optional)
        If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(
        self,
        root,
        csv_occurence_path,
        csv_separator,
        csv_col_class_id,
        csv_col_occurence_id,
        patch_data_ext,
        subset,
        *,
        patch_data="all",
        use_rasters=True,
        patch_extractor=None,
        use_localisation=False,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.subset = subset
        self.patch_data = patch_data
        self.csv_occurence_path = csv_occurence_path
        self.patch_data_ext = patch_data_ext
        self.use_localisation = use_localisation
        self.transform = transform
        self.target_transform = target_transform
        self.n_classes = 10

        #if download:
        #    self.download()

        #if not self._check_integrity():
        #    raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        df = pd.read_csv(
            self.csv_occurence_path,                                             # Ben
            #str(self.root) + '/' + self.root.name + "_subset.csv",              # Ben
            sep=csv_separator,                                                   # Ben
            index_col=csv_col_occurence_id)   
                                                         # Ben

        # Ne parle pas du test set car c'est automatiquement ce qu'il reste après avoir 
        # trier le train et val
        if subset != "train+val":                       
            ind = df.index[df["subset"] == subset]
        else:
            ind = df.index[np.isin(df["subset"], ["train", "val"])]
        df = df.loc[ind]

        self.observation_ids = df.index
        #self.coordinates = df[["latitude", "longitude"]].values
        #self.targets = df["species_id"].values
        self.coordinates = df[["SiteLat", "SiteLong"]].values                  # Ben 
        self.targets = df[csv_col_class_id].values                            # Ben

        if use_rasters:
            if patch_extractor is None:
                patch_extractor = PatchExtractor(self.root / "rasters", size=256)
                patch_extractor.add_all_rasters()

            self.patch_extractor = patch_extractor
        else:
            self.patch_extractor = None

    #def _check_integrity(self):
    #    return (self.root / "micro_geolifeclef_observations.csv").exists()
    #
    #def download(self):
    #    if self._check_integrity():
    #        print("Files already downloaded and verified")
    #        return
    #
    #    download_and_extract_archive(
    #        "https://lab.plantnet.org/seafile/f/b07039ce11f44072a548/?dl=1",
    #        self.root,
    #        filename="micro_geolifeclef.zip",
    #        md5="ff27b08b624c91b1989306afe97f2c6d",
    #        remove_finished=True,
    #    )

    def __len__(self):
        """Returns the number of observations in the dataset."""
        return len(self.observation_ids)

    def __getitem__(self, index):
        latitude = self.coordinates[index][0]
        longitude = self.coordinates[index][1]
        observation_id = self.observation_ids[index]

        patches = load_patch(observation_id, str(self.root) + "/patches", data=self.patch_data, patch_data_ext = self.patch_data_ext)

        # Extracting patch from rasters
        if self.patch_extractor is not None:
            environmental_patches = self.patch_extractor[(latitude, longitude)]
            patches["environmental_patches"] = environmental_patches

        if self.use_localisation:
            patches["localisation"] = np.asarray([latitude, longitude], dtype=np.float32)

        if self.transform:
            patches = self.transform(patches)

        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return patches, target
