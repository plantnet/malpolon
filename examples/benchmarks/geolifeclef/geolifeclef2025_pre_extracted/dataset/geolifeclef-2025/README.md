### A. Spatially splitting the dataset
To split the observation dataset in _train_ and _val_ while avoiding spatial auto-corelation, we use Malpolon's toolbox method `split_obs_spatially.py` based on the library `verde`. The method takes as input an observation CSV file with **lon**, **lat** columns, and evenly splits the data subsets wrt a spacing radius (be default: 10/60 degrees).

The radius value can be whatever real, but it should be coherent with the CRS of the dataset to split. In the case of GLC25, observations coordinates are registered in WGS84 EPSG:4326. So inputting 10/60 as spacing value corresponds to ~0.16 degrees, or 10 arcminutes. Over France, this corresponds to a spacing of around 17km.

In this repository, we chose to split with a spacing of 0.01 degrees, or 0.6 arcminutes which, over France, corresponds to a spacing of around 1.1km.

### B. Computing dataset moments
To compute the mean and standard deviation values of each modality of our dataset, we use the method `compute_mean_std_iteratively_from_sample.py` Malpolon's toolbox which approximates the real values of mean & std with an iterative computation based on a list of path files.

1. Produce text files containg the filepaths to each data element of the dataset for each modality.

In a Python terminal session:
```python
import os
import pandas as pd

def construct_patch_path(data_path, survey_id):
    path = data_path
    for d in (str(survey_id)[-2:], str(survey_id)[-4:-2]):
        path = os.path.join(path, d)
    path = os.path.join(path, f"{survey_id}.tiff")
    return path

df_train = pd.read_csv('GLC25_PA_metadata_train_train-0.6min.csv')
df_val = pd.read_csv('GLC25_PA_metadata_train_val-0.6min.csv')

# Example for bioclim rasters
fps_train_bioclim = list(df_train['surveyId'].apply(lambda x: f'BioclimTimeSeries/cubes/PA-train/GLC25-PA-train-bioclimatic_monthly_{x}_cube.pt').values)
with open('fps_bioclim_train_train-0.6min.txt', 'w') as f:
    for string in fps_train_bioclim:
        f.write(string + '\n')
    
# Example for landsat time series
fps_train_landsat = list(df_train['surveyId'].apply(lambda x: f'SatelliteTimeSeries-Landsat/cubes/PA-train/GLC25-PA-train-landsat-time-series_{x}_cube.pt').values)
with open('fps_landsat_train_train-0.6min.txt', 'w') as f:
    for string in fps_train_landsat:
        f.write(string + '\n')
    
# Example for satellite patches
fps_val_satellite = list(df_val['surveyId'].apply(lambda x: construct_patch_path('SatellitePatches/PA-train/', x)).values)
with open('fps_satellite_train_val-0.6min.txt', 'w') as f:
    for string in fps_val_satellite:
        f.write(string + '\n')
```

2. Run the moments computation script.

```bash
python ../../../../../../toolbox/compute_mean_std_iteratively_from_sample.py -p fps_bioclim_train_val-0.6min.txt -o Stats_bioclim_val.csv --type tiff --max_items 10000
```

For Satellite patches (Sentinel-2A), add the argumen `--per_channel` to compute the moments for each of the 4 channels: red, green, blue, nir. The output CSV contains the values for those channels in the same order row-wise. You can verify the order of bands with the command `gdalinfo <path>/<patch_name>.tiff`.

### Glossary
- fps: filepaths
- PA: Presence Absence
- PO: Presence Only
- CRS: Coordinate Reference System
- xxx\_train\_train-[train,val]-\d.\dmin: spatial split, either the train or validation part, of the observation dataset, with a spatial spacing of \d.\d minutes (wrt to WGS84 CRS)
