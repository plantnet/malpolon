.. Malpolon documentation master file, created by
   sphinx-quickstart on Wed Apr 20 17:15:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

*******************
Experiment examples
*******************

.. toctree::
   :maxdepth: 10


Ecologists scenario
*******************

I have a dataset of my own and I want to train a model on it. I want to be able to easily customize the training process and the model architecture.

- *Drop and play* : I have an observations file (.csv) and I want to train a model on different environmental variables (rasters, satellite imagery) without having to worry about the data loading.

- *Custom dataset* : I have my own dataset consisting of pre-extracted image patches and/or rasters and I want to train a model on it.

Sentinel-2A
===========

`See Sentinel-2a (training) GitHub README 🔗 <https://github.com/plantnet/malpolon/tree/main/examples/ecologists/sentinel_2a>`_

MicroGeoLifeCLEF2022
====================

`See MicroGeoLifeCLEF2022 (training) GitHub README 🔗 <https://github.com/plantnet/malpolon/tree/main/examples/ecologists/micro_geolifeclef2022>`_


Inference scenario
******************

I have an observations file (.csv) and I want to predict the presence of species on a given area using a model I trained previously and a selected dataset or a shapefile I would provide.

Sentinel-2A
===========

`See Sentinel-2a (inference) GitHub README 🔗 <https://github.com/plantnet/malpolon/tree/main/examples/inference/sentinel_2a>`_

MicroGeoLifeCLEF2022
====================

`See MicroGeoLifeCLEF2022 (inference) GitHub README 🔗 <https://github.com/plantnet/malpolon/tree/main/examples/inference/micro_geolifeclef2022>`_


Kaggle scenario
***************

I am a potential kaggle participant on the GeoLifeClef challenge. I want to train a model on the provided datasets without having to worry about the data loading, starting from a plug-and-play example.

- *GeoLifeClef2022* : contains a fully functionnal example of a model training on the GeoLifeClef2022 dataset, from data download, to training and prediction.
  
- *GeoLifeClef2023* : contains dataloaders for the GeoLifeClef2023 dataset (different from the GLC2022 dataloaders). The training and prediction scripts are not provided.



GeoLifeCLEF 2022
================

`See GLC22 (kaggle) GitHub README 🔗 <https://github.com/plantnet/malpolon/tree/main/examples/kaggle/geolifeclef2022>`_

GeoLifeCLEF 2023
================

`See GLC23 (kaggle) GitHub README 🔗 <https://github.com/plantnet/malpolon/tree/main/examples/kaggle/geolifeclef2023>`_