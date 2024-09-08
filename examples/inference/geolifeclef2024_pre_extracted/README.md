# GeoLifeCLEF 2024 (pre-extracted) - Inference mode

<div align="center">
  <a href="https://www.kaggle.com/competitions/geolifeclef-2024"><img src="../../../docs/resources/GLC2024_thumbnail.png" alt="glc24_thumbnail" style="width: 200px;  margin-top: 15px; margin-right: 50px;"></a>
</div>

This repository is related to the GeoLifeCLEF challenge.

The details of the challenge, the data, and all other useful information are present on the challenge page: [https://www.kaggle.com/competitions/geolifeclef-2024](https://www.kaggle.com/competitions/geolifeclef-2024)

## Codes
In this repository you will find a ready-to-use code to run one of the baseline models for the GeoLifeCLEF 2024 challenge: the [MultimodelEnsemble baseline](https://www.kaggle.com/code/picekl/sentinel-landsat-bioclim-baseline-0-31626) in **inference mode** to predict the presence/absence of species (or habitats) in a given area, based on a test dataset. By default, the datasets and the model's weights (resulting from one of our trainings) will be automatically downloaded via the following 2 config key-values:
  1. `data.download_data: True`
  2. `models.model_kwargs.pretrained: True`

This example can be run using the following command:
```bash
python glc24_cnn_multimodal_ensemble.py
```

and uses a custom dataset from `malpolon.data.datasets.geolifeclef2024_pre_extracted` as well as a custom model from `malpolon.models.geolifeclef2024_multimodal_ensemble`.

## Data
The data used in this example is the pre-extracted data from the GeoLifeCLEF 2024 challenge. The data is available on the [https://www.kaggle.com/competitions/geolifeclef-2024/data](challenge "data" page).

It consists of points extractions from Sentinel-2A patches, Landsat time series, and Bioclim time series for each species occurrence of Presence/Absence (PA) surveys.

## Models
The model used in this example is the Multimodal Ensemble model. This model is a simple ensemble of 3 models:
- ResNet18 for Landsat time series extractions
- ReNet18 for Bioclim time series extractions
- Swin V2 Transformer for Sentinel-2A patches extractions

These 3 models are then merged with a simple late fusion strategy (concatenation of features + Multi-Layer Perceptron (MLP)). A single loss is used for the training of the 3 models.