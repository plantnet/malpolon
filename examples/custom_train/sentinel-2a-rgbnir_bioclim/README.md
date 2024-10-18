<a name="readme-top"></a>

# Sentinel-2A patches + Bioclim rasters example (training)

This `torchgeo` based example performs multi-label (by default), multi-class or binary classification of species using a CNN model on a combination of Sentinel-2A pre-extracted patches + Bioclim raster data and geolocated plant observations.

Sentinel-2A satellite data is hosted and available on [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) (MPC).\
By default, the pre-extracted patches are taken from the pre-processed tiles from [Ecodatacube](https://stac.ecodatacube.eu/) of the year 2021, and consist of 4 bands (RGB-IR).

Bioclim raster data is hosted and available on [CHELSA](https://chelsa-climate.org/bioclim/). By default this example uses bioclimatic variables 1 to 4.\

## Data

### Sample data

The sample data used in this example consists of:
- **Satellite images**: the RGB-IR bands of the tile `T31TEJ` from the Sentinel-2A satellite (which contains the city of Montpellier, France). Each band is a GeoTIFF file with a resolution of 10m.

<div align="center">
  <figure>
    <a href="https://planetarycomputer.microsoft.com/explore?c=4.0129%2C43.6370&z=10.30&v=2&d=sentinel-2-l2a&m=cql%3A17367ba270405507e8f9aa7772327681&r=Natural+color&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0">
      <img src="../../../docs/resources/S2A_MSIL2A_20190801T104031_R008_T31TEJ_20201004T190635_preview.jpg" alt="Sentinel2A_T31TEJ_preview" width="300"></a>
      <br/>
     <figcaption>Sentinel-2A tile <code>T31TEJ</code> at 01/08/2019 (dd/mm/yy)</figcaption>
  </figure>
</div>

- **Bioclim rasters**: bioclimatic [variables 1 to 4](https://chelsa-climate.org/bioclim/) from the CHELSA database. Each variable is a GeoTIFF file with a resolution of 30 arc-seconds (~1 km), representing the average of its values over the years 1980-2010.

<div align="center">
  <figure>
    <a href="https://planetarycomputer.microsoft.com/explore?c=4.0129%2C43.6370&z=10.30&v=2&d=sentinel-2-l2a&m=cql%3A17367ba270405507e8f9aa7772327681&r=Natural+color&s=false%3A%3A100%3A%3Atrue&sr=desc&ae=0">
      <img src="../../../docs/resources/bio_3_crop_montpellier.jpg" alt="Bio3_preview" width="300"></a>
      <br/>
     <figcaption>Bioclim variable <code>bio3</code> over the region of Montpellier (same as the Sentinel-2A tile)</figcaption>
  </figure>
</div>

- **Observations**: a CSV file containing a list of geolocated plant observations from the [Pl@ntNet](https://plantnet.org/) database. The CSV file contains the following columns:
  - `survey_id`: unique identifier of the observation
  - `species_id`: unique identifier of the plant species
  - `GBIF_species_name`: GBIF species name
  - `latitude`: latitude of the observation
  - `longitude`: longitude of the observation
  - `subset`: subset of the observation (`train`, `val` or `test`)

  (The CSV is a dummie file based on the Pl@ntNet observation database so species IDs and geographic coordinates do not reflect real observations.)

Species include:

| Species | Himantoglossum hircinum | Mentha suaveolens | Ophrys apifera | Orchis purpurea | Stachys byzantina |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Photo |![Himantoglossum_hircinum](../../../docs/resources/Himantoglossum_hircinum.jpg "Himantoglossum hircinum") | ![Mentha_suaveolens](../../../docs/resources/Mentha_suaveolens.jpg "Mentha suaveolens") | ![Ophrys_apifera](../../../docs/resources/Ophrys_apifera.jpg "Ophrys apifera") | ![Orchis_purpurea](../../../docs/resources/Orchis_purpurea.jpg "Orchis purpurea") | ![Stachys_byzantina](../../../docs/resources/Stachys_byzantina.jpg "Stachys byzantina") |
| Source |[Wikipedia: Himantoglossum hircinum](https://en.wikipedia.org/wiki/Himantoglossum_hircinum) | [Wikipedia: Mentha suaveolens](https://en.wikipedia.org/wiki/Mentha_suaveolens) | [Wikipedia: Ophrys apifera](https://en.wikipedia.org/wiki/Ophrys_apifera) | [Wikipedia: Orchis purpurea](https://en.wikipedia.org/wiki/Orchis_purpurea) | [Wikipedia: Stachys byzantina](https://en.wikipedia.org/wiki/Stachys_byzantina) |
| Author | [Jörg Hempel](https://commons.wikimedia.org/wiki/User:LC-de) <br> (24/05/2014) | [Broly0](https://commons.wikimedia.org/wiki/User:Smithh05) <br> (12/05/2009) |  [Orchi](https://commons.wikimedia.org/wiki/User:Orchi) <br> (15/06/2005) | [Francesco Scelsa](https://commons.wikimedia.org/w/index.php?title=User:Francesco_Scelsa&action=edit&redlink=1) <br> (10/05/2020) | [Jean-Pol GRANDMONT](https://commons.wikimedia.org/wiki/User:Jean-Pol_GRANDMONT) <br> (09/06/2010) |
| License | CC BY-SA 3.0 de | CC0 | CC BY-SA 3.0 | CC BY-SA 4.0 | CC BY-SA 3.0 |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Data loading and adding more data

This example uses the class `malpolon.data.datasets.torchgeo_concat.ConcatPatchRasterDataset` to load both pre-extracted patches (.jpeg) and geo-located rasters (.tif) together and handle them under a single class. This class then called by the datamodule `malpolon.data.datasets.torchgeo_concat.ConcatTorchGeoDataModule` which is used in the training script.

Users can specify each sub-dataset's parameters in the configuration file under the section `data.dataset_kwargs`.\
By default, 2 datasets are called:
1. `malpolon.data.dataset.torchgeo_datasets.RasterBioclim` which will load the bioclimatic rasters
2. `malpolon.data.dataset.geolifeclef2024.PatchesDataset` which will load the pre-extracted patches

```yaml
  dataset_kwargs:
    - callable: "RasterBioclim"
          kwargs:
            root: "dataset/bioclim_rasters/"
            labels_name: "../sample_obs.csv"
            query_units: "pixel"
            query_crs: 4326
            patch_size: 128
            filename_regex: '(?P<band>bio_[\d])_crop_sample'  # single quotes are mandatory
            bands: ["bio_1", "bio_2", "bio_3", "bio_4"]
            obs_data_columns: {'x': 'longitude',
                              'y': 'latitude',
                              'index': 'surveyId',
                              'species_id': 'speciesId',
                              'split': 'subset'}
        - callable: "PatchesDataset"
          kwargs:
            occurrences: "dataset/sample_obs.csv"
            providers:
              - callable: "JpegPatchProvider"
                kwargs:
                  root_path: "dataset/satellite_patches/"
                  dataset_stats: 'jpeg_patches_sample_stats.csv'
                  id_getitem: "surveyId"
                  size: 128
                  select: ['red', 'green', 'blue', 'nir']
            item_columns: ['longitude', 'latitude', 'surveyId']
```

- **Bioclimatic rasters**

The bioclimatic raster files are looked for in the `<path_to_example>/dataset/bioclim_rasters` directory and they are loaded based on the naming regex rule set by the config parameter `filename_regex`.

_e.g.: here the regex rule contains a [named capturing group](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions/Named_capturing_group) named "band" which matches the string 'bio\_' followed by any amount of digits. Then the regex rule looks if the filename ends with "\_crop_sample". So "bio_1_crop_sample.tif" will be matched and the band name will be "bio_1"._

The `bands` parameter then acts as a selectio tool, telling the class to only use the specified bands.

To extend your dataset, simply drop more files in the `<path_to_example>/dataset` directory with the same naming convention, and adapt your rules to select the new bands and/or tiles (see [RasterBioclim documentation](https://plantnet.github.io/malpolon/api.html#malpolon.data.datasets.torchgeo_sentinel2.RasterBioclim) for more details on the class parameters).

- **Sentinel-2A pre-extracted patches**

The satellite patches are looked for in the directory `<path_to_example>/dataset/satellite_patches` and are loaded based on the values of the column `surveyId` in the observations CSV file. Each patch should be located at `<root_path>/YZ/WX/<surveyId>.jpeg` with surveyId being equal to `ABCDWXYZ`.

_e.g.: an observation has a surveyId of 123456789. Then it's pre-extracted patch is located at `<root_path>/89/67/123456789.jpeg`._


- **Observations**

The observations are loaded from the `<path_to_example>/dataset/sample_obs.csv` file. The name of the file can vary so long as it matches the name specified in the configuration file.

To extend your dataset, simply add more observations to the CSV file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Examples are **ready-to-use scripts** that can be executed by a simple Python command. Every data, model and training parameters are specified in a `.yaml` configuration file located in the `config/` directory.

### Training

To train an example's model such as `resnet18` in `cnn_on_rgbnir_concat.py`, run the following command:

```script
python cnn_on_rgbnir_concat.py
```

You can also specify any of your config parameters within your command through arguments such as:

```script
python cnn_on_rgbnir_concat.py data.dataset_path=<DATASET_PATH> trainer.gpus=1
```

The model's weights, logs and metrics are saved in the `outputs/cnn_on_rgbnir_concat/<date_of_run>/` directory

Config parameters provided in this example are listed in the [Parameters](#parameters) section.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Prediction

This example is configured to run in training mode by default but if you want to re-use it for prediction, follow these steps:

- Change config file parameter `run.predict` to `true`
- Specify a path to your model checkpoint in parameter `run.checkpoint_path`

Note that any of these parameters can also be passed through command line like shown in the previous section and overrule those of the config file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Parameters

All hyperparameters are specified in a `.yaml` configuration file located in a `config/` directory, which is read and transformed into a dictionary by the [**Hydra**](https://hydra.cc/docs/intro/) library.

You can parametrize your models and your training routine through your `.yaml` config file which is split in main sections:

- **run**: parameters related to prediction and transfer learning\
  This section is passed on to your PyTorchLightning checkpoint loading method.
- **data**: data related information such as the path to your dataset or batch size.\
  This section is passed on to your data module _(e.g. `Sentinel2TorchGeoDataModule`)_.
- **task**: defines the type of deep learning task chosen for your experiment (currently only supporting any of `['classification_binary', 'classification_multiclass', 'classification_multilabel']`)\
  This section is passed on to your prediction system _(e.g. `ClassificationSystem`)_.
- **trainer**: parameters to tweak your training session via PyTorchLightning Trainer class\
  This section is passed on to your PyTorchLightning trainer.
- **model**: defines which model you want to load, from which source, and contains models hyperparameters. You can pass any model hyperparameter listed in your provider's model builder.\
  This section is passed on to your prediction system _(e.g. `ClassificationSystem`)_.
- **optimizer**: your loss parameters optimizer, scheduler and metrics hyperparameters.\
  This section is passed on to your prediction system _(e.g. `ClassificationSystem`)_.

Hereafter is a detailed list of every sub parameters:

<details>
  <summary><i><u>Click here to expand sub parameters</u></i></summary>

- **run**
  - **predict** _(bool)_: If set to `true`, runs your example in inference mode; if set to `false`, runs your example in training mode.
  - **checkpoint\_path** _(str)_: Path to the PyTorch checkpoint you wish to load weights from either for inference mode, for resuming training or perform transfer learning.

- **data**
  - **num_classes** _(int)_: Number of classes for your classification task. This argument acts as a variable which is re-used through the config file for convenience.
  - **dataset\_path** _(str)_: path to the dataset. At the moment, patches and rasters should be directly put in this directory.
  - **labels\_name** _(str)_: name of the file containing the labels which should be located in the same directory as the data.
  - **download\_data\_sample** _(bool)_: If `true`, a small sample of the example's dataset will be downloaded (if not already on your machine); if `false`, will not.
  - **train\_batch\_size** _(int)_: size of train batches.
  - **inference\_batch\_size** _(int)_: size of inference batches.
  - **num\_workers** _(int)_: number of worker processes to use for loading the data. When you set the “number of workers” parameter to a value greater than 0, the DataLoader will load data in parallel using multiple worker processes.
  - **units** _(str)_: unit system of the queries performed on the dataset. This value should be equal to the units of your observations, which can be different from you dataset's unit system. Takes any value in [`'crs'`, `'pixel'`, `'m'`, `'meter'`, `'metre'`] as input.
  - **crs** _(int)_: coordinate reference system of the queries performed on the dataset. This value should be equal to the CRS of your observations, which can be different from your dataset's CRS.
  - **dataset_kwargs**\
    Parameters forwarded to the dataset constructor. You may add any parameter in this section belonging to your dataset's constructor. Leave empty (None) to use the dataset's default parameter value.\
    In this example, the dataset is a concatenation of two datasets: the `RasterBioclim` and the `PatchesDataset`, passed as a list of dictionaries.
    - **item n°k**
      - **callable** _(str)_: String containing the name of the class you want to call. Can be any class of `geolifeclef2024`, `torchgeo_datasets` or `torchgeo_sentinel2` modules.
      - **kwargs** _(dict)_: Dictionary containing the parameters you want to pass to your callable class.
    - ...

- **task**
  - **task** _(str)_: deep learning task to be performed. At the moment, can take any value in [`'classification_binary'`, `'classification_multiclass'`, `'classification_multilabel'`].

- **trainer**
  - **accelerator** _(str)_: Selects the type of hardware you want your example to run on. Either `'gpu'` or `'cpu'`.
  - **devices** _(int)_: Defines how many accelerator devices you want to use for parallelization.
  - **max_epochs** _(int)_: The maximum number of training epochs.
  - **val_check_interval** _(int)_: How often within one training epoch to check the validation set.
  - **check_val_every_n_epoch** _(int)_: Defines the interval of epochs on which validation should be performed throughout training.
  - **log_every_n_steps** _(int)_: How often to log within one training step (defaults to 50).

- **model**
  - **provider\_name** _(str)_: Defines the source you want to load your models from. Models from the timm and torchvision repositories can be downloaded with or without pre-trained weights and are fully PyTorch compatible. Either `'timm'` or `'torchvision'`.
  - **model\_name** _(str)_: Name of the model you wish your provider to retrieve. For a complete list of available models, please refer to [timm's](https://timm.fast.ai/) and [torchvision's](https://pytorch.org/vision/stable/models.html) documentations.
  - **model_kwargs**\
    Parameters forwarded to the model constructor. You may add any parameter in this section belonging to your model's constructor. Leave empty (None) to use the model's default parameter value.
    - **pretrained** _(bool)_: If `true`, your model will be retrieved with pre-trained weights; if `false`, your model will be retrieved with no weights and training will have to be conducted from scratch.
    - **num_classes** _(int)_: Number of classes for you classification task.
    - **in\_chans** _(int)_: Number of input channels.
    - **output\_stride** _(int)_: Output stride value for CNN models. This parameter defines how much the convolution window is shifted when performing convolution.
    - **global\_pool** _(str)_: Type of global pooling. Takes any value in [`'avg'`, `'max'`, `'avgmax'`, `'catavgmax'`].
    - ...
  - **modifiers**\
    Malpolon's modifiers you can call to modify your model's structure or behavior.
    - **change\_first\_convolutional\_layer**
      - **num\_input\_channels** _(int)_: Number of input channels you would like your model to take instead of its default value.
    - **change_last_layer**
      - **num\_outputs** _(int)_: Number of output channels you would like your model to have instead of its default value.

- **optimizer**
  - **loss_kwargs** (optional): any key-value arguments compatible with the selected loss function. See [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#loss-functions) for the complete list of kwargs to your loss function.
  - **optimizer** (optional): Name of your optimizer. If not provided, by default SGD is selected with the following arguments `[lr=1e-2, momentum=0.9, nesterov=True]`
    - **_\<optimizer name\>_** (optional) _(str)_: Name of an optimizer you want to call. Can either be a custom name or one of the keys listed in `malpolon.models.utils.OPTIMIZERS_CALLABLES`
      - **callable** (optional) _(str)_: Name of the optimizer you want to call.
      - **kwargs** (optional): any key-value arguments compatible with the selected optimizer such as `lr` (learning rate). See [PyTorch documentation](https://pytorch.org/docs/stable/optim.html) for the complete list of kwargs to your optimizer.
  - **metrics**
    - **_\<metric name\>_**: The name of your metric. Can either be a custom name or one of the keys listed in `malpolon.models.utils.FMETRICS_CALLABLES`. In the latter case, the _callable_ argument is not required.
      - **callable** (optional) _(str)_: Name of the TorchMetrics functional metric to call _(e.g.: `'torchmetrics.functional.classification.multiclass_accuracy'`)_. Find all functional metrics on the TorchMetrics documentation page such as [here](https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html#functional-interface) in the "functional Interface" section. Learn more about functional metrics [here](https://lightning.ai/docs/torchmetrics/stable/pages/quickstart.html#functional-metrics).
      - **_kwargs_** (optional): any key-value arguments compatible with the selected metric such as `num_classes` or `threshold`. See [TorchMetrics documentation](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html) for the complete list of kwargs to your metric.





</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Transfer learning

- **Resuming training (same model)**

To resume a training or perform transfer learning with the same model without changing its channels, update your configuration file checkpoint path, and run your script in training mode:

```yaml
run:
  predict: false
  checkpoint_path: <PATH_TO_CHECKPOINT>
```

A new output folder will be generated.

- **Transfer with model modifications**

Be aware that for now there are no tools provided to easily freeze or manage intermediate layers during training. Thus you may encounter challenges when trying to train a model with pre-trained weights _(e.g. from ImageNet)_ on 4-channels (or more) data like RGB-IR as most of the pre-trained models are done over 3-channels RGB images.

However, Malpolon provides methods to modify your **first** and **last** model layers. These methods are located in `malpolon.models.model_builder.py`:

- `change_first_convolutional_layer_modifier()`
- `change_last_layer_modifier()`
- `change_last_layer_to_identity_modifier()`

Furthermore to perform transfer learning with model modifications you can:
- Train from scratch by setting config hyperparameter `model.model_kwargs.pretrained` to false
- Manually change your model and use a freeze strategy before `trainer.fit` (in your main script) to only train 3 bands at once
- Restrain your trainings to 3 bands and merge several output features

Future updates will aim at making this step easier.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Additional notes
### Debugging

For debugging purposes, using the `trainer.fast_dev_run=true` and `hydra.job.name=test` parameters can be handy:

```bash
python cnn_on_rgb_patches.py data.dataset_path=<DATASET_PATH> trainer.gpus=1 +trainer.fast_dev_run=true +hydra.job.name=test
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>