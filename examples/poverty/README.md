<a name="readme-top"></a>

# Landsat rasters example (training)

This example performs regression of the Asset Wealth Index (AWI) using a CNN on Landsat raster over Africa between 2013 and 2020.

## Data

### Sample data

The sample data used in this example consists of:
- **Satellite images**: MS (Multi-Spectral) 7 bands Landsat8 image satellite. Numerous GeoTiff files of resolution XXXm  distributed across various countries, years and clusters.

<div align="center">
  <figure>
      <img src="../../docs/resources/angola_2015.png" width="300">
      <br/>
     <figcaption>Landsat8 composite patch of a village in Angola in 2015 </figcaption>
  </figure>
</div>

- **Observations**: a CSV file containing all labels and corresponding data necessary for matching the GeoTiffs. The CSV file contains the following columns:
  - `country`, (ex : angola, etc.)
  - `year`, (2013 to 2019)
  - `cluster`, cluster ID
  - `lat` : latitude of the cluster,
  - `lon` : longitude of the cluster,
  - `households` : number of households in the cluster,
  - `wealthpooled`, the poverty indicator we want to regress
  - `urban_rural`, 0 if rural, 1 if urban
  - `subset`, train, validation or test

The sample data is based on the Demographic and Health Surveys undertaken in Africa since 2013.


### Data loading

- **Satellite images**

The LandSat patches are looked for in the `example/poverty/dataset` directory, and they are loaded based on  a `PovertyDataModule` and `MSDataset` (cf. [python file](datamodule/landsat_poverty.py) ).
They are to be downloaded beforehand in a TIF format and placed in the `dataset` directory.
The images were downloaded from the Google Earth Engine platform and preprocessed using [this method](https://github.com/mpa-poverty/2-mpa/tree/main/preprocessing).

## Usage

Examples are **ready-to-use scripts** that can be executed by a simple Python command. Every data, model and training parameters are specified in a `.yaml` configuration file located in the `config/` directory.
As stated in the previous section, the data needs to be downloaded and placed in the `dataset` directory.

### Training

To train an example's model such as `resnet18` in `cnn_on_ms_poverty.py`, run the following command:

```script
python examples/poverty/cnn_on_ms_poverty.py
```

You can also specify any of your config parameters within your command through arguments such as:

```script
python examples/poverty/cnn_on_ms_poverty.py data.dataset_path=<DATASET_PATH> trainer.gpus=1
```

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

You can parametrize your models and your training routine through your `.yaml` config file which is split in main sections :

- **trainer** : parameters to tweak your training session via PyTorchLightning Trainer class\
  This section is passed on to your PyTorchLightning trainer.
- **run** : parameters related to prediction and transfer learning\
  This section is passed on to your PyTorchLightning checkpoint loading method.
- **model** : defines which model you want to load, from which source, and contains models hyperparameters. You can pass any model hyperparameter listed in your provider's model builder.\
  This section is passed on to your prediction system _(e.g. `RegressionSystem`)_.
- **optimizer** : your optimizer and metrics hyperparameters.\
  This section is passed on to your prediction system _(e.g. `RegressionSystem`)_.
- **task** : defines the type of deep learning task chosen for your experiment (currently only supporting any of `['classification_binary', 'classification_multiclass', 'classification_multilabel', 'regression']`)\
  This section is passed on to your prediction system _(e.g. `RegressionSystem`)_.
- **data** : data related information such as the path to your dataset or batch size.\
  This section is passed on to your data module _(e.g. `PovertyDataModule`)_.

Hereafter is a detailed list of every sub parameters :

<details>
  <summary><i><u>Click here to expand sub parameters</u></i></summary>

- **trainer**
  - **accelerator** _(str)_ : Selects the type of hardware you want your example to run on. Either `'gpu'` or `'cpu'`.
  - **devices** _(int)_ : Defines how many accelerator devices you want to use for parallelization.
  - **max_epochs** _(int)_ : The maximum number of training epochs.
  - **check_val_every_n_epoch** _(int)_ : Defines the interval of epochs on which validation should be performed throughout training.
- **run**
  - **predict** _(bool)_ : If set to `true`, runs your example in inference mode; if set to `false`, runs your example in training mode.
  - **checkpoint\_path** _(str)_ : Path to the PyTorch checkpoint you wish to load weights from either for inference mode, for resuming training or perform transfer learning.
- **model**
  - **provider\_name** _(str)_ : Defines the source you want to load your models from. Models from the timm and torchvision repositories can be downloaded with or without pre-trained weights and are fully PyTorch compatible. Either `'timm'` or `'torchvision'`.
  - **model\_name** _(str)_ : Name of the model you wish your provider to retrieve. For a complete list of available models, please refer to [timm's](https://timm.fast.ai/) and [torchvision's](https://pytorch.org/vision/stable/models.html) documentations.
  - **model_kwargs**\
    Parameters forwarded to the model constructor. You may add any parameter in this section belonging to your model's constructor. Leave empty (None) to use the model's default parameter value.
    - **pretrained** _(bool)_ : If `true`, your model will be retrieved with pre-trained weights; if `false`, your model will be retrieved with no weights and training will have to be conducted from scratch.
    - **num_classes** _(int)_ : Number of classes for you classification task.
    - **in\_chans** _(int)_ : Number of input channels.
    - **output\_stride** _(int)_ : Output stride value for CNN models. This parameter defines how much the convolution window is shifted when performing convolution.
    - **global\_pool** _(str)_ : Type of global pooling. Takes any value in [`'avg'`, `'max'`, `'avgmax'`, `'catavgmax'`].
    - ...
  - **modifiers**\
    Malpolon's modifiers you can call to modify your model's structure or behavior.
    - **change_last_layer**
      - **num\_outputs** _(int)_ : Number of output channels you would like your model to have instead of its default value.

- **optimizer**
  - **lr** (_float)_ : Learning rate.
  - **weight\_decay** _(float)_ : Model's regularization parameter that penalizes large weights. Takes any floating value in `[0, 1]`.
  - **momentum** _(float)_ : Model's momentum factor which acts on the model's gradient descent by minimizing its oscillations thus accelerating the convergence and avoiding being trapped in local minimas. Takes ano floating value in `[0, 1]`.
  - **nesterov** _(bool)_ : If `true`, adopts nesterov momentum; if `false`, adopts PyTorch's default strategy.
  - **metrics**
    - **_\<metric name\>_** : The name of your metric. Can either be a custom name or one of the keys listed in `malpolon.models.utils.FMETRICS_CALLABLES`. In the latter case, the _callable_ argument is not required.
      - **callable** (optional) _(str)_ : Name of the TorchMetrics functional metric to call _(e.g.: `'torchmetrics.functional.classification.multiclass_accuracy'`)_. Find all functional metrics on the TorchMetrics documentation page such as [here](https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html#functional-interface) in the "functional Interface" section. Learn more about functional metrics [here](https://lightning.ai/docs/torchmetrics/stable/pages/quickstart.html#functional-metrics). Takes a string as input.
      - **_kwargs_** (optional) : any key-value arguments compatible with the selected metric such as `num_classes` or `threshold`. See [TorchMetrics documentation](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html) for the complete list of kwargs to your metric.

- **task**
  - **task** _(str)_ : deep learning task to be performed. At the moment, can take any value in [`'classification_binary'`, `'classification_multiclass'`, `'classification_multilabel'`].

- **data**
  - **dataset\_path** _(str)_ : path to the dataset. At the moment, patches and rasters should be directly put in this directory.
  - **train\_batch\_size** _(int)_ : size of train batches.
  - **inference\_batch\_size** _(int)_ : size of inference batches.
  - **num\_workers** _(int)_ : number of worker processes to use for loading the data. When you set the “number of workers” parameter to a value greater than 0, the DataLoader will load data in parallel using multiple worker processes.

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

Furthermore to perform transfer learning with model modifications you can :
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
```