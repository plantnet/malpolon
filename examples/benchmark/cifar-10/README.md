<a name="readme-top"></a>

# Cifar-10 example (training)

This example performs multi-class classification of images using a CNN model on the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

The datase consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are completely mutually exclusive. There are 50,000 training images and 10,000 test images. These images are compiled in binary pickle files, 5 train files (data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5) and 1 test file (test_batch).

## Data

For more details about the classes and example image please visit the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) page.

<div align="center">
  <figure>
    <a href="https://www.cs.toronto.edu/~kriz/cifar.html">
      <img src="../../../docs/resources/cifar10_preview.jpg" alt="cifar10_preview" width="300"></a>
      <br/>
     <figcaption>CIFAR-10 patches</figcaption>
  </figure>
</div>

### Data loading and adding more data

- **Image patches**

The dataset image patches are loaded from the `<path_to_example>/dataset/cifar-10-batches-py/` directory through `torchvision.datasets.CIFAR10` class. Data is downloaded automatically if missing.

To extend your dataset, you will need to import and call other CIFAR-related torchvision datasets sucha s `torchvision.datasets.CIFAR100` and adjust your config parameters to match the new amount of classes.

## Usage

Examples are **ready-to-use scripts** that can be executed by a simple Python command. Every data, model and training parameters are specified in a `.yaml` configuration file located in the `config/` directory.

### Training

To train an example's model such as `resnet18` in `cnn_cifar10.py`, run the following command:

```script
python cnn_cifar10.py
```

You can also specify any of your config parameters within your command through arguments such as:

```script
python cnn_cifar10.py data.dataset_path=<DATASET_PATH> trainer.gpus=1
```

The model's weights, logs and metrics are saved in the `outputs/cnn_cifar10/<date_of_run>/` directory

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
- **optimizer**: your optimizer and metrics hyperparameters.\
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
  - **train\_batch\_size** _(int)_: size of train batches.
  - **inference\_batch\_size** _(int)_: size of inference batches.
  - **num\_workers** _(int)_: number of worker processes to use for loading the data. When you set the “number of workers” parameter to a value greater than 0, the DataLoader will load data in parallel using multiple worker processes.

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
  - **lr** (_float)_: Learning rate.
  - **weight\_decay** _(float)_: Model's regularization parameter that penalizes large weights. Takes any floating value in `[0, 1]`.
  - **momentum** _(float)_: Model's momentum factor which acts on the model's gradient descent by minimizing its oscillations thus accelerating the convergence and avoiding being trapped in local minimas. Takes ano floating value in `[0, 1]`.
  - **nesterov** _(bool)_: If `true`, adopts nesterov momentum; if `false`, adopts PyTorch's default strategy.
  - **metrics**
    - **_\<metric name\>_**: The name of your metric. Can either be a custom name or one of the keys listed in `malpolon.models.utils.FMETRICS_CALLABLES`. In the latter case, the _callable_ argument is not required.
      - **callable** (optional) _(str)_: Name of the TorchMetrics functional metric to call _(e.g.: `'torchmetrics.functional.classification.multiclass_accuracy'`)_. Find all functional metrics on the TorchMetrics documentation page such as [here](https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html#functional-interface) in the "functional Interface" section. Learn more about functional metrics [here](https://lightning.ai/docs/torchmetrics/stable/pages/quickstart.html#functional-metrics). Takes a string as input.
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