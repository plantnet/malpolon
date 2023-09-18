# Sentinel-2 Raster (using torchgeo) example

This example serves as a Getting Started code with our `torchgeo` based pipeline.
It uses the Sentinel-2A data which can be downloaded separately from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) consisting of multi-spectral raster files (.tif) of the planet taken from the Sentinel-2 program, spanning from 2016 to nowadays. By default, sample data is downloaded from MPC, consisting of one 4-bands (RGB-IR) tile over Montpellier, France.

This visual data is coupled with an observation file containing plant references observed at different geographical locations. The observation file is an extract from the Pl@ntNet observation database where cell values might not always correspond to the right species since this is a dummie file.
Species include:

| Species | Himantoglossum hircinum | Mentha suaveolens | Ophrys apifera | Orchis purpurea | Stachys byzantina |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Photo |![Himantoglossum_hircinum](../../docs/resources/Himantoglossum_hircinum.jpg "Himantoglossum hircinum") | ![Mentha_suaveolens](../../docs/resources/Mentha_suaveolens.jpg "Mentha suaveolens") | ![Ophrys_apifera](../../docs/resources/Ophrys_apifera.jpg "Ophrys apifera") | ![Orchis_purpurea](../../docs/resources/Orchis_purpurea.jpg "Orchis purpurea") | ![Stachys_byzantina](../../docs/resources/Stachys_byzantina.jpg "Stachys byzantina") |
| Source |[Wikipedia: Himantoglossum hircinum](https://en.wikipedia.org/wiki/Himantoglossum_hircinum) | [Wikipedia: Mentha suaveolens](https://en.wikipedia.org/wiki/Mentha_suaveolens) | [Wikipedia: Ophrys apifera](https://en.wikipedia.org/wiki/Ophrys_apifera) | [Wikipedia: Orchis purpurea](https://en.wikipedia.org/wiki/Orchis_purpurea) | [Wikipedia: Stachys byzantina](https://en.wikipedia.org/wiki/Stachys_byzantina) | 
| Author | [Jörg Hempel](https://commons.wikimedia.org/wiki/User:LC-de) <br> (24/05/2014) | [Broly0](https://commons.wikimedia.org/wiki/User:Smithh05) <br> (12/05/2009) |  [Orchi](https://commons.wikimedia.org/wiki/User:Orchi) <br> (15/06/2005) | [Francesco Scelsa](https://commons.wikimedia.org/w/index.php?title=User:Francesco_Scelsa&action=edit&redlink=1) <br> (10/05/2020) | [Jean-Pol GRANDMONT](https://commons.wikimedia.org/wiki/User:Jean-Pol_GRANDMONT) <br> (09/06/2010) |
| License | CC BY-SA 3.0 de | CC0 | CC BY-SA 3.0 | CC BY-SA 4.0 | CC BY-SA 3.0 |




## Running the examples

Examples or ready-to-use scripts that can be executed by a simple Python command. For instance, to run an example such as `cnn_on_rgbnir_torchgeo.py`, use:

```script
python cnn_on_rgbnir_torchgeo.py
```

You can also parametrize your command with arguments such as:

```script
python cnn_on_rgbnir_torchgeo.py data.dataset_path=<DATASET_PATH> trainer.gpus=1
```

where the example is ran on a single GPU using the `<DATASET_PATH>` as path to the dataset (which, by default, is downloaded automatically if not already on your machine).

You can pass any argument listed in the configuration file associated with your example, in your command line (thus overruling the values of the configuration file), or simply modify your configuration file and run your python command without any additional argument. Example parameters are listed in the following section.
### Parametrization

All hyperparameters are specified in a `.yaml` configuration file located in the `config/` directory, which is read and transformed into a dictionary by **Hydra** when running your scripts.

You can parametrize your models and your training routine through your `.yaml` config file which is split in 5 main sections :

- **trainer** : contains parameters to tweak your training session via pytorchlightning Trainer class
- **run** : contains parameters related to prediction and transfer learning
- **model** : defines which model you want to load, from which source, and contains models hyperparameters. You can pass any model hyperparameter listed in your provider's model builder.
- **optimizer** : contains your optimizer's and metrics hyperparameters.
  - **metrics** keys name must match existing **TorchMetrics** metrics and contain the following sub key-value pairs:
    1. `callable` key with a functional metric method as value (unless the callable is listed in the FMETRICS_CALLABLES constant dictionary at the start of your main script, in which case this argument is optional)
    2. `kwargs` key with the functional metric method's input arguments
- **data** : contains data related information such as the path to your dataset and batch size.
- **task** : defines the type of deep learning task chosen for your experiment (currently only supporting any of `['classification_binary', 'classification_multiclass', 'classification_multilabel']`)

Key-value pairs from **data** and **task** are passed as input arguments of your data module _(e.g. `Sentinel2TorchGeoDataModule`)_.\
Key-value pairs from **model**, **optimizer** and **task** are passed as input arguments of your prediction system _(e.g. `ClassificationSystem`)_.\
Key-value pairs from **trainer** are passed as input arguments of your pytorchlightning trainer.\
Key-value pairs from **run** are passed as input arguments of your PyTorchLightning checkpoint loading method.

Hereafter is a detailed list of every sub parameters :

- **trainer**
  - _accelerator_ : Selects the type of hardware you want your example to run from. Either `'gpu'` or `'cpu'`.
  - _devices_ : Defines how many accelerator devices you want to use for parallelization. Takes an integer as input.
  - _max_epochs_ : The maximum number of training epochs. Takes an integer as input.
  - _check_val_every_n_epoch_ : Defines the interval of epochs on which validation should be performed throughout training. Takes an integer as input.
- **run**
  - _predict_ : If set to `true`, runs your example in inference mode; if set to `false`, runs your example in training mode. Boolean parameter.
  - _checkpoint\_path_ : Path to the PyTorch checkpoint you wish to load weights from either for inference mode, for resuming training or perform transfer learning. Takes a string as input.
- **model**
  - _provider\_name_ : Defines the source you want to load your models from. Models from the timm and torchvision repositories can be downloaded with or without pre-trained weights and are fully PyTorch compatible. Either `'timm'` or `'torchvision'`.
  - _model\_name_ : Name of the model your wish to retrieve from your provider. For a complete list of available models, please refer to [timm's]() and [torchvision's]() documentations. Takes a string as input.
  - **model_kwargs** (parameters forwarded to the model constructor. You may add any parameter in this section belonging to your model's constructor. Leave empty (None) to use the model's default parameter value.)
    - _pretrained_ : If `true`, your model will be retrieved with pre-trained weights; if `false`, your model will be retrieved with no weights and training will have to be conducted from scratch. Boolean parameter.
    - _num_classes_ : Number of classes for you classification task. Takes an integer as input.
    - _in\_chans_ : Number of input channels. Takes an integer as input.
    - _output\_stride_ : Output stride value for CNN models. This parameter defines how much the convolution window is shifted when performing convolution. Takes an integer as input.
    - _global\_pool_ : Type of global pooling. Takes any value in [`avg`, `max`, `avgmax`, `catavgmax`].
    - ...
  - **modifiers** (malpolon's modifiers you can call to modify your model's structure or behavior)
    - **change\_first\_convolutional\_layer**
      - _num\_input\_channels_ : Number of input channels you would like your model to take instead of its default value. Takes an integer as input.
    - **change_last_layer**
      - _num\_outputs_ : Number of output channels you would like your model to have instead of its default value. Takes an integer as input.
- **optimizer**
  - _lr_ : learning rate. Takes a float as input.
  - _weight\_decay_ : model's weight decay. Takes a float as input.
  - _momentum_ : model's momentum factor. Takes a float as input.
  - _nesterov_ : If `true`, adopts nesterov momentum; if `false`, adopts PyTorch's default strategy. Boolean parameter.
  - **metrics**
    - **_\<metric name\>_** : The name of an actual TorchMetrics metric. Some of them are automatically tied to functional callables in the malpolon framework.
      - _callable (optional)_ : If the metric's name is not tied to a functional callable, you can specify it here. You can find functional callable's names on the TorchMetrics documentation page such as [here](https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html#functional-interface)
      - _kwargs_ : any key-value arguments compatible with the selected metric.
        - _num\_classes_ : number of classes in the dataset. Takes an integer as input.
        - ...
  - **data**
    - _dataset\_path_ : path to the dataset. At the moment, patches and rasters should be directly put in this directory. Takes a string as input.
    - _labels\_name_ : name of the file containing the labels which should be located in the same directory as the data. Takes a string as input.
    - _download\_data\_sample_ : If `true`, a small sample of the example's dataset will be downloaded (if not already on your machine); if `false`, will not. Boolean parameter.
    - _train\_batch\_size_ : size of train batches. Takes an integer as input.
    - _inference\_batch\_size_ : size of inference batches. Takes an integer as input.
    - _num\_workers_ : number of worker processes to use for loading the data. When you set the “number of workers” parameter to a value greater than 0, the DataLoader will load data in parallel using multiple worker processes. Takes an integer as input.
    - _units_ : unit of the dataset. Takes a string as input.
    - _crs_ : coordinate reference system of the dataset. Takes an integer as input.
- **task**
  - _task_ : deep learning task to be performed. At the moment, can taks any value in [`'classification_binary'`, `'classification_multiclass'`, `'classification_multilabel'`].  Takes a string as input.

Note that any of these parameters can also be passed through command line like shown in the previous section and overrule those of the config file.

### Inference

Switch running mode from training to prediction by setting the config file parameter `run.predict` to `true` and specify a path to your model checkpoint. Both training and prediction mode are embedded in the example file.

## Additional notes
### Transfer learning
Be aware that for now there are no tools provided to easily freeze or manage layers during training. Thus you may encounter errors when trying to train a model with pre-trained weights _(e.g. from ImageNet)_ on RGB-IR data as most of pre-trained models are done over 3 RGB images.

To avoid such issue, either :
- train from scratch by setting hyperparameter `model.model_kwargs.pretrained` to false
- manually change your model and freeze strategy before `trainer.fit` (in your main script) to only train 3 bands at once
- restrain your trainings to 3 bands and merge several trainings output features

Future updates will aim at making this step easier.

Otherwise, to simply resume a training or perform transfer learning on 3-channels models, simply update the path to your model checkpoint in your configuration file, and run your script in training mode. A new output folder will be generated.

### Debugging

For debugging purposes, using the `trainer.fast_dev_run=true` and `hydra.job.name=test` parameters can be handy:
```script
python cnn_on_rgb_patches.py data.dataset_path=<DATASET_PATH> trainer.gpus=1 +trainer.fast_dev_run=true +hydra.job.name=test
```

Be careful when using any path argument like `data.dataset_path`, since each `.yaml` file contains a `hydra.run.dir` argument set with a default value of `outputs/<hydra job name>/<date>` (with `<hydra job name>` itself defaulting to the name of the file executed), the current working directory will be changed to said path once the config file is read and loaded. Therefore any other path argument should be written relatively to that `hydra.run.dir` path.
