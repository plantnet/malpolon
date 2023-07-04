# Sentinel-2 Raster (using torchgeo) example

This example serves as a Getting Started code with our `torchgeo` based pipeline.
It uses the Sentinel-2A data which can be downloaded separately on [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a).
It consists of multi-spectral raster files (.tif) of the planet taken from the Sentinel-2 program, spanning from 2016 to nowadays. These visual data is coupled with an observation file containing plant references observed at different geographical locations. The observation file is an extract from the Pl@ntNet observation database where cell values might not always correspond to the right species since this is a dummie file.
Species include:
- _Himantoglossum hircinum_
- _Mentha suaveolens_
- _Ophrys apifera_
- _Orchis purpurea_
- _Stachys byzantina_


## Running the examples

To run the example `cnn_on_rgbnir_torchgeo.py` on a single GPU using the `<DATASET_PATH>` as path to the dataset (will be downloaded automatically), use:
```script
python cnn_on_rgb_patches.py data.dataset_path=<DATASET_PATH> trainer.gpus=1
```

### Inference

Switch running mode from training to prediction by setting the config file parameter `inference.predict` to `true` and specify a path to your model checkpoint. Both training and prediciton mode are embedded in the example file.

## Parametrization

You can parametrize your models and your training routine through your `.yaml` config file which is split in 5 main sections :

- **trainer** : contains parameters to tweak your training session via pytorchlightning Trainer class
- **model** : defines which model you want to load, from which source, and contains models hyperparameters. You can pass any model hyperparameter listed in your provider's model builder.
- **optimizer** : contains your optimizer's hyperparameters.
- **data** : contains data related information such as the path to your dataset and batch size.

## Additional notes

For debugging purposes, using the `trainer.fast_dev_run=true` and `hydra.job.name=test` parameters can be handy:
```script
python cnn_on_rgb_patches.py data.dataset_path=<DATASET_PATH> trainer.gpus=1 +trainer.fast_dev_run=true +hydra.job.name=test
```

Be careful when using any path argument like `data.dataset_path`, since each `.yaml` file contains a `hydra.run.dir` argument set with a default value of `outputs/<hydra job name>/<date>` (with `<hydra job name>` itself defaulting to the name of the file executed), the current working directory will be changed to said path once the config file is read and loaded. Therefore any other path argument should be written relatively to that `hydra.run.dir` path.
