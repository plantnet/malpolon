# MicroGeoLifeCLEF example

This example serves as a Getting Started code.
It uses the MicroGeoLifeCLEF 2022 dataset which can be downloaded separately [here](https://lab.plantnet.org/seafile/f/b07039ce11f44072a548/?dl=1) (the code downloads it automatically if it was not downloaded before).
It consists of a subset of 10,000 observations from the GeoLifeCLEF 2022 dataset, retaining 1,000 observations in France of the 10 species most present in the original dataset belonging to two families (_Lamiaceae_ and _Orchidaceae_):
- _Himantoglossum hircinum_
- _Himantoglossum robertianum_
- _Melittis melissophyllum_
- _Mentha suaveolens_
- _Ophrys apifera_
- _Orchis purpurea_
- _Orchis mascula_
- _Perovskia atriplicifolia_
- _Stachys byzantina_
- _Vitex agnus-castus_


## Running the examples

To run the example `cnn_on_rgb_patches.py` on a single GPU using the `<DATASET_PATH>` as path to the dataset (will be downloaded automatically), use:
```script
python cnn_on_rgb_patches.py data.dataset_path=<DATASET_PATH> trainer.gpus=1
```

Similarly for the example `cnn_on_rgb_nir_patches.py`:
```script
python cnn_on_rgb_nir_patches.py data.dataset_path=<DATASET_PATH> trainer.gpus=1
```


## Additional notes

For debugging purposes, using the `trainer.fast_dev_run=true` and `hydra.job.name=test` parameters can be handy:
```script
python cnn_on_rgb_patches.py data.dataset_path=<DATASET_PATH> trainer.gpus=1 +trainer.fast_dev_run=true +hydra.job.name=test
```
