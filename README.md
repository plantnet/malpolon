# Malpolon

Malpolon is a framework facilitating the training and sharing of Deep Species Distribution models using various types of input covariates including biodclimatic rasters, remote sensing images, land-use rasters, etc.

## Installation

Currently, only the development version is available.
First make sure that the dependences listed in the `requirements.txt` file are installed.

One way to do so is to use `conda`

```script
conda env create -n <name> -f environment.yml
conda activate <name>
```

`malpolon` can then be installed via `pip` using

```script
git clone https://github.com/plantnet/malpolon.git
cd malpolon
pip install -e .
```

To check that the installation went well, use the following command

```script
python -m malpolon.check_install
```

which, if you have CUDA properly installed, should output something similar to

```script
Using PyTorch version 1.13.0
CUDA available: True (version: 11.6)
cuDNN available: True (version: 8302)
Number of CUDA-compatible devices found: 1
```

The **dev** branch is susceptible to have more up-to-date content such as newer examples and experimental features. To switch to the dev branch locally, run:

```script
git checkout dev
```

## Examples

Examples using the GeoLifeCLEF 2022 and 2023 datasets, as well as Sentinel-2A rasters are provided in the `examples` folder. Instructions about how to train and perform predictions with your models can be found in the README file of each example in said folder.


## Documentation

To generate the documention, additional dependences contained in `docs/docs_requirements.txt` must be installed using

```script
pip install -r docs/docs_requirements.txt
```

The documentation can then be generated using

```script
make -C docs html
```

The result can be found in `docs/_build/html`.
