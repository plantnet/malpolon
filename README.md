<p align="center">
  <a href="https://github.com/plantnet/malpolon/issues"><img src="https://img.shields.io/github/issues/plantnet/malpolon" alt="GitHub issues"></a>
  <a href="https://github.com/plantnet/malpolon/pulls"><img src="https://img.shields.io/github/issues-pr/plantnet/malpolon" alt="GitHub pull requests"></a>
  <a href="https://github.com/plantnet/malpolon/graphs/contributors"><img src="https://img.shields.io/github/contributors/plantnet/malpolon" alt="GitHub contributors"></a>
  <a href="https://github.com/plantnet/malpolon/network/members"><img src="https://img.shields.io/github/forks/plantnet/malpolon" alt="GitHub forks"></a>
  <a href="https://github.com/plantnet/malpolon/stargazers"><img src="https://img.shields.io/github/stars/plantnet/malpolon" alt="GitHub stars"></a>
  <a href="https://github.com/plantnet/malpolon/watchers"><img src="https://img.shields.io/github/watchers/plantnet/malpolon" alt="GitHub watchers"></a>
  <a href="https://github.com/plantnet/malpolon/blob/main/LICENSE"><img src="https://img.shields.io/github/license/plantnet/malpolon" alt="License"></a>
</p>

<div align="center">
  <img src="docs/resources/Malpolon_transparent.png" alt="Project logo" width="300">
  <p align="center">A deep learning framework to help you build your species distribution models</p>
  <a href="https://github.com/plantnet/malpolon">View framework</a>
  ·
  <a href="https://github.com/plantnet/malpolon/issues/new?assignees=aerodynamic-sauce-pan&labels=bug&projects=&template=bug_report.md&title=%5BBUG%5D">Report Bug</a>
  ·
  <a href="https://github.com/plantnet/malpolon/issues/new?assignees=aerodynamic-sauce-pan&labels=enhancement&projects=&template=enhancement.md&title=%5BEnhancement%5D">Request Feature</a>
  <h1></h1>
</div>

# Malpolon

Malpolon is a framework facilitating the training and sharing of Deep Species Distribution models using various types of input covariates including biodclimatic rasters, remote sensing images, land-use rasters, etc.

## Roadmap

This roadmap outlines the planned features and milestones for the project. Please note that the roadmap is subject to change and may be updated as the project progress.

- [ ] Data support
    - [x] Images (pre-extracted patches)
    - [x] Rasters
    - [ ] Time series
      - [x] Via GLC23 loaders (.csv)
      - [ ] Via generic loader
    - [ ] Shapefiles
    - [ ] Fuse several data types in one training
- [ ] Deep learning tasks
  - [x] Binary classification
  - [x] Multi-class classification
  - [x] Multi-label classification
  - [ ] Regression (abundance prediction)
  - [ ] Ordinal
- [ ] Supported models
  - [x] CNN
  - [ ] LSTM
  - [ ] Transformers
- [ ] Training flexibility
  - [x] Add model head/tail modifiers
  - [ ] Allow easy (un-)freeze of layers
  - [ ] Allow dataset intersections and unions
- [ ] Allow data parallel training
  - [x] Multithreading
  - [ ] Multiprocessing
    - Issues may arise depending on hardware

## Installation

To install malpolon, you will first need to install **Python 3.8, 3.9 or 3.10**, and several python packages. To do so, it is best practice to create a virtual environment containing all these packages locally.

⚠️ **macOS** installation may be bumpy for now. More instructions will be added shortly. It is recommended to stick to Linux for the time being. ⚠️

### 0. Requirements

Before proceeding, please make sure the following packages are installed on your system:

- [3.8 ≥ Python ≥ 3.10](https://www.python.org/downloads/)
- [`pip`](https://pip.pypa.io/en/stable/installation/)
- [`git`](https://git-scm.com/downloads)
- `libgeos-dev` (dependency of Python library `Cartopy`)
  - On Linux (Ubuntu): `sudo apt install libgeos-dev`
  - On MacOS: `brew install geos`
- `cmake`
  - On Linux (Ubuntu): `sudo apt install cmake`
  - On MacOS: `brew install cmake`
- `cuda` (if you intend to run your models on GPU)
  - [`CUDA Installation guide`](https://docs.nvidia.com/cuda/index.html)
  - [`CuDNN Installation guide`](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)


### 1. Clone the repository

Clone the Malpolon repository using `git` in the directory of your choice:
```script
git clone https://github.com/plantnet/malpolon.git
```

---

### 2. Create your virtual environment

- **Via `virtualenv` (recommended)**
  
We recommend handling your virtual environment using [`virtualenv`](https://virtualenv.pypa.io/en/stable/) (or similar) and installing the packages via `pip`.

First create your virtual environment using the proper python version, and activate it _(note that in this example, the virtual environment "malpolon_env" will be installed in the current directory)_.

```script
virtualenv -p /usr/bin/python3.8 ./malpolon_env
source ./malpolon_env/bin/activate
```

Once the env is activated, install the python packages listed in `requirements.txt`:
```script
pip install --upgrade setuptools
pip install -r requirements.txt
```

- **Via `conda`**

You can also use `conda` to install your packages.

```script
conda env create -n <name> -f environment.yml
conda activate <name>
```

---

### 3. Install Malpolon as a python package

The malpolon repository can also be installed in your virtual environment as a package. This allows you to import `malpolon` anywhere in your scripts without having to worry about file paths. It can be installed via `pip` using:

```script
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

---

The **dev** branch is susceptible to have more up-to-date content such as newer examples and experimental features. To switch to the dev branch locally, run:

```script
git checkout dev
```

## Librairies
Here is an overview of the main Python librairies used in this project. 

* [![PyTorch](https://img.shields.io/badge/PyTorch-%23ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/) - To handle deep learning loops and dataloaders
* [![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-%23792EE5.svg?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/) - Deep learning framework which simplifies the usage of PyTorch elements
* [![Numpy](https://img.shields.io/badge/Numpy-%234D77CF.svg?logo=numpy&logoColor=white)](https://numpy.org/) - For common computational operations
* [![Torchgeo](https://img.shields.io/badge/Torchgeo-%23EE4C2C.svg?logo=torchgeo&logoColor=white)](https://torchgeo.readthedocs.io/en/stable/) - To handle data rasters
* [![Matplotlib](https://img.shields.io/badge/Matplotlib-%2311557C.svg?logo=matplotlib&logoColor=white)](https://matplotlib.org/) - For displaying purposes
* [![Hydra](https://img.shields.io/badge/Hydra-%23729DB1.svg?logo=hydra&logoColor=white)](https://hydra.cc/docs/intro/) - To handle models' hyperparameters

## Examples

Examples using the GeoLifeCLEF 2022 and 2023 datasets, as well as Sentinel-2A rasters are provided in the `examples` folder. Instructions about how to train and perform predictions with your models can be found in the README file of each example in said folder.

## Documentation

An online code documentation is available via GitHub pages at [this link](https://plantnet.github.io/malpolon/). This documentation is updated each time new content is pushed to the `main` branch.

Alternatively, you can generate the documention locally by following these steps :

1. Install the additional dependences contained in `docs/docs_requirements.txt` must be installed

```script
pip install -r docs/docs_requirements.txt
```

2. Generated the documentation

```script
make -C docs html
```

The result can be found in `docs/_build/html`.

## Licensing
This framework is ditributed under the [MIT license](https://opensource.org/license/mit/), as is the Pl@ntNet project. See LICENSE.md for more information.
