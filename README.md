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

Malpolon is a framework facilitating the training and sharing of Deep Species Distribution models using various types of input covariates including bioclimatic rasters, remote sensing images, land-use rasters, etc...

## Acknowledgments

This work is made possible through public financing by the [European Commission](https://commission.europa.eu/index_en) on european projects [MAMBO](https://www.mambo-project.eu/) and [GUARDEN](https://guarden.org/).

<div align="center">
  <a href="https://www.mambo-project.eu/"><img src="docs/resources/mambo_logo.png" alt="MAMBO_logo" style="width: 200px;  margin-top: 15px; margin-right: 50px;"></a>
  <a href="https://guarden.org/"><img src="docs/resources/guarden_logo.png" alt="GUARDEN_logo" style="width: 230px; height: auto; margin-right: 50px;"></a>
</div>
<div align="center">
  <a href="https://commission.europa.eu/index_en"><img src="docs/resources/logo-ec--en.svg" alt="europ_commission_logo" style="width: 300px;  margin-top: 20px; margin-bottom: 15px;"></a>
</div>

This work is currently under development and maintained by the [Pl@ntNet](https://plantnet.org/) team within the [INRIA](https://www.inria.fr/en) research institute.

<div align="center">
  <a href="https://www.inria.fr/en"><img src="docs/resources/inria.png" alt="MAMBO_logo" style="width: 150px;  margin-top: 15px; margin-right: 50px;"></a>
  <a href="https://plantnet.org/"><img src="docs/resources/plantnet_logo.png" alt="GUARDEN_logo" style="width: 250px; height: auto; margin-right: 50px;"></a>
</div>

## Roadmap

This roadmap outlines the planned features and milestones for the project. Please note that the roadmap is subject to change and may be updated as the project progress.

<details open>
  <summary><i><u>Click here to toggle roadmap</u></i></summary>
  <br>

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

</details>

## Installation

To install malpolon, you will first need to install **Python 3.8, 3.9 or 3.10**, and several python packages. To do so, it is best practice to create a virtual environment containing all these packages locally.

⚠️ **macOS** installation does not yet include instructions on how to properly set up GPU usage for GPU-enabled mac. For training purposes we recommend sticking to Linux for the time being. ⚠️

<details>
  <summary><i><u>Click here to expand instructions</u></i></summary>

### 0. Requirements

Before proceeding, please make sure the following packages are installed on your system:

- [3.8 ≤ Python ≤ 3.10](https://www.python.org/downloads/)
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

The following instructions show installation commands for Python 3.10, but can be adapted for any of the compatible Python versions metionned above by simply changing the version number.

### 1. Clone the repository

Clone the Malpolon repository using `git` in the directory of your choice:
```script
git clone https://github.com/plantnet/malpolon.git
```

---

### 2. Create your virtual environment

- **Via `virtualenv`**
  
We recommend handling your virtual environment using [`virtualenv`](https://virtualenv.pypa.io/en/stable/) (or similar) and installing the packages via `pip`.

First create your virtual environment using the proper python version, and activate it _(note that in this example, the virtual environment "malpolon_env" will be installed in the current directory)_.

```script
virtualenv -p /usr/bin/python3.10 ./malpolon_3.10
source ./malpolon_3.10/bin/activate
```

Once the env is activated, install the python packages listed in `requirements_python3.10.txt`:
```script
pip install --upgrade setuptools
pip install -r requirements_python3.10.txt
```

- **Via `conda`**

You can also use `conda` to install your packages.

```script
conda env create -f environment_python3.10.yml
conda activate malpolon_3.10
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

</details>

## Usage

Malpolon is destined to be used by various user profiles, some more experimented than others. To this end, we provide several examples of usage of the framework, organized by use case or _scenarios_. These examples can be found in the `examples` folder of the repository, each with a README file for more details on how to use the scripts.

Here is a list of the currently available scenarios:

- [**Kaggle**](examples/kaggle/) : I am a potential kaggle participant on the GeoLifeClef challenge. I want to train a model on the provided datasets without having to worry about the data loading, starting from a plug-and-play example.
  - [<u>GeoLifeClef2022</u>](examples/kaggle/geolifeclef2022/) : contains a fully functionnal example of a model training on the GeoLifeClef2022 dataset, from data download, to training and prediction.
  - [<u>GeoLifeClef2023</u>](examples/kaggle/geolifeclef2023/) : contains dataloaders for the GeoLifeClef2023 dataset (different from the GLC2022 dataloaders). The training and prediction scripts are not provided.
- [**Ecologists**](examples/ecologists/) : I have a dataset of my own and I want to train a model on it. I want to be able to easily customize the training process and the model architecture.
  - <u>Drop and play</u> : I have an observations file (.csv) and I want to train a model on different environmental variables (rasters, satellite imagery) without having to worry about the data loading.
  - <u>Custom dataset</u> : I have my own dataset consisting of pre-extracted image patches and/or rasters and I want to train a model on it.
- [**Inference**](examples/inference/) : I have an observations file (.csv) and I want to predict the presence of species on a given area using a model I trained previously and a selected dataset or a shapefile I would provide.

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

## Librairies
Here is an overview of the main Python librairies used in this project. 

* [![PyTorch](https://img.shields.io/badge/PyTorch-%23ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/) - To handle deep learning loops and dataloaders
* [![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-%23792EE5.svg?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/) - Deep learning framework which simplifies the usage of PyTorch elements
* [![Numpy](https://img.shields.io/badge/Numpy-%234D77CF.svg?logo=numpy&logoColor=white)](https://numpy.org/) - For common computational operations
* [![Torchgeo](https://img.shields.io/badge/Torchgeo-%23EE4C2C.svg?logo=torchgeo&logoColor=white)](https://torchgeo.readthedocs.io/en/stable/) - To handle data rasters
* [![Matplotlib](https://img.shields.io/badge/Matplotlib-%2311557C.svg?logo=matplotlib&logoColor=white)](https://matplotlib.org/) - For displaying purposes
* [![Hydra](https://img.shields.io/badge/Hydra-%23729DB1.svg?logo=hydra&logoColor=white)](https://hydra.cc/docs/intro/) - To handle models' hyperparameters
* [![Cartopy](https://img.shields.io/badge/Cartopy-%2300A1D9.svg?logo=cartopy&logoColor=white)](https://scitools.org.uk/cartopy/docs/latest/) - To handle geographical data
 

## Licensing
This framework is ditributed under the [MIT license](https://opensource.org/license/mit/), as is the Pl@ntNet project. See LICENSE.md for more information.
