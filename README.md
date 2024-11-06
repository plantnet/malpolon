<p align="center">
  <a href="https://pypi.org/project/malpolon/"><img src="https://img.shields.io/pypi/v/malpolon" alt="Python version"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/pypi/pyversions/malpolon" alt="Python version"></a>
  <a href="https://github.com/plantnet/malpolon/issues"><img src="https://img.shields.io/github/issues/plantnet/malpolon" alt="GitHub issues"></a>
  <a href="https://github.com/plantnet/malpolon/pulls"><img src="https://img.shields.io/github/issues-pr/plantnet/malpolon" alt="GitHub pull requests"></a>
  <a href="https://github.com/plantnet/malpolon/graphs/contributors"><img src="https://img.shields.io/github/contributors/plantnet/malpolon" alt="GitHub contributors"></a>
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

Developed as part of the European GUARDEN (ID: 101060693) and MAMBO (ID: 101060639) projects, Malpolon is a framework facilitating the training and sharing of Deep Species Distribution models using various types of input covariates including bioclimatic rasters, remote sensing images, land-use rasters, etc...

If you're not a deep learning or PyTorch expert but nonetheless want to use visual deep learning models on satellite and/or bioclimatic rasters to predict the presence of species on a given area, this framework is for you.

## 🧭 Usage

Malpolon is destined to be used by various user profiles, some more experimented than others. To this end, we provide several examples of usage of the framework, organized by use case or _scenarios_. These examples can be found in the `examples` folder of the repository, each with a README file for more details on how to use the scripts. Additionally, check out our guide "[**Getting started with examples**](examples/)".

Here is a list of the currently available scenarios:

- [**Benchmarks**](examples/benchmarks/) : I want to compare the performance of different models on a given known dataset;\
  or I am a potential kaggle participant on the GeoLifeClef challenge. I want to train a model on the provided datasets without having to worry about the data loading, starting from a plug-and-play example.
  - [<u>GeoLifeClef2022</u>](examples/benchmarks/geolifeclef2022/) : contains a fully functionnal example of a model training on the GeoLifeClef2022 dataset, from data download, to training and prediction.
  - [<u>GeoLifeClef2023</u>](examples/benchmarks/geolifeclef2023/) : contains dataloaders for the GeoLifeClef2023 dataset (different from the GLC2022 dataloaders). The training and prediction scripts are not provided.
  - [<u>GeoLifeClef2024 (pre-extracted)</u>](examples/benchmarks/geolifeclef2024_pre_extracted/) : contains a fully functional example of a multimodal ensemble model used to provide a strong baseline for the [GeoLifeClef2024 kaggle competition](https://www.kaggle.com/competitions/geolifeclef-2024). The example uses unique dataloaders and models to handle pre-extracted values from satellite patches, satellite time series and bioclimatic time series.
- [**Train (custom datasets)**](examples/custom_train/) : I have a dataset of my own and I want to train a model on it. I want to be able to easily customize the training process and the model architecture.
  - <u>Drop and play</u> : I have an observations file (.csv) and I want to train a model on different environmental variables (rasters, satellite imagery) without having to worry about the data loading.
  - <u>Custom dataset</u> : I have my own dataset consisting of pre-extracted image patches and/or rasters and I want to train a model on it.
- [**Inference**](examples/inference/) : I have an observations file (.csv) and I want to predict the presence of species on a given area using a model I trained previously and a selected dataset or a shapefile I would provide.

## ⚙️ Installation

To install malpolon, you will first need to install **Python ≥ 3.10**, and several python packages. To do so, it is best practice to create a virtual environment containing all these packages locally.

⚠️ **macOS** installation does not yet include instructions on how to properly set up GPU usage for GPU-enabled mac. For training purposes we recommend sticking to Linux for the time being. ⚠️

<details>
  <summary><i><u>Click here to expand instructions</u></i></summary>

### Requirements

Before proceeding, please make sure the following packages are installed on your system:

- [Python ≥ 3.10](https://www.python.org/downloads/)
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

### Install from `PyPi`
The backend side of malpolon is distributed as a package on `PyPi`. To install it, simply run the following command:

```script
pip install malpolon
```

However, versions available on PyPi are non-experimental and possibly behind the repository's `main` and `dev` branches. To know which version you want download, please refer to the *tags* section of the repository and match it with PyPi.
Furthermore, the PyPi package does not include the examples and the documentation. If you want to install the full repository, follow the next steps.

### Install from `GitHub`
#### 1. Clone the repository

Clone the Malpolon repository using `git` in the directory of your choice:
```script
git clone https://github.com/plantnet/malpolon.git
```

---

#### 2. Create your virtual environment

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

⚠️ Be aware that conda recently changed its licensing and you may subject to fees, or be limited in downloads. Sources: [anaconda website](https://www.anaconda.com/blog/update-on-anacondas-terms-of-service-for-academia-and-research),
[datacamp blog recap](https://www.datacamp.com/blog/navigating-anaconda-licensing) ⚠️

You can also use `conda` to install your packages.

```script
conda env create -f environment_python3.10.yml
conda activate malpolon_3.10
```

---

#### 3. Install Malpolon as a python package

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
Using PyTorch version 2.1.0+cu121
CUDA available: True (version: 12.1)
cuDNN available: True (version: 8902)
Number of CUDA-compatible devices found: 1
```

---

The **dev** branch is susceptible to have more up-to-date content such as newer examples and experimental features. To switch to the dev branch locally, run:

```script
git checkout dev
```

</details>

## 📄 Documentation

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


## ⚒️ Troubleshooting
Commonly encountered errors when using the framework are compiled [here](examples/README.md#⚒️-troubleshooting).

## 🚀 Contributing
### **Guidelines**

Issues and PR templates are provided to help you start a contribution to the project.

A checking script is also provided and can run checks relative to the 2 next sections with the following command:

```bash
./checkMyCode all
```

### **Unit tests**
<details>
  <summary><i><u>Click here to expand instructions</u></i></summary>

When submitting, make sure the unit tests all pass without errors. These tests are located at `malpolon/tests/` and can be ran all at once, with a code coverage estimation, via command line:

```bash
./checkMyCode.sh t  # or `pytest malpolon/tests/`
```
Specify a file path as argument to run a single test file:

```bash
./checkMyCode.sh malpolon/tests/<TEST_FILE>.py  # or `pytest malpolon/tests/<TEST_FILE>.py`
```

Run individual test functions via `python malpolon/tests/test_<module>.py` by modifying the files beforehand to call the functions you want to test with:

```python
if __name__ == '__main__':
  test_my_function()
```

**This is especially useful for `malpolon/tests/test_examples.py` which tests all the provided examples**, ensuring they do not crash. However, these **require having all the datasets and take a while to run**. Some data you might not have local access to.\
To skip a test function, add a decorator `@pytest.mark.skip()` above the function definition.

</details>

### **Linting**

<details>
  <summary><i><u>Click here to expand instructions</u></i></summary>

Likewise, do care about writing a clean code. The project uses `flake8`, `Pylint` and `Pydocstyle` to check the good formatting and documentation of your code. To run linters check on your code you can either run each of these library independently or use the checking script:

```bash
./checkMyCode.sh l
```

Run linters on non-test file(s) :

```bash
./checkMyCode.sh <FILE_PATH_1> <FILE_PATH_2>
```
</details>

## 🚆 Roadmap

This roadmap outlines the planned features and milestones for the project. Please note that the roadmap is subject to change and may be updated as the project progress.

<details>
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

## Libraries
Here is an overview of the main Python librairies used in this project.

* [![PyTorch](https://img.shields.io/badge/PyTorch-%23ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/) - To handle deep learning loops and dataloaders
* [![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-%23792EE5.svg?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/) - Deep learning framework which simplifies the usage of PyTorch elements
* [![Numpy](https://img.shields.io/badge/Numpy-%234D77CF.svg?logo=numpy&logoColor=white)](https://numpy.org/) - For common computational operations
* [![Torchgeo](https://img.shields.io/badge/Torchgeo-%23EE4C2C.svg?logo=torchgeo&logoColor=white)](https://torchgeo.readthedocs.io/en/stable/) - To handle data rasters
* [![Matplotlib](https://img.shields.io/badge/Matplotlib-%2311557C.svg?logo=matplotlib&logoColor=white)](https://matplotlib.org/) - For displaying purposes
* [![Hydra](https://img.shields.io/badge/Hydra-%23729DB1.svg?logo=hydra&logoColor=white)](https://hydra.cc/docs/intro/) - To handle models' hyperparameters
* [![Cartopy](https://img.shields.io/badge/Cartopy-%2300A1D9.svg?logo=cartopy&logoColor=white)](https://scitools.org.uk/cartopy/docs/latest/) - To handle geographical data

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

## Licensing
This framework is ditributed under the [MIT license](https://opensource.org/license/mit/), as is the Pl@ntNet project. See LICENSE.md for more information.

## Citation & credits
Malpolon is a project developed by the [Pl@ntNet](https://plantnet.org/) team within the [INRIA](https://www.inria.fr/en) research institute. If you use this framework in your research, please cite this repository in your paper.

Authors include :
- [Théo Larcher](https://github.com/tlarcher) (current lead developper) ([email](mailto:theo.larcher@inria.fr))
- [Maximilien Servajean](https://github.com/maximiliense)
- [Alexis Joly](https://github.com/alexisjoly)

Former developpers include :
- [Titouan Lorieul](https://github.com/tlorieul) (former lead developper) ([email](mailto:titouan.lorieul@gmail.com))
