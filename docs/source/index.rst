.. Malpolon documentation master file, created by
   sphinx-quickstart on Wed Apr 20 17:15:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

=====================================
Welcome to Malpolon's documentation !
=====================================

.. figure:: ../resources/Malpolon_transparent.png
  :width: 300
  :alt: Malpolon_logo
  :align: center


Malpolon is a framework facilitating the training and sharing of Deep Species Distribution models using various types of input covariates including bioclimatic rasters, remote sensing images, land-use rasters, etc...

If you're not a deep learning or PyTorch expert but nonetheless want to use visual deep learning models on satellite and/or bioclimatic rasters to predict the presence of species on a given area, this framework is for you.

.. toctree::
   :maxdepth: 1

   api
   examples

🔧 Installation
===============

To install malpolon, you will first need to install **Python ≥ 3.10**, and several python packages. To do so, it is best practice to create a virtual environment containing all these packages locally.

⚠️ **macOS** installation does not yet include instructions on how to properly set up GPU usage for GPU-enabled mac. For training purposes we recommend sticking to Linux for the time being. ⚠️

Requirements
````````````

Before proceeding, please make sure the following packages are installed on your system:

- `Python ≥ 3.10 <https://www.python.org/downloads/>`_
- `pip <https://pip.pypa.io/en/stable/installation/>`_
- `git <https://git-scm.com/downloads>`_
- ``libgeos-dev`` (dependency of Python library ``Cartopy``)

  - On Linux (Ubuntu): ``sudo apt install libgeos-dev``

  - On MacOS: ``brew install geos``

- ``cmake``

  - On Linux (Ubuntu): ``sudo apt install cmake``

  - On MacOS: ``brew install cmake``

- ``cuda`` (if you intend to run your models on GPU)

  - `CUDA Installation guide <https://docs.nvidia.com/cuda/index.html>`_
  
  - `CuDNN Installation guide <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>`_

The following instructions show installation commands for Python 3.10, but can be adapted for any of the compatible Python versions metionned above by simply changing the version number.

Install from ``PyPi``
`````````````````````
The backend side of malpolon is distributed as a package on `PyPi`. To install it, simply run the following command::


   pip install malpolon


However, versions available on PyPi are non-experimental and possibly behind the repository's `main` and `dev` branches. To know which version you want download, please refer to the *tags* section of the repository and match it with PyPi.
Furthermore, the PyPi package does not include the examples and the documentation. If you want to install the full repository, follow the next steps.


Install from ``GitHub``
```````````````````````

1. Clone the repository
'''''''''''''''''''''''
Clone the Malpolon repository using ``git`` in the directory of your choice::

   git clone https://github.com/plantnet/malpolon.git


2. Create your virtual environment
''''''''''''''''''''''''''''''''''

- **Via** ``virtualenv`` **(recommended)**
  
We recommend handling your virtual environment using [``virtualenv``](https://virtualenv.pypa.io/en/stable/) (or similar) and installing the packages via `pip`.

First create your virtual environment using the proper python version, and activate it _(note that in this example, the virtual environment "malpolon_env" will be installed in the current directory)_.::

   virtualenv -p /usr/bin/python3.10 ./malpolon_3.10
   source ./malpolon_3.10/bin/activate

Once the env is activated, install the python packages listed in ``requirements_python3.10.txt``::

   pip install --upgrade setuptools
   pip install -r requirements_python3.10.txt


- **Via** ``conda``

You can also use ``conda`` to install your packages::

   conda env create -n <name> -f environment.yml
   conda activate <name>

3. Install Malpolon as a python package
'''''''''''''''''''''''''''''''''''''''

The malpolon repository can also be installed in your virtual environment as a package. This allows you to import ``malpolon`` anywhere in your scripts without having to worry about file paths. It can be installed via ``pip`` using::

   cd malpolon
   pip install -e .

To check that the installation went well, use the following command::

   python -m malpolon.check_install

which, if you have CUDA properly installed, should output something similar to::

   Using PyTorch version 1.13.0
   CUDA available: True (version: 11.6)
   cuDNN available: True (version: 8302)
   Number of CUDA-compatible devices found: 1

----

The **dev** branch is susceptible to have more up-to-date content such as newer examples and experimental features. To switch to the dev branch locally, run::

   git checkout dev

Quick start
===========

For a quick overview of the framework, you can run the following commands::

   cd examples/ecologists/sentinel-2a
   python cnn_on_rgbnir_torchgeo.py trainer.accelerator=cpu run.predict=false trainer.max_epochs=5

This will train a ResNet-50 CNN on a 1 tile, 4 modalities (RGB + IR) sample of the Sentinel-2A dataset for 5 epochs, using the CPU. The model will be saved in the ``outputs`` folder. To run the prediction on the test set, run::

   python cnn_on_rgbnir_torchgeo.py trainer.accelerator=cpu run.predict=true checkpoint_path=../<PATH_TO_YOUR_CHECKPOINT_FILE>

Examples
========

Examples using the GeoLifeCLEF 2022 and 2023 datasets, as well as Sentinel-2A rasters are provided in the ``examples`` folder. Instructions about how to train and perform predictions with your models can be found in the README file of each example in said folder.

Build your own scripts by modifying the provided examples!


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
