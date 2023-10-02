.. Malpolon documentation master file, created by
   sphinx-quickstart on Wed Apr 20 17:15:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

Welcome to Malpolon's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installation
============

To install malpolon, you will first need to install **Python 3.8 or newer**, and several python packages. To do so, it is best practice to create a virtual environment containing all these packages locally.

⚠️ **macOS** installation may be bumpy for now. More instructions will be added shortly. It is recommended to stick to Linux for the time being. ⚠️

0. Requirements
---------------

Before proceeding, please make sure the following packages are installed on your system:

- [Python3.8](https://www.python.org/downloads/) (or newer)
- [``pip``](https://pip.pypa.io/en/stable/installation/)
- [``git``](https://git-scm.com/downloads)
- ``libgeos-dev`` (dependency of Python library ``Cartopy``)
  - On Linux (Ubuntu): ``sudo apt install libgeos-dev``
  - On MacOS: ``brew install geos``
- ``cmake``
  - On Linux (Ubuntu): ``sudo apt install cmake``
  - On MacOS: ``brew install cmake``
- ``cuda`` (if you intend to run your models on GPU)
  - [``CUDA Installation guide``](https://docs.nvidia.com/cuda/index.html)
  - [``CuDNN Installation guide``](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

1. Clone the repository
-----------------------

Clone the Malpolon repository using ``git`` in the directory of your choice::
   
	git clone https://github.com/plantnet/malpolon.git

2. Create your virtual environment
----------------------------------

- **Via** ``virtualenv`` **(recommended)**
  
We recommend handling your virtual environment using [``virtualenv``](https://virtualenv.pypa.io/en/stable/) (or similar) and installing the packages via ``pip``.

First create your virtual environment using the proper python version, and activate it _(note that in this example, the virtual environment "malpolon_env" will be installed in the current directory)_.::

   virtualenv -p /usr/bin/python3.8 ./malpolon_env
   source ./malpolon_env/bin/activate

Once the env is activated, install the python packages listed in ``requirements.txt``::
   
   pip install --upgrade setuptools
   pip install -r requirements.txt

- **Via** ``conda``
You can also use ``conda`` to install your packages.::

   conda env create -n <name> -f environment.yml
   conda activate <name>

3. Install Malpolon as a python package
---------------------------------------

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

   cd examples/sentinel2_raster_torchgeo
   python cnn_on_rgbnir_torchgeo.py trainer.accelerator=cpu run.predict=false trainer.max_epochs=5

This will train a ResNet-50 CNN on a 1 tile, 4 modalities (RGB + IR) sample of the Sentinel-2A dataset for 5 epochs, using the CPU. The model will be saved in the ``outputs`` folder. To run the prediction on the test set, run::

   python cnn_on_rgbnir_torchgeo.py trainer.accelerator=cpu run.predict=true checkpoint_path=../<PATH_TO_YOUR_CHECKPOINT_FILE>

Examples
========

Examples using the GeoLifeCLEF 2022 and 2023 datasets, as well as Sentinel-2A rasters are provided in the ``examples`` folder. Instructions about how to train and perform predictions with your models can be found in the README file of each example in said folder.

Build your own scripts by modifying the provided examples!

API documentation
=================

Data
----

.. automodule:: malpolon.data
    :members:


Datasets
--------

.. automodule:: malpolon.data.datasets
    :members:


Models
------

.. automodule:: malpolon.models
   :members:


Indices and tables
==================

* :ref:``genindex``
* :ref:``modindex``
* :ref:``search``
