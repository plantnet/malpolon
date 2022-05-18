.. Malpolon documentation master file, created by
   sphinx-quickstart on Wed Apr 20 17:15:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Malpolon's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installation
============

Currently, only the development version is available.
First make sure that the dependances listed in the ``requirements.txt`` file are installed.
malpolon can then be installed via ``pip`` using::

	git clone https://github.com/plantnet/malpolon.git
	cd malpolon
	pip install -e .


Quick start
===========

 .. todo::

	Complete this section once framework is sufficiently advanced.
	

Examples
========

Examples using the GeoLifeCLEF 2022 dataset is provided in the ``examples`` folder.


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

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
