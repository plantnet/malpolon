.. Malpolon documentation master file, created by
   sphinx-quickstart on Wed Apr 20 17:15:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

*****************
API documentation
*****************

.. toctree::
   :maxdepth: 10


malpolon.models
***************

malpolon.models.model_builder
=============================
.. automodule:: malpolon.models.model_builder
   :members:

malpolon.models.standard_prediction_systems
===========================================
.. automodule:: malpolon.models.standard_prediction_systems
   :members:

malpolon.models.utils
=====================
.. automodule:: malpolon.models.utils
   :members:

malpolon.models.custom_models.glc2024_multimodal_ensemble_model_modality_specific
=================================================================================
.. automodule:: malpolon.models.custom_models.glc2024_multimodal_ensemble_model_modality_specific
   :members:

malpolon.models.custom_models.glc2024_multimodal_ensemble_model
===============================================================
.. automodule:: malpolon.models.custom_models.glc2024_multimodal_ensemble_model
   :members:

malpolon.models.custom_models.glc2024_pre_extracted_prediction_system
=====================================================================
.. automodule:: malpolon.models.custom_models.glc2024_pre_extracted_prediction_system
   :members:

malpolon.models.custom_models.multi_modal
=========================================
.. automodule:: malpolon.models.custom_models.multi_modal
   :members:


malpolon.data
*************

malpolon.data.data_module
=========================
.. automodule:: malpolon.data.data_module
   :members:

malpolon.data.environmental_raster
==================================
.. automodule:: malpolon.data.environmental_raster
   :members:

malpolon.data.get_jpeg_patches_stats
====================================
.. automodule:: malpolon.data.get_jpeg_patches_stats
   :members:

malpolon.data.utils
===================
.. automodule:: malpolon.data.utils
   :members:

malpolon.data.datasets
======================

malpolon.data.datasets.torchgeo_datasets
----------------------------------------
.. automodule:: malpolon.data.datasets.torchgeo_datasets
   :members:

malpolon.data.datasets.torchgeo_sentinel2
-----------------------------------------
.. automodule:: malpolon.data.datasets.torchgeo_sentinel2
   :members:

malpolon.data.datasets.torchgeo_concat
--------------------------------------
.. automodule:: malpolon.data.datasets.torchgeo_concat
   :members:

malpolon.data.datasets.geolifeclef2022
--------------------------------------

GeoLifeCLEF2022: datasets
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: malpolon.data.datasets.geolifeclef2022.GeoLifeCLEF2022Dataset
.. autoclass:: malpolon.data.datasets.geolifeclef2022.MiniGeoLifeCLEF2022Dataset
.. automodule:: malpolon.data.datasets.geolifeclef2022.MicroGeoLifeCLEF2022Dataset

malpolon.data.datasets.geolifeclef2023
--------------------------------------

GeoLifeCLEF2023: datasets
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: malpolon.data.datasets.geolifeclef2023.PatchesDataset
.. autoclass:: malpolon.data.datasets.geolifeclef2023.PatchesDatasetMultiLabel
.. autoclass:: malpolon.data.datasets.geolifeclef2023.TimeSeriesDataset

GeoLifeCLEF2023: providers
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: malpolon.data.datasets.geolifeclef2023.PatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2023.MetaPatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2023.RasterPatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2023.MultipleRasterPatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2023.JpegPatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2023.TimeSeriesProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2023.MetaTimeSeriesProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2023.CSVTimeSeriesProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2023.MultipleCSVTimeSeriesProvider

malpolon.data.datasets.geolifeclef2024
--------------------------------------

GeoLifeCLEF2024: datasets
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: malpolon.data.datasets.geolifeclef2024.PatchesDataset
.. autoclass:: malpolon.data.datasets.geolifeclef2024.PatchesDatasetMultiLabel
.. autoclass:: malpolon.data.datasets.geolifeclef2024.TimeSeriesDataset

GeoLifeCLEF2024: providers
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: malpolon.data.datasets.geolifeclef2024.PatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2024.MetaPatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2024.RasterPatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2024.MultipleRasterPatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2024.JpegPatchProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2024.TimeSeriesProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2024.MetaTimeSeriesProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2024.CSVTimeSeriesProvider
.. autoclass:: malpolon.data.datasets.geolifeclef2024.MultipleCSVTimeSeriesProvider

malpolon.data.datasets.geolifeclef2024_pre_extracted
----------------------------------------------------
.. autofunction:: malpolon.data.datasets.geolifeclef2024_pre_extracted.construct_patch_path
.. autofunction:: malpolon.data.datasets.geolifeclef2024_pre_extracted.load_landsat
.. autofunction:: malpolon.data.datasets.geolifeclef2024_pre_extracted.load_bioclim
.. autofunction:: malpolon.data.datasets.geolifeclef2024_pre_extracted.load_sentinel

GeoLifeCLEF2024 pre-extracted: datamodules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: malpolon.data.datasets.geolifeclef2024_pre_extracted.GLC24Datamodule
   :exclude-members: __len__, __getitem__
   :members:

.. autoclass:: malpolon.data.datasets.geolifeclef2024_pre_extracted.GLC24DatamoduleHabitats
   :exclude-members: __len__, __getitem__
   :members:

GeoLifeCLEF2024 pre-extracted: datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: malpolon.data.datasets.geolifeclef2024_pre_extracted.TrainDataset
.. autoclass:: malpolon.data.datasets.geolifeclef2024_pre_extracted.TestDataset
.. autoclass:: malpolon.data.datasets.geolifeclef2024_pre_extracted.TrainDatasetHabitat
.. autoclass:: malpolon.data.datasets.geolifeclef2024_pre_extracted.TestDatasetHabitat


malpolon.plot
*************

malpolon.plot.history
=====================
.. automodule:: malpolon.plot.history
   :members:

malpolon.plot.map
=================
.. automodule:: malpolon.plot.map
   :members:

malpolon.logging
****************
.. autoclass:: malpolon.logging.Summary
   :exclude-members: __len__, __getitem__
   :members:

.. autofunction:: malpolon.logging.str_object

malpolon.check_install
**********************
.. automodule:: malpolon.check_install
   :members:
