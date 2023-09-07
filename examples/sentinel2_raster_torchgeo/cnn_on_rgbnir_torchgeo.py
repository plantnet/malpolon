"""Main script to run training or inference on torchgeo datasets.

This script runs the RasterSentinel2 dataset class by default.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@inria.fr>
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import hydra
import omegaconf
import planetary_computer
import pystac
import pytorch_lightning as pl
import torchmetrics.functional as Fmetrics
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import Units
from torchvision import transforms

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.torchgeo_datasets import (RasterSentinel2,
                                                      Sentinel2GeoSampler)
from malpolon.logging import Summary
from malpolon.models import FinetuningClassificationSystem

FMETRICS_CALLABLES = {'binary_accuracy': Fmetrics.accuracy,
                      'multiclass_accuracy': Fmetrics.classification.multiclass_accuracy,
                      'multilabel_accuracy': Fmetrics.classification.multilabel_accuracy, }


class Sentinel2TorchGeoDataModule(BaseDataModule):
    """Data module for Sentinel-2A dataset."""
    def __init__(
        self,
        dataset_path: str,
        labels_name: str = 'labels.csv',
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        size: int = 200,
        units: Units = Units.CRS,
        crs: int = 4326,
        binary_positive_classes: list = [],
        task: str = 'classification_multiclass',  # ['classification_binary', 'classification_multiclass', 'classification_multilabel']
        download_data_sample: bool = False
    ):
        """Class constructor.

        Parameters
        ----------
        dataset_path : str
            path to the directory containing the data
        labels_name : str, optional
            labels file name, by default 'labels.csv'
        train_batch_size : int, optional
            train batch size, by default 32
        inference_batch_size : int, optional
            inference batch size, by default 256
        num_workers : int, optional
            how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the
            main process, by default 8
        size : int, optional
            size of the 2D extracted patches. Patches can either be
            square (int/float value) or rectangular (tuple of int/float).
            Defaults to a square of size 200, by default 200
        units : Units, optional
             The dataset's unit system, must have a value in
             ['pixel', 'crs'], by default Units.CRS
        crs : int, optional
            `coordinate reference system (CRS)` to warp to
            (defaults to the CRS of the first file found), by default 4326
        binary_positive_classes : list, optional
            labels' classes to consider valid in the case of binary
            classification with multi-class labels (defaults to all 0),
            by default []
        task : str, optional
            machine learning task (used to format labels accordingly),
            by default 'classification_multiclass'
        """
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.labels_name = labels_name
        self.size = size
        self.units = units
        self.crs = crs
        self.sampler = Sentinel2GeoSampler
        self.task = task
        self.binary_positive_classes = binary_positive_classes
        if download_data_sample:
            self.download_data_sample()

    def download_data_sample(self):
        """Download 4 Sentinel-2A tiles from MPC.

        This method is useful to quickly download a sample of
        Sentinel-2A tiles via Microsoft Planetary Computer (MPC).
        The referenced of the tile downloaded are specified by the
        `tile_id` and `timestamp` variables. Tiles are not downloaded
        if they already have been and are found locally.
        """
        tile_id = 'T31TEJ'
        timestamp = '20190801T104031'
        item_urls = [f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2A_MSIL2A_{timestamp}_R008_{tile_id}_20201004T190635"]
        for item_url in item_urls:
            item = pystac.Item.from_file(item_url)
            signed_item = planetary_computer.sign(item)
            for band in ["B08", "B03", "B02", "B04"]:
                asset_href = signed_item.assets[band].href
                filename = urlparse(asset_href).path.split("/")[-1]
                download_url(asset_href, self.dataset_path, filename)

    def get_dataset(self, split, transform, **kwargs):
        dataset = RasterSentinel2(
            self.dataset_path,
            labels_name=self.labels_name,
            split=split,
            task=self.task,
            binary_positive_classes=self.binary_positive_classes,
            transforms_data=transform,
            **kwargs
        )
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_train,
            sampler=self.sampler(self.dataset_train, size=self.size, units=self.units, crs=self.crs),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_val,
            sampler=self.sampler(self.dataset_val, size=self.size, units=self.units, crs=self.crs),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_test,
            sampler=self.sampler(self.dataset_test, size=self.size, units=self.units, crs=self.crs),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_predict,
            sampler=self.sampler(self.dataset_predict, size=self.size, units=self.units),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.2], std=[0.229, 0.224, 0.225, 0.2]
                ),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                transforms.CenterCrop(size=128),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.2], std=[0.229, 0.224, 0.225, 0.2]
                ),
            ]
        )


class ClassificationSystem(FinetuningClassificationSystem):
    """Classification task class."""
    def __init__(
        self,
        model: dict,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        metrics: dict,
        task: str = 'classification_multiclass',
        hparams_preprocess: bool = True,
    ):
        """Class constructor.

        Parameters
        ----------
        model : dict
            _description_
        lr : float
            learning rate
        weight_decay : float
            weight decay
        momentum : float
            value of momentum
        nesterov : bool
            if True, uses Nesterov's momentum
        metrics : dict
            dictionnary containing the metrics to compute.
            Keys must match metrics' names and have a subkey with each
            metric's functional methods as value. This subkey is either
            created from the FMETRICS_CALLABLES constant or supplied,
            by the user directly.
        task : str, optional
            machine learning task (used to format labels accordingly),
            by default 'classification_multiclass'
        hparams_preprocess : bool, optional
            if True performs preprocessing operations on the hyperparameters,
            by default True
        """
        if hparams_preprocess:
            task = task.split('classification_')[1]
            try:
                metrics = omegaconf.OmegaConf.to_container(metrics)
                for k, v in metrics.items():
                    if 'callable' in v:
                        metrics[k]['callable'] = eval(v['callable'])
                    else:
                        metrics[k]['callable'] = FMETRICS_CALLABLES[k]
            except ValueError as e:
                print('\n[WARNING]: Please make sure you have registered'
                      ' a dict-like value to your "metrics" key in your'
                      ' config file. Defaulting metrics to None.\n')
                print(e, '\n')
                metrics = None
            except KeyError as e:
                print('\n[WARNING]: Please make sure the name of your metrics'
                      ' registered in your config file match an entry'
                      ' in constant FMETRICS_CALLABLES.'
                      ' Defaulting metrics to None.\n')
                print(e, '\n')
                metrics = None

        super().__init__(
            model,
            lr,
            weight_decay,
            momentum,
            nesterov,
            metrics,
            task,
        )


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_on_rgbnir_torchgeo_config")
def main(cfg: DictConfig) -> None:
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """
    # cfg.data.dataset_pathstate_dict = state_dict.model.state_dict = '../../../' + cfg.data.dataset_path  # Uncomment if value contains only the name of the dataset folder. Only works with a 3-folder-deep hydra job path.
    logger = pl.loggers.CSVLogger(".", name="", version="")
    logger.log_hyperparams(cfg)

    datamodule = Sentinel2TorchGeoDataModule(**cfg.data, **cfg.task)
    model = ClassificationSystem(cfg.model, **cfg.optimizer, **cfg.task)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}-{step}-{" + f"val_{next(iter(model.metrics.keys()))}" + ":.4f}",
            monitor=f"val_{next(iter(model.metrics.keys()))}",
            mode="max",
        ),
    ]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    if cfg.run.predict:
        model_loaded = ClassificationSystem.load_from_checkpoint(cfg.run.checkpoint_path,
                                                                 model=model.model,
                                                                 hparams_preprocess=False)

        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        print('Test dataset prediction (extract) : ', predictions[:10])

        # Option 2: Predict 1 data point (Pytorch)
        test_data = datamodule.get_test_dataset()
        query_point = {'lon': test_data.coordinates[0][0], 'lat': test_data.coordinates[0][1],
                       'crs': datamodule.crs,
                       'size': datamodule.size,
                       'units': datamodule.units}
        test_data_point = test_data[query_point][0]
        test_data_point = test_data_point.resize_(1, *test_data_point.shape)
        prediction = model_loaded.predict_point(cfg.run.checkpoint_path,
                                                test_data_point,
                                                ['model.', ''])
        print('Point prediction : ', prediction)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
        trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
