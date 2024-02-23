"""Main script to run training on microlifeclef2022 dataset.

Uses RGB pre-extracted patches from the dataset.
This script was created for Kaggle participants of the GeoLifeCLEF 2022
challenge.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
"""

import hydra
import pytorch_lightning as p
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from transforms import RGBDataTransform

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.geolifeclef2022 import (GeoLifeCLEF2022Dataset,
                                                    MiniGeoLifeCLEF2022Dataset)
from malpolon.logging import Summary
from malpolon.models.standard_prediction_systems import ClassificationSystem


class GeoLifeCLEF2022DataModule(BaseDataModule):
    r"""
    Data module for GeoLifeCLEF 2022.

    Parameters
    ----------
        dataset_path: Path to dataset
        minigeolifeclef: if True, loads MiniGeoLifeCLEF 2022, otherwise loads GeoLifeCLEF2022
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation, testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        dataset_path: str,
        minigeolifeclef: bool = False,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        download: bool = False,
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.minigeolifeclef = minigeolifeclef
        self.download = download

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                lambda data: RGBDataTransform()(data["rgb"]),
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                lambda data: RGBDataTransform()(data["rgb"]),
                transforms.CenterCrop(size=224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_dataset(self, split, transform, **kwargs):
        if self.minigeolifeclef:
            dataset_cls = MiniGeoLifeCLEF2022Dataset
        else:
            dataset_cls = GeoLifeCLEF2022Dataset

        dataset = dataset_cls(
            self.dataset_path,
            split,
            patch_data=["rgb"],
            use_rasters=False,
            transform=transform,
            download=self.download,
            **kwargs
        )
        return dataset


@hydra.main(version_base="1.3", config_path="config", config_name="mono_modal_3_channels_model")
def main(cfg: DictConfig) -> None:

    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = pl.loggers.CSVLogger(log_dir, name=False, version="")
    logger.log_hyperparams(cfg)

    datamodule = GeoLifeCLEF2022DataModule(**cfg.data)

    model = ClassificationSystem(cfg.model, **cfg.optimizer, **cfg.task)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="checkpoint-{epoch:02d}-{step}-{val_top_30_multiclass_accuracy:.4f}",
            monitor="val_top_30_multiclass_accuracy",
            mode="max",
        ),
    ]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)
    trainer.fit(model, datamodule=datamodule)

    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
