from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator, Optional, Tuple, Union

import hydra
import pytorch_lightning as pl
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers.constants import Units
from torchgeo.samplers import GeoSampler
import torchmetrics.functional as Fmetrics
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from transforms import RGBDataTransform
from torch.utils.data import DataLoader
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box, tile_to_chips

import torch
from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.geolifeclef2022 import MicroGeoLifeCLEF2022Dataset
from malpolon.logging import Summary
from malpolon.models import FinetuningClassificationSystem
from torchgeo.samplers import RandomGeoSampler, Units
from malpolon.data.datasets.torchgeo_datasets import RasterTorchGeoDataset
import pandas as pd
from malpolon.data.utils import is_point_in_bbox

if TYPE_CHECKING:
    from typing import Any, Callable, Mapping, Optional, Union


class Sentinel2GeoSampler(GeoSampler):
    def __init__(self,
                 dataset: GeoDataset,
                 size: Union[Tuple[float, float], float],
                 length: Optional[int],
                 roi: Optional[BoundingBox] = None,
                 units: Units = Units.PIXELS,
    ) -> None:
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.coordinates = dataset.coordinates
        self.length = 0
        self.bounds = (dataset.bounds.minx, dataset.bounds.maxx,
                       dataset.bounds.miny, dataset.bounds.maxy)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        if length is not None:
            self.length = len(dataset.observation_ids)

    def __iter__(self) -> Iterator[BoundingBox]:
        for _ in range(len(self)):
            coords = tuple(self.coordinates[_])
            if is_point_in_bbox(coords, self.bounds):
                yield self.coordinates[_]


class Sentinel2TorchGeoDataModule(BaseDataModule):
    r"""
    Data module for MicroGeoLifeCLEF 2022.

    Parameters
    ----------
        dataset_path: Path to dataset
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation, testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        dataset_path: str,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        size: int = 256,
        units: Units = Units.CRS
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.sampler = Sentinel2GeoSampler
        self.size = size
        self.units = units

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

    # def prepare_data(self):
    #     MicroGeoLifeCLEF2022Dataset(
    #         self.dataset_path,
    #         subset="train",
    #         use_rasters=False,
    #         download=True,
    #     )

    def get_dataset(self, split, transform, **kwargs):
        dataset = RasterTorchGeoDataset(
            self.dataset_path,
            split,
            bands=["rgb"],
            use_rasters=False,
            transform=transform,
            **kwargs
        )
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_train,
            sampler=self.sampler(self.dataset_train, size=self.size, units=self.units),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_val,
            sampler=self.sampler(self.dataset_val, size=self.size, units=self.units),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_test,
            sampler=self.sampler(self.dataset_test, size=self.size, units=self.units),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader


class ClassificationSystem(FinetuningClassificationSystem):
    def __init__(
        self,
        model: dict,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
    ):
        metrics = {
            "accuracy": Fmetrics.accuracy,
        }

        super().__init__(
            model,
            lr,
            weight_decay,
            momentum,
            nesterov,
            metrics,
        )


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_on_rgb_patches_config")
def main(cfg: DictConfig) -> None:
    # cfg.data.dataset_pathstate_dict = state_dict.model.state_dict = '../../../' + cfg.data.dataset_path  # Uncomment if value contains only the name of the dataset folder. Only works with a 3-folder-deep hydra job path.
    logger = pl.loggers.CSVLogger(".", name="", version="")
    logger.log_hyperparams(cfg)

    datamodule = Sentinel2TorchGeoDataModule(**cfg.data)

    model = ClassificationSystem(cfg.model, **cfg.optimizer)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}-{step}-{val_accuracy:.4f}",
            monitor="val_accuracy",
            mode="max",
        ),
    ]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    if cfg.inference.predict:
        model_loaded = ClassificationSystem.load_from_checkpoint(cfg.inference.checkpoint_path, model=model.model)

        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        print('Test dataset prediction (extract) : ', predictions[:10])

        # Option 2: Predict 1 data point (Pytorch)
        test_data = datamodule.get_test_dataset()
        test_data_point = test_data[0][0]
        test_data_point = test_data_point.resize_(1, *test_data_point.shape)
        prediction = model_loaded.predict_point(cfg.inference.checkpoint_path,
                                                test_data_point,
                                                ['model.', ''])
        print('Point prediction : ', prediction)
    else:
        trainer.fit(model, datamodule=datamodule)
        trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
