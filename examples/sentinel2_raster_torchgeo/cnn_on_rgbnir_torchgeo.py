from __future__ import annotations

import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torchmetrics.functional as Fmetrics
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from transforms import RGBDataTransform

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.torchgeo_datasets import (RasterSentinel2,
                                                      Sentinel2GeoSampler)
from malpolon.logging import Summary
from malpolon.models import FinetuningClassificationSystem
from torchgeo.samplers import Units
from torch.utils.data import DataLoader


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
        labels_name: str = 'labels.csv',
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        size: int = 200,
        units: Units = Units.CRS,
        crs: int = 4326,
        task: str = 'classification_multiclass',  # ['classification_binary', 'classification_multiclass', 'classification_multilabel']
        binary_positive_classes: list = []
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.labels_name = labels_name
        self.size = size
        self.units = units
        self.crs = crs
        self.sampler = Sentinel2GeoSampler
        self.task = task
        self.binary_positive_classes = binary_positive_classes

    def get_dataset(self, split, transform, **kwargs):
        dataset = RasterSentinel2(
            self.dataset_path,
            labels_name=self.labels_name,
            split=split,
            task=self.task,
            binary_positive_classes=self.binary_positive_classes,
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
        # return transforms.Compose(
        #     [
        #         lambda data: RGBDataTransform()(data["rgb"]),self.coordinates
        #         transforms.RandomRotation(degrees=45, fill=1),
        #         transforms.RandomCrop(size=224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )
        pass

    @property
    def test_transform(self):
        # return transforms.Compose(
        #     [
        #         lambda data: RGBDataTransform()(data["rgb"]),
        #         transforms.CenterCrop(size=224),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )
        pass


class ClassificationSystem(FinetuningClassificationSystem):
    def __init__(
        self,
        model: dict,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        metrics: dict,
        binary: bool,
    ):
        metrics = omegaconf.OmegaConf.to_container(metrics)
        for k, v in metrics.items():
            metrics[k]['callable'] = eval(v['callable'])

        super().__init__(
            model,
            lr,
            weight_decay,
            momentum,
            nesterov,
            metrics,
            binary,
        )


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_on_rgbnir_torchgeo_config")
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
            filename="checkpoint-{epoch:02d}-{step}-{" + f"val_{next(iter(model.metrics.keys()))}" + ":.4f}",
            monitor=f"val_{next(iter(model.metrics.keys()))}",
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
        query_point = {'lon': test_data.coordinates[0][0], 'lat': test_data.coordinates[0][1],
                       'crs': datamodule.crs,
                       'size': datamodule.size,
                       'units': datamodule.units}
        test_data_point = test_data[query_point][0]
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
