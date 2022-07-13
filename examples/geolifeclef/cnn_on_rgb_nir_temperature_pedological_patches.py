import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torchmetrics.functional as Fmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

from malpolon.data.data_module import BaseDataModule
from malpolon.data.environmental_raster import PatchExtractor
from malpolon.data.datasets.geolifeclef import GeoLifeCLEF2022Dataset, MiniGeoLifeCLEF2022Dataset
from malpolon.models.multi_modal import HomogeneousMultiModalModel
from malpolon.models.standard_prediction_systems import GenericPredictionSystem
from malpolon.logging import Summary

from transforms import RGBDataTransform, NIRDataTransform, TemperatureDataTransform, PedologicalDataTransform


class PreprocessRGBNIRTemperaturePedologicalData:
    def __call__(self, data):
        rgb_data = data["rgb"]
        nir_data = data["near_ir"]
        raster_data = data["environmental_patches"]
        temp_data, pedo_data = raster_data[:3], raster_data[3:]

        rgb_data = RGBDataTransform()(rgb_data)
        nir_data = NIRDataTransform()(nir_data)
        temp_data = TemperatureDataTransform()(temp_data)
        pedo_data = PedologicalDataTransform()(pedo_data)

        return torch.concat((rgb_data, nir_data, temp_data, pedo_data))


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
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.minigeolifeclef = minigeolifeclef

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                PreprocessRGBNIRTemperaturePedologicalData(),
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406] * 4,
                    std=[0.229, 0.224, 0.225] * 4,
                ),
                transforms.Lambda(lambda x: {
                    "rgb": x[:3],
                    "near_ir": x[3:6],
                    "temperature": x[6:9],
                    "pedological": x[9:12],
                }),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                PreprocessRGBNIRTemperaturePedologicalData(),
                transforms.CenterCrop(size=224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406] * 4,
                    std=[0.229, 0.224, 0.225] * 4,
                ),
                transforms.Lambda(lambda x: {
                    "rgb": x[:3],
                    "near_ir": x[3:6],
                    "temperature": x[6:9],
                    "pedological": x[9:12],
                }),
            ]
        )

    def get_dataset(self, split, transform, **kwargs):
        if self.minigeolifeclef:
            dataset_cls = MiniGeoLifeCLEF2022Dataset
        else:
            dataset_cls = GeoLifeCLEF2022Dataset

        patch_extractor = PatchExtractor(Path(self.dataset_path) / "rasters", size=20)
        patch_extractor.append("bio_1", nan=-12.0)
        patch_extractor.append("bio_2", nan=1.0)
        patch_extractor.append("bio_7", nan=1.0)
        patch_extractor.append("orcdrc", nan=-1.0)
        patch_extractor.append("phihox", nan=31.0)
        patch_extractor.append("sltppt", nan=-1.0)

        dataset = dataset_cls(
            self.dataset_path,
            split,
            patch_data=["rgb", "near_ir"],
            use_rasters=True,
            patch_extractor=patch_extractor,
            transform=transform,
            **kwargs
        )
        return dataset


class ClassificationSystem(GenericPredictionSystem):
    def __init__(
        self,
        modalities_model: dict,
        feature_space_size: int,
        num_outputs: int,
        lr: float = 1e-2,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = True,
    ):
        model = HomogeneousMultiModalModel(
            ["rgb", "near_ir", "temperature", "pedological"],
            modalities_model,
            torch.nn.Linear(feature_space_size, num_outputs),
        )
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
        metrics = {
            "accuracy": Fmetrics.accuracy,
            "top_30_accuracy": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=30),
        }

        super().__init__(model, loss, optimizer, metrics)


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_on_rgb_temperature_patches_config")
def main(cfg: DictConfig) -> None:
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)

    datamodule = GeoLifeCLEF2022DataModule(**cfg.data)

    model = ClassificationSystem(cfg.model, **cfg.optimizer)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}-{step}-{val_top_30_accuracy:.4f}",
            monitor="val_top_30_accuracy",
            mode="max",
        ),
    ]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)
    trainer.fit(model, datamodule=datamodule)

    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
