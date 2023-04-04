import os

import hydra
import pytorch_lightning as pl
import torch
import torchmetrics.functional as Fmetrics
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from transforms import NIRDataTransform, RGBDataTransform

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.geolifeclef2022 import MicroGeoLifeCLEF2022Dataset
from malpolon.logging import Summary
from malpolon.models import FinetuningClassificationSystem


class RGBNIRDataPreprocessing:
    def __call__(self, data):
        rgb, nir = data["rgb"], data["near_ir"]
        rgb = RGBDataTransform()(rgb)
        nir = NIRDataTransform()(nir)[[0]]
        return torch.concat((rgb, nir))


class MicroGeoLifeCLEF2022DataModule(BaseDataModule):
    r"""
    Data module for MicroGeoLifeCLEF 2022.

    Parameters
    ----------
        dataset_path: Path to dataset
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation,
                              testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        dataset_path: str,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        download: bool = True,
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.download = download

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                RGBNIRDataPreprocessing(),
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485],
                    std=[0.229, 0.224, 0.225, 0.229],
                ),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                RGBNIRDataPreprocessing(),
                transforms.CenterCrop(size=224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485],
                    std=[0.229, 0.224, 0.225, 0.229],
                ),
            ]
        )

    def prepare_data(self):
        # download, split, etc...
        MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            subset="train",
            use_rasters=False,
            download=self.download,
        )

    def get_dataset(self, split, transform, **kwargs):
        dataset = MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            split,
            patch_data=["rgb", "near_ir"],
            use_rasters=False,
            transform=transform,
            **kwargs
        )
        return dataset


class NewConvolutionalLayerInitFuncStrategy:
    def __init__(self, strategy, rescaling=False):
        self.strategy = strategy
        self.rescaling = rescaling

    def __call__(self, old_layer, new_layer):
        with torch.no_grad():
            if self.strategy == "random_init":
                new_layer.weight[:, :3] = old_layer.weight
            elif self.strategy == "red_pretraining":
                new_layer.weight[:] = old_layer.weight[:, [0, 1, 2, 0]]

            if self.rescaling:
                new_layer.weight *= 3 / 4

            if hasattr(new_layer, "bias"):
                new_layer.bias = old_layer.bias


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


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_on_rgb_nir_patches_config")
def main(cfg: DictConfig) -> None:
    # cfg.data.dataset_path = '../../../' + cfg.data.dataset_path  # Uncomment if value contains only the name of the dataset folder. Only works with a 3-folder-deep hydra job path.
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)

    datamodule = MicroGeoLifeCLEF2022DataModule(**cfg.data)

    cfg_model = hydra.utils.instantiate(cfg.model)
    model = ClassificationSystem(cfg_model, **cfg.optimizer)

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
    trainer.fit(model, datamodule=datamodule)

    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
