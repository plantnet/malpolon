from argparse import Namespace
from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics.functional as Fmetrics
from torchvision import transforms

from malpolon.data.data_module import BaseDataModule
from malpolon.data.environmental_raster import PatchExtractor
from malpolon.data.datasets.geolifeclef import GeoLifeCLEF2022Dataset, MiniGeoLifeCLEF2022Dataset
from malpolon.models.standard_classification_models import StandardFinetuningClassificationSystem


class ReplaceChannelsByBIOTEMPTransform:
    def __call__(self, data):
        mu = np.asarray([-12.0, 1.0, 1.0], dtype=np.float32)[:, None, None]
        sigma = np.asarray([40.0, 22.0, 51.0], dtype=np.float32)[:, None, None]
        data = (data - mu) / sigma
        data = torch.as_tensor(data, dtype=torch.float32)
        data = transforms.functional.resize(data, 256)
        return data


class GeoLifeCLEF2022DataModule(BaseDataModule):
    r"""
    Data module for GeoLifeCLEF 2022.

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
                ReplaceChannelsByBIOTEMPTransform(),
                transforms.RandomRotation(degrees=45, fill=255),
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
                ReplaceChannelsByBIOTEMPTransform(),
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

        patch_extractor = PatchExtractor(Path(self.dataset_path) / "rasters", size=20)
        patch_extractor.append("bio_1", nan=-12.0)
        patch_extractor.append("bio_2", nan=1.0)
        patch_extractor.append("bio_7", nan=1.0)

        dataset = dataset_cls(
            self.dataset_path,
            split,
            patch_data=[],
            use_rasters=True,
            patch_extractor=patch_extractor,
            transform=transform,
            **kwargs
        )
        return dataset


class ClassificationSystem(StandardFinetuningClassificationSystem):
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 100,
        pretrained: bool = True,
        lr: float = 1e-2,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = True,
    ):
        super().__init__(model_name, num_classes, pretrained, lr, weight_decay, momentum, nesterov)

        self.metrics = {
            "accuracy": Fmetrics.accuracy,
            "top_k_accuracy": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=30),
        }


@hydra.main(config_path=".", config_name="cnn_on_rgb_patches_config")
def main(cls: DictConfig):
    logger = pl.loggers.CSVLogger(".", name="logs")

    datamodule = GeoLifeCLEF2022DataModule.from_argparse_args(Namespace(**cls.data))

    model = ClassificationSystem.from_argparse_args(Namespace(**cls.model))
    print(model.model)

    trainer = pl.Trainer(gpus=1, logger=logger, **cls.params)
    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
