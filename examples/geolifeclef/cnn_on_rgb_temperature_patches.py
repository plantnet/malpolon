from __future__ import annotations
from typing import Optional

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torchmetrics.functional as Fmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torchvision import transforms

from malpolon.data.data_module import BaseDataModule
from malpolon.data.environmental_raster import PatchExtractor
from malpolon.data.datasets.geolifeclef import GeoLifeCLEF2022Dataset, MiniGeoLifeCLEF2022Dataset
from malpolon.models.standard_classification_models import load_standard_classification_model, StandardClassificationSystem
from malpolon.logging import Summary

from cnn_on_temperature_patches import ReplaceChannelsByBIOTEMPTransform


def change_last_classification_layer_to_identity(model: torch.nn.Module) -> None:
    """
    Removes the last layer of a classification model and replaces it by an nn.Identity layer.

    Parameters
    ----------
    model: torch.nn.Module
        Model to adapt.

    Returns
    -------
    num_features: int
        Size of the feature space.
    """
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        submodule = model
        layer_name = "fc"
    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear):
        submodule = model
        layer_name = "classifier"
    elif (
        hasattr(model, "classifier")
        and isinstance(model.classifier, torch.nn.Sequential)
        and isinstance(model.classifier[-1], torch.nn.Linear)
    ):
        submodule = model.classifier
        layer_name = str(len(model.classifier)-1)
    else:
        raise ValueError(
            "not supported architecture {}".format(model.__class__.__name__)
        )

    num_features = getattr(submodule, layer_name).in_features
    new_layer = nn.Identity()
    setattr(submodule, layer_name, new_layer)

    return num_features


class PreprocessRGBTemperature:
    def __call__(self, data):
        rgb_data, temp_data = data

        rgb_data = transforms.ToTensor()(rgb_data)
        temp_data = ReplaceChannelsByBIOTEMPTransform()(temp_data)

        return torch.concat((rgb_data, temp_data))


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
                PreprocessRGBTemperature(),
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
                ),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                PreprocessRGBTemperature(),
                transforms.CenterCrop(size=224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
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
            patch_data=["rgb"],
            use_rasters=True,
            patch_extractor=patch_extractor,
            transform=transform,
            **kwargs
        )
        return dataset


class MultiModalModel(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        backbone_model_name: str,
        backbone_model_pretrained: bool,
        num_classes: int,
        final_classifier: Optional[nn.Module] = None,
        multigpu: bool = False,
    ):
        super().__init__()
        self.multigpu = multigpu

        backbone_models = []
        for _ in range(num_modalities):
            model = load_standard_classification_model(
                backbone_model_name,
                pretrained=backbone_model_pretrained,
            )
            num_features = change_last_classification_layer_to_identity(model)
            backbone_models.append(model)
        self.backbone_models = backbone_models

        if final_classifier is None:
            self.final_classifier = nn.Linear(num_modalities * num_features, num_classes)
        else:
            self.final_classifier = final_classifier

        if self.multigpu:
            self.num_gpus = torch.cuda.device_count()
            self.device_allocation = torch.arange(num_modalities) % self.num_gpus

            for i in range(num_modalities):
                device = self.device_allocation[i]
                self.backbone_models[i] = self.backbone_models[i].to(f"cuda:{device}")

            self.final_classifier = self.final_classifier.to("cuda:0")
        else:
            # Can not use nn.ModuleList with multigpu=True and modules on different devices for some reason
            self.backbone_models = nn.ModuleList(backbone_models)

        self.input_channels = 3 * torch.arange(num_modalities+1)

    def forward(self, x):
        features = []

        for i, model in enumerate(self.backbone_models):
            x_i = x[:, self.input_channels[i]:self.input_channels[i+1]]

            if self.multigpu:
                device = self.device_allocation[i]
                x_i = x_i.to(f"cuda:{device}")
                out = model(x_i)
                out = out.to(f"cuda:0")
            else:
                out = model(x_i)

            features.append(out)

        features = torch.concat(features, dim=-1)
        return self.final_classifier(features)


class ClassificationSystem(StandardClassificationSystem):
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 100,
        pretrained: bool = True,
        lr: float = 1e-2,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = True,
        multigpu: bool = False,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        model = MultiModalModel(
            2,
            self.model_name,
            self.pretrained,
            self.num_classes,
            multigpu=multigpu,
        )
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        metrics = {
            "accuracy": Fmetrics.accuracy,
            "top_30_accuracy": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=30),
        }

        super().__init__(model, loss, optimizer, metrics)


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_on_rgb_patches_config")
def main(cfg: DictConfig) -> None:
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)

    datamodule = GeoLifeCLEF2022DataModule(**cfg.data)

    model = ClassificationSystem(**cfg.model)

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

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
