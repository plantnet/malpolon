import argparse
from typing import Optional

import pytorch_lightning as pl
from torchvision import transforms

from ...data_module import BaseDataModule
from ...datasets.geolifeclef import GeoLifeCLEF2022Dataset
from ...model import StandardClassificationSystem


class RGBPatchesGeoLifeCLEF22DataModule(BaseDataModule):
    def __init__(
        self,
        dataset_path: str,
        train_batch_size: Optional[int] = 64,
        inference_batch_size: Optional[int] = 256,
        num_workers: Optional[int] = 8,
    ):
        r"""Data module for GeoLifeCLEF22.

        Args:
            dataset_path: Path to dataset
            train_batch_size: Size of batch for training
            inference_batch_size: Size of batch for inference (validation, testing, prediction)
            num_workers: Number of workers to use for data loading
        """
        super().__init__(
            train_batch_size,
            inference_batch_size,
            num_workers,
        )
        self.dataset_path = dataset_path

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=45, fill=255),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_dataset(self, split, transform, **kwargs):
        dataset = GeoLifeCLEF2022Dataset(
            self.dataset_path,
            split,
            patch_data=["rgb"],
            use_rasters=False,
            transform=transform,
            **kwargs
        )
        return dataset


class ReplaceBlueChannelByNearIRTransform:
    def __call__(self, data):
        rgb, near_ir = data
        rgb[:, :, 2] = near_ir
        return rgb


class RGNIRPatchesGeoLifeCLEF22DataModule(RGBPatchesGeoLifeCLEF22DataModule):
    @property
    def train_transform(self):
        return transforms.Compose(
            [
                ReplaceBlueChannelByNearIRTransform(),
                super().train_transform
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                ReplaceBlueChannelByNearIRTransform(),
                super().test_transform
            ]
        )

    def get_dataset(self, split, transform, **kwargs):
        dataset = GeoLifeCLEF2022Dataset(
            self.dataset_path,
            split,
            patch_data=["rgb", "near_ir"],
            use_rasters=False,
            transform=transform,
            **kwargs
        )
        return dataset


class ClassificationSystem(StandardClassificationSystem):
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        lr: float = 1e-2,
        weight_decay: float = 0,
        num_classes: int = 17037,
    ):
        metrics = {
            "accuracy": Fmetrics.accuracy,
            "top_k_accuracy": lambda y_hat, y: Fmetrics.accuracy(y_hat, y, top_k=30),
        }
        super().__init__(
            model_name,
            pretrained,
            num_classes,
            lr=lr,
            weight_decay=weight_decay,
            metrics=metrics,
        )


if __name__ == "__main__":
    """
    from pytorch_lightning.utilities.cli import LightningArgumentParser
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(Model, "model")
    parser.add_lightning_class_args(GeoLifeCLEF22DataModule, "data")
    """
    parser = argparse.ArgumentParser(
        description="Example of training and testing on GeoLifeCLEF2022."
    )
    parser = RGBPatchesGeoLifeCLEF22DataModule.add_argparse_args(parser, use_argument_group=False)
    parser = ClassificationSystem.add_argparse_args(parser, use_argument_group=False)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "--data",
        choices=["rgb", "rg-nir"],
    )
    """
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="number of training epochs",
    )
    """
    args = parser.parse_args()

    logger = pl.loggers.CSVLogger("logs")
    logger.log_hyperparams(args)

    if args.data == "rgb":
        datamodule = RGBPatchesGeoLifeCLEF22DataModule.from_argparse_args(args)
    else:
        datamodule = RGNIRPatchesGeoLifeCLEF22DataModule.from_argparse_args(args)

    model = ClassificationSystem.from_argparse_args(args)
    print(model.model)

    #trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs, logger=logger)
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, logger=logger)
    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)
