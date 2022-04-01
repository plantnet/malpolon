import argparse

import attrs
import pytorch_lightning as pl
import torchmetrics.functional as Fmetrics
from torchvision import transforms

from ..data.data_module import BaseDataModule
from ..data.datasets.geolifeclef import MiniGeoLifeCLEF2022Dataset
from ..model import StandardFinetuningClassificationSystem


@attrs.define(slots=False)
class MiniGeoLifeCLEF22DataModule(BaseDataModule):
    r"""Data module for MiniGeoLifeCLEF22.

    Args:
        dataset_path: Path to dataset
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation, testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    dataset_path: str

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
        dataset = MiniGeoLifeCLEF2022Dataset(
            self.dataset_path,
            split,
            patch_data=["rgb"],
            use_rasters=False,
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


if __name__ == "__main__":
    """
    from pytorch_lightning.utilities.cli import LightningArgumentParser
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(Model, "model")
    parser.add_lightning_class_args(MiniGeoLifeCLEF22DataModule, "data")
    """
    parser = argparse.ArgumentParser(
        description="Example of training and testing on MiniGeoLifeCLEF2022."
    )
    parser = MiniGeoLifeCLEF22DataModule.add_argparse_args(parser, use_argument_group=False)
    parser = ClassificationSystem.add_argparse_args(parser, use_argument_group=False)
    parser = pl.Trainer.add_argparse_args(parser)
    """
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="number of training epochs",
    )
    """
    args = parser.parse_args()

    """
    logger = pl.loggers.CSVLogger("logs")
    logger.log_hyperparams(args)
    """
    logger = None

    datamodule = MiniGeoLifeCLEF22DataModule.from_argparse_args(args)

    model = ClassificationSystem.from_argparse_args(args)
    print(model.model)

    #trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs, logger=logger)
    trainer = pl.Trainer.from_argparse_args(args, gpus=1, logger=logger)
    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)
