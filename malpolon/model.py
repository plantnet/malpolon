from argparse import ArgumentParser, Namespace
from typing import Any, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities.argparse import add_argparse_args, from_argparse_args
from torchvision import models

from .utils import change_last_layer


def load_standard_classification_model(model_name, pretrained, n_classes):
    model = getattr(models, model_name)
    model = model(pretrained=pretrained)
    change_last_layer(model, n_classes=n_classes)

    return model


class StandardClassificationSystem(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        lr: float = 1e-2,
        weight_decay: float = 0,
        num_classes: int = 17037,
    ):
        r"""Data module for MNIST.

        Args:
            model_name: name of model to use
            pretrained: load weights pretrained on ImageNet
            lr: learning rate
            weight_decay: weight decay value (0 to disable)
            num_classes: number of classes
        """
        super().__init__()

        model = load_standard_classification_model(model_name, pretrained, num_classes)

        self.lr = lr
        self.weight_decay = weight_decay
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

        val_accuracy = torchmetrics.functional.accuracy(y_hat, y)
        self.log("val_accuracy", val_accuracy)

        val_top_k_accuracy = torchmetrics.functional.accuracy(y_hat, y, top_k=30)
        self.log("val_top_k_accuracy", val_top_k_accuracy)

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)

        test_accuracy = torchmetrics.functional.accuracy(y_hat, y)
        self.log("test_accuracy", test_accuracy)

        test_top_k_accuracy = torchmetrics.functional.accuracy(y_hat, y, top_k=30)
        self.log("test_top_k_accuracy", test_top_k_accuracy)

        return test_loss

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.SGD(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        return optimizer

    @classmethod
    def from_argparse_args(
        cls: Any,
        args: Union[Namespace, ArgumentParser],
        **kwargs,
    ) -> Any:
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs,
    ) -> ArgumentParser:
        return add_argparse_args(cls, parent_parser, **kwargs)
