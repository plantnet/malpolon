from typing import Optional
from argparse import ArgumentParser, Namespace
from typing import Any, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics.functional as Fmetrics
from pytorch_lightning.utilities.argparse import add_argparse_args, from_argparse_args
from torchvision import models

from .utils import change_last_layer


def load_standard_classification_model(model_name, pretrained, n_classes):
    model = getattr(models, model_name)
    model = model(pretrained=pretrained)
    change_last_layer(model, n_classes=n_classes)

    return model


class StandardClassificationSystem(pl.LightningModule):
    r"""
    Args:
        model_name: name of model to use
        num_classes: number of classes
        pretrained: load weights pretrained on ImageNet
        lr: learning rate
        momentum: value of momentum
        nesterov: if True, uses Nesterov's momentum
        weight_decay: weight decay value (0 to disable)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        binary_classification: bool = False,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.binary_classification = binary_classification

        if self.binary_classification:
            self.loss = F.binary_cross_entropy_with_logits
        else:
            self.loss = F.cross_entropy

        self.metrics = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss)

        for metric_name, metric_func in self.metrics.items():
            val_score = metric_func(y_hat, y)
            self.log("val_{}".format(metric_name), val_score)

        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        test_loss = self.loss(y_hat, y)
        self.log("test_loss", test_loss)

        for metric_name, metric_func in self.metrics.items():
            test_score = metric_func(y_hat, y)
            self.log("test_{}".format(metric_name), test_score)

        return test_loss

    def configure_optimizers(self):
        print(self.optimizer)
        return self.optimizer

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


class StandardFinetuningClassificationSystem(StandardClassificationSystem):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        lr: float = 1e-2,
        weight_decay: Optional[float] = None,
        momentum: float = 0.9,
        nesterov: bool = True,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        model = load_standard_classification_model(
            self.model_name,
            self.pretrained,
            self.num_classes,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

        super().__init__(model, optimizer)

        self.metrics = {
            "accuracy": Fmetrics.accuracy,
        }
