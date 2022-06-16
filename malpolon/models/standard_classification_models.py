from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics.functional as Fmetrics
from torchvision import models


def change_last_layer(
    model: torch.nn.Module,
    n_classes: int,
) -> torch.nn.Module:
    """
    Removes the last layer of a classification model and replaces it by a new dense layer with the provided number of classes.

    Parameters
    ----------
    model: torch.nn.Module
        Model to adapt.
    n_classes: integer
        Number of classes, used to update the last classification layer.

    Returns
    -------
    model: torch.nn.Module
        Newly created last dense classification layer.
    """
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
        return model.fc
    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear):
        num_ftrs = model.classifier
        model.classifier = torch.nn.Linear(num_ftrs, n_classes)
        return model.classifier
    elif (
        hasattr(model, "classifier")
        and isinstance(model.classifier, torch.nn.Sequential)
        and isinstance(model.classifier[-1], torch.nn.Linear)
    ):
        num_ftrs = model.classifier[-1]
        model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)
        return model.classifier[-1]
    else:
        raise ValueError(
            "not supported architecture {}".format(model.__class__.__name__)
        )


def load_standard_classification_model(
    model_name: str,
    n_classes: int,
    pretrained: bool = False,
) -> torch.nn.Module:
    """
    Loads a standard classification neural network architecture from `torchvision`, removes the last layer and replaces it by a new dense layer with the provided number of classes.

    Parameters
    ----------
    model_name: string
        Name of the model, should match one of the models in `torchvision.models`.
    n_classes: integer
        Number of classes, used to update the last classification layer.
    pretrained: boolean
        If True, load weights from pretrained model learned on ImageNet dataset.

    Returns
    -------
    model: torch.nn.Module
        Pytorch model.
    """
    model = getattr(models, model_name)
    model = model(pretrained=pretrained)
    change_last_layer(model, n_classes=n_classes)

    return model


class StandardClassificationSystem(pl.LightningModule):
    r"""
    Basic classification system providing standard methods.

    Parameters
    ----------
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
            self.num_classes,
            self.pretrained,
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
