from typing import Callable, Optional

import pytorch_lightning as pl
import torch
import torchmetrics.functional as Fmetrics
from torchvision import models


def change_last_layer(
    model: torch.nn.Module,
    num_classes: int,
) -> torch.nn.Module:
    """
    Removes the last layer of a classification model and replaces it by a new dense layer with the provided number of classes.

    Parameters
    ----------
    model: torch.nn.Module
        Model to adapt.
    num_classes: integer
        Number of classes, used to update the last classification layer.

    Returns
    -------
    model: torch.nn.Module
        Newly created last dense classification layer.
    """
    if num_classes == 2:
        num_classes = 1

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
    new_layer = torch.nn.Linear(num_features, num_classes)
    setattr(submodule, layer_name, new_layer)

    return model


def load_standard_classification_model(
    model_name: str,
    pretrained: bool = False,
    num_classes: Optional[int] = None,
) -> torch.nn.Module:
    """
    Loads a standard classification neural network architecture from `torchvision`, removes the last layer and replaces it by a new dense layer with the provided number of classes.

    Parameters
    ----------
    model_name: string
        Name of the model, should match one of the models in `torchvision.models`.
    num_classes: integer
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

    if num_classes is not None:
        change_last_layer(model, num_classes=num_classes)

    return model


class StandardClassificationSystem(pl.LightningModule):
    r"""
    Basic classification system providing standard methods.

    Parameters
    ----------
    model: torch.nn.Module
        Model to use.
    loss: torch.nn.modules.loss._Loss
        Loss used to fit the model.
    optimizer: torch.optim.Optimizer
        Optimization algorithm used to train the model.
    metrics: dict
        Dictionary containing the metrics to monitor during the training and to compute at test time.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        metrics: Optional[dict[str, Callable]] = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or {}

    def forward(self, x):
        return self.model(x)

    def _step(self, split, batch, batch_idx):
        if split == "train":
            log_kwargs = {"on_step": False, "on_epoch": True}
        else:
            log_kwargs = {}

        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        self.log(f"{split}_loss", loss, **log_kwargs)

        for metric_name, metric_func in self.metrics.items():
            score = metric_func(y_hat, y)
            self.log(f"{split}_{metric_name}", score, **log_kwargs)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx)

    def configure_optimizers(self):
        return self.optimizer


class StandardFinetuningClassificationSystem(StandardClassificationSystem):
    r"""
    Basic finetuning classification system.

    Parameters
    ----------
        model_name: name of model to use
        pretrained: load weights pretrained on ImageNet
        num_classes: number of classes
        lr: learning rate
        weight_decay: weight decay value
        momentum: value of momentum
        nesterov: if True, uses Nesterov's momentum
        metrics: dictionnary containing the metrics to compute
    """
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        lr: float = 1e-2,
        weight_decay: Optional[float] = None,
        momentum: float = 0.9,
        nesterov: bool = True,
        metrics: Optional[dict[str, Callable]] = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
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
        if num_classes <= 2:
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = torch.nn.CrossEntropyLoss()

        if metrics is None:
            metrics = {
                "accuracy": Fmetrics.accuracy,
            }

        super().__init__(model, loss, optimizer, metrics)
