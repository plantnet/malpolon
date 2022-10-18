from __future__ import annotations
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
import torchmetrics.functional as Fmetrics

from .utils import check_loss, check_model, check_optimizer

if TYPE_CHECKING:
    from typing import Any, Callable, Mapping, Optional, Union
    from torch import Tensor


class GenericPredictionSystem(pl.LightningModule):
    r"""
    Generic prediction system providing standard methods.

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
        model: Union[torch.nn.Module, Mapping],
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        metrics: Optional[dict[str, Callable]] = None,
    ):
        super().__init__()

        self.model = check_model(model)
        self.optimizer = check_optimizer(optimizer)
        self.loss = check_loss(loss)
        self.metrics = metrics or {}

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def _step(
        self, split: str, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
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

    def training_step(
        self, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        return self._step("train", batch, batch_idx)

    def validation_step(
        self, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        return self._step("val", batch, batch_idx)

    def test_step(
        self, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        return self._step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer


class FinetuningClassificationSystem(GenericPredictionSystem):
    r"""
    Basic finetuning classification system.

    Parameters
    ----------
        model: model to use
        lr: learning rate
        weight_decay: weight decay value
        momentum: value of momentum
        nesterov: if True, uses Nesterov's momentum
        metrics: dictionnary containing the metrics to compute
        binary: if True, uses binary classification loss instead of multi-class one
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        lr: float = 1e-2,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = True,
        metrics: Optional[dict[str, Callable]] = None,
        binary: bool = False,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        model = check_model(model)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        if binary:
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = torch.nn.CrossEntropyLoss()

        if metrics is None:
            metrics = {
                "accuracy": Fmetrics.accuracy,
            }

        super().__init__(model, loss, optimizer, metrics)
