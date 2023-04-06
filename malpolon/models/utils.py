"""This file compiles useful functions related to models."""

from __future__ import annotations
from typing import Mapping, Union

from torch import nn, optim

from .model_builder import ModelBuilder


def check_loss(loss: nn.modules.loss._Loss) -> nn.modules.loss._Loss:
    """Ensure input loss is a pytorch loss.

    Args:
        loss (nn.modules.loss._Loss): input loss.

    Raises:
        ValueError: if input loss isn't a pytorch loss object.

    Returns:
        nn.modules.loss._Loss: the pytorch input loss itself.
    """
    if isinstance(loss, nn.modules.loss._Loss):  # pylint: disable=protected-access  # noqa
        return loss
    raise ValueError(f"Loss must be of type nn.modules.loss. "
                     f"Loss given type {type(loss)} instead")


def check_model(model: Union[nn.Module, Mapping]) -> nn.Module:
    """Ensure input model is a pytorch model.

    Args:
        model (Union[nn.Module, Mapping]): input model.

    Raises:
        ValueError:  if input model isn't a pytorch model object.

    Returns:
        nn.Module: the pytorch input model itself.
    """
    if isinstance(model, nn.Module):
        return model
    if isinstance(model, Mapping):
        return ModelBuilder.build_model(**model)
    raise ValueError(
        "Model must be of type nn.Module or a mapping used to call "
        f"ModelBuilder.build_model(), given type {type(model)} instead"
    )


def check_optimizer(optimizer: optim.Optimizer) -> optim.Optimizer:
    """Ensure input optimizer is a pytorch optimizer.

    Args:
        optimizer (optim.Optimizer): input optimizer.

    Raises:
        ValueError: if input optimizer isn't a pytorch optimizer object.

    Returns:
        optim.Optimizer: the pytorch input optimizer itself.
    """
    if isinstance(optimizer, optim.Optimizer):
        return optimizer
    raise ValueError(
        "Optimizer must be of type optim.Optimizer,"
        f"given type {type(optimizer)} instead"
    )
