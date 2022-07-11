from __future__ import annotations
from typing import Mapping, Union

from torch import nn, optim

from .model_builder import ModelBuilder


def check_loss(loss: nn.modules.loss._Loss) -> nn.modules.loss._Loss:
    if isinstance(loss, nn.modules.loss._Loss):
        return loss
    else:
        raise ValueError(
            "loss must be of type nn.modules.loss._Loss,"
            "given type {} instead".format(type(loss))
        )


def check_model(model: Union[nn.Module, Mapping]) -> nn.Module:
    if isinstance(model, nn.Module):
        return model
    elif isinstance(model, Mapping):
        return ModelBuilder.build_model(**model)
    else:
        raise ValueError(
            "model must be of type nn.Module or a mapping used to call ModelBuilder.build_model(),"
            "given type {} instead".format(type(model))
        )


def check_optimizer(optimizer: optim.Optimizer) -> optim.Optimizer:
    if isinstance(optimizer, optim.Optimizer):
        return optimizer
    else:
        raise ValueError(
            "optimizer must be of type optim.Optimizer,"
            "given type {} instead".format(type(optimizer))
        )
