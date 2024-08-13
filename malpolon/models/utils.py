"""This file compiles useful functions related to models.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@gmail.com>
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Mapping, Union

import torchmetrics.functional as Fmetrics
from omegaconf import OmegaConf
from torch import nn, optim

from .model_builder import ModelBuilder

FMETRICS_CALLABLES = {'binary_accuracy': Fmetrics.classification.binary_accuracy,
                      'multiclass_accuracy': Fmetrics.classification.multiclass_accuracy,
                      'multilabel_accuracy': Fmetrics.classification.multilabel_accuracy, }

class CrashHandler():
    """Saves the model in case of unexpected crash or user interruption."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.ckpt_dir_path = Path(trainer.logger.log_dir) / "crash_latest_checkpoint.ckpt"
        signal.signal(signal.SIGINT, self.signal_handler)

    def save_checkpoint(self):
        """Save the latest checkpoint."""
        print("Saving lastest checkpoint...")
        self.trainer.save_checkpoint(self.ckpt_dir_path)

    def signal_handler(self, sig, frame):
        """Attempt to save the latest checkpoint in case of crash."""
        print(f"Received signal {sig}. Performing cleanup...")
        self.save_checkpoint()
        sys.exit(0)


def check_metric(metrics: OmegaConf) -> OmegaConf:
    """Ensure user's model metrics are valid.

    Users can either choose from a list of predefined metrics or
    define their own custom metrics. This function binds the user's
    metrics with their corresponding callable function from
    torchmetrics, by reading the values in `metrics` which is a
    dict-like structure returned by hydra when reading the config
    file.
    If the user chose predefined metrics, the function will
    automatically bind the corresponding callable function from
    torchmetrics.
    If the user chose custom metrics, the function checks that they
    also provided the callable function to compute the metric.

    Parameters
    ----------
    metrics: OmegaConf
        user's input metrics, read from the config file via hydra, in
        a dict-like structure

    Returns
    -------
    OmegaConf
        user's metrics with their corresponding callable function
    """
    try:
        metrics = OmegaConf.to_container(metrics, resolve=True)
        for k, v in metrics.items():
            if 'callable' in v:
                metrics[k]['callable'] = eval(v['callable'])
            else:
                metrics[k]['callable'] = FMETRICS_CALLABLES[k]
    except ValueError as e:
        print('\n[WARNING]: Please make sure you have registered'
              ' a dict-like value to your "metrics" key in your'
              ' config file. Defaulting metrics to None.\n')
        print(e, '\n')
        metrics = None
    except KeyError as e:
        print('\n[WARNING]: Please make sure the name of your metrics'
              ' registered in your config file match an entry'
              ' in constant FMETRICS_CALLABLES.'
              ' Defaulting metrics to None.\n')
        print(e, '\n')
        metrics = None
    return metrics


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
