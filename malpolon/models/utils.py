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
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from .model_builder import ModelBuilder

FMETRICS_CALLABLES = {'binary_accuracy': Fmetrics.classification.binary_accuracy,
                      'multiclass_accuracy': Fmetrics.classification.multiclass_accuracy,
                      'multilabel_accuracy': Fmetrics.classification.multilabel_accuracy, }

OPTIMIZERS_CALLABLES = {'sgd': optim.SGD,
                        'adam': optim.Adam,
                        'adamw': optim.AdamW,
                        'adadelta': optim.Adadelta,
                        'adagrad': optim.Adagrad,
                        'adamax': optim.Adamax,
                        'rmsprop': optim.RMSprop, }

SCHEDULER_CALLABLES = {'step_lr': lr_scheduler.StepLR,
                       'reduce_lr_on_plateau': lr_scheduler.ReduceLROnPlateau,
                       'cosine_annealing_lr': lr_scheduler.CosineAnnealingLR, }


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


def check_scheduler(scheduler: Union[LRScheduler, dict],
                    optimizer: optim.Optimizer) -> dict:
    """Ensure input scheduler is a pytorch scheduler.

    Input can either be an Omegaconf mapping (passed through a hydra config
    file) or a pytorch scheduler object. Several scheduler can be passed as
    input through an Omegaconf mapping which will be instantiated and returned
    as a list of scheduler.

    Parameters
    ----------
    scheduler : Union[LRScheduler, dict]
        input scheduler(s)
    optimizer : optim.Optimizer
        associated optimizer

    Returns
    -------
    dict
        dictionary of LR scheduler config
    """
    if scheduler is None:
        return None

    lr_sch_config = {'scheduler': None}

    if isinstance(scheduler, LRScheduler):
        lr_sch_config['scheduler'] = scheduler
        return lr_sch_config

    try:
        k, v = next(iter(scheduler.items()))  # Get 1st key & value of scheduler dict as there can only be 1 scheduler per optimizer
        if 'lr_scheduler_config' in v and v['lr_scheduler_config'] is not None:
            lr_sch_config = lr_sch_config | v['lr_scheduler_config']
        if 'callable' in v:
            v['callable'] = eval(v['callable'])
        else:
            v['callable'] = SCHEDULER_CALLABLES[k]
        scheduler = v['callable'](optimizer, **v['kwargs'])
    except ValueError as e:
        print('\n[ERROR]: Please make sure you have registered'
              ' a dict-like value to your "scheduler" key in your'
              ' config file.\n')
        print(e, '\n')
        raise e
    except KeyError as e:
        print('\n[ERROR]: Please make sure the name of your scheduler'
              ' registered in your config file match an entry'
              ' in constant SCHEDULER_CALLABLES; or that you have provided a'
              ' callable function if your scheduler\'s name is not pre-registered'
              ' in SCHEDULER_CALLABLES.\n')
        print(e, '\n')
        raise e

    lr_sch_config['scheduler'] = scheduler
    return lr_sch_config


def check_optimizer(optimizer: Union[Optimizer, OmegaConf],
                    model: nn.Module) -> Optimizer:
    """Ensure input optimizer is a pytorch scheduler.

    Input can either be an Omegaconf mapping (passed through a hydra config
    file) or a pytorch optimizer object. Several optimizers can be passed as
    input through an Omegaconf mapping which will be instantiated and returned
    as a list of optimizers.

    Parameters
    ----------
    optimizer : Union[Optimizer, OmegaConf]
        input scheduler(s)
    model : nn.Module, optional
        associated model

    Returns
    -------
    Optimizer
        list of instantiated optimizer(s) and corresponding scheduler(s)
        (for each optimizer with no scheduler, None is the corresponding value
        in the schedulers list).
    """
    optim_list = []
    scheduler_list = []

    if isinstance(optimizer, Optimizer):
        return [optimizer], [None]

    try:
        if optimizer is not None:
            optimizer = OmegaConf.to_container(optimizer, resolve=True)
            # Loop over all optimizers
            for k, v in optimizer.items():
                if 'callable' in v:
                    optimizer[k]['callable'] = eval(v['callable'])
                else:
                    optimizer[k]['callable'] = OPTIMIZERS_CALLABLES[k]
                optim_list.append(optimizer[k]['callable'](model.parameters(), **optimizer[k]['kwargs']))
                scheduler_list.append(check_scheduler(v.get('scheduler'), optim_list[-1]))
    except (TypeError, ValueError) as e:
        print('\n[ERROR]: Please make sure you have registered'
              ' a non-empty dict-like value to your "optimizer" key in your'
              ' config file. Your optimizer dict might be empty (NoneType).')
        print(e, '\n')
        raise e
    except KeyError as e:
        print('\n[ERROR]: Please make sure the name of your optimizer'
              ' registered in your config file match an entry'
              ' in constant OPTIMIZERS_CALLABLES; or that you have provided a'
              ' callable function if your optimizer\'s name is not pre-registered'
              ' in OPTIMIZERS_CALLABLES.\n'
              ' Please make sure your optimizer\'s and scheduler\'s kwargs keys'
              ' are valid.\n')
        print(e, '\n')
        raise e

    return optim_list, scheduler_list
