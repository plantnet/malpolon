"""This module provides classes for advanced model building.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
        Theo Larcher <theo.larcher@inria.fr>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pytorch_lightning.strategies import SingleDeviceStrategy, StrategyRegistry
from pytorch_lightning.utilities import move_data_to_device
from torch import nn

from malpolon.models.utils import check_model

if TYPE_CHECKING:
    from typing import Any, Mapping, Optional, Union


class MultiModalModel(nn.Module):
    """Base multi-modal model.

    This class builds an aggregation of multiple models from the passed
    on config file values, one for each modality, splits the training
    routine per modality and then aggregates the features from each
    modality after each forward pass.
    """
    def __init__(
        self,
        modality_models: Union[nn.Module, Mapping],
        aggregator_model: Union[nn.Module, Mapping],
    ):
        """Class constructor.

        Parameters
        ----------
        modality_models : Union[nn.Module, Mapping]
            dictionary of modality names and their respective models to
            pass on to the model builder
        aggregator_model : Union[nn.Module, Mapping]
            Model strategy to aggregate the features from each modality.
            Can either be a PyTorch module directly (in this case, the
            module will be directly called), or a mapping in the same
            fashion as for buiding the modality models, in which case
            the model builder will be called again.
        """
        super().__init__()

        for modality_name, model in modality_models.items():
            modality_models[modality_name] = check_model(model)
        self.modality_models = nn.ModuleDict(modality_models)

        self.aggregator_model = check_model(aggregator_model)

    def forward(self, x: list[Any]) -> Any:
        features = []

        for modality_name, model in self.modality_models.items():
            out = model(x[modality_name])
            out = out.to(next(self.aggregator_model.parameters()).device)
            features.append(out)

        features = torch.concat(features, dim=-1)
        return self.aggregator_model(features)


class HomogeneousMultiModalModel(MultiModalModel):
    """Straightforward multi-modal model."""
    def __init__(
        self,
        modality_names: list,
        modalities_model: dict,
        aggregator_model: Union[nn.Module, Mapping],
    ):
        """Class constructor.

        Parameters
        ----------
        modality_names : list
            list of modalities names
        modalities_model : dict
            dictionary of modality names and their respective models to
            pass on to the model builder
        aggregator_model : Union[nn.Module, Mapping]
            Model strategy to aggregate the features from each modality.
            Can either be a PyTorch module directly (in this case, the
            module will be directly called), or a mapping in the same
            fashion as for buiding the modality models, in which case
            the model builder will be called again.
        """
        self.modality_names = modality_names
        self.modalities_model = modalities_model

        modalities_models = {
            modality_name: dict(modalities_model) for modality_name in modality_names
        }
        super().__init__(modalities_models, aggregator_model)


class ParallelMultiModalModelStrategy(SingleDeviceStrategy):
    """Model parallelism strategy for multi-modal models.

    WARNING: STILL UNDER DEVELOPMENT.
    """
    strategy_name = "parallel_multi_modal_model"

    def __init__(
        self,
        accelerator=None,
        parallel_devices=None,
        checkpoint_io=None,
        precision_plugin=None,
    ):
        super().__init__("cuda:0", accelerator, checkpoint_io, precision_plugin)

    def model_to_device(self) -> None:
        """TODO: Docstring."""
        model = self.model.model
        self.modalites_names = model.modalities_models.keys()
        num_modalities = len(self.modalities_names)

        self.num_gpus = torch.cuda.device_count()
        device_allocation = torch.arange(num_modalities) % self.num_gpus
        self.device_allocation = dict(zip(
            self.modalities_names,
            map(lambda i: f"cuda:{i}", device_allocation)
        ))
        self.root_device = "cuda:0"

        for modality_name in self.modalities_names:
            device = self.device_allocation[modality_name]
            model.modalities_models[modality_name] = model.modalities_models[modality_name].to(device)

        model.aggregator_model = model.aggregator_model.to(self.root_device)

    def batch_to_device(
        self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0
    ) -> Any:
        """TODO: Docstring."""
        x, target = batch

        for modality_name in self.modalities_models:
            device = self.device_allocation[modality_name]
            x[modality_name] = move_data_to_device(x[modality_name], device)

        target = move_data_to_device(target, self.root_device)
        return (x, target)


StrategyRegistry.register(
    ParallelMultiModalModelStrategy.strategy_name,
    ParallelMultiModalModelStrategy,
    description="Model parallelism strategy for multi-modal models",
)
