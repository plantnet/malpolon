from __future__ import annotations
from typing import Any, Optional

import torch
from torch import nn
from pytorch_lightning.strategies import SingleDeviceStrategy, StrategyRegistry
from pytorch_lightning.utilities.apply_func import move_data_to_device

from .model_builder import ModelBuilder


def change_last_classification_layer_to_identity(model: torch.nn.Module) -> int:
    """
    Removes the last layer of a classification model and replaces it by an nn.Identity layer.

    Parameters
    ----------
    model: torch.nn.Module
        Model to adapt.

    Returns
    -------
    num_features: int
        Size of the feature space.
    """
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
        layer_name = str(len(model.classifier) - 1)
    else:
        raise ValueError(
            "not supported architecture {}".format(model.__class__.__name__)
        )

    num_features = getattr(submodule, layer_name).in_features
    new_layer = nn.Identity()
    setattr(submodule, layer_name, new_layer)

    return num_features


class MultiModalModel(nn.Module):
    def __init__(
        self,
        num_modalities: int,
        backbone_model_name: str,
        backbone_model_pretrained: bool,
        num_classes: int,
        final_classifier: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_modalities = num_modalities

        backbone_models = []
        for _ in range(num_modalities):
            model = ModelBuilder.build_model(
                "torchvision",
                backbone_model_name,
                [backbone_model_pretrained],
            )
            num_features = change_last_classification_layer_to_identity(model)
            backbone_models.append(model)
        self.backbone_models = nn.ModuleList(backbone_models)

        if final_classifier is None:
            self.final_classifier = nn.Linear(
                num_modalities * num_features, num_classes
            )
        else:
            self.final_classifier = final_classifier

        self.input_channels = 3 * torch.arange(num_modalities + 1)

    def forward(self, x: list[Any]) -> Any:
        features = []

        for i, model in enumerate(self.backbone_models):
            out = model(x[i])
            out = out.to(next(self.final_classifier.parameters()).device)
            features.append(out)

        features = torch.concat(features, dim=-1)
        return self.final_classifier(features)


class ParallelMultiModalModelStrategy(SingleDeviceStrategy):
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
        model = self.model.model
        self.num_modalities = model.num_modalities
        self.input_channels = model.input_channels

        self.num_gpus = torch.cuda.device_count()
        self.device_allocation = torch.arange(self.num_modalities) % self.num_gpus
        self.device_allocation = list(
            map(lambda i: f"cuda:{i}", self.device_allocation)
        )

        for i in range(self.num_modalities):
            device = self.device_allocation[i]
            model.backbone_models[i] = model.backbone_models[i].to(device)

        model.final_classifier = model.final_classifier.to(self.root_device)

    def batch_to_device(
        self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0
    ) -> Any:
        x, target = batch

        for i in range(self.num_modalities):
            device = self.device_allocation[i]
            x[i] = move_data_to_device(x[i], device)

        target = move_data_to_device(target, self.root_device)
        return (x, target)


StrategyRegistry.register(
    ParallelMultiModalModelStrategy.strategy_name,
    ParallelMultiModalModelStrategy,
    description="Model parallelism strategy for multi-modal models",
)
