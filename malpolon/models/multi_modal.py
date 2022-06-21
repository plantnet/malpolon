from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from .standard_classification_models import load_standard_classification_model


def change_last_classification_layer_to_identity(model: torch.nn.Module) -> None:
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
        multigpu: bool = False,
    ):
        super().__init__()
        self.multigpu = multigpu

        backbone_models = []
        for _ in range(num_modalities):
            model = load_standard_classification_model(
                backbone_model_name,
                pretrained=backbone_model_pretrained,
            )
            num_features = change_last_classification_layer_to_identity(model)
            backbone_models.append(model)
        self.backbone_models = backbone_models

        if final_classifier is None:
            self.final_classifier = nn.Linear(num_modalities * num_features, num_classes)
        else:
            self.final_classifier = final_classifier

        if self.multigpu:
            self.num_gpus = torch.cuda.device_count()
            self.device_allocation = torch.arange(num_modalities) % self.num_gpus

            for i in range(num_modalities):
                device = self.device_allocation[i]
                self.backbone_models[i] = self.backbone_models[i].to(f"cuda:{device}")

            self.final_classifier = self.final_classifier.to("cuda:0")
        else:
            # Can not use nn.ModuleList with multigpu=True and modules on different devices for some reason
            self.backbone_models = nn.ModuleList(backbone_models)

        self.input_channels = 3 * torch.arange(num_modalities + 1)

    def forward(self, x):
        features = []

        for i, model in enumerate(self.backbone_models):
            x_i = x[:, self.input_channels[i]:self.input_channels[i+1]]

            if self.multigpu:
                device = self.device_allocation[i]
                x_i = x_i.to(f"cuda:{device}")
                out = model(x_i)
                out = out.to(f"cuda:0")
            else:
                out = model(x_i)

            features.append(out)

        features = torch.concat(features, dim=-1)
        return self.final_classifier(features)
