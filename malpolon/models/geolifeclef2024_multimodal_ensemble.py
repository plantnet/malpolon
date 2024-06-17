"""This module provides a Multimodal Ensemble model for GeoLifeCLEF2024 data.

Author: Lukas Picek <lukas.picek@inria.fr>
        Theo Larcher <theo.larcher@inria.fr>

License: GPLv3
Python version: 3.10.6
"""
from typing import Any, Callable, Mapping, Optional, Union

import torch
from torch import Tensor, nn
from torchvision import models

from malpolon.models import ClassificationSystem


class ClassificationSystemGLC24(ClassificationSystem):
    """Classification task class for GLC24_pre-extracted.

    Inherits ClassificationSystem.
    """
    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        lr: float = 1e-2,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = True,
        metrics: Optional[dict[str, Callable]] = None,
        task: str = 'classification_binary',
        loss_kwargs: Optional[dict] = {},
        hparams_preprocess: bool = True
    ):
        super().__init__(model, lr, weight_decay, momentum, nesterov, metrics, task, loss_kwargs, hparams_preprocess)
        print('a')

    def forward(self, x, y, z):  # noqa: D102 pylint: disable=C0116
        return self.model(x, y, z)

    def _step(
        self, split: str, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        if split == "train":
            log_kwargs = {"on_step": True, "on_epoch": True, "sync_dist": True}
        else:
            log_kwargs = {"on_step": True, "on_epoch": True, "sync_dist": True}

        x_landsat, x_bioclim, x_sentinel, y, species_id = batch
        y_hat = self(x_landsat, x_bioclim, x_sentinel)

        pos_weight = y * self.model.positive_weigh_factor
        self.loss.pos_weight = pos_weight  # Proper way would be to forward pos_weight to loss instantiation via loss_kwargs, but pos_weight must be a tensor, i.e. have access to y -> Not possible in Malpolon as datamodule and optimizer instantiations are separate

        loss = self.loss(y_hat, self._cast_type_to_loss(y))  # Shape mismatch for binary: need to 'y = y.unsqueeze(1)' (or use .reshape(2)) to cast from [2] to [2,1] and cast y to float with .float()
        self.log(f"loss/{split}", loss, **log_kwargs)

        for metric_name, metric_func in self.metrics.items():
            if isinstance(metric_func, dict):
                score = metric_func['callable'](y_hat, y, **metric_func['kwargs'])
            else:
                score = metric_func(y_hat, y)
            self.log(f"{metric_name}/{split}", score, **log_kwargs)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  # noqa: D102 pylint: disable=C0116
        x_landsat, x_bioclim, x_sentinel, y, species_id = batch
        return self(x_landsat, x_bioclim, x_sentinel)


class MultimodalEnsemble(nn.Module):
    """Multimodal ensemble model processing Sentinel-2A, Landsat & Bioclimatic data.

    Inherits torch nn.Module.
    """
    def __init__(self, num_classes=11255, positive_weigh_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.positive_weigh_factor = positive_weigh_factor

        self.landsat_model = models.resnet18(weights=None)
        self.landsat_norm = nn.LayerNorm([6, 4, 21])
        # Modify the first convolutional layer to accept 6 channels instead of 3
        self.landsat_model.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.landsat_model.maxpool = nn.Identity()

        self.bioclim_model = models.resnet18(weights=None)
        self.bioclim_norm = nn.LayerNorm([4, 19, 12])
        # Modify the first convolutional layer to accept 4 channels instead of 3
        self.bioclim_model.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bioclim_model.maxpool = nn.Identity()

        self.sentinel_model = models.swin_t(weights="IMAGENET1K_V1")
        # Modify the first layer to accept 4 channels instead of 3
        self.sentinel_model.features[0][0] = nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4))
        self.sentinel_model.head = nn.Identity()

        self.ln1 = nn.LayerNorm(1000)
        self.ln2 = nn.LayerNorm(1000)
        self.fc1 = nn.Linear(2768, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, y, z):  # noqa: D102 pylint: disable=C0116
        x = self.landsat_norm(x)
        x = self.landsat_model(x)
        x = self.ln1(x)

        y = self.bioclim_norm(y)
        y = self.bioclim_model(y)
        y = self.ln2(y)

        z = self.sentinel_model(z)

        xyz = torch.cat((x, y, z), dim=1)
        xyz = self.fc1(xyz)
        xyz = self.dropout(xyz)
        out = self.fc2(xyz)
        return out
