"""This module provides a model to align features from multiscale geo-tagged data.

Author: Theo Larcher <theo.larcher@inria.fr>
        Alexis Joly <alexis.joly@inria.fr>

License: GPLv3
Python version: 3.12.9
"""
from typing import Any, Callable, Mapping, Optional, Union

import omegaconf
import torch
from omegaconf import OmegaConf
from torch import Tensor

from malpolon.models.standard_prediction_systems import ClassificationSystem

class MultiScaleGeoContrastiveSystem(ClassificationSystem):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def def _step(
        self, split: str, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass
