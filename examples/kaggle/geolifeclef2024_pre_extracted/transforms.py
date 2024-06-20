"""Collection of custom PyTorch friendly transform classes.

These transform classes can be called during training loops to perform
data augmentation.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
        Theo Larcher <theo.larcher@inria.fr>
"""

import numpy as np
import torch
from torchvision import transforms


class Standardize:
    def __call__(self, data):
        data = torch.as_tensor(data, dtype=torch.float32)
        data -= data.min()
        data = data / data.max()
        return data
