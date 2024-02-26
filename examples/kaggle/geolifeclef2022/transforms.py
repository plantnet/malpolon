"""Collection of custom PyTorch friendly transform classes.

These transform classes can be called during training loops to perform
data augmentation.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
        Theo Larcher <theo.larcher@inria.fr>
"""

import numpy as np
import torch
from torchvision import transforms


class RGBDataTransform:
    def __call__(self, data):
        return transforms.functional.to_tensor(data)


class NIRDataTransform:
    def __call__(self, data):
        data = np.tile(data[:, :, None], 3)
        data = transforms.functional.to_tensor(data)
        return data


class RasterDataTransform:
    def __init__(self, mu, sigma, resize=None):
        self.mu = np.asarray(mu, dtype=np.float32)[:, None, None]
        self.sigma = np.asarray(sigma, dtype=np.float32)[:, None, None]
        self.resize = resize

    def __call__(self, data):
        data = torch.as_tensor(data, dtype=torch.float32)
        data = (data - self.mu) / self.sigma
        if self.resize:
            data = transforms.functional.resize(data, self.resize)
        return data


class TemperatureDataTransform(RasterDataTransform):
    def __init__(self):
        mu = [-12.0, 1.0, 1.0]
        sigma = [40.0, 22.0, 51.0]
        super().__init__(mu, sigma, resize=256)


class PrecipitationDataTransform(RasterDataTransform):
    def __init__(self):
        mu = [43.0, -1.0, 4.0]
        sigma = [3410.0, 177.0, 139.0]
        super().__init__(mu, sigma, resize=256)


class PedologicalDataTransform(RasterDataTransform):
    def __init__(self):
        mu = [-1.0, 31.0, -1.0]
        sigma = [526.0, 68.0, 88.0]
        super().__init__(mu, sigma, resize=256)


class DataAugmentation(transforms.Compose):
    def __init__(self, train):
        if train:
            super().__init__([
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        else:
            super().__init__([
                transforms.CenterCrop(size=224),
            ])


class Normalization(transforms.Normalize):
    def __init__(self, num_modalities):
        super().__init__(
            mean=[0.485, 0.456, 0.406] * num_modalities,
            std=[0.229, 0.224, 0.225] * num_modalities,
        )

class PreprocessRGBTemperatureData:
    def __call__(self, data):
        rgb_data, temp_data = data["rgb"], data["environmental_patches"]

        rgb_data = RGBDataTransform()(rgb_data)
        temp_data = TemperatureDataTransform()(temp_data)

        return torch.concat((rgb_data, temp_data))