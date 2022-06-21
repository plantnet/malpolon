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
        data = (data - self.mu) / self.sigma
        data = torch.as_tensor(data, dtype=torch.float32)
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
