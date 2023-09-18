"""Collection of custom PyTorch friendly transform classes.

These transform classes can be called during training loops to perform
data augmentation.

Author: Titouan Lorieul <titouan.lorieul@inria.fr>
        Theo Larcher <theo.larcher@inria.fr>
"""
import numpy as np

import torch
from torchvision import transforms


class RGBDataTransform:
    """PyTorch friendly callable transform class for RGB data."""
    def __call__(self, data: np.ndarray) -> torch.tensor:
        """Call method.

        Converts data to tensor.

        Parameters
        ----------
        data : np.ndarray
            input data.

        Returns
        -------
        torch.tensor
            transformed data (tensor).
        """
        return transforms.functional.to_tensor(data)


class NIRDataTransform:
    """PyTorch friendly callable transform class for NIR data."""
    def __call__(self, data: np.ndarray) -> torch.tensor:
        """Call method.

        Stacks up the Near Infra-Red layer 3 times to make it compatible with
        models learned over 3 channels (usually RGB).

        Parameters
        ----------
        data : np.ndarray
            input data.

        Returns
        -------
        torch.tensor
            transformed data (tensor).
        """
        data = np.tile(data[:, :, None], 3)
        data = transforms.functional.to_tensor(data)
        return data


class RasterDataTransform:
    """PyTorch friendly callable transform class for Raster data."""
    def __init__(
        self,
        mu: float,
        sigma: float,
        resize: bool = None
    ) -> None:
        """Init method.

        Parameters
        ----------
        mu : float
            dataset mean.
        sigma : float
            dataset standard deviation.
        resize : bool, optional
            resize the input image to the given size, by default None.
        """
        self.mu = np.asarray(mu, dtype=np.float32)[:, None, None]
        self.sigma = np.asarray(sigma, dtype=np.float32)[:, None, None]
        self.resize = resize

    def __call__(self, data: np.ndarray) -> torch.tensor:
        """Call method.

        Standardizes the data using the dataset's moments 0 (mean) and 1 (std).

        Parameters
        ----------
        data : np.ndarray
            input data.

        Returns
        -------
        torch.tensor
            transformed data (tensor).
        """
        data = torch.as_tensor(data, dtype=torch.float32)
        data = (data - self.mu) / self.sigma
        if self.resize:
            data = transforms.functional.resize(data, self.resize)
        return data


class TemperatureDataTransform(RasterDataTransform):
    """PyTorch friendly callable transform class for Temperature data."""
    def __init__(self) -> None:
        """Init method."""
        mu = [-12.0, 1.0, 1.0]
        sigma = [40.0, 22.0, 51.0]
        super().__init__(mu, sigma, resize=256)


class PrecipitationDataTransform(RasterDataTransform):
    """PyTorch friendly callable transform class for Precipitation data."""
    def __init__(self) -> None:
        """Init method."""
        mu = [43.0, -1.0, 4.0]
        sigma = [3410.0, 177.0, 139.0]
        super().__init__(mu, sigma, resize=256)


class PedologicalDataTransform(RasterDataTransform):
    """PyTorch friendly callable transform class for Pedological data."""
    def __init__(self) -> None:
        """Init method."""
        mu = [-1.0, 31.0, -1.0]
        sigma = [526.0, 68.0, 88.0]
        super().__init__(mu, sigma, resize=256)
