""" GeoCLIP
A reimplementation of: `GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective
Worldwide Geo-localization`
    - https://arxiv.org/pdf/2309.16020

Code and weights originates from https://github.com/VicenteVivan/geo-clip, under MIT licence.

Modifications and additions for malpolon put together by Lukas Picek. Copyright 2024
"""
# --------------------------------------------------------
# GeoCLIP
# Copyright (c) 2023
# Licensed under The MIT License
# Developed by
# Vicente Vivanco Cepeda
# --------------------------------------------------------

import os
import warnings
from typing import List, Optional

import requests
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


def equal_earth_projection(location: Tensor) -> Tensor:
    """
    Apply the Equal Earth projection to a set of geographical coordinates.

    The Equal Earth projection is a pseudocylindrical projection for creating maps
    of the world that is neither equal-area nor conformal.

    Parameters
    ----------
    location : torch.Tensor
        A 2D tensor with shape (N, 2) where each row represents
        a pair of geographical coordinates in degrees [latitude, longitude].

    Returns
    -------
    torch.Tensor
        A 2D tensor with shape (N, 2) where each row represents
        the projected coordinates [x, y].

    Constants
    ---------
    A1 : float
        Coefficient for the Equal Earth projection.
    A2 : float
        Coefficient for the Equal Earth projection.
    A3 : float
        Coefficient for the Equal Earth projection.
    A4 : float
        Coefficient for the Equal Earth projection.
    SF : float
        Scaling factor for the projection.
    """
    # Constants
    A1 = 1.340264
    A2 = -0.081106
    A3 = 0.000893
    A4 = 0.003796
    SF = 66.50336

    latitude = location[:, 0]
    longitude = location[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)

    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)

    denominator = 3 * (
        9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1
    )

    x = (
        2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)
    ) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta

    return (torch.stack((x, y), dim=1) * SF) / 180


def sample_b(sigma: float, size: tuple) -> Tensor:
    """
    Generate a matrix of specified size sampled from a normal distribution with mean 0 and
    variance sigma^2.

    Parameters
    ----------
    sigma : float
        Standard deviation of the normal distribution.
    size : tuple
        Size of the matrix to be sampled.

    Returns
    -------
    Tensor
        A tensor of specified size sampled from the normal dist. with mean 0 and variance sigma^2.

    See Also
    --------
    rff.layers.GaussianEncoding : For more details on usage.
    """
    return torch.randn(size) * sigma


@torch.jit.script
def gaussian_encoding(v: Tensor, b: Tensor) -> Tensor:
    r"""
    Computes the Gaussian encoding of the input tensor.

    The encoding is given by the function:
    :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}}, \sin{2 \pi \mathbf{B} \mathbf{v}})`

    Parameters
    ----------
    v : Tensor
        Input tensor of shape (N, *, input_size).
    b : Tensor
        Projection matrix of shape (encoded_layer_size, input_size).

    Returns
    -------
    Tensor
        Mapped tensor of shape (N, *, 2 * encoded_layer_size).
    """
    vp = 2 * torch.pi * torch.matmul(v, b.T)
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class GaussianEncoding(nn.Module):
    """
    Layer for mapping coordinates using random Fourier features.

    This layer applies a Gaussian encoding to the input tensor using a random projection matrix.
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[int] = None,
        encoded_size: Optional[int] = None,
        b: Optional[Tensor] = None,
    ):
        """
        Initialize the GaussianEncoding layer.

        Parameters
        ----------
        sigma : Optional[float]
            Standard deviation of the normal distribution used to sample the projection matrix.
        input_size : Optional[int]
            The number of input dimensions.
        encoded_size : Optional[int]
            The number of dimensions the `b` matrix maps to.
        b : Optional[Tensor]
            Optionally specify a pre-sampled projection matrix.

        Raises
        ------
        ValueError
            If `b` is provided and one of `sigma`, `input_size`, or `encoded_size` is also provided.
            If `b` is not provided and one of `sigma`, `input_size`, or `encoded_size` is not provided.
        """
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required if "b" is not provided.'
                )
            b = sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it exclusively.')

        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        r"""
        Compute the Gaussian encoding of the input tensor.

        The encoding is given by the function:
        :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}}, \sin{2 \pi \mathbf{B} \mathbf{v}})`

        Parameters
        ----------
        v : Tensor
            Input tensor of shape (N, *, input_size).

        Returns
        -------
        Tensor
            Tensor mapped using random Fourier features of shape (N, *, 2 * encoded_size).
        """
        return gaussian_encoding(v, self.b)


class LocationEncoderCapsule(nn.Module):
    """
    Location Encoder Capsule using Gaussian Encoding and a series of linear layers.
    """

    def __init__(self, sigma: float):
        """
        Initialize the LocationEncoderCapsule.

        Parameters
        ----------
        sigma : float
            Standard deviation for the Gaussian encoding.
        """
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the LocationEncoderCapsule.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, 2).

        Returns
        -------
        Tensor
            Encoded tensor of shape (N, 512).
        """
        x = self.capsule(x)
        x = self.head(x)
        return x


class LocationEncoder(nn.Module):
    """
    Location Encoder using multiple LocationEncoderCapsules.
    """

    def __init__(
        self,
        sigma: List[float] = [2**0, 2**4, 2**8],
        from_pretrained: bool = True,
    ):
        """
        Initialize the LocationEncoder.

        Parameters
        ----------
        sigma : list of float, optional
            List of standard deviations for the Gaussian encoding capsules,
            by default [2**0, 2**4, 2**8].
        from_pretrained : bool, optional
            Whether to load pre-trained weights, by default True.
        """
        super(LocationEncoder, self).__init__()
        self.sigma = sigma

        for i, s in enumerate(self.sigma):
            self.add_module("LocEnc" + str(i), LocationEncoderCapsule(sigma=s))

        if from_pretrained:
            self._load_weights()

    def _load_weights(self, weight_folder: str = "./weights"):
        """
        Load pre-trained weights for the LocationEncoder.

        Parameters
        ----------
        weight_folder : str, optional
            Folder to load the weights from, by default "./weights".
        """
        weight_path = os.path.join(weight_folder, "location_encoder_weights.pth")

        if not os.path.exists(weight_folder):
            os.makedirs(weight_folder)

        if not os.path.exists(weight_path):
            url = "https://github.com/VicenteVivan/geo-clip/raw/main/geoclip/model/weights/location_encoder_weights.pth"
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                with open(weight_path, "wb") as f, tqdm(
                    desc=weight_path,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)
            else:
                raise Exception(f"Failed to download weights from {url}")

        # Load the weights
        self.load_state_dict(torch.load(weight_path))

        # self.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

    def forward(self, location: Tensor) -> Tensor:
        """
        Forward pass for the LocationEncoder.

        Parameters
        ----------
        location : Tensor
            Input tensor of geographical coordinates of shape (N, 2).

        Returns
        -------
        Tensor
            Encoded tensor of shape (N, 512).
        """
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], 512, device=location.device)

        for i in range(len(self.sigma)):
            location_features += self._modules["LocEnc" + str(i)](location)

        return location_features
