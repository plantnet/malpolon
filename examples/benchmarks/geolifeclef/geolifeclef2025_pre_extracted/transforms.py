"""Collection of custom PyTorch friendly transform classes.

These transform classes can be called during training loops to perform
data augmentation.

Author: Theo Larcher <theo.larcher@inria.fr>
        Lukas Picek <lukas.picek@inria.fr>
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt  # pylint: disable=W0611 # noqa: F401
from PIL import Image  # pylint: disable=W0611 # noqa: F401
from torchvision import transforms
from tqdm import tqdm


class SafeRescaleTo255:
    """Rescale an image band to [0, 255] and clip values."""
    def __call__(
        self,
        band: np.ndarray,
    ):
        """Call method.

        Args:
            band (np.ndarray): 2D array to normalize

        Returns:
            (np.ndarray): normalized array
        """
        return np.clip(band * 255, 0, 255).astype(np.uint8)


class MinMaxNormalize:
    """Normalize an image band to [0, 1]."""
    def __call__(
        self,
        band: np.ndarray,
    ):
        """Call method.

        Args:
            band (np.ndarray): 2D array to normalize

        Returns:
            (np.ndarray): normalized array
        """
        normalized = (band - band.min()) / (band.max() - band.min())
        return normalized


class Standardize:
    """Standardize an image band."""
    def __call__(
        self,
        band: np.ndarray,
    ):
        """Call method.

        Args:
            band (np.ndarray): 2D array to normalize

        Returns:
            (np.ndarray): normalized array
        """
        standardized = (band - band.mean()) / band.std()
        return np.clip(standardized, 0, 1)


class LogNormalize:
    """Log normalize an image band."""
    def __call__(
        self,
        band: np.ndarray,
    ):
        """Call method.

        Args:
            band (np.ndarray): 2D array to normalize

        Returns:
            (np.ndarray): normalized array
        """
        normalized = np.log1p(band - band.min()) / np.log1p(band.max() - band.min())
        return normalized


class QuantileLinearNormalize:
    """Normalize an image band based on quantiles."""
    def __call__(
        self,
        band: np.ndarray,
        low: int = 2,
        high: int = 98,
    ):
        """Call method.

        Args:
            band (np.ndarray): 2D array to normalize.
            low (int, optional): low quantile threshold to cut. Defaults to 2.
            high (int, optional): high quantile threshold to cut. Defaults to 98.

        Returns:
            (np.ndarray): normalized array
        """
        sorted_band = np.sort(band.flatten())
        quantiles = np.percentile(sorted_band, np.linspace(low, high, len(sorted_band)))
        normalized_band = np.interp(band.flatten(), sorted_band, quantiles).reshape(band.shape)
        min_val = np.min(normalized_band)
        max_val = np.max(normalized_band)
        return (normalized_band - min_val) / (max_val - min_val)


def pre_compute_quantile_linear_on_dataset(
    root_path: str = "dataset/geolifeclef-2025/SatelitePatches/",
    low: int = 2,
    high: int = 98,
    subset: str = "train",
    data_type: str = "tiff",
    output_path: str = "dataset/geolifeclef-2025/Stats/",
    max_iter: int = 100,
) -> None:
    """Pre-compute the quantiles over the satellite dataset to perform global normalization.

    This method uses numpy's percentile function with mode "linear" to compute
    the quantiles. It is memory hungry as it requires all the arrays stacked up to compute
    the percentiles. However it is more intuitive.
    Saves quantiles and min/max values as numpy objects to be used for normalization.
    Satellite patches shape is (64, 64)
    Quantiles shape is (4, 4096) and min/max shape is (4, 2) where the 1st element of
    axis 1 is the min and the 2nd element is the max.

    Args:
        root_path (str, optional): path to the satellite root folder. Defaults to "data/geolifeclef2025_pre_extracted/SatelitePatches/".
        low (int, optional): low percentile to compute quantiles from. Defaults to 2.
        high (int, optional): hight percentile to compute quantiles from. Defaults to 98.
        subset (str, optional): dataset subset. Takes values in ['train', 'test']. Defaults to "train".
        data_type (str, optional): data format. Defaults to "tiff".
        output_path (str, optional): output path of saved quantiles and min/max. Defaults to "data/geolifeclef2025_pre_extracted/SatelitePatches/".
        max_iter (int, optional): max files to considered to compute the quantiles from. Defaults to 100.
    """
    fps = (Path(root_path) / Path(f'PA-{subset}')).rglob(f"*.{data_type}")
    data = []
    quantiles = {'r': None, 'g': None, 'b': None, 'nir': None}
    min_max_val = {'r': [0, 1], 'g': [0, 1], 'b': [0, 1], 'nir': [0, 1]}
    k = 0
    pbar = tqdm(total=max_iter)
    while k < max_iter:
        try:
            fp = next(fps)
            if os.path.isfile(fp) and fp.suffix == '.tiff':
                img = rasterio.open(fp).read(out_dtype=np.float32)
                data.append(img)
                k += 1
                pbar.update(1)
        except StopIteration:
            print(f'Max files ({k}) reached before max iterations ({max_iter}). Quantiles will be computed on {k} files.')
            pbar.update(max_iter)
            k = np.inf
    data = np.stack(data, axis=0)
    for i, k_band in enumerate(quantiles.keys()):
        band = data[:, i]
        sorted_band = np.sort(band.flatten())
        quantiles[k_band] = np.percentile(sorted_band, np.linspace(low, high, len(sorted_band) // max_iter))
        min_max_val[k_band] = [band.min(), band.max()]
    np.save(Path(output_path) / Path(f'Satellite_quantiles_linear_approx-{max_iter}.npy'), np.array(list(quantiles.values())))
    np.save(Path(output_path) / Path(f'Satellite_min-max_values_linear_approx-{max_iter}.npy'), np.array(list(min_max_val.values())))


# WARNING: Using inverted_cdf needs Numpy >= 2.0.0. Check that all other dependencies are compatible.
def pre_compute_quantile_inverted_cdf_on_dataset(
    root_path: str = "dataset/geolifeclef-2025/SatelitePatches/",
    low: int = 2,
    high: int = 98,
    subset: str = "train",
    data_type: str = "tiff",
    output_path: str = "dataset/geolifeclef-2025/Stats/",
    max_iter: int = 1000,
) -> None:
    """Pre-compute the quantiles over the satellite dataset to perform global normalization.

    This method uses numpy's percentile function with mode "inverted_cdf" to compute
    the quantiles. It is memory efficient as it only requires the unique
    values in the dataset's images, and an associated weight array representing their distribution.
    Saves quantiles and min/max values as numpy objects to be used for normalization.
    Satellite patches shape is (64, 64)
    Quantiles shape is (4, 4096) and min/max shape is (4, 2) where the 1st element of
    axis 1 is the min and the 2nd element is the max.

    Args:
        root_path (str, optional): path to the satellite root folder. Defaults to "data/geolifeclef2025_pre_extracted/SatelitePatches/".
        low (int, optional): low percentile to compute quantiles from. Defaults to 2.
        high (int, optional): hight percentile to compute quantiles from. Defaults to 98.
        subset (str, optional): dataset subset. Takes values in ['train', 'test']. Defaults to "train".
        data_type (str, optional): data format. Defaults to "tiff".
        output_path (str, optional): output path of saved quantiles and min/max. Defaults to "data/geolifeclef2025_pre_extracted/SatelitePatches/".
        max_iter (int, optional): max files to considered to compute the quantiles from. Defaults to 1000.
    """
    fps = (Path(root_path) / Path(f'PA-{subset}')).rglob(f"*.{data_type}")
    value_counts = {'r': defaultdict(int), 'g': defaultdict(int), 'b': defaultdict(int), 'nir': defaultdict(int)}
    quantiles = {'r': None, 'g': None, 'b': None, 'nir': None}
    min_max_val = {'r': [0, 1], 'g': [0, 1], 'b': [0, 1], 'nir': [0, 1]}
    k = 0
    pbar = tqdm(total=max_iter)
    while k < max_iter:
        try:
            fp = next(fps)
            img = rasterio.open(fp).read(out_dtype=np.float32)
            for band, k_band in zip(img, quantiles.keys()):
                unique_vals, counts = np.unique(band.flatten(), return_counts=True)
                for val, count in zip(unique_vals, counts):
                    value_counts[k_band][val] += count
            k += 1
            pbar.update(1)
        except StopIteration:
            pbar.update(max_iter)
            k = np.inf
    for k_band in quantiles:
        value_counts_sorted = dict(sorted(value_counts[k_band].items()))  # apparently useless
        quantiles[k_band] = np.percentile(list(value_counts_sorted.keys()),
                                          np.linspace(low, high, img.shape[1] * img.shape[2]),
                                          method='inverted_cdf',
                                          weights=np.array(list(value_counts_sorted.values())) / np.array(list(value_counts_sorted.values())).sum())
        min_max_val[k_band] = [np.min(quantiles[k_band]), np.max(quantiles[k_band])]
    np.save(Path(output_path) / Path('Satellite_quantiles_inverted-cdf.npy'), np.array(list(quantiles.values())))
    np.save(Path(output_path) / Path('Satellite_min-max_values_inverted-cdf.npy'), np.array(list(min_max_val.values())))


class QuantileNormalizeFromPreComputedDatasetPercentiles:
    """Apply quantile normalization from pre-computed quantiles and min/max."""
    def __call__(
        self,
        img: np.ndarray,
        fp_quantiles: Union[str, Path] = "dataset/geolifeclef-2025/Stats/Satellite_quantiles_linear_approx-100.npy",
        fp_min_max: Union[str, Path] = "dataset/geolifeclef-2025/Stats/Satellite_min-max_values_linear_approx-100.npy",
    ):
        """Call method.

        Args:
            img (np.ndarray): image to normalize
            fp_quantiles (Union[str, Path]): file path to pre-computed quantiles
            fp_min_max (Union[str, Path]): file path to pre-computed min/max

        Returns:
            (np.ndarray): quantile-normalized image
        """
        quantiles = np.load(fp_quantiles)  # Quantiles shape: (4, 4096), 4 bands, 4096 quantile values. Bands order: [r, g, b, nir]
        min_max_val = np.load(fp_min_max)  # Min/max shape: (4, 2) 4 bands, 2 values (min, max)
        min_val, max_val = min_max_val[:, 0][:, np.newaxis, np.newaxis], min_max_val[:, 1][:, np.newaxis, np.newaxis]
        normalized_img = img.copy()
        for (k, band), quantile in zip(enumerate(img), quantiles):
            flat_band = band.flatten()
            sorted_band = np.sort(flat_band)
            normalized_img[k] = np.interp(flat_band, sorted_band, quantile).reshape(band.shape)
        return (normalized_img - min_val) / (max_val - min_val)


class GLC25CustomNormalize:
    """Return custom GLC25 normalization based on data modality."""
    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        subset: str = "train",
        modality: str = "landsat"
    ) -> dict:
        """Call method.

        The normalization values are pre-computed from the training dataset
        (pre-extracted values) for each modality.

        Args:
            img (np.ndarray): image to normalize.
            modality (str): modality. Takes values in ['landsat', 'bioclim', 'sentinel'].
            subset (str, optional): dataset subset. Takes values in ['train', , 'val', 'test']. Defaults to "train".

        Returns
        -------
        (dict)
            dictionary of transform functions for each data modality.
        """
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        transfo_dict = {'train': {'landsat': transforms.Normalize(mean=[30.654] * 6, std=[25.702] * 6),
                                  'bioclim': transforms.Normalize(mean=[3914.847] * 4, std=[3080.644] * 4),
                                  'sentinel': transforms.Normalize(mean=[629.624, 691.815, 460.605] + [2959.370],
                                                                   std=[435.995, 371.396, 342.897] + [925.369])},
                        'val': {'landsat': transforms.Normalize(mean=[30.269] * 6, std=[25.212] * 6),
                                'bioclim': transforms.Normalize(mean=[3955.529] * 4, std=[3234.002] * 4),
                                'sentinel': transforms.Normalize(mean=[633.110, 692.764, 462.189] + [2950.603],
                                                                 std=[465.046, 398.975, 370.759] + [927.021])},
                        'test': {'landsat': transforms.Normalize(mean=[26.188] * 6, std=[29.624] * 6),
                                 'bioclim': transforms.Normalize(mean=[3932.149] * 4, std=[3490.368] * 4),
                                 'sentinel': transforms.Normalize(mean=[517.786, 565.655, 376.777] + [2289.862],
                                                                  std=[530.537, 497.530, 427.435] + [1510.104])}}
        normalized = transfo_dict[subset][modality](img)
        return normalized


# # Example usage
# if __name__ == "__main__":
#     # pre_compute_quantile_inverted_cdf_on_dataset(
#     #   "dataset/geolifeclef-2025/SatelitePatches/",
#     #   subset="train",
#     #   data_type="tiff"
#     # )
#     pre_compute_quantile_linear_on_dataset(
#         "dataset/geolifeclef-2025/SatelitePatches/",
#         subset="train",
#         data_type="tiff"
#     )
#     patch = rasterio.open("dataset/geolifeclef-2025/SatelitePatches/PA-train/00/00/3440000.tiff").read(out_dtype=np.float32)
#     minmax_norm = np.array([MinMaxNormalize()(band) for band in patch])
#     standard_norm = np.array([Standardize()(band) for band in patch])
#     log_norm = np.array([LogNormalize()(band) for band in patch])
#     quantile_norm = np.array([QuantileLinearNormalize()(band) for band in patch])
#     quantile_norm_precomp = QuantileNormalizeFromPreComputedDatasetPercentiles()(
#         patch,
#         "dataset/geolifeclef-2025/Stats/Satellite_quantiles_linear_approx-100.npy",
#         "dataset/geolifeclef-2025/Stats/Satellite_min-max_values_linear_approx-100.npy"
#     )
#     custom_norm = GLC25CustomNormalize()(patch, modality='sentinel').numpy()

#     Image.fromarray(SafeRescaleTo255()(np.transpose(patch[:3], (1,2,0)))).save('original patch.jpeg')
#     Image.fromarray(SafeRescaleTo255()(np.transpose(minmax_norm[:3], (1,2,0)))).save('minmax_norm.jpeg')
#     Image.fromarray(SafeRescaleTo255()(np.transpose(standard_norm[:3], (1,2,0)))).save('standard_norm.jpeg')
#     Image.fromarray(SafeRescaleTo255()(np.transpose(log_norm[:3], (1,2,0)))).save('log_norm.jpeg')
#     Image.fromarray(SafeRescaleTo255()(np.transpose(quantile_norm[:3], (1,2,0)))).save('quantile_norm_linear.jpeg')
#     Image.fromarray(SafeRescaleTo255()(np.transpose(quantile_norm_precomp[:3], (1,2,0)))).save('quantile_norm_inverted-cdf.jpeg')
#     Image.fromarray(SafeRescaleTo255()(np.transpose(custom_norm[:3], (1,2,0)))).save('custom_norm.jpeg')

#     fig, ax = plt.subplots(4, 7, figsize=(20, 12))
#     for i in range(4):
#         ax[i, 0].imshow(patch[i], cmap='gray')
#         ax[i, 0].set_title(f'Band {i} original')

#         ax[i, 1].imshow(minmax_norm[i], cmap='gray')
#         ax[i, 1].set_title(f'Band {i} minmax')

#         ax[i, 2].imshow(standard_norm[i], cmap='gray')
#         ax[i, 2].set_title(f'Band {i} standard_norm')

#         ax[i, 3].imshow(log_norm[i], cmap='gray')
#         ax[i, 3].set_title(f'Band {i} log_norm')

#         ax[i, 4].imshow(quantile_norm[i], cmap='gray')
#         ax[i, 4].set_title(f'Band {i} quantile linear')

#         ax[i, 5].imshow(quantile_norm_precomp[i], cmap='gray')
#         ax[i, 5].set_title(f'Band {i} quantile inverted_cdf')

#         ax[i, 6].imshow(custom_norm[i], cmap='gray')
#         ax[i, 6].set_title(f'Band {i} custom_norm diff')
#     fig.savefig('all.png', dpi=300)
