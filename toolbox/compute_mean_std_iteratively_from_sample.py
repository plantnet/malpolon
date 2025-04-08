"""This script computes the mean and std of a dataset.

This dataset can be a representative sample of a bigger dataset.
"""

import argparse
from time import time
from typing import Callable
import warnings
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from tqdm import tqdm

INFO = '\033[93m'
RESET = '\033[0m'
LINK = '\033[94m'


def load_raster(fp: str):
    """Load an raster from a file path.

    Parameters
    ----------
    fp : str
        file path to the image.

    Returns
    -------
    (array)
        raster as a numpy array.
    """
    with rasterio.open(fp) as dataset:
        raster = dataset.read(out_dtype=np.float32)
    raster = np.transpose(raster, (1, 2, 0))  # Move the first axis to the last axis
    return raster


def load_img(fp: str):
    """Load an image from a file path.

    Parameters
    ----------
    fp : str
        file path to the image.

    Returns
    -------
    (array)
        image as a numpy array.
    """
    return np.array(Image.open(fp)).astype(np.float32)


def load_csv(fp: str):
    """Load a csv file from a file path.

    The CSV file is expected to contain pre-extracted observations (PA or
    PO), from any source (bioclimatic, landsat, sentinel...) with
    the data contained in columns 1 to end. Column 0 is skipped.

    Parameters
    ----------
    fp : str
        file path to the csv file.

    Returns
    -------
    (array)
        pre-extracted obs as a numpy array.
    """
    df = pd.read_csv(fp)
    return df.iloc[:, 1:].values.astype(np.float32)


def load_pt(fp: str):
    """Load a PyTorch cube from a file path.

    Parameters
    ----------
    fp : str
        file path to the PyTorch cube.

    Returns
    -------
    (array)
        numpy array of the PyTorch cube.
    """
    return torch.load(fp).numpy()


def iterative_mean_std(fps: list,
                       load_fun: Callable,
                       compare_numpy: bool = False):
    """Compute the mean and std of a dataset iteratively.

    Parameters
    ----------
    fps : str
        list of paths to the dataset file.
    load_fun : callable
        loading function to load the data from the file (depends on the
        type of file).
    compare_numpy : bool, optional
        if True, computes the numpy mean and std by loading all dataset
        files content in a single numpy array to compare with the
        iteratively computed values.
        By default False

    Returns
    -------
    (tuple)
        tuple of iterative mean and std of the dataset as float values.
    """
    mean = 0
    mean2 = 0
    data = []
    n_skips = 0
    for k, fp in tqdm(enumerate(fps), total=len(fps)):
        x = load_fun(fp)  # Giving a large type is important to avoid value overflow with mean squared
        if compare_numpy:
            data.append(x)
        nanmean = np.nanmean(x)
        if np.isnan(nanmean):
            n_skips += 1
            warnings.warn(f'File {fp} contains only NaN values. Skipping...')
            continue
        mean += (nanmean - mean) / (k + 1)
        mean2 += (np.nanmean(x**2) - mean2) / (k + 1)
    var = mean2 - mean**2
    if compare_numpy:
        print(f'Numpy mean: {INFO}{np.mean(data)}{RESET}, Numpy std: {INFO}{np.std(data)}{RESET}')
    if n_skips > 0:
        print(f'Skipped {INFO}{n_skips}{RESET} files due to: containing only NaN values.')
    return mean, np.sqrt(var)

def iterative_mean_std_img_per_channel(fps: list,
                                       load_fun: Callable,
                                       compare_numpy: bool = False):
    """Compute the mean and std of a dataset iteratively per channel.

    This method is intended to compute the mean and std for each
    individual channel of 3D matrices.

    Parameters
    ----------
    fps : str
        list of paths to the dataset file.
    load_fun : callable
        loading function to load the data from the file (depends on the
        type of file).
    compare_numpy : bool, optional
        if True, computes the numpy mean and std by loading all dataset
        files content in a single numpy array to compare with the
        iteratively computed values.
        By default False

    Returns
    -------
    (tuple)
        tuple of iterative mean and std of the dataset as float values.
    """
    mean, mean2, var = (np.zeros(1),) * 3
    data = []
    for k, fp in tqdm(enumerate(fps), total=len(fps)):
        try:
            x = load_fun(fp)  # Giving a large type is important to avoid value overflow with mean squared
            assert len(x.shape) >= 3
            if k == 0:
                mean = np.zeros(x.shape[-1]).astype(np.float32)
                mean2 = np.zeros(x.shape[-1]).astype(np.float32)
            if compare_numpy:
                data.append(x)
            mean += (np.nanmean(x, axis=(0,1)) - mean) / (k + 1)
            mean2 += (np.nanmean(x**2, axis=(0,1)) - mean2) / (k + 1)
        except AssertionError:
            print(f'File {fp} does not contain 3D data. Passing...')
    var = mean2 - mean**2
    if compare_numpy:
        print(f'Numpy mean: {INFO}{np.nanmean(data, axis=(0,1))}{RESET}, Numpy std: {INFO}{np.nanstd(data, axis=(0,1))}{RESET}')
    return mean.tolist(), np.sqrt(var).tolist()

def main(paths_file: str,
         output: str = None,
         data_type: str = 'img',
         max_items: int = None,
         per_channel: bool = False,
         compare_numpy: bool = False):
    """Run the main function.

    This method calls the correct loading functions and contrains the
    max amount of items to compute the mean/std on.

    Parameters
    ----------
    paths_file : str
        path to a file containing the paths to the files to process.
    output : str, optional
        path to the output file to store the mean/std values. This file
        is expected to be a CSV of 1 line and 2 columns.
        By default None
    data_type : str, optional
        type of file to process, by default 'image'
    max_items : _type_, optional
        maximum number of items to compute the mean/std on.
        By default None
    per_channel : bool, optionnal
        if True, calls computaiton of mean / std seperately for each
        data channel
    compare_numpy : bool, optional
        if True, the numpy mean and std will also be computed for
        comparison.
        By default False

    Raises
    ------
    ValueError
        triggers when the type is not recognized.
    """
    t1 = time()
    with open(paths_file, 'r', encoding="utf-8") as f:
        fps = f.read().splitlines()
    # fps = fps[:max_items]
    fps = np.array(fps)[np.random.choice(len(fps), size=min(len(fps), max_items), replace=False)]
    ims = iterative_mean_std
    if per_channel and data_type in ['img', 'tiff']:
        ims = iterative_mean_std_img_per_channel

    if data_type == 'img':
        it_mean, it_std = ims(fps, load_img, compare_numpy)
        print(f'Processed {INFO}{len(fps)}{RESET} images.')
    if data_type == 'tiff':
        it_mean, it_std = ims(fps, load_raster, compare_numpy)
        print(f'Processed {INFO}{len(fps)}{RESET} raster.')
    elif data_type == 'csv':
        it_mean, it_std = ims(fps, load_csv, compare_numpy)
        print(f'Processed {INFO}{len(fps)}{RESET} csv pre-extracted obs files.')
    elif data_type == 'pt':
        it_mean, it_std = ims(fps, load_pt, compare_numpy)
        print(f'Processed {INFO}{len(fps)}{RESET} pytorch cubes.')
    else:
        raise ValueError(f"Type {data_type} not recognized.")
    print(f'Iterative mean: {INFO}{it_mean}{RESET}, Iterative std: {INFO}{it_std}{RESET} in {LINK}{(time() - t1):.3f}{RESET}s')

    if output:
        it_mean = [it_mean] if not isinstance(it_mean, list) else it_mean
        it_std = [it_std] if not isinstance(it_std, list) else it_std
        df = pd.DataFrame({'mean': it_mean, 'std': it_std})
        df.to_csv(output, index=False, sep=',')
        print(f'Stats saved to {INFO}{output}{RESET}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths_file",
                        help="Path a text file containing the paths to the files to process.",
                        default=None,
                        type=str)
    parser.add_argument("-o", "--output",
                        help="Output path for the csv containing the mean and std of the dataset. If None: doesn't output anything.",
                        default=None,
                        type=str)
    parser.add_argument("--max_items",
                        help="Max number of items to process. Default is 1000.",
                        default=1000,
                        type=int)
    parser.add_argument("--type",
                        help="Type of files to process.",
                        choices=['img', 'tiff', 'csv', 'pt'],
                        type=str)
    parser.add_argument("--per_channel",
                        help="Compute mean/std over each channel seperately.",
                        action='store_true')
    parser.add_argument("--compare_numpy",
                        help="If true, computes the Numpy mean and std for comparison. WARNING: this will load all the items in memory, only use with a reasonable value of --max_items.",
                        action='store_true')
    args = parser.parse_args()
    main(args.paths_file, args.output, data_type=args.type, max_items=args.max_items,
         per_channel=True,
         compare_numpy=args.compare_numpy)
