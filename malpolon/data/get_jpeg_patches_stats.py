"""Script / module used to compute the mean and std on JPEG files.

When dealing with a large amount of files it should be run
only once, and the statistics should be stored in a separate .csv
for later use.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from malpolon.data.utils import get_files_path_recursively


def standardize(root_path: str = 'sample_data/SatelliteImages/',
                ext: str = ['jpeg', 'jpg'],
                output: str = 'root_path'):
    """Perform standardization over images.

    Returns and stores the mean and standard deviation of an image
    dataset organized inside a root directory for computation
    purposes like deep learning.

    Args:
        root_path (str): root dir. containing the images.
                         Defaults to './sample_data/SatelliteImages/'.
        ext (str, optional): the images extensions to consider.
                             Defaults to 'jpeg'.
        output (str, optional): output path where to save the csv containing
                                the mean and std of the dataset.
                                If None: doesn't output anything.
                                Defaults to root_path.

    Returns:
        (tuple): tuple of mean and std fo the jpeg images.
    """
    fps = get_files_path_recursively(root_path, *ext)
    imgs = []
    stats = {'mean': [], 'std': []}
    for fpath in fps:
        img = np.array(Image.open(fpath, mode='r'))
        if len(img.shape) == 3:
            img = np.transpose(img, (2, 0, 1))
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        imgs.append(img)
    imgs = np.concatenate(imgs, axis=0)
    stats['mean'].append(np.nanmean(imgs))
    stats['std'].append(np.nanstd(imgs))
    if output:
        output = os.path.join(root_path, 'standardize_stats.csv') if output == 'root_path' else output
        df = pd.DataFrame(stats)
        df.to_csv(output, index=False, sep=';')
    return stats['mean'][0], stats['std'][0]


def standardize_by_parts(fps_fp: str,
                         output: str = 'glc23_stats.csv',
                         max_imgs_per_computation: int = 100000):
    """Perform standardization over images part by part.

    With too many images, memory can overflow. This function addresses
    this problem by performing the computation in parts.
    Downside: the computed standard deviation is an mean approximation
    of the true value.

    Args:
        fps_fp (str): file path to a text file containing the paths to
                      the images.
        output (str, optional): output path where to save the csv containing
                                the mean and std of the dataset.
                                If None: doesn't output anything.
                                Defaults to root_path.
        max_imgs_per_computation (int, optional): maximum number of images to hold in memory.
                                                  Defaults to 100000.

    Returns:
        (tuple): tuple of mean and std fo the jpeg images.
    """
    with open(fps_fp, 'r', encoding="utf-8") as f:
        fps = f.readlines()
    imgs = []
    stats = {'mean': [], 'std': []}
    i = 0
    for fpath in tqdm(fps):
        img = np.array(Image.open(fpath[:-1], mode='r'))
        if len(img.shape) == 3:
            img = np.transpose(img, (2, 0, 1))
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        imgs.append(img)
        stats['mean'].append(np.nanmean(imgs))
        stats['std'].append(np.nanstd(imgs))
        i += 1
        if i % max_imgs_per_computation == 0 and i > 0:
            imgs = []
            stats = {'mean': [np.mean(stats['mean'])], 'std': [np.std(stats['std'])]}
    if output:
        df = pd.DataFrame(stats)
        df.to_csv(output, index=False, sep=';')
    return stats['mean'][0], stats['std'][0]


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--root_path',
                        nargs=1,
                        type=str,
                        default=['sample_data/SatelliteImages/'],
                        help='Rooth path.')
    PARSER.add_argument('--ext',
                        nargs=1,
                        type=str,
                        default=['jpg', 'jpeg'],
                        help='File extension.')
    PARSER.add_argument('--out',
                        nargs=1,
                        type=str,
                        default=['sample_data/SatelliteImages/jpeg_patches_sample_stats.csv'],
                        help='Output path.')
    ARGS = PARSER.parse_args()
    standardize(ARGS.root_path[0], ext=ARGS.ext, output=ARGS.out[0])
