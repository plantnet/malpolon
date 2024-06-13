"""This script computes the mean and std of a dataset.

This dataset can be a representative sample of a bigger dataset.
"""

import argparse
from time import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

INFO = '\033[93m'
RESET = '\033[0m'
LINK = '\033[94m'

def load_img(fp):
    return np.array(Image.open(fp)).astype(np.float32)

def load_csv(fp):
    df = pd.read_csv(fp)
    return df.iloc[:, 1:].values.astype(np.float32)

def load_pt(fp):
    return torch.load(fp).numpy()

def iterative_mean_std(fps, load_fun, compare_numpy=False):
    mean = 0
    mean2 = 0
    data = []
    for k, fp in tqdm(enumerate(fps), total=len(fps)):
        x = load_fun(fp)  # Giving a large type is important to avoid value overflow with mean squared
        if compare_numpy:
            data.append(x)
        mean += (np.nanmean(x) - mean) / (k + 1)
        mean2 += (np.nanmean(x**2) - mean2) / (k + 1)
    var = mean2 - mean**2
    if compare_numpy:
        print(f'Numpy mean: {INFO}{np.mean(data)}{RESET}, Numpy std: {INFO}{np.std(data)}{RESET}')
    return mean, np.sqrt(var)

def main(paths_file, output=None, type='image', max_items=None, compare_numpy=False):
    t1 = time()
    with open(paths_file, 'r', encoding="utf-8") as f:
        fps = f.read().splitlines()
    fps = fps[:max_items]

    if type == 'img':
        it_mean, it_std = iterative_mean_std(fps, load_img, compare_numpy)
        print(f'Processed {INFO}{len(fps)}{RESET} images. Iterative mean: {INFO}{it_mean}{RESET}, Iterative std: {INFO}{it_std}{RESET} in {LINK}{(time() - t1):.3f}{RESET}s')
    elif type == 'csv':
        it_mean, it_std = iterative_mean_std(fps, load_csv, compare_numpy)
        print(f'Processed {INFO}{len(fps)}{RESET} csv pre-extracted obs files. Iterative mean: {INFO}{it_mean}{RESET}, Iterative std: {INFO}{it_std}{RESET} in {LINK}{(time() - t1):.3f}{RESET}s')
    elif type == 'pt':
        it_mean, it_std = iterative_mean_std(fps, load_pt, compare_numpy)
        print(f'Processed {INFO}{len(fps)}{RESET} pytorch cubes. Iterative mean: {INFO}{it_mean}{RESET}, Iterative std: {INFO}{it_std}{RESET} in {LINK}{(time() - t1):.3f}{RESET}s')
    else:
        raise ValueError(f"Type {type} not recognized.")

    if output:
        df = pd.DataFrame({'mean': [it_mean], 'std': [it_std]})
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
                        default=None,
                        type=int)
    parser.add_argument("--type",
                        help="Type of files to process.",
                        choices=['img', 'csv', 'pt'],
                        type=str)
    parser.add_argument("--compare_numpy",
                        help="If true, computes the Numpy mean and std for comparison. WARNING: this will load all the items in memory, only use with a reasonable value of --max_items.",
                        action='store_true')
    args = parser.parse_args()
    main(args.paths_file, args.output, type=args.type, max_items=args.max_items,
         compare_numpy=args.compare_numpy)
