"""This file compiles useful functions related to data and file handling.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Union

import numpy as np
import pandas as pd
from shapely import Point, Polygon
from torchgeo.datasets import BoundingBox
from verde import train_test_split as spatial_tts

from malpolon.plot.map import plot_observation_dataset as plot_od


def is_bbox_contained(
    bbox1: Union[Iterable, BoundingBox],
    bbox2: Union[Iterable, BoundingBox],
    method: str = 'shapely'
) -> bool:
    """Determine if a 2D bbox in included inside of another.

    Returns a boolean answering the question "Is bbox1 contained inside
    bbox2 ?".
    With methods 'shapely' and 'manual', bounding boxes must
    follow the format: [xmin, ymin, xmax, ymax].
    With method 'torchgeo', bounding boxes must be of type:
    `torchgeo.datasets.utils.BoundingBox`.

    Parameters
    ----------
    bbox1 : Union[Iterable, BoundingBox]
        Bounding box n°1.
    bbox2 : Union[Iterable, BoundingBox]
        Bounding box n°2.
    method : str
        Method to use for comparison. Can take any value in
        ['shapely', 'manual', 'torchgeo'], by default 'shapely'.

    Returns
    -------
    boolean
        True if bbox1 ⊂ bbox2, False otherwise.
    """
    if method == "manual":
        is_contained = (bbox1[0] >= bbox2[0] and bbox1[0] <= bbox2[2]
                        and bbox1[2] >= bbox2[0] and bbox1[2] <= bbox2[2]
                        and bbox1[1] >= bbox2[1] and bbox1[1] <= bbox2[3]
                        and bbox1[3] >= bbox2[1] and bbox1[3] <= bbox2[3])
    elif method == "shapely":
        polygon1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[0], bbox1[3]),
                            (bbox1[2], bbox1[3]), (bbox1[2], bbox1[1])])
        polygon2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[0], bbox2[3]),
                            (bbox2[2], bbox2[3]), (bbox2[2], bbox2[1])])
        is_contained = polygon2.contains(polygon1)
    elif method == "torchgeo":
        is_contained = bbox1 in bbox2
    return is_contained


def is_point_in_bbox(
    point: Iterable,
    bbox: Iterable,
    method: str = 'shapely'
) -> bool:
    """Determine if a 2D point in included inside of a 2D bounding box.

    Returns a boolean answering the question "Is point contained inside
    bbox ?".
    Point must follow the format: [x, y]
    Bounding box must follow the format: [xmin, ymin, xmax, ymax]

    Parameters
    ----------
    point : Iterable
        Point in the format [x, y].
    bbox : Iterable
        Bounding box in the format [xmin, xmax, ymin, ymax].
    method : str
        Method to use for comparison. Can take any value in
        ['shapely', 'manual'], by default 'shapely'.

    Returns
    -------
    boolean
        True if point ⊂ bbox, False otherwise.
    """
    if method == "manual":
        is_contained = (point[0] >= bbox[0] and point[0] <= bbox[2]
                        and point[1] >= bbox[1] and point[1] <= bbox[3])
    elif method == "shapely":
        point = Point(point)
        polygon2 = Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]),
                            (bbox[2], bbox[3]), (bbox[2], bbox[1])])
        is_contained = polygon2.contains(point)
    return is_contained


def to_one_hot_encoding(
    labels_predict: int | list,
    labels_target: list,
) -> list:
    """Return a one-hot encoding of class-index predicted labels.

    Converts a single label value or a vector of labels into a vector
    of one-hot encoded labels. The labels order follow that of input
    labels_target.

    Parameters
    ----------
    labels_predict : int | list
        Labels to convert to one-hot encoding.
    labels_target : list
        All existing labels, in the right order.

    Returns
    -------
    list
        One-hot encoded labels.
    """
    labels_predict = [labels_predict] if isinstance(labels_predict, int) else labels_predict
    n_classes = len(labels_target)
    one_hot_labels = np.zeros(n_classes, dtype=np.float32)
    one_hot_labels[np.in1d(labels_target, labels_predict)] = 1
    return one_hot_labels


def get_files_path_recursively(path, *args, suffix='') -> list:
    """Retrieve specific files path recursively from a directory.

    Retrieve the path of all files with one of the given extension names,
    in the given directory and all its subdirectories, recursively.
    The extension names should be given as a list of strings. The search for
    extension names is case sensitive.

    Parameters
    ----------
    path : str
        root directory from which to search for files recursively
    *args : list
        list of file extensions to be considered.

    Returns
    -------
    list list of paths of every file in the directory and all its
         subdirectories.
    """
    exts = list(args)
    for ext_i, ext in enumerate(exts):
        exts[ext_i] = ext[1:] if ext[0] == '.' else ext
    ext_list = "|".join(exts)
    result = [os.path.join(dp, f)
              for dp, dn, filenames in os.walk(path)
              for f in filenames
              if re.search(rf"^.*({suffix})\.({ext_list})$", f)]
    return result


def split_obs_spatially(input_name: str,
                        spacing: float = 10 / 60,
                        plot: bool = False,
                        val_size: float = 0.15):
    """Perform a spatial train/val split on the input csv file.

    Parameters
    ----------
    input_name : str
        obs CSV input file's name without the .csv extension.
    spacing : float, optional
        size of the spatial split in degrees (or whatever unit the coordinates are in),
        by default 10/60
    plot : bool, optional
        if true, plots the train/val split on a 2D map,
        by default False
    val_size : float, optional
        size of the validaiton split, by default 0.15
    """
    df = pd.read_csv(f'{input_name}.csv')
    coords, data = {}, {}
    for col in df.columns:
        if col in ['lon', 'lat']:
            coords[col] = df[col].to_numpy()
        else:
            data[col] = df[col].to_numpy()
    train_split, val_split = spatial_tts(tuple(coords.values()), tuple(data.values()),
                                         spacing=spacing, test_size=val_size)

    df_train = pd.DataFrame({'lon': train_split[0][0], 'lat': train_split[0][1]})
    df_val = pd.DataFrame({'lon': val_split[0][0], 'lat': val_split[0][1]})
    df_train['subset'] = ['train'] * len(df_train)
    df_val['subset'] = ['val'] * len(df_val)
    for train_data, val_data, col in zip(train_split[1], val_split[1], data.keys()):
        df_train[col] = train_data
        df_val[col] = val_data

    df_train_val = pd.concat([df_train, df_val])

    df_train_val.to_csv(f'{input_name}_train_val-{spacing*60}min.csv', index=False)
    df_train.to_csv(f'{input_name}_train-{spacing*60}min.csv', index=False)
    df_val.to_csv(f'{input_name}_val-{spacing*60}min.csv', index=False)

    if plot:
        plot_od(df=df_train_val, show_map=True)
