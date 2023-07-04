"""This file compiles useful functions related to data and file handling."""
from __future__ import annotations

import os
import re
import numpy as np
from typing import TYPE_CHECKING, Iterable, Union

from shapely import Polygon, Point


def is_bbox_contained(bbox1: Iterable,
                      bbox2: Iterable,
                      method: str['shapely', 'manual', 'torchgeo'] = 'shapely') -> bool:
    """Determine if a 2D bbox in included inside of another.

    Returns a boolean answering the question "Is bbox1 contained inside
    bbox2 ?".
    With methods 'shapely' and 'manual', bounding boxes must
    follow the format: [xmin, ymin, xmax, ymax].
    With method 'torchgeo', bounding boxes must be of type:
    `torchgeo.datasets.utils.BoundingBox`.

    Parameters
    ----------
    bbox1 : iterable
        bounding box n°1.
    bbox2 : iterable
        bounding box n°2.

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


def is_point_in_bbox(point: tuple[int],
                     bbox2: Iterable,
                     method: str['shapely', 'manual'] = 'shapely') -> bool:
    """Determine if a 2D point in included inside of a 2D bounding box.

    Returns a boolean answering the question "Is point contained inside
    bbox ?".
    Point must follow the format: [x, y]
    Bounding boxe must follow the format: [xmin, ymin, xmax, ymax]

    Parameters
    ----------
    point : iterable
        point.
    bbox : iterable
        bounding box.

    Returns
    -------
    boolean
        True if point ⊂ bbox, False otherwise.
    """
    if method == "manual":
        is_contained = (point[0] >= bbox2[0] and point[0] <= bbox2[2]
                        and point[1] >= bbox2[1] and point[1] <= bbox2[3])
    elif method == "shapely":
        point = Point(point)
        polygon2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[0], bbox2[3]),
                            (bbox2[2], bbox2[3]), (bbox2[2], bbox2[1])])
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

    Returns
    -------
    list
        One-hot encoded labels.
    """
    n_classes = len(labels_target)
    one_hot_labels = np.zeros(n_classes)
    one_hot_labels[np.array(labels_predict) == labels_target] = 1
    return one_hot_labels


def get_files_path_recursively(path, *args, suffix=''):
    """Retrieve specific files path recursively from a directory.

    Retrieve the path of all files with one of the given extension names,
    in the given directory and all its subdirectories, recursively.
    The extension names should be given as a list of strings. The search for
    extension names is case sensitive.

    Args:
        path (str): root directory from which to search for files recursively
        *args: list of file extensions to be considered.

    Returns:
        list(str): list of paths of every file in the directory and all its
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
