
from __future__ import annotations

import numpy as np
from torchgeo.datasets import BoundingBox

from malpolon.data.utils import (get_files_path_recursively, is_bbox_contained,
                                 is_point_in_bbox, to_one_hot_encoding)


def test_is_bbox_contained() -> None:
    bbox1 = [1, 2, 9, 8]
    bbox2 = [0, 0, 10, 10]
    ibc_manual = is_bbox_contained(bbox1, bbox2, method='manual')
    ibc_shapely = is_bbox_contained(bbox1, bbox2, method='shapely')

    tg_bbox1 = BoundingBox(bbox1[0], bbox1[2], bbox1[1], bbox1[3], 0, 10)
    tg_bbox2 = BoundingBox(bbox2[0], bbox2[2], bbox2[1], bbox2[3], 0, 10)
    ibc_torchgeo = is_bbox_contained(tg_bbox1, tg_bbox2, method='torchgeo')
    assert all([ibc_manual, ibc_shapely, ibc_torchgeo])

def test_is_point_in_bbox() -> None:
    point = (5, 5)
    bbox = [0, 0, 10, 10]
    ibc_manual = is_point_in_bbox(point, bbox, method='manual')
    ibc_shapely = is_point_in_bbox(point, bbox, method='shapely')
    assert all([ibc_manual, ibc_shapely])

def test_to_one_hot_encoding() -> None:
    labels_predict = [0, 3, 4]
    labels_target = [0, 1, 2, 3, 4]
    expected1 = np.array([1, 0, 0, 1, 1])
    res1 = to_one_hot_encoding(labels_predict, labels_target)
    expected2 = np.array([1, 0, 0, 0, 0])
    res2 = to_one_hot_encoding(labels_predict[0], labels_target)
    assert np.array_equal(res1, expected1)
    assert np.array_equal(res2, expected2)

def test_get_files_path_recursively() -> None:
    """Test get_files_path_recursively.

    This test assumes to be ran from malpolon's root directory. If not,
    please adapt `path` value.
    """
    path = 'malpolon/tests/data/fp_test_dir/'
    res1 = get_files_path_recursively(path, 'txt')
    res2 = get_files_path_recursively(path, 'txt', 'md', 'rtf')
    res3 = get_files_path_recursively(path, 'rtf', suffix='_deep')
    expected1 = ['fp_test_file1.txt',
                 'fp_test_subdir2/fp_test_file3_deep.txt']
    expected2 = ['fp_test_file1.txt',
                 'fp_test_subdir1/fp_test_file2.md',
                 'fp_test_subdir2/fp_test_file3_deep.txt',
                 'fp_test_subdir2/fp_test_file4_deep.rtf']
    expected3 = ['fp_test_subdir2/fp_test_file4_deep.rtf']
    for expected in [expected1, expected2, expected3]:
        for k, v in enumerate(expected):
            expected[k] = path + v
    assert set(res1) == set(expected1)
    assert set(res2) == set(expected2)
    assert set(res3) == set(expected3)
