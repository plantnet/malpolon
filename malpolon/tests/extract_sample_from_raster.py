"""This script helps extracting small rasters from big rasters.

This script is intended for testing purposes.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

import rasterio
import rioxarray
import numpy as np
from rasterio.windows import Window
from matplotlib import pyplot as plt
from matplotlib import patches


def extract_sample_from_raster(
    input_path: str,
    output_path: str,
    size: tuple,
    xy_offset: tuple = None,
) -> None:
    """Extract a sample raster from a big raster.

    By defining a starting point (coordinate offset) and a size
    (width, height)pix_miny_proj in rasterio's Window, this method extracts a
    sample from an input raster, preserving metadata and CRS values.
    The coordinate offset must take into account the size of the sample
    raster as the rectangle computed to extract the sample is not
    drawn center-wise but corner-wise (minx, miny corner).

    Parameters
    ----------
    input_path : str
        Path to the input raster to extract from.
    output_path : str
        Path to the extracted sample raster.
    size : tuplecoords_4326
    """
    with rasterio.open(input_path) as src:
        xy_offset = (src.shape[1] // 2 - size[0] // 2, src.shape[0] // 2 - size[1] // 2) if xy_offset is None else xy_offset
        xy_offset_proj = [xy_offset[0], xy_offset[1]]*src.transform
        pix_size = abs(src.bounds[2]-src.bounds[0])/src.shape[1], abs(src.bounds[3]-src.bounds[1])/src.shape[0]
        window = Window(*xy_offset, size[0], size[1])  # (x,y), x, y
        sample_data = src.read(window=window)
        profile = src.profile
        profile.update(width=sample_data.shape[2], height=sample_data.shape[1],
                       transform=rasterio.transform.from_origin(xy_offset_proj[0], xy_offset_proj[1], pix_size[0], pix_size[1]))  # sample_data.shape -> (1, rows (y), cols (x))
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(sample_data)
    print("Sample extracted and saved as a new raster image.")


def compare_rasters(
    full_fp: str,
    sample_fp: str,
    size: tuple,
    xy_offset: tuple = None,
    ) -> None:
    """Display full and sample rasters in a window for comparison.

    Parameters
    ----------
    full_fp : str
        full raster file path
    sample_fp : str
        sample raster file path
    size : tuple
        Size of the sample to extract (in pixels)
    xy_offset : tuple
        Coordinate offset from where to extract the rectangular sample.
        If None, will be set to the center of the full raster.
        Defaults to None.
    """
    with rasterio.open(full_fp) as src:
        full = src.read()
        full_crs = src.crs
    with rasterio.open(sample_fp) as src:
        sample = src.read()
        sample_crs = src.crs
    xy_offset = (full.shape[2] // 2 - size[0] // 2, full.shape[1] // 2 - size[1] // 2) if xy_offset is None else xy_offset
    assert full_crs.to_epsg() == sample_crs.to_epsg()
    assert full_crs == sample_crs
    # assert np.array_equal(full[0, xy_offset[0]:xy_offset[0]+size[0], xy_offset[1]:xy_offset[1]+size[1]], sample[0])
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(full[0], cmap='gray', vmin=np.min(sample), vmax=np.max(sample))
    rect = patches.Rectangle(xy_offset, size[0], size[1], linewidth=3, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title('Original raster')
    ax2.imshow(sample[0], cmap='gray', vmin=np.min(sample), vmax=np.max(sample))
    ax2.set_title('Sample raster')
    plt.show()
    print('')


def main() -> None:
    """Run main script."""
    # Sentinel-2
    # input_path = './torchgeo_sentinel2_test_full.tif'
    # output_path = 'a.tif'
    # size = (600, 600)
    # xy_offset = (6728, 6795)

    # MicroLifeClef
    input_path = './torchgeo_mlc_test_full.tif'
    output_path = 'torchgeo_mlc_test_sample.tif'
    size = (400, 400)
    xy_offset = None

    extract_sample_from_raster(input_path, output_path, size, xy_offset=xy_offset)
    compare_rasters(input_path, output_path, size, xy_offset=xy_offset)


if __name__ == "__main__":
    main()
