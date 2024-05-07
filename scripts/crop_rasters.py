import os

import numpy as np
import pyproj
import rasterio
from pyproj import CRS, Transformer
from rasterio.mask import mask
from shapely.geometry import box
from tqdm import tqdm


def main(fps, data_crs, coords_crs):
    """Clip and export a window from raster files.
    
    Also possible to do via command line:
    `rio input_raster output_raster --bounds "xmin, ymin xmax ymax"`
    """
    # Define the coordinates of the area you want to crop
    minx, miny = 499980.0, 4790220.0  # EPSG 32631
    maxx, maxy = 609780.0, 4900020.0  # EPSG 32631
    if data_crs != coords_crs:
        transformer = Transformer.from_crs(pyproj.CRS.from_epsg(coords_crs), pyproj.CRS.from_epsg(data_crs), always_xy=True)
        minx, miny, maxx, maxy = transformer.transform_bounds(minx, miny, maxx, maxy)
    bbox = box(minx-0.8, miny-0.8, maxx+0.8, maxy+0.8)

    for k, v in tqdm(fps.items()):
        # Open the raster file
        with rasterio.open(v) as src:
            # Crop the raster using the bounding box
            out_image, out_transform = mask(src, [bbox], crop=True)

            # Update the metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            })

        # Write the cropped raster to a new file
        with rasterio.open(f'{k}_crop_sample.tif', "w", **out_meta) as dest:
            dest.write(out_image)
        
if __name__ == '__main__':
    root = './'
    fps = {'bio_1': root + 'bio_1/bio_1_FR.tif',
           'bio_5': root + 'bio_5/bio_5_FR.tif',
           'bio_6': root + 'bio_6/bio_6_FR.tif',
           'bio_12': root + 'bio_12/bio_12_FR.tif',
          }
    data_crs = '4326'
    coords_crs = '32631'
    main(fps, data_crs, coords_crs)