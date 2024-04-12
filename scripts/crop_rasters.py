import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import numpy as np
from tqdm import tqdm
import pyproj
import os

def main(fps):
    """Clip and export a window from raster files.
    
    Also possible to do via command line:
    `rio input_raster output_raster --bounds "xmin, ymin xmax ymax"`
    """
    # Define the coordinates of the area you want to crop
    minx, miny = 3.78891, 43.483567  # EPSG 3035
    maxx, maxy = 3.956451, 43.700148  # EPSG 3035
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
    fps = {'bio_1': root + 'bio_1.tif',
           'bio_2': root + 'bio_2.tif',
           'bio_3': root + 'bio_3.tif',
           'bio_4': root + 'bio_4.tif',
          }
    main(fps)
