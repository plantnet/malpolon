import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import numpy as np

def main():
    # Define the coordinates of the area you want to crop
    minx, miny = 3.048729, 49.991  # replace with your values
    maxx, maxy = 6.898029, 52.13476  # replace with your values
    bbox = box(minx, miny, maxx, maxy)

    for i in np.arange(19):
        # Open the raster file
        with rasterio.open(f'BioClimatic_Average_1981-2010/bio{i+1}.tif') as src:
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
        with rasterio.open(f'BioClimatic_Average_1981-2010/belgium_crop/bio{i+1}_belgium.tif', "w", **out_meta) as dest:
            dest.write(out_image)
        
if __name__ == '__main__':
    main()
