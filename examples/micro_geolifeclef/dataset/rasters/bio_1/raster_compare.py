from pathlib import Path

# coding: utf-8
import rasterio
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler, Units

from malpolon.data.environmental_raster import PatchExtractor, Raster

with rasterio.open('bio_1_FR.tif') as f:
    raster_rasterio = f.read()
    meta = f.meta
    wholefile = f
    d = dir(f)
    meta = {attr_name: getattr(f, attr_name)
            for attr_name in dir(f) if not attr_name.startswith('_')}

raster_malpolon = Raster('../bio_1/', 'FR')  # [(-7.0, 52.0)], [(3.687344766, 43.755105938)]

print(raster_malpolon.dataset)
print(len(raster_malpolon))
print(raster_malpolon.size)
print(raster_malpolon.dataset.count)
print(raster_malpolon.dataset.bounds)
print(raster_malpolon.dataset.crs)
print(raster_malpolon.dataset.count)
print(raster_malpolon.dataset.bounds)
coordinates = (3.687344766, 43.755105938)
lon = coordinates[0]
lat = coordinates[1]
row, col = raster_malpolon.dataset.index(lon, lat)
patch_malpolon = raster_malpolon[(lat, lon)]
print(row, col)
print(raster_malpolon.dataset.shape)

plt.figure()
plt.imshow(patch_malpolon[0])
plt.show()

"""
Pytorchgeo
"""
class Glc(RasterDataset):
    filename = "bio_1_FR.tif"
    is_image = True
    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()
        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig

dataset = Glc('./mtp')
dataset.rgb_bands = []
print(dataset)
dataset.all_bands = ['bio1']
dataset.rgb_bands = ['bio1']

sampler = RandomGeoSampler(dataset, size=256, length=1, units=Units.PIXELS)
dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

for batch in dataloader:
    sample = unbind_samples(batch)[0]
    dataset.plot(sample)
    plt.axis("off")
    plt.show()
