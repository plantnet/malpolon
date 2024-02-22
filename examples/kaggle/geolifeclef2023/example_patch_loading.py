"""GeoLifeCLEF23 patch example module.

This module provides an example of how to load and use GeoLifeCLEF2023 patch
datasets using the framework developped for the challenge and incorporated
in malpolon at `malpolon.data.datasets.geolifeclef2023`.
"""

import random

from malpolon.data.datasets.geolifeclef2023 import (
    JpegPatchProvider, MultipleRasterPatchProvider, PatchesDataset,
    PatchesDatasetMultiLabel, RasterPatchProvider)


def main():
    """Run GLC23 patch example script."""
    data_path = 'dataset/sample_data/'  # root path of the data

    # configure providers
    p_rgb = JpegPatchProvider(data_path + 'SatelliteImages/',
                              dataset_stats='jpeg_patches_sample_stats.csv',
                              id_getitem='patchID')  # take all sentinel imagery layer (r,g,b,nir)
    p_hfp_d = MultipleRasterPatchProvider(data_path + 'EnvironmentalRasters/HumanFootprint/detailed/')  # take all rasters from human footprint detailed
    p_bioclim = MultipleRasterPatchProvider(data_path + 'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/',
                                            select=['bio1', 'bio2'])  # take only bio1 and bio2 from bioclimatic rasters
    p_hfp_s = RasterPatchProvider(data_path + 'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif')  # take the human footprint 2009 summurized raster

    # create dataset
    dataset = PatchesDataset(occurrences=data_path + 'Presence_only_occurrences/Presences_only_train_sample.csv',
                             providers=[p_hfp_d, p_bioclim, p_hfp_s, p_rgb],
                             item_columns=['lat', 'lon', 'patchID'])
    dataset_multi = PatchesDatasetMultiLabel(occurrences=data_path + 'Presence_only_occurrences/Presences_only_train_sample.csv',
                                             providers=[p_hfp_d, p_bioclim, p_hfp_s, p_rgb],
                                             item_columns=['lat', 'lon', 'patchID'],
                                             id_getitem='patchID')

    # print random tensors from dataset
    ids = [random.randint(0, len(dataset) - 1) for i in range(5)]
    for i in ids:
        tensor, label = dataset[i]
        label_multi = dataset_multi[i][1]
        print(f'Tensor type: {type(tensor)}, tensor shape: {tensor.shape}, '
              f'label: {label}, \nlabel_multi: {label_multi}')
        dataset.plot_patch(i)


if __name__ == '__main__':
    main()
