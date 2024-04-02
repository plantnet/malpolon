import pyproj

from malpolon.data.datasets.geolifeclef2023 import (
    JpegPatchProvider, MultipleRasterPatchProvider, PatchesDataset,
    PatchesDatasetMultiLabel, RasterPatchProvider)
from malpolon.data.datasets.torchgeo_sentinel2 import \
    Sentinel2TorchGeoDataModule


def main():
    data_path = "dataset/satellite_patches/"
    p_rgb = JpegPatchProvider(data_path + 'SatelliteImages/',
                              dataset_stats='jpeg_patches_sample_stats.csv',
                              id_getitem='patchID')  # take all sentinel imagery layer (r,g,b,nir)
    ds_s2_p = PatchesDataset(occurrences=data_path + 'Presence_only_occurrences/Presences_only_train_sample.csv',
                             providers=[p_rgb],
                             item_columns=['lat', 'lon', 'patchID'])

    crs_4326 = pyproj.CRS.from_epsg("4326")
    crs_32631 = pyproj.CRS.from_epsg("32631")
    transformer = pyproj.Transformer.from_crs(crs_4326, crs_32631, always_xy=True)

    montpellier_gps = (43.611285, 3.870814)
    montpellier_gps_32631 = transformer.transform(montpellier_gps[1], montpellier_gps[0])
    delta = 10000
    s2_ds = Sentinel2_custom(paths='dataset', crs=None, res=10, transforms=None, cache=True)
    bbox = BoundingBox(minx=montpellier_gps_32631[0]-delta,
                       maxx=montpellier_gps_32631[0]+delta,
                       miny=montpellier_gps_32631[1]-delta,
                       maxy=montpellier_gps_32631[1]+delta,
                       mint=0, maxt=0)
    sample = s2_ds[bbox]
    s2_ds.plot(sample)


if __name__ == "__main__":
    main()
