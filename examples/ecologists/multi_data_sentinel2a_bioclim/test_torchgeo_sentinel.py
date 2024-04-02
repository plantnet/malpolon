import pyproj
from matplotlib import pyplot as plt
from torchgeo.datasets import BoundingBox, Sentinel2


def main():
    class Sentinel2_custom(Sentinel2):
        filename_glob = "T*_B0*_10m.tif"
        filename_regex = r"T31TEJ_20190801T104031_(?P<band>B0[\d])"
        date_format = "%Y%m%dT%H%M%S"
        is_image = True
        separate_files = True
        all_bands = ["B02", "B03", "B04", "B08"]
        plot_bands = ["B04", "B03", "B02"]
        rgb_bands = ["B08", "B08", "B08"]

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
    print('Plotting sample for 3 seconds...')
    s2_ds.plot(sample, suptitle='Plotting sample for 3 seconds...')
    plt.pause(3)


if __name__ == "__main__":
    main()
