#!/usr/bin/env python3

import argparse
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Point
from pathlib import Path


def sample_raster(input_csv, output_file, lon_col, lat_col, id_col, csv_crs, raster_path):
    # --- Load CSV ---
    df_complete = pd.read_csv(input_csv, low_memory=False)
    df = df_complete[[id_col, lon_col, lat_col]]
    df_complete = df_complete[df_complete.columns.difference([id_col, lon_col, lat_col])]

    # --- Convert to GeoDataFrame ---
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=csv_crs)

    # --- Open raster and sample ---
    with rasterio.open(raster_path) as src:
        # Reproject if needed
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        coords = [(geom.x, geom.y) for geom in gdf.geometry]

        nodata = src.nodata
        values = []
        for val in src.sample(coords):
            v = val[0]
            if nodata is not None and v == nodata:
                values.append(None)
            else:
                values.append(v)

        gdf["gpn_habitat_id"] = values
    gdf = gdf.dropna(subset=["gpn_habitat_id"])  # remove rows where sampling failed (e.g., outside raster bounds, in lakes etc...)
    gdf["gpn_habitat_id"] = gdf["gpn_habitat_id"].astype(np.int16)

    # if "18" in input_csv:
    #     cols = ['fid', 'point_id', 'survey_grass_gps_lon', 'survey_grass_gps_lat', 'survey_grass_eunis_habitat', 'gpn_habitat_id', 'survey_grass_gps_ew', 'distancetothloc', 'point_altitude', 'geometry']
    #     dtypes = [int, int, float, float, 'object', int, int, float, float]
    #     gdf = gdf[cols]
    #     gdf['survey_grass_eunis_habitat'] = gdf['survey_grass_eunis_habitat'].fillna('Unknown').astype('object')
    # elif "22" in input_csv:
    #     cols = ['point_id', 'point_long', 'point_lat', 'survey_grass_eunis_habitat_type', 'gpn_habitat_id', 'point_altitude', 'geometry']
    #     gdf = gdf[cols]
    #     gdf['survey_grass_eunis_habitat_type'] = gdf['survey_grass_eunis_habitat_type'].fillna('Unknown').astype('object')
    
    # --- Save output ---
    # gdf.to_file(f"{Path(output_file).stem}.gpkg", driver="GPKG")
    df_out = pd.DataFrame(gdf)
    df_out = pd.concat([df_out, df_complete], axis=1)
    df_out.to_csv(f'{Path(output_file).stem}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Sample raster values at point locations from a CSV file."
    )
    parser.add_argument("--raster", "-r",
                        required=True,
                        type=str,
                        help="Path to raster file")
    parser.add_argument("--input_csv", "-i",
                        nargs='?',
                        required=True,
                        type=str,
                        help="Input CSV file with coordinates")
    parser.add_argument("--output_name", "-o",
                        nargs='?',
                        required=True,
                        type=str,
                        help="Output file (CSV, Shapefile, GPKG, etc.)")
    parser.add_argument("--lon_col",
                        default="lon",
                        type=str,
                        required=False, help="Longitude column name")
    parser.add_argument("--lat_col",
                        default='lat',
                        type=str,
                        required=False, help="Latitude column name")
    parser.add_argument("--id_col",
                        default='id',
                        type=str,
                        required=False, help="Sample ID column name")
    parser.add_argument("--csv_crs",
                        default="EPSG:4326",
                        type=str,
                        required=False, help="CRS of input CSV (e.g., EPSG:4326)"
    )

    args = parser.parse_args()

    sample_raster(
        input_csv=args.input_csv,
        output_file=args.output_name,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        id_col=args.id_col,
        csv_crs=args.csv_crs,
        raster_path=args.raster,
    )


if __name__ == "__main__":
    main()
