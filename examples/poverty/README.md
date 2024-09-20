<a name="readme-top"></a>

# Landsat rasters example (training)

This example performs regression of poverty  using a 2-mpa model on Landsat raster and socio-climatic data.

## Data

### Sample data

The sample data used in this example consists of:
- **Satellite images**: MS (Multi-Spectral) 7 bands Landsat8 image satellite. Numerous GeoTiff files of resolution XXXm  distributed across various countries, years and clusters.

- **Observations**: a CSV file containing all labels and corresponding data necessary for matching the GeoTiffs. The CSV file contains the following columns:
  - `country`, (ex : angola, etc.)
  - `year`, (2013 to 2019)
  - `cluster`, cluster ID
  - `lat` : latitude of the cluster,
  - `lon` : longitude of the cluster,
  - `households` : number of households in the cluster,
  - `wealthpooled`, the poverty indicator we want to regress
  - `urban_rural`, 0 if rural, 1 if urban
  - `fold`, fold ID (for a cross validation)


### Data loading

- **Satellite images**

The LandSat tiles are looked for in the `example/poverty/dataset` directory and they are loaded based on  a `PovertyDataModule` and `MSDataset` (cf. [python file](datamodule/landsat_poverty.py) ).


```script
python examples/poverty/cnn_on_ms_poverty.py
```