<a name="readme-top"></a>

# Landsat rasters example (training)

This example performs regression of poverty  using a 2-mpa model on Landsat raster and socio-climatic data.

## Data

### Sample data

The sample data used in this example consists of:
- **Satellite images**: MS (Multi-Spectral) 7 bands Landsat7 image satellite. Numerous GeoTiff files of resolution XXXm  distributed across various countries, years and clusters.

- **Observations**: a CSV file containing all labels and correspondings informations necessary for matching the GeoTiffs. The CSV file contains the following columns:
  - `country`, (ex : angola, etc.)
  - `years`, (2013 to 2019)
  - `cluster`, cluster ID
  - `lat`, `lon`
  - `households`, XXXXXXXX
  - `wealthpooled`, the poverty indicator we want to regress
  - `urban_rural`, 0 or 1 for XXXXXXXXXX
  - `fold`, fold ID (for a cross validation)


### Data loading

- **Satellite images**

The LandSat tiles are looked for in the `example/poverty/dataset` directory and they are loaded based on  a `PovertyDataModule` and `MSDataset` (cf. [python file](datamodule/landsat_poverty.py) ).


```python 
  dm = PovertyDataModule("dataset/observation_2013+.csv", "dataset/landsat_7_less")
  dm.setup()
  dl = dm.train_dataloader()
```




