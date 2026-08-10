## General

LUCAS photos (orthogonal & cover) go here.

The metadata file to be used should be a slice of the LUCAS exif file, over the `id` column.

The file paths are directly taken from columns `file_path_gisco_<view>` as the file structure is kept form the year of the survey and down.
GPS coordinates are registered in columns `['gps_long', 'gps_lat']`.

## France subset
- Train file: `lucas_harmo_cover_exif_fixed_gps_France-nuts0_expanded_essentials_exists_train-500m.csv`
- Tets file: `lucas_harmo_cover_exif_fixed_gps_France-nuts0_expanded_essentials_exists_test-500m.csv`

The subset was obtained by:
1. Manually slicing `lucas_harmo_cover_exif` over col `nuts0`
2. Running `pre-processing.py` with these methods:
```python
    df = pd.read_csv(df_fp)
    df = clean_bad_file_paths(df)
    df = cut_file_path_to_local_path(df)
    # df = gather_file_paths(df)
    df = expand(df)
    # df = check_existing_file_paths(df)
    df = keep_essentials_cols(df)
    df = rename_cols(df)
    df.to_csv(f'{fp_out}', index=False)
```
3. [Optional] Verifying existing files on disk: export col `file_path` to a txt file, run `check_file_exist_on_disk_multithread_simple_progressbar.sh`, remove the still missing and/or corrupted files based on `check_files_on_disk_MISSING_FILES_France-nuts0.txt` and `check_files_on_disk_SMALL_FILES_France-nuts0.txt`.
4. Spliting the data to train/test:
    4.1. Manually create a version of the dataframe with unique `point_id` (as there as are multiple duplicates because 1 row = set(point_id, year))
    4.2. Run `toolbox/split_obs_spatially.py` on that version
    4.3. Split the original dataframe from step 3. based on col `subset` of the unique-sites version.

## Filepaths formats
`lucas_photos_all_before_2022/`: <ID><view initial> (`26381956C.jpg`)
`lucas_photos_all_2022`: <ID><mission acronym>\_<view all letters> (`202240682432LCLU_North.jpg`, `202240682434GRASS_TransectAbove.jpg`)
`lucas_cover`: <ID>\_<Cover> (`LUCAS2015_38023110_Cover.jpg`)
`lucas_missing`: 
  - Before 2022: <ID><view initial> (`53162380N.jpg`)
  - 2022: <ID><mission acronym>\_<view all letters> (`202238002646LCLU_East.jpg`, `202238002646LCLU_West.jpg`)

## Cleaning
### GPS values
The **extent** of LUCAS surveys, as registered in metadata file lucas_harmo_cover_exif.csv  is so large that verde makes the memory explode while trying to create the raster on which to perform the split. Therefore the split cannot be made.
So we needed to split it in random spatial chunks first, apply the spatial split on them, and merge them back. BUT, if we randomize the survey points before chunkizing the file, we end up with an even spatial representation which doesn't help. HOWEVER, we can first sort the CSV by lon then lat so that the chunks are naturally small.
-> ACTUALLY: this is due to wrong values in th_long/th_lat which are not bound to [-180, 180]. They are either missing a comma, or in a different CRS.

In [21]: df[(df['th_long']>180) | (df['th_long']<-180) | (df['th_lat']>180) | (df['th_lat']<-180)][['th_long', 'th_lat', 'gps_long', 'gps_lat']]
Out[21]: 
         th_long     th_lat  gps_long   gps_lat
85198  1530000.0  4774000.0  15.03480  36.70606
85199  2186000.0  4398000.0  10.94428  42.78383
85200  2396000.0  3088000.0   5.33272  43.53012
85201  2422000.0  4512000.0  12.40444  44.88513
85202  2430000.0  4520000.0  12.49047  44.96730
85203  2434000.0  4518000.0  12.47960  44.97789
85204  2442000.0  4502000.0  12.29106  45.05623
85205  2448000.0  4502000.0  12.28795  45.12130
85206  2520000.0  4566000.0  13.14498  45.75114
85207  3182000.0  3916000.0   4.14189  51.59725
85208  3184000.0  3894000.0   3.82833  51.60393
85209  3186000.0  3918000.0   4.17128  51.63873

**CONCLUSION: for these indices, overwrite th_long/th_lat values with gps_long/gps_lat values**

Cleaning `lucas_harmo_cover_exif.csv` with bash, by removing rows if not at least 1 filepath exist among the 6 posisble views, would take approx 5 to 6 hours directly on nef-devel.

### Filepaths
Images in lucas_missing/2006, 2009 and 2018 are nested in a folder `Data/`.
However for 2018, the folder `Data/` exists among direct countries folders and there are some duplicates.
