LUCAS photos (orthogonal & cover) go here.

The metadata file to be used should be a slice of the LUCAS exif file, over the `id` column.

The file paths are directly taken from columns `file_path_gisco_<view>` as the file structure is kept form the year of the survey and down.
GPS coordinates are registered in columns `['gps_long', 'gps_lat']`.


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
