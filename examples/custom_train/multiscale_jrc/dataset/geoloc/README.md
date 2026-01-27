Listing of geolocations (lon, lat) taken from other geo-located datasets go here for separate embedding.

## GLC24_PA
Statistics on the private test set of GLC24_PA.

Percentages are given relative to the raw file in 1st line.

|Version|n_plots <br>(`surveyId`)|n_obs|n_species <br>(`speciesId`)|Comments|
|-|-|-|-|-|
|glc24_pa_test_private|4 717|93 518|2 958||
|glc24_pa_test_private_CBN-Med|855 (~18.13%)|14 137 (~15.12%)|1 484 (~50.17%)||
|glc24_pa_test_private_CBN-med_matching-LUCAS-500m|64 (~1.36%)|1 252 (~1.34%)|458 (~15.49%)|1 row = 1 obs and `lucas_matching_ids` are strings of compatible lucas IDs|
|glc24_pa_test_private_CBN-med_matching-LUCAS-500m_exploded|64|1 252 (2 824 rows)|458|Each row with mutiple LUCAS IDs has been exploded to one row with a single LUCAS ID.|
|glc24_pa_test_private_CBN-med_matching-LUCAS-500m_exploded_merged|64|1 252 (16 880 rows)|458|Duplicated rows to integrate every possible file paths (N, S, E, W, C, P). Approx 6X more rows for the same nb of obs|
