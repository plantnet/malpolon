## Commands
1. Pull GPN's habitat predictions from its raster and insert it in the input LUCAS CSV.
`python extract_labels_from_habitat_raster.py  -r GPN/v1/habitat.tif -i LUCAS_habitats/data/output/csv/LUCAS_metadata_labels_merged_S3-10%_extended_CBN-Med.csv -o out.csv --id_col "point_id"`

2. Merge the habitat\_code and name from the raster's habitat\_IDs
`python merge_gpn_habitats_to_lucas.py -i out.csv -g GPN/v1/geoplantnet_v1_habitat_eunis2020_metadata.csv -o LUCAS_habitats/data/output/csv/LUCAS_metadata_labels_merged_S3-10%_extended_CBN-Med_GPN-labels.csv`

## Other
Raw dataset: 21 002 089 rows (obs) with 1 175 186 sites, where each site is expanded for every species it contains. It contains 9290 species.
eva\_data\_known\_habitats\_collapsed\_species\_small: 559 115 rows, where 1 row = 1 site.

Unique values of n\_species per eva\_id:
array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
        27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
        40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
        53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
        66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
        79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
        92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
       105, 106, 107, 108, 110, 111, 112, 113, 114, 116, 117, 118, 121,
       123, 127, 131])

