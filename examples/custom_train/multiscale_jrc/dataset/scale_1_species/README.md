Pl@ntNet & Gbif illustrations of individual plant species go here.

The metadata file to be used is the merge of the `multimedia.tsv` and `PN_gbif_France_2005-2025_illustrated.csv` extract, based on the column `gbifID`, filtered on the desired zone. The 1st file contains the urls to fetch images online; while the 2nd file contains the coordinates.

The URLs to the images are under the column `identifier`.
The file paths are formed by the `gbifID` as the images are dumped directly in the dataset folder.
The GPS coordinates are retrieved from columns `['decimalLongitude', 'decimalLatitude']`.
