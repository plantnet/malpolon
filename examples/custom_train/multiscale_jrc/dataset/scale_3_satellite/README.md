Satellite imagery (patches or rasters) go here.

The metadata file to be used is the same one as in GLC24_pre-extracted.

The file paths are determined by the values in `surveyId` through the Malpolon dataset.
The coordinates are retrieved through the same dataset class, by asking it to return the columns `['lat', 'lon']` as labels.

# Statistics
A contrastive learning scheme pairwaise doesn't care for speciesId because we only want to match a surveyId with an image. So there is no need to keep rows duplicates of surveyId in the obs file for the sake keeping speciesId.
In the JRC satellite dataset, we filter the dataset to only keep the 1st unique surveyId values' rows. Initially, `glc24_pa_train_CBN-med_surveyId_split-10.0%_train` has 65k rows, but only 4453 unique surveyId. With an average of ~15 rows per surveyId.
