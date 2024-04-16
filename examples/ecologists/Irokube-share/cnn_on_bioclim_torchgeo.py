"""Main script to run training or inference on torchgeo datasets.

This script runs the RasterSentinel2 dataset class by default.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

from __future__ import annotations

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchgeo.samplers import GeoSampler, Units
from torchvision import transforms

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.geolifeclef2024 import (
    JpegPatchProvider, MultipleRasterPatchProvider, PatchesDataset,
    PatchesDatasetMultiLabel)
from malpolon.data.datasets.torchgeo_sentinel2 import (
    RasterSentinel2, Sentinel2GeoSampler, Sentinel2TorchGeoDataModule)
from malpolon.logging import Summary
from malpolon.models import ClassificationSystem
from malpolon.models.utils import CrashHandler

MEANS = [763.2619718472749,
         834.9697242928007,
         595.0890138994772,
         3247.0304854146166,
         2094.7135797725805,
         1442.4245984008298]
STDS = [418.6450262773753,
        314.494360753153,
        282.4347110246691,
        879.1942841719269,
        644.4186875914867,
        718.8127599088908]


class Sentinel2BelgiumTorchGeoDataModule(BaseDataModule):
    """Data module for Sentinel-2A dataset."""
    def __init__(
        self,
        dataset_path: str,
        labels_name: str = 'labels.csv',
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 0,
        size: int = 200,
        units: str = 'pixel',
        crs: int = 4326,
        binary_positive_classes: list = [],
        task: str = 'classification_multiclass',  # ['classification_binary', 'classification_multiclass', 'classification_multilabel']
        download_data_sample: bool = False
    ):
        """Class constructor.

        Parameters
        ----------
        dataset_path : str
            path to the directory containing the data
        labels_name : str, optional
            labels file name, by default 'labels.csv'
        train_batch_size : int, optional
            train batch size, by default 32
        inference_batch_size : int, optional
            inference batch size, by default 256
        num_workers : int, optional
            how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the
            main process, by default 8
        size : int, optional
            size of the 2D extracted patches. Patches can either be 
            square (int/float value) or rectangular (tuple of int/float).
            Defaults to a square of size 200, by default 200
        units : Units, optional
             The queries' unit system, must have a value in
             ['pixel', 'crs', 'm', 'meter', 'metre]. This sets the unit you want
             your query to be performed in, even if it doesn't match
             the dataset's unit system, by default Units.CRS
        crs : int, optional
            The queries' `coordinate reference system (CRS)`. This
            argument sets the CRS of the dataset's queries. The value
            should be equal to the CRS of your observations. It takes
            any EPSG integer code, by default 4326
        binary_positive_classes : list, optional
            labels' classes to consider valid in the case of binary
            classification with multi-class labels (defaults to all 0),
            by default []
        task : str, optional
            machine learning task (used to format labels accordingly),
            by default 'classification_multiclass'
        """
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.labels_name = labels_name
        self.size = size
        self.units = units
        self.crs = crs
        self.sampler = Sentinel2IrokubeGeoSampler
        self.task = task
        self.binary_positive_classes = binary_positive_classes
        if download_data_sample:
            self.download_data_sample()

    def get_dataset(self, split, transform, **kwargs):
        BelgiumRasterSentinel2 = RasterSentinel2
        BelgiumRasterSentinel2.filename_glob = "s2_*_2021_crop_belgium.tif"
        BelgiumRasterSentinel2.filename_regex = r"s2_(?P<band>blue|red|green|nir|swir1|swir2)_2021_crop_belgium"
        BelgiumRasterSentinel2.all_bands = ["red", "green", "blue", "nir", "swir1", "swir2"]
        BelgiumRasterSentinel2.plot_bands = ["red", "green", "blue"]

        dataset = BelgiumRasterSentinel2(
            self.dataset_path,
            labels_name=self.labels_name,
            split=split,
            task=self.task,
            binary_positive_classes=self.binary_positive_classes,
            transforms_data=transform,
            **kwargs
        )
        return dataset

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_train,
            sampler=self.sampler(self.dataset_train, size=self.size, units=self.units, crs=self.crs),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_val,
            sampler=self.sampler(self.dataset_val, size=self.size, units=self.units, crs=self.crs),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_test,
            sampler=self.sampler(self.dataset_test, size=self.size, units=self.units, crs=self.crs),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_predict,
            sampler=self.sampler(self.dataset_predict, size=self.size, units=self.units),
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return dataloader

    @property
    def target_transform(self):
        """Applying a transformation to the target.

        Labels in the PA/PO csv files range from 1 to num_classes. However
        PyTorch CELoss expects labels to range from 0 to num_classes - 1 so
        this transform makes to adjustment without modifying the label file.

        Returns
        -------
        Callable
            callable function to apply to the target
        """
        return lambda x: int(x) - 1

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=MEANS,
                    std=STDS
                ),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                transforms.CenterCrop(size=128),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=MEANS,
                    std=STDS
                ),
            ]
        )


class Sentinel2IrokubeGeoSampler(GeoSampler):
    """Custom sampler for RasterSentinel2.

    This custom sampler is used by RasterSentinel2 to query the dataset
    with the fully constructed dictionary. The sampler is passed to and
    used by PyTorch dataloaders in the training/inference workflow.

    Inherits GeoSampler.

    NOTE: this sampler is compatible with any class inheriting
          RasterTorchGeoDataset's `__getitem__` method so the name of
          this sampler may become irrelevant when more dataset-specific
          classes inheriting RasterTorchGeoDataset are created.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: Optional[int] = None,
        roi: Optional[BoundingBox] = None,
        units: Units = 'pixel',
        crs: str = 'crs',
    ) -> None:
        super().__init__(dataset, roi)
        self.units = units
        self.crs = crs
        self.size = (size, size) if isinstance(size, (int, float)) else size
        self.coordinates = dataset.coordinates
        self.observation_ids = dataset.observation_ids.values
        df = pd.DataFrame({'obs_id': self.observation_ids},
                          index=np.arange(len(self.observation_ids)))
        self.unique_obs_ids = df.drop_duplicates(subset='obs_id').index
        self.length = length if length is not None else len(self.unique_obs_ids)
        

    def __iter__(self) -> Iterator[BoundingBox]:
        """Yield a dict to iterate over a RasterTorchGeoDataset dataset.

        Yields
        ------
        Iterator[BoundingBox]
            dataset input query
        """
        for _ in self.unique_obs_ids:
            coords = tuple(self.coordinates[_])
            obs_id = self.observation_ids[_]
            yield {'lon': coords[0], 'lat': coords[1],
                   'crs': self.crs,
                   'size': self.size,
                   'units': self.units,
                   'obs_id': obs_id}

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length

@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_bioclim_torchgeo.yaml")
def main(cfg: DictConfig) -> None:
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name="tensorboard_logs", version="")
    logger_tb.log_hyperparams(cfg)

    datamodule = Sentinel2BelgiumTorchGeoDataModule(**cfg.data, **cfg.task)
    model = ClassificationSystem(cfg.model, **cfg.optimizer, **cfg.task)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="checkpoint-{epoch:02d}-{step}-{" + f"Loss/val" + ":.4f}",
            monitor=f"Loss/val",
            mode="min",
            save_on_train_epoch_end=True,
            save_last=True,
            every_n_train_steps=20,
        ),
    ]

    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, num_sanity_val_steps=0, **cfg.trainer)
    if cfg.run.predict:
        model_loaded = ClassificationSystem.load_from_checkpoint(cfg.run.checkpoint_path,
                                                                 model=model.model,
                                                                 hparams_preprocess=False)

        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           np.arange(0, max(datamodule.get_test_dataset().targets)+1))
        datamodule.export_predict_csv(preds,
                                      probas,
                                      out_name='predictions_test_dataset',
                                      out_dir=log_dir,
                                      top_k=100)
        print('Test dataset prediction (extract) : ', predictions[:1])

        # Option 2: Predict 1 data point (Pytorch)
        test_data = datamodule.get_test_dataset()
        test_data_point = test_data[0]
        query_point = {'lon': test_data.coordinates[0][0], 'lat': test_data.coordinates[0][1],
                       'species_id': test_data_point[1],
                       'crs': 4326}
        test_data_point = test_data_point[0].resize_(1, *test_data_point[0].shape)
        prediction = model_loaded.predict_point(cfg.run.checkpoint_path,
                                                test_data_point,
                                                ['model.', ''])
        preds, probas = datamodule.predict_logits_to_class(prediction,
                                                           np.arange(0, max(datamodule.get_test_dataset().targets)+1))
        datamodule.export_predict_csv(preds,
                                      probas,
                                      single_point_query=query_point,
                                      out_name='prediction_point',
                                      out_dir=log_dir,
                                      return_csv=True)
        print('Point prediction : ', prediction.shape, prediction)
    else:
#        CrashHandler(trainer)
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
        trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
