"""Main script to run training or inference on torchgeo datasets.

This script runs the RasterSentinel2 dataset class by default.

Author: Theo Larcher <theo.larcher@inria.fr>
"""

from __future__ import annotations

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torchgeo.samplers import Units
from torchvision import transforms

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.geolifeclef2023 import (JpegPatchProvider,
                                                    PatchesDataset,
                                                    PatchesDatasetMultiLabel)
from malpolon.logging import Summary
from malpolon.models import ClassificationSystem
from malpolon.models.utils import CrashHandler


class Sentinel2PatchesDataModule(BaseDataModule):
    """Data module for Sentinel-2A dataset."""
    def __init__(
        self,
        dataset_path: str,
        labels_name: str = 'labels.csv',
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        size: int = 200,
        units: Units = Units.CRS,
        crs: int = 4326,
        binary_positive_classes: list = [],
        task: str = 'classification_multiclass',
        **kwargs,
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
             The dataset's unit system, must have a value in
             ['pixel', 'crs'], by default Units.CRS
        crs : int, optional
            `coordinate reference system (CRS)` to warp to
            (defaults to the CRS of the first file found), by default 4326
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
        self.sampler = None
        self.task = task
        self.binary_positive_classes = binary_positive_classes
        self.pin_memory = False

    def get_dataset(self, split, transform, target_transform=None, **kwargs):
        jpp_rgbnir = JpegPatchProvider(self.dataset_path + 'SatelliteImages/',
                                       dataset_stats='jpeg_patches_sample_stats.csv',
                                       id_getitem='patchID')  # 'dataset/jpeg_patches_sample_stats_bidon.csv')
        if 'multiclass' in self.task:
            dataset = PatchesDataset(
                occurrences=self.labels_name,
                providers=[jpp_rgbnir],
                transform=transform,
                target_transform=target_transform,
                item_columns=['lat', 'lon', 'patchID'],
                **kwargs
            )
        else:
            dataset = PatchesDatasetMultiLabel(
                occurrences=self.labels_name,
                providers=[jpp_rgbnir],
                transform=transform,
                target_transform=target_transform,
                id_getitem='patchID',
                item_columns=['lat', 'lon', 'patchID'],
                n_classes='max',
                **kwargs
            )
        dataset.coordinates = dataset.items[['lon', 'lat']].values
        return dataset

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
        return lambda x: x - 1

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.2],
                    std=[0.229, 0.224, 0.225, 0.2]
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
                    mean=[0.485, 0.456, 0.406, 0.2],
                    std=[0.229, 0.224, 0.225, 0.2]
                ),
            ]
        )


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_rgbnir_glc23_patches_train_multiclass.yaml")
def main(cfg: DictConfig) -> None:
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """
    # Loggers
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log_dir = log_dir.split(hydra.utils.get_original_cwd())[1][1:]  # Transforming absolute path to relative path
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name="tensorboard_logs", version="")
    logger_tb.log_hyperparams(cfg)

    # Datamodule & Model
    datamodule = Sentinel2PatchesDataModule(**cfg.data, **cfg.task)
    classif_system = ClassificationSystem(cfg.model, **cfg.optim, **cfg.task,
                                          checkpoint_path=cfg.run.checkpoint_path)

    # Lightning Trainer
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="checkpoint-{epoch:02d}-{step}-{" + f"{next(iter(classif_system.metrics.keys()))}/val" + ":.4f}",
            monitor=f"{next(iter(classif_system.metrics.keys()))}/val",
            mode="max",
            save_on_train_epoch_end=True,
            save_last=True,
        ),
    ]
    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, num_sanity_val_steps=0, **cfg.trainer)

    # Run
    if cfg.run.predict:
        model_loaded = ClassificationSystem.load_from_checkpoint(cfg.run.checkpoint_path,
                                                                 model=classif_system.model,
                                                                 hparams_preprocess=False,
                                                                 weights_dir=log_dir)

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
                       'species_id': [test_data_point[1]],
                       'crs': 4326}
        test_data_point = test_data_point[0].resize_(1, *test_data_point[0].shape)
        prediction = model_loaded.predict_point(cfg.run.checkpoint_path,
                                                test_data_point)
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
        CrashHandler(trainer)
        trainer.fit(classif_system, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
        trainer.validate(classif_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
