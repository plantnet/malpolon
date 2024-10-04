"""Main script to run training or inference on torchgeo datasets.

This script runs the RasterSentinel2 dataset class by default.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@gmail.com>
"""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.torchgeo_datasets import (RasterBioclim,
                                                      RasterGeoDataModule,
                                                      RasterGeoSampler,
                                                      RasterTorchGeoDataset)
from malpolon.logging import Summary
from malpolon.models import ClassificationSystem

# class RasterPyrenees(RasterBioclim):
#     """Raster dataset adapted for Sentinel-2 data.

#     Inherits RasterTorchGeoDataset.
#     """
#     filename_glob = "bio_*.tif"
#     filename_regex = r"(?P<band>bio_[\d][\d])"
#     date_format = "%Y%m%dT%H%M%S"
#     is_image = True
#     separate_files = True
#     all_bands = ['bio_' + str(x) for x in range(1, 43)]
#     for k, v in enumerate(all_bands):
#         if len(v) <= 5:
#             all_bands[k] = v.replace('bio_', 'bio_0')
#     plot_bands = all_bands


class PyreneesDataModule(RasterGeoDataModule):
    def __init__(
        self,
        dataset_path: str,
        labels_name: str = 'labels.csv',
        train_batch_size: int = 32,
        inference_batch_size: int = 16,
        num_workers: int = 8,
        size: int = 200,
        units: str = 'pixel',
        crs: int = 4326,
        binary_positive_classes: list = [],
        task: str = 'classification_multiclass',  # ['classification_binary', 'classification_multiclass', 'classification_multilabel']
        dataset_kwargs: dict = {},
        **kwargs,
    ):
        """Class constructor."""
        super().__init__(dataset_path, labels_name, train_batch_size, inference_batch_size, num_workers, size, units, crs, binary_positive_classes, task, dataset_kwargs, **kwargs)

    def get_dataset(self, split, transform, **kwargs):
        dataset = RasterBioclim(
            self.dataset_path,
            labels_name=self.labels_name,
            split=split,
            task=self.task,
            binary_positive_classes=self.binary_positive_classes,
            transform=transform,
            **self.dataset_kwargs
        )
        return dataset

    @property
    def train_transform(self):
        pass

    @property
    def test_transform(self):
        pass


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_rgbnir_torchgeo_config")
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
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name="tensorboard_logs", version="")
    logger_tb.log_hyperparams(cfg)

    # Datamodule & Model
    datamodule = PyreneesDataModule(**cfg.data, **cfg.task)
    classif_system = ClassificationSystem(cfg.model, **cfg.optimizer, **cfg.task,
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
            every_n_train_steps=10,
        ),
    ]
    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, **cfg.trainer)

    # Run
    if cfg.run.predict:
        model_loaded = ClassificationSystem.load_from_checkpoint(classif_system.checkpoint_path,
                                                                 model=classif_system.model,
                                                                 hparams_preprocess=False)

        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           datamodule.get_test_dataset().unique_labels)
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='predictions_test_dataset', top_k=3, return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:1])

        # Option 2: Predict 1 data point (Pytorch)
        test_data = datamodule.get_test_dataset()
        query_point = {'lon': test_data.coordinates[0][0], 'lat': test_data.coordinates[0][1],
                       'crs': 4326,
                       'size': datamodule.size,
                       'units': datamodule.units,
                       'species_id': [test_data.targets[0]]}
        test_data_point = test_data[query_point][0]
        test_data_point = test_data_point.resize_(1, *test_data_point.shape)

        prediction = model_loaded.predict_point(cfg.run.checkpoint_path,
                                                test_data_point)
        preds, probas = datamodule.predict_logits_to_class(prediction,
                                                           datamodule.get_test_dataset().unique_labels)
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='prediction_point', single_point_query=query_point, return_csv=True)
        print('Point prediction : ', prediction.shape, prediction)
    else:
        trainer.fit(classif_system, datamodule=datamodule, ckpt_path=classif_system.checkpoint_path)
        trainer.validate(classif_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
