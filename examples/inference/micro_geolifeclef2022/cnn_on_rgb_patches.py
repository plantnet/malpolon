"""Main script to run inference on microlifeclef2022 dataset.

Uses RGB pre-extracted patches from the dataset.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
        Theo Larcher <theo.larcher@inria.fr>
"""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from transforms import RGBDataTransform

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.geolifeclef2022 import MicroGeoLifeCLEF2022Dataset
from malpolon.logging import Summary
from malpolon.models import ClassificationSystem


class MicroGeoLifeCLEF2022DataModule(BaseDataModule):
    r"""
    Data module for MicroGeoLifeCLEF 2022.

    Parameters
    ----------
        dataset_path: Path to dataset
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation, testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        dataset_path: str,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        download: bool = True,
        task: str = 'classification_multiclass',
        **kwargs
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.download = download
        self.task = task

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                lambda data: RGBDataTransform()(data["rgb"]),
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                lambda data: RGBDataTransform()(data["rgb"]),
                transforms.CenterCrop(size=224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def prepare_data(self):
        MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            subset="train",
            use_rasters=False,
            download=self.download,
        )

    def get_dataset(self, split, transform, **kwargs):
        dataset = MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            split,
            patch_data=["rgb"],
            use_rasters=False,
            transform=transform,
            **kwargs
        )
        return dataset


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_rgb_patches_config")
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
    datamodule = MicroGeoLifeCLEF2022DataModule(**cfg.data)
    classif_system = ClassificationSystem(cfg.model, **cfg.optim, **cfg.task)
    model_loaded = ClassificationSystem.load_from_checkpoint(cfg.run.checkpoint_path,
                                                                model=classif_system.model,
                                                                hparams_preprocess=False,
                                                                weights_dir=log_dir)

    # Lightning Trainer
    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="checkpoint-{epoch:02d}-{step}-{" + f"{next(iter(classif_system.metrics.keys()))}/val" + ":.4f}",
            monitor=f"{next(iter(classif_system.metrics.keys()))}/val",
            mode="max",
        ),
    ]
    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, **cfg.trainer)

    # Run
    if cfg.run.predict_type == 'test_dataset':
        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           list(range(datamodule.dataset_test.n_classes)))
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name="predictions_test_dataset", top_k=3, return_csv=True)
        print("Test dataset prediction (extract) : ", predictions[:1])

    elif cfg.run.predict_type == 'test_point':
        test_data = datamodule.get_test_dataset()
        query_point = {'observation_id': test_data.observation_ids[0],
                       'lon': test_data.coordinates[0][0], 'lat': test_data.coordinates[0][1],
                       'crs': 4326,
                       'species_id': [test_data[0][1]]}
        test_data_point = test_data[0][0]
        test_data_point = test_data_point.resize_(1, *test_data_point.shape)

        prediction = model_loaded.predict_point(cfg.run.checkpoint_path,
                                                test_data_point)
        preds, probas = datamodule.predict_logits_to_class(prediction,
                                                           list(range(test_data.n_classes)))
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='prediction_point', single_point_query=query_point, return_csv=True)
        print('Point prediction : ', prediction.shape, prediction)


if __name__ == "__main__":
    main()
