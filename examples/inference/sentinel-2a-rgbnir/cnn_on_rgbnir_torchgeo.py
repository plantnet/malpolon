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

from malpolon.data.datasets.torchgeo_sentinel2 import \
    Sentinel2TorchGeoDataModule
from malpolon.logging import Summary
from malpolon.models import ClassificationSystem


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
    log_dir = log_dir.split(hydra.utils.get_original_cwd())[1][1:]  # Transforming absolute path to relative path
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name="tensorboard_logs", version="")
    logger_tb.log_hyperparams(cfg)

    # Datamodule & Model
    datamodule = Sentinel2TorchGeoDataModule(**cfg.data, **cfg.task)
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
        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           datamodule.get_test_dataset().unique_labels)
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='predictions_test_dataset', top_k=3, return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:1])

    elif cfg.run.predict_type == 'test_point':
        # Option 2: Predict 1 data point (Pytorch)
        test_data = datamodule.get_test_dataset()
        query_point = {'lon': test_data.coordinates[0][0], 'lat': test_data.coordinates[0][1],
                       'crs': 4326,
                       'size': datamodule.size,
                       'units': datamodule.units,
                       'species_id': [test_data.targets[0]]}  # Adjust value in case of multilabel classification
        test_data_point = test_data[query_point][0]
        test_data_point = test_data_point.resize_(1, *test_data_point.shape)

        prediction = model_loaded.predict_point(cfg.run.checkpoint_path,
                                                test_data_point)
        preds, probas = datamodule.predict_logits_to_class(prediction,
                                                           datamodule.get_test_dataset().unique_labels)
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='prediction_point', single_point_query=query_point, return_csv=True)
        print('Point prediction : ', prediction.shape, prediction)


if __name__ == "__main__":
    main()
