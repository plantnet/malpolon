"""Main script to run training or inference on torchgeo datasets.

This script runs the RasterSentinel2 dataset class by default.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@gmail.com>
"""

from __future__ import annotations

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from malpolon.data.datasets.torchgeo_concat import ConcatTorchGeoDataModule
from malpolon.logging import Summary
from malpolon.models import ClassificationSystem


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_rgbnir_concat_config")
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
    datamodule = ConcatTorchGeoDataModule(**cfg.data, **cfg.task)
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
            every_n_train_steps=10,
            verbose=True
        ),
    ]
    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, **cfg.trainer)

    # Run
    if cfg.run.predict:
        model_loaded = ClassificationSystem.load_from_checkpoint(classif_system.checkpoint_path,
                                                                 model=classif_system.model,
                                                                 hparams_preprocess=False,
                                                                 weights_dir=log_dir)

        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           np.arange(cfg.data.num_classes))
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='predictions_test_dataset', return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:1])

        # Option 2: Predict 1 data point (Pytorch)
        test_data = datamodule.get_test_dataset()
        query_point = {'observation_id': test_data.observation_ids[0],
                       'lon': test_data.coordinates[0][0], 'lat': test_data.coordinates[0][1],
                       'crs': cfg.data.dataset_kwargs[0].kwargs.query_crs,
                       'species_id': [test_data[0][1].astype(int)]}
        test_data_point = test_data[0][0]
        test_data_point = test_data_point.resize_(1, *test_data_point.shape)

        prediction = model_loaded.predict_point(cfg.run.checkpoint_path,
                                                test_data_point)
        preds, probas = datamodule.predict_logits_to_class(prediction,
                                                           np.arange(cfg.data.num_classes))
        datamodule.export_predict_csv(preds, probas,
                                      out_dir=log_dir, out_name='prediction_point', single_point_query=query_point, return_csv=True)
        print('Point prediction : ', prediction.shape, prediction)
    else:
        trainer.fit(classif_system, datamodule=datamodule, ckpt_path=classif_system.checkpoint_path)
        trainer.validate(classif_system, datamodule=datamodule)


if __name__ == "__main__":
    main()
