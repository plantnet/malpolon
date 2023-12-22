"""Main script to run training or inference on torchgeo datasets.

This script runs the RasterSentinel2 dataset class by default.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@inria.fr>
"""
from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from malpolon.data.datasets.torchgeo_sentinel2 import \
    Sentinel2TorchGeoDataModule
from malpolon.logging import Summary
from malpolon.models import ClassificationSystem


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_on_rgbnir_torchgeo_config")
def main(cfg: DictConfig) -> None:
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """
    # cfg.data.dataset_pathstate_dict = state_dict.model.state_dict = '../../../' + cfg.data.dataset_path  # Uncomment if value contains only the name of the dataset folder. Only works with a 3-folder-deep hydra job path.
    logger = pl.loggers.CSVLogger(".", name="", version="")
    logger.log_hyperparams(cfg)

    datamodule = Sentinel2TorchGeoDataModule(**cfg.data, **cfg.task)
    model = ClassificationSystem(cfg.model, **cfg.optimizer, **cfg.task)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}-{step}-{" + f"val_{next(iter(model.metrics.keys()))}" + ":.4f}",
            monitor=f"val_{next(iter(model.metrics.keys()))}",
            mode="max",
        ),
    ]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    model_loaded = ClassificationSystem.load_from_checkpoint(cfg.run.checkpoint_path,
                                                             model=model.model,
                                                             hparams_preprocess=False)

    if cfg.run.predict_type == 'test_dataset':
        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions)
        datamodule.export_predict_csv(preds, probas, out_name='predictions_test_dataset', return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:10])

    elif cfg.run.predict_type == 'test_point':
        # Option 2: Predict 1 data point (Pytorch)
        test_data = datamodule.get_test_dataset()
        query_point = {'lon': test_data.coordinates[0][0], 'lat': test_data.coordinates[0][1],
                        'crs': 4326,
                        'size': datamodule.size,
                        'units': datamodule.units}
        test_data_point = test_data[query_point][0]
        test_data_point = test_data_point.resize_(1, *test_data_point.shape)
        prediction = model_loaded.predict_point(cfg.run.checkpoint_path,
                                                test_data_point,
                                                ['model.', ''])
        preds, probas = datamodule.predict_logits_to_class(prediction)
        datamodule.export_predict_csv(preds, probas, single_point_query=query_point, out_name='prediction_point', return_csv=True)
        print('Point prediction : ', prediction.shape, prediction)


if __name__ == "__main__":
    main()
