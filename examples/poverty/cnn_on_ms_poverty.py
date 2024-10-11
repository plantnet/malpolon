"""Main script to run training or inference on Poverty Marbec Dataset.

This script will run the Poverty dataset by default.

Author: Auguste Verdier <auguste.verdier@umontpellier.fr>
        Isabelle Mornard <isabelle.mornard@umontpellier.fr>
"""

from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

torch.set_float32_matmul_precision('medium')
from malpolon.data.datasets import PovertyDataModule
from malpolon.logging import Summary
from malpolon.models import RegressionSystem

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def main(cfg: DictConfig) -> None:
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """

    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log_dir = os.path.join(log_dir)
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name="tensorboard_logs", version="")
    logger_tb.log_hyperparams(cfg)

    datamodule = PovertyDataModule(**cfg.data, **cfg.task)
    model = RegressionSystem(cfg.model, **cfg.optimizer, **cfg.task)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="{epoch:02d}-{step}-{" + f"{next(iter(model.metrics.keys()))}_val" + ":.4f}",
            monitor=f"{next(iter(model.metrics.keys()))}_val",
            mode="max",
            save_on_train_epoch_end=True,
            save_last=True,
            every_n_train_steps=10,
        ),
        LearningRateMonitor()
    ]
    print(cfg.trainer)
    trainer = pl.Trainer(logger=[logger_csv, logger_tb], log_every_n_steps=1, callbacks=callbacks,
                         **cfg.trainer)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
