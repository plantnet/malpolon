"""Main script to run training or inference on Poverty Marbec Dataset.

This script will runs the Poverty dataset by default.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@gmail.com>
        Auguste Verdier <auguste.verdier@umontpellier.fr>
"""
# TODO : implement data ajustment for Poverty / Regression task
# TODO : CHECK WHY BAND 6 IS NANA AFTER JITTER


from __future__ import annotations

import os
import sys
from tqdm import tqdm
import json


# Force work with the malpolon github package localled at the root of the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

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
    for fold in range(5):
        print("Training fold ", fold + 1)

        log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        log_dir = os.path.join(log_dir, f"fold_{fold + 1}")
        logger_csv = pl.loggers.CSVLogger(log_dir, name="", version="")
        logger_csv.log_hyperparams(cfg)
        logger_tb = pl.loggers.TensorBoardLogger(log_dir, name="tensorboard_logs", version="")
        logger_tb.log_hyperparams(cfg)

        datamodule = PovertyDataModule(**cfg.data, **cfg.task, fold=fold + 1)
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
                             **cfg.trainer)  #

        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def calcul_mean_std(cfg: DictConfig) -> None:
    datamodule = PovertyDataModule(**cfg.data, **cfg.task)  #

    datamodule.setup()

    data_loader = datamodule.train_dataloader()

    mean = torch.zeros(7)
    std = torch.zeros(7)

    total_images_count = 0
    for images, _ in tqdm(data_loader):
        batch_images_count = images.size(0)
        images = images.view(batch_images_count, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_images_count

    mean /= total_images_count
    std /= total_images_count
    print(mean, std)
    json.dump({'mean': mean.tolist(), 'std': std.tolist()}, open('examples/poverty/mean_std_normalize.json', 'w'))


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def test(cfg: DictConfig) -> None:
    dataM = PovertyDataModule(**cfg.data, **cfg.task, fold=5)
    dataM.setup()

    model = RegressionSystem.load_from_checkpoint(
        checkpoint_path='outputs/cnn_on_ms_poverty/2024-08-29_17-14-16/fold_5/checkpoint-epoch=15-step=3160-regression_R2score/val=0.5259.ckpt')
    trainer = pl.Trainer(logger=False, log_every_n_steps=1, **cfg.trainer)
    trainer.test(model, datamodule=dataM)

    predictions = model.predict(dataM, trainer)
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    dataM.export_predict_csv(predictions, out_dir=log_dir, out_name=f'predictions_test_dataset_D', top_k=3,
                             return_csv=True)
    print('Test dataset prediction (extract) : ', predictions[:1])


if __name__ == "__main__":
    main()
