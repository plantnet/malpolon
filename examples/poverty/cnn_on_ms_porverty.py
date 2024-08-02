"""Main script to run training or inference on Poverty Marbec Dataset.

This script will runs the Poverty dataset by default.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@gmail.com>
        Auguste Verdier <auguste.verdier@umontpellier.fr>
"""
# TODO : implement data ajustment for Poverty / Regression task

from __future__ import annotations

import os
import sys
# Force work with the malpolon github package localled at the root of the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from malpolon.data.datasets import PovertyDataModule
from malpolon.logging import Summary
from malpolon.models import RegressionSystem,ClassificationSystem


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
    logger_csv = pl.loggers.CSVLogger(log_dir, name="", version="")
    logger_csv.log_hyperparams(cfg)
    logger_tb = pl.loggers.TensorBoardLogger(log_dir, name="tensorboard_logs", version="")
    logger_tb.log_hyperparams(cfg)

    datamodule = PovertyDataModule()#**cfg.data, **cfg.task
    model = RegressionSystem(cfg.model, **cfg.optimizer, **cfg.task)

    callbacks = [
        # Summary()#,
        # ModelCheckpoint(
        #     dirpath=log_dir,
        #     filename="checkpoint-{epoch:02d}-{step}-{" + f"{next(iter(model.metrics.keys()))}/val" + ":.4f}",
        #     monitor=f"{next(iter(model.metrics.keys()))}/val",
        #     mode="max",
        #     save_on_train_epoch_end=True,
        #     save_last=True,
        #     every_n_train_steps=10,
        # ),
    ]
    print(cfg.trainer)
    trainer = pl.Trainer(logger=[logger_csv, logger_tb],log_every_n_steps=1, **cfg.trainer)#, callbacks=callbacks
    # datamodule.setup()
    # batch_1 = next(iter(datamodule.train_dataloader()))
    # print(batch_1[0].shape)
    # pred = model(batch_1[0])

    # print(pred.shape)
    # print(batch_1[1].shape)
    # print(model)
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
    # trainer.validate(model, datamodule=datamodule)

@hydra.main(version_base="1.3", config_path="config", config_name="cnn_on_ms_torchgeo_config")
def test(cfg: DictConfig) -> None:


    
    # datamodule = PovertyDataModule()#**cfg.data, **cfg.task
    model = RegressionSystem(cfg.model, **cfg.optimizer, **cfg.task)

    x = torch.load('0.pt')
    y = torch.randn(x.size(0)).unsqueeze(-1)
    loss= model._step(split='val', batch=(x,y),batch_idx=0)
   
    # datamodule.setup()

    # data_loader = iter(datamodule.train_dataloader())

    

    # with torch.no_grad():

        
        # for i in range(100):

        #     # batch = next(data_loader)

        #     # input_containe_nan = torch.isnan(batch[0])
        #     print(batch[0])
        #     loss,score= model._step(split='predict', batch=batch, batch_idx=i)
        #     print(loss)
        #     print(score)
        
        
        


if __name__ == "__main__":
    main()
    # model = RegressionSystem(cfg.model, **cfg.optimizer, **cfg.task)