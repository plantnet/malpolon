"""Main script to run training or inference on torchgeo datasets.

This script runs the RasterSentinel2 dataset class by default.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@gmail.com>
"""

from __future__ import annotations

import os
from typing import Any, Callable
import hydra
import pandas as pd
from torch import tensor
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from malpolon.logging import Summary
from malpolon.models import ClassificationSystem
from torch.utils.data import Dataset, DataLoader
from malpolon.data import BaseDataModule
from torchvision.datasets import CIFAR10
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from malpolon.models.utils import check_metric


class Cifar10Datamodule(BaseDataModule):
    """DataModule for the CIFAR-10 dataset.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the CIFAR-10 dataset is stored.
    batch_size : int
        Size of the mini-batches.
    num_workers : int
        Number of workers to use for loading data.
    """

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.__dict__.update(kwargs)


    def prepare_data(self) -> None:
        """Download the CIFAR-10 dataset."""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        """Set up the CIFAR-10 dataset."""
        cifar10_train = CIFAR10(self.data_dir, train=True, transform=ToTensor())
        train_size = int(0.8 * len(cifar10_train))
        val_size = len(cifar10_train) - train_size
        cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10_train, [train_size, val_size])
        self.cifar10_train = cifar10_train
        self.cifar10_val = cifar10_val
        self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=ToTensor())
    
    def get_dataset(self, split: str, transform: Callable, **kwargs: Any) -> Dataset:
        if split == 'train':
            return self.cifar10_train
        elif split == 'val':
            return self.cifar10_val
        elif split == 'test':
            return self.cifar10_test
        
    def train_transform(self, **kwargs: Any) -> Callable:
        pass

    def test_transform(self, **kwargs: Any) -> Callable:
        pass

    def train_dataloader(self) -> DataLoader:
        """Return the CIFAR-10 training dataloader."""
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Return the CIFAR-10 validation dataloader."""
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the CIFAR-10 test dataloader."""
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


@hydra.main(version_base="1.3", config_path="config", config_name="cnn_cifar10")
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

    datamodule = Cifar10Datamodule(**cfg.data, **cfg.task)
    model = ClassificationSystem(cfg.model, **cfg.optimizer, **cfg.task)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=log_dir,
            filename="checkpoint-{epoch:02d}-{step}-{" + f"loss/val" + ":.4f}",
            monitor=f"loss/val",
            mode="min",
            save_on_train_epoch_end=True,
            save_last=True,
            every_n_train_steps=300,
        ),
    ]
    trainer = pl.Trainer(logger=[logger_csv, logger_tb], callbacks=callbacks, **cfg.trainer)

    if cfg.run.predict:
        model_loaded = ClassificationSystem.load_from_checkpoint(cfg.run.checkpoint_path,
                                                                 model=model.model,
                                                                 hparams_preprocess=False)

        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions,
                                                           list(datamodule.get_test_dataset().class_to_idx.values()))
        preds_test = datamodule.export_predict_csv_basic(preds, datamodule.get_test_dataset().targets, probas, np.arange(len(preds)),
                                                         out_dir=log_dir, out_name='predictions_test_dataset', top_k=3, return_csv=True)
        metrics = check_metric(cfg.optimizer.metrics)
        scores = {'metric_name': [], 'score': []}
        for metric_name, metric_func in metrics.items():
            res = metric_func['callable'](tensor(preds_test['predictions'].astype(int)),
                                          tensor(preds_test['targets'].astype(int)),
                                          **metric_func['kwargs'])
            scores['metric_name'].append(metric_name)
            scores['score'].append(res.item())
        df = pd.DataFrame(scores)
        df.to_csv(os.path.join(log_dir, 'scores_test_dataset.csv'), index=False)
        print('Test dataset prediction (extract) : ', predictions[:1])
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
        trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
