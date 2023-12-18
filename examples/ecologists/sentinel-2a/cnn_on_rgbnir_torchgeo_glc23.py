"""Main script to run training or inference on torchgeo datasets.

This script runs the RasterSentinel2 dataset class by default.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@inria.fr>
"""
from __future__ import annotations

import os
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from malpolon.data.datasets.torchgeo_sentinel2 import (
    RasterSentinel2, Sentinel2TorchGeoDataModule)
from malpolon.models import ClassificationSystem

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class TorchgeoGLC23DataModule(Sentinel2TorchGeoDataModule):
    """Torchgeo data module for GLC23 dataset"""
    def get_val_dataset(self) -> Dataset:
        dataset = self.get_dataset(
            split="test",
            transform=self.test_transform,
        )
        return dataset

    def get_dataset(self, split, transform, **kwargs):
        dataset = RasterSentinel2GLC23(
            self.dataset_path,
            labels_name=self.labels_name,
            split=split,
            task=self.task,
            binary_positive_classes=self.binary_positive_classes,
            transforms_data=transform,
            **kwargs
        )
        return dataset


class RasterSentinel2GLC23(RasterSentinel2):
    """Adaptation of RasterSentinel2 for new GLC23 observations"""
    filename_glob = "*.tif"
    filename_regex = r"(?P<band>red|green|blue|nir)_2021"
    all_bands = ["red", "green", "blue", "nir"]
    plot_bands = ["red", "green", "blue"]

    def _load_observation_data(
        self,
        root: Path = None,
        obs_fn: str = None,
        subsets: str = ['train', 'test', 'val'],
    ) -> pd.DataFrame:
        if any([root is None, obs_fn is None]):
            return pd.DataFrame(columns=['longitude', 'latitude', 'species_id', 'subset'])
        labels_fp = obs_fn if len(obs_fn.split('.csv')) >= 2 else f'{obs_fn}.csv'
        labels_fp = root / labels_fp
        df = pd.read_csv(
            labels_fp,
            sep=";",
        )
        df.rename(columns={'lon': 'longitude',
                           'lat': 'latitude',
                           'speciesId': 'species_id',
                           'glcID': 'observation_id'}, inplace=True)
        df['subset'] = df['test'].apply(lambda x: 'test' if x else 'train')

        train_inds = np.argwhere(df['subset'] == 'train')
        np.random.shuffle(train_inds)
        val_inds = train_inds[:ceil(len(train_inds) * 0.2)]
        df.loc[np.ravel(val_inds), 'subset'] = 'val'
        self.unique_labels = np.sort(np.unique(df['species_id']))
        try:
            subsets = [subsets] if isinstance(subsets, str) else subsets
            ind = df.index[df["subset"].isin(subsets)]
            df = df.loc[ind]
        except ValueError as e:
            print('Unrecognized subset name.\n'
                  'Please use one or several amongst: ["train", "test", "val"], as a string or list of strings.\n',
                  {e})
        return df


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_on_rgbnir_torchgeo_glc23_config")
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

    datamodule = TorchgeoGLC23DataModule(**cfg.data, **cfg.task)
    model = ClassificationSystem(cfg.model, **cfg.optimizer, **cfg.task)

    callbacks = [
        # Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}-{step}-{" + f"val_{next(iter(model.metrics.keys()))}" + ":.4f}",
            monitor=f"val_{next(iter(model.metrics.keys()))}",
            mode="max",
            every_n_train_steps=1,  # Careful, if > than the actual number steps taken, no checkpoint will be saved
            save_on_train_epoch_end=True,
            save_last=True,
        ),
    ]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    if cfg.run.predict:
        model_loaded = ClassificationSystem.load_from_checkpoint(cfg.run.checkpoint_path,
                                                                 model=model.model,
                                                                 hparams_preprocess=False)

        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        preds, probas = datamodule.predict_logits_to_class(predictions)
        datamodule.export_predict_csv(preds, probas, out_name='predictions_test_dataset', return_csv=True)
        print('Test dataset prediction (extract) : ', predictions[:10])

        # Option 2: Predict 1 data point (Pytorch)
        test_data = datamodule.get_test_dataset()
        query_point = {'lon': test_data.coordinates[0][0], 'lat': test_data.coordinates[0][1],
                       'crs': datamodule.crs,
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
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
        trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
