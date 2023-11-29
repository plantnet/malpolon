from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Union

    from torch.utils.data import Dataset


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers

        # TODO check if uses GPU or not before using pin memory
        self.pin_memory = True

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_predict = None
        self.sampler = None

    @property
    @abstractmethod
    def train_transform(self) -> Callable:
        pass

    @property
    @abstractmethod
    def test_transform(self) -> Callable:
        pass

    @abstractmethod
    def get_dataset(self, split: str, transform: Callable, **kwargs: Any) -> Dataset:
        pass

    def get_train_dataset(self) -> Dataset:
        dataset = self.get_dataset(
            split="train",
            transform=self.train_transform,
        )
        return dataset

    def get_val_dataset(self) -> Dataset:
        dataset = self.get_dataset(
            split="val",
            transform=self.test_transform,
        )
        return dataset

    def get_test_dataset(self) -> Dataset:
        dataset = self.get_dataset(
            split="test",
            transform=self.test_transform,
        )
        return dataset

    # called for every GPU/machine
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.dataset_train = self.get_train_dataset()
            self.dataset_val = self.get_val_dataset()

        if stage == "test":
            self.dataset_test = self.get_test_dataset()

        if stage == "predict":
            self.dataset_predict = self.get_test_dataset()

    def prepare_data(self) -> None:
        """Called once on CPU. Class states defined here are lost afterwards.
        This method is intended for data downloading, tokenization,
        permanent transformation..."""

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_train,
            sampler=self.sampler,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_val,
            sampler=self.sampler,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_test,
            sampler=self.sampler,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_predict,
            sampler=self.sampler,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def predict_logits_to_class(self, predictions: Union[np.ndarray, Tensor]) -> Tensor:
        device = torch.device("cpu")
        activation = torch.nn.Softmax(dim=1)
        probas = activation(predictions)
        probas, indices = torch.sort(probas, descending=True)
        probas, indices = probas.to(device), indices.to(device)
        predictions = self.get_test_dataset().unique_labels[indices]
        return predictions, probas

    def export_predict_csv(self,
                           predictions,
                           probas=None,
                           single_point_query: dict = None,
                           out_name: str = "predictions",
                           return_csv: bool = False,
                           **kwargs: Any) -> Any:
        out_name = out_name + ".csv" if not out_name.endswith(".csv") else out_name
        fp = Path('./') / Path(out_name)
        if single_point_query:
            df = pd.DataFrame({'observation_id': [single_point_query['observation_id'] if 'observation_id' in single_point_query else None],
                               'lon': [single_point_query['lon']],
                               'lat': [single_point_query['lat']],
                               'crs': [single_point_query['crs']],
                               'target_species_id': [single_point_query['species_id'] if 'species_id' in single_point_query else None],
                               'predictions': predictions[:, 0],
                               'probas': probas[:, 0]})
        else:
            test_ds = self.get_test_dataset()
            df = pd.DataFrame({'observation_id': test_ds.observation_ids,
                               'lon': test_ds.coordinates[:, 0],
                               'lat': test_ds.coordinates[:, 1],
                               'target_species_id': test_ds.targets,
                               'predictions': predictions[:, 0],
                               'probas': [None] * len(predictions)})
        if probas is not None:
            df['probas'] = probas[:, 0]
        df.to_csv(fp, index=False, sep=',', **kwargs)
        if return_csv:
            return df
