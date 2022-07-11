from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

import pytorch_lightning as pl
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from typing import Any, Callable, Optional
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

    def get_train_dataset(self, test: bool) -> Dataset:
        dataset = self.get_dataset(
            split="train",
            transform=self.train_transform,
        )
        return dataset

    def get_test_dataset(self, test: bool) -> Dataset:
        split = "test" if test else "val"
        dataset = self.get_dataset(
            split=split,
            transform=self.test_transform,
        )
        return dataset

    # called for every GPU/machine
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.dataset_train = self.get_train_dataset(test=False)
            self.dataset_val = self.get_test_dataset(test=False)

        if stage == "test":
            self.dataset_test = self.get_test_dataset(test=True)

        if stage == "predict":
            self.dataset_predict = self.get_test_dataset(test=True)

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_val,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_predict,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader
