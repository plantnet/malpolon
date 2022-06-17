from abc import ABC, abstractmethod

import pytorch_lightning as pl
from torch.utils.data import DataLoader


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
    def train_transform(self):
        pass

    @property
    @abstractmethod
    def test_transform(self):
        pass

    @abstractmethod
    def get_dataset(self, train, transform, **kwargs):
        pass

    def get_train_dataset(self, test):
        dataset = self.get_dataset(
            split="train",
            transform=self.train_transform,
        )
        return dataset

    def get_test_dataset(self, test):
        split = "test" if test else "val"
        dataset = self.get_dataset(
            split=split,
            transform=self.test_transform,
        )
        return dataset

    # called for every GPU/machine
    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.dataset_train = self.get_train_dataset(test=False)
            self.dataset_val = self.get_test_dataset(test=False)

        if stage == "test":
            self.dataset_test = self.get_test_dataset(test=False)

        if stage == "predict":
            self.dataset_test = self.get_test_dataset(test=True)

    def train_dataloader(self):
        dataloader = DataLoader(
            self.dataset_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.dataset_val,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.dataset_test,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(
            self.dataset_predict,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader
