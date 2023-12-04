"""This module provides a base class for data modules.

Author: Theo Larcher <theo.larcher@inria.fr>
        Titouan Lorieul <titouan.lorieul@inria.fr>

"""

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
    """Base class for data modules.

    This class inherits pytorchlightining's LightningDataModule class
    and provides a base class for data modules by re-defining steps
    methods as well as adding new data manipulation methods.
    """
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
        """Return train data transforms.

        Returns
        -------
        Callable
            train data transforms
        """

    @property
    @abstractmethod
    def test_transform(self) -> Callable:
        """Return test data transforms.

        Returns
        -------
        Callable
            test data transforms
        """

    @abstractmethod
    def get_dataset(self, split: str, transform: Callable, **kwargs: Any) -> Dataset:
        """Return the dataset corresponding to the split.

        Parameters
        ----------
        split : str
            Type of dataset. Values must be on of ["train", "val",
            "test"]
        transform : Callable
            data transforms to apply when loading the dataset

        Returns
        -------
        Dataset
            dataset corresponding to the split
        """

    def get_train_dataset(self) -> Dataset:
        """Call self.get_dataset to return the train dataset.

        Returns
        -------
        Dataset
            train dataset
        """
        dataset = self.get_dataset(
            split="train",
            transform=self.train_transform,
        )
        return dataset

    def get_val_dataset(self) -> Dataset:
        """Call self.get_dataset to return the validation dataset.

        Returns
        -------
        Dataset
            validation dataset
        """
        dataset = self.get_dataset(
            split="val",
            transform=self.test_transform,
        )
        return dataset

    def get_test_dataset(self) -> Dataset:
        """Call self.get_dataset to return the test dataset.

        Returns
        -------
        Dataset
            test dataset
        """
        dataset = self.get_dataset(
            split="test",
            transform=self.test_transform,
        )
        return dataset

    # called for every GPU/machine
    def setup(self, stage: Optional[str] = None) -> None:
        """Register the correct datasets to the class attributes.

        Depending on the trainer's stage, this method will retrieve
        the train, val or test dataset and register it as a class
        attribute. The "predict" stage calls for the test dataset.

        Parameters
        ----------
        stage : Optional[str], optional
            trainer's stage, by default None (train)
        """
        if stage in (None, "fit"):
            self.dataset_train = self.get_train_dataset()
            self.dataset_val = self.get_val_dataset()

        if stage == "test":
            self.dataset_test = self.get_test_dataset()

        if stage == "predict":
            self.dataset_predict = self.get_test_dataset()

    def prepare_data(self) -> None:
        """Prepare data.

        Called once on CPU. Class states defined here are lost afterwards.
        This method is intended for data downloading, tokenization,
        permanent transformation...
        """

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader instantiated with class attributes.

        Returns
        -------
        DataLoader
            train dataloader
        """
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
        """Return validation dataloader instantiated with class attributes.

        Returns
        -------
        DataLoader
            Validation dataloader
        """
        dataloader = DataLoader(
            self.dataset_val,
            sampler=self.sampler,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader instantiated with class attributes.

        Returns
        -------
        DataLoader
            test dataloader
        """
        dataloader = DataLoader(
            self.dataset_test,
            sampler=self.sampler,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        """Return predict dataloader instantiated with class attributes.

        Returns
        -------
        DataLoader
            predict dataloader
        """
        dataloader = DataLoader(
            self.dataset_predict,
            sampler=self.sampler,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def predict_logits_to_class(self,
                                predictions: Union[np.ndarray, Tensor],
                                activation_fn: torch.nn.modules.activation = torch.nn.Softmax(dim=1)) -> Tensor:
        """Convert the model's predictions to class labels.

        This method applies an activation function to the model's
        predictions and returns the corresponding class labels.

        Parameters
        ----------
        predictions : Union[np.ndarray, Tensor]
            model's predictions (raw logits), by default Softmax(dim=1)

        Returns
        -------
        tuple[Union[np.ndarray, Tensor], Tensor]
            _description_
        """
        device = torch.device("cpu")
        probas = activation_fn(predictions)
        probas, indices = torch.sort(probas, descending=True)
        probas, indices = probas.to(device), indices.to(device)
        predictions = self.get_test_dataset().unique_labels[indices]
        return predictions, probas

    def export_predict_csv(self,
                           predictions: np.ndarray,
                           probas: Union[Tensor, np.ndarray] = None,
                           single_point_query: dict = None,
                           out_name: str = "predictions",
                           return_csv: bool = False,
                           **kwargs: Any) -> Any:
        """Export predictions to csv file.

        This method is used to export predictions to a csv file.
        It can be used with a single point query or with the whole
        test dataset.
        This method is adapted for a classification task with an
        observations file and multi-modal data.
        Keys in the csv file match the ones used to inistantiate the
        `RasterTorchGeoDataset` class, that is to say :
        `observation_id`, `lon`, `lat`, `target_species_id`. The `crs`
        key is also mandatory in the case of singl-point query.

        Parameters
        ----------
        predictions : np.ndarray
            model's predictions.
        probas : Union[Tensor, np.ndarray], optional
            predictions' raw logits or logits passed through an
            activation function, by default None
        single_point_query : dict, optional
            query dictionnary of the single-point prediction, by default
            None
        out_name : str, optional
            output CSV file name, by default "predictions"
        return_csv : bool, optional
            if true, the method returns the CSV as a pandas DataFrame,
            by default False

        Returns
        -------
        pandas.DataFrame
            CSV content as a pandas DataFrame if `return_csv` is True
        """
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
        return None
