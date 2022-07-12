from __future__ import annotations
from typing import TYPE_CHECKING

import logging

from pytorch_lightning.callbacks import Callback

if TYPE_CHECKING:
    from typing import Any
    import pytorch_lightning as pl
    from .models.standard_prediction_systems import GenericPredictionSystem


def str_object(obj: Any) -> str:
    """
    Formats an object to printing by returning a string containing the
    class name and attributes (both name and values)

    Parameters
    ----------
    obj: object to print.

    Returns
    -------
    str: string containing class name and attributes.
    """

    class_name = obj.__class__.__name__
    attributes = obj.__dict__

    filtered_attributes = []
    for key, val in attributes.items():
        # Test if its a private attribute
        if not key.startswith("_"):
            # Test if is not builtin type
            if hasattr(val, "__module__"):
                val = "<object>"
            filtered_attributes.append((key, val))

    formatted_attributes = ", ".join(
        map(lambda x: "{}={}".format(*x), filtered_attributes)
    )
    return "{}(\n    {}\n)".format(class_name, formatted_attributes)


class Summary(Callback):
    """
    FIXME handle multi validation data loaders, combined datasets
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("malpolon")

    def _log_data_loading_summary(self, data_loader, split: str) -> None:
        logger = self.logger

        if split == "Train":
            dataset = data_loader.dataset.datasets
        else:
            dataset = data_loader.dataset

        from torch.utils.data import Subset

        if isinstance(dataset, Subset):
            dataset = dataset.dataset

        logger.info("{} dataset: {}".format(split, dataset))
        logger.info("{} set size: {}".format(split, len(dataset)))

        if split == "Train" and hasattr(dataset, "n_classes"):
            logger.info("Number of classes: {}".format(dataset.n_classes))

        if hasattr(dataset, "transform"):
            logger.info("{} data transformations: {}".format(split, dataset.transform))

        if hasattr(dataset, "target_transform"):
            logger.info(
                "{} data target transformations: {}".format(
                    split, dataset.target_transform
                )
            )

        logger.info(
            "{} data sampler: {}".format(split, str_object(data_loader.sampler))
        )

        if hasattr(data_loader, "loaders"):
            batch_sampler = data_loader.loaders.batch_sampler
        else:
            batch_sampler = data_loader.batch_sampler
        logger.info(
            "{} data batch sampler: {}".format(split, str_object(batch_sampler))
        )

    def on_train_start(self, trainer: pl.Trainer, model: GenericPredictionSystem) -> None:
        logger = self.logger

        logger.info("\n# Model specification")
        logger.info(model.model)
        logger.info(model.loss)
        logger.info(model.optimizer)
        logger.info("Metrics: {}".format(model.metrics))

        logger.info("\n# Data loading information")
        logger.info("\n## Training data")
        self._log_data_loading_summary(trainer.train_dataloader, "Train")

        logger.info("\n## Validation data")
        self._log_data_loading_summary(trainer.val_dataloaders[0], "Validation")

        logger.info("\n# Strategy information")
        logger.info(trainer.strategy)
