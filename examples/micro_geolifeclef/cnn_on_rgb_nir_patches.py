from __future__ import annotations

import os

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchmetrics.functional as Fmetrics
import warnings
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from transforms import NIRDataTransform, RGBDataTransform

from malpolon.data.data_module import BaseDataModule
from malpolon.data.datasets.geolifeclef2022 import MicroGeoLifeCLEF2022Dataset
from malpolon.logging import Summary
from malpolon.models import FinetuningClassificationSystem

FMETRICS_CALLABLES = {'binary_accuracy': Fmetrics.classification.binary_accuracy,
                      'multiclass_accuracy': Fmetrics.classification.multiclass_accuracy,
                      'multilabel_accuracy': Fmetrics.classification.multilabel_accuracy, }


class RGBNIRDataPreprocessing:
    def __call__(self, data):
        rgb, nir = data["rgb"], data["near_ir"]
        rgb = RGBDataTransform()(rgb)
        nir = NIRDataTransform()(nir)[[0]]
        return torch.concat((rgb, nir))


class MicroGeoLifeCLEF2022DataModule(BaseDataModule):
    r"""
    Data module for MicroGeoLifeCLEF 2022.

    Parameters
    ----------
        dataset_path: Path to dataset
        train_batch_size: Size of batch for training
        inference_batch_size: Size of batch for inference (validation,
                              testing, prediction)
        num_workers: Number of workers to use for data loading
    """
    def __init__(
        self,
        dataset_path: str,
        train_batch_size: int = 32,
        inference_batch_size: int = 256,
        num_workers: int = 8,
        download: bool = True,
    ):
        super().__init__(train_batch_size, inference_batch_size, num_workers)
        self.dataset_path = dataset_path
        self.download = download

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                RGBNIRDataPreprocessing(),
                transforms.RandomRotation(degrees=45, fill=1),
                transforms.RandomCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485],
                    std=[0.229, 0.224, 0.225, 0.229],
                ),
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                RGBNIRDataPreprocessing(),
                transforms.CenterCrop(size=224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.485],
                    std=[0.229, 0.224, 0.225, 0.229],
                ),
            ]
        )

    def prepare_data(self):
        # download, split, etc...
        MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            subset="train",
            use_rasters=False,
            download=self.download,
        )

    def get_dataset(self, split, transform, **kwargs):
        dataset = MicroGeoLifeCLEF2022Dataset(
            self.dataset_path,
            split,
            patch_data=["rgb", "near_ir"],
            use_rasters=False,
            transform=transform,
            **kwargs
        )
        return dataset


class NewConvolutionalLayerInitFuncStrategy:
    def __init__(self, strategy, rescaling=False):
        self.strategy = strategy
        self.rescaling = rescaling

    def __call__(self, old_layer, new_layer):
        with torch.no_grad():
            if self.strategy == "random_init":
                new_layer.weight[:, :3] = old_layer.weight
            elif self.strategy == "red_pretraining":
                new_layer.weight[:] = old_layer.weight[:, [0, 1, 2, 0]]

            if self.rescaling:
                new_layer.weight *= 3 / 4

            if hasattr(new_layer, "bias"):
                new_layer.bias = old_layer.bias

class ClassificationSystem(FinetuningClassificationSystem):
    """Classification task class."""
    def __init__(
        self,
        model: dict,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        metrics: dict,
        task: str = 'classification_multiclass',
        hparams_preprocess: bool = True,
    ):
        """Class constructor.

        Parameters
        ----------
        model : dict
            _description_
        lr : float
            learning rate
        weight_decay : float
            weight decay
        momentum : float
            value of momentum
        nesterov : bool
            if True, uses Nesterov's momentum
        metrics : dict
            dictionnary containing the metrics to compute.
            Keys must match metrics' names and have a subkey with each
            metric's functional methods as value. This subkey is either
            created from the FMETRICS_CALLABLES constant or supplied,
            by the user directly.
        task : str, optional
            machine learning task (used to format labels accordingly),
            by default 'classification_multiclass'
        hparams_preprocess : bool, optional
            if True performs preprocessing operations on the hyperparameters,
            by default True
        """
        if hparams_preprocess:
            task = task.split('classification_')[1]
            try:
                metrics = omegaconf.OmegaConf.to_container(metrics)
                for k, v in metrics.items():
                    if 'callable' in v:
                        metrics[k]['callable'] = eval(v['callable'])
                    else:
                        metrics[k]['callable'] = FMETRICS_CALLABLES[k]
            except ValueError as e:
                print('\n[WARNING]: Please make sure you have registered'
                      ' a dict-like value to your "metrics" key in your'
                      ' config file. Defaulting metrics to None.\n')
                print(e, '\n')
                metrics = None
            except KeyError as e:
                print('\n[WARNING]: Please make sure the name of your metrics'
                      ' registered in your config file match an entry'
                      ' in constant FMETRICS_CALLABLES.'
                      ' Defaulting metrics to None.\n')
                print(e, '\n')
                metrics = None

        super().__init__(
            model,
            lr,
            weight_decay,
            momentum,
            nesterov,
            metrics,
            task,
        )


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_on_rgb_nir_patches_config")
def main(cfg: DictConfig) -> None:
    """Run main script used for either training or inference.

    Parameters
    ----------
    cfg : DictConfig
        hydra config dictionary created from the .yaml config file
        associated with this script.
    """
    # cfg.data.dataset_path = '../../../' + cfg.data.dataset_path  # Uncomment if value contains only the name of the dataset folder. Only works with a 3-folder-deep hydra job path.
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)

    datamodule = MicroGeoLifeCLEF2022DataModule(**cfg.data)

    cfg_model = hydra.utils.instantiate(cfg.model)
    model = ClassificationSystem(cfg_model, **cfg.optimizer, **cfg.task)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}-{step}-{" + f"val_{next(iter(model.metrics.keys()))}" + ":.4f}",
            monitor=f"val_{next(iter(model.metrics.keys()))}",
            mode="max",
        ),
    ]
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    if cfg.run.predict:
        model_loaded = ClassificationSystem.load_from_checkpoint(cfg.run.checkpoint_path,
                                                                 model=model.model,
                                                                 hparams_preprocess=False)

        # Option 1: Predict on the entire test dataset (Pytorch Lightning)
        predictions = model_loaded.predict(datamodule, trainer)
        print('Test dataset prediction (extract) : ', predictions[:10])

        # Option 2: Predict 1 data point (Pytorch)
        test_data = datamodule.get_test_dataset()
        test_data_point = test_data[0][0]
        test_data_point = test_data_point.resize_(1, *test_data_point.shape)
        prediction = model_loaded.predict_point(cfg.run.checkpoint_path,
                                                test_data_point,
                                                ['model.', ''])
        print('Point prediction : ', prediction)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.run.checkpoint_path)
        trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
