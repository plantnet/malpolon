"""This module provides classes wrapping pytorchlightning training modules.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
        Theo Larcher <theo.larcher@inria.fr>
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
import torchmetrics.functional as Fmetrics
from torchvision.datasets.utils import (download_and_extract_archive,
                                        download_url)

from malpolon.models.utils import check_metric

from .utils import check_loss, check_model, check_optimizer, check_scheduler

if TYPE_CHECKING:
    from typing import Any, Callable, Mapping, Optional, Union

    from torch import Tensor


class GenericPredictionSystem(pl.LightningModule):
    """Generic prediction system providing standard methods.


    """

    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        loss: torch.nn.modules.loss._Loss,
        optimizer: Union[torch.optim.Optimizer, Mapping],
        scheduler: Union[torch.optim.Optimizer] = None,
        metrics: Optional[dict[str, Callable]] = None,
        save_hyperparameters: Optional[bool] = True,
    ):
        """Class constructor.

        Parameters
        ----------
        model : Union[torch.nn.Module, Mapping]
            Model to use.
        loss : torch.nn.modules.loss._Loss
             Loss used to fit the model.
        optimizer : Union[torch.optim.Optimizer, Mapping]
            Optimization algorithm(s) used to train the model. There can be
            several optimizers passed as an Omegaconf mapping.
        scheduler : Union[torch.optim.Optimizer, Mapping], optional
            Learning rate scheduler(s) used to train the model. There can be
            several schedulers passed as an Omegaconf mapping., by default None
        metrics : Optional[dict[str, Callable]], optional
            Dictionary containing the metrics to monitor during the training and
            to compute at test time., by default None
        save_hyperparameters : Optional[bool], optional
            Save arguments to hparams attribute., by default True
        """
        if save_hyperparameters:
            self.save_hyperparameters(ignore=['model', 'loss'])
        # Must be placed before the super call (or anywhere in other inheriting
        # class of GenericPredictionSystem). Otherwise the script pauses
        # indefinitely after returning self.optimizer. It is unclear why.

        super().__init__()
        self.checkpoint_path = None if not hasattr(self, 'checkpoint_path') else self.checkpoint_path  # Avoids overwriting the attribute. This class will need to be re-written properly alongside ClassificationSystem
        self.model = check_model(model)
        self.optimizer, config_scheduler = check_optimizer(optimizer, self.model)
        self.scheduler = config_scheduler if scheduler is None else check_scheduler(scheduler, self.optimizer)
        self.loss = check_loss(loss)
        self.metrics = metrics or {}
        if len(self.optimizer) > 1:
            print('[INFO] Multiple optimizers detected: setting automatic optimization to False... you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()`` of your prediction system.')
            self.automatic_optimization = False

    def _check_integrity(self, fp: str) -> bool:
        return (fp).exists()

    def download_weights(
        self,
        url: str,
        out_path: str,
        filename: str,
        md5: Optional[str] = None,
    ):
        """Download pretrained weights from a remote repository.

        Downloads weights and ajusts self.checkpoint_path accordingly.
        This method is intended to be used to perform transfer learning
        or resume a model training later on and/or on a different
        machine.
        Downloaded content can either be a single file or a pre-zipped
        directory containing all training filee, in which case the
        value of checkpoint_path is updated to point inside that
        unzipped folder.

        Parameters
        ----------
        url : str
            url to the path or directory to download
        out_path : str
            local root path where to to extract the downloaded content
        filename : str
            name of the file (in case of a single file download) or the
            directory (in case of a zip download) on local disk
        md5 : Optional[str], optional
            checksum value to verify the integrity of the downloaded
            content, by default None
        """
        path = self.checkpoint_path
        if Path(filename).suffix == '.zip':
            path = Path(out_path) / Path(filename).stem / 'pretrained.ckpt'
            if self._check_integrity(path):
                print("Files already downloaded and verified")
                return
            download_and_extract_archive(
                url,
                out_path,
                filename=filename,
                md5=md5,
                remove_finished=True,
            )
        else:
            path = Path(out_path) / 'pretrained.ckpt'
            if self._check_integrity(path):
                print("Files already downloaded and verified")
                return
            download_url(
                url,
                out_path,
                filename=filename,
                md5=md5,
            )
        self.checkpoint_path = path

    def _cast_type_to_loss(self, y):
        if isinstance(self.loss, torch.nn.CrossEntropyLoss) and len(y.shape) == 1 or\
           isinstance(self.loss, torch.nn.NLLLoss):
            y = y.to(torch.int64)
        else:
            y = y.to(torch.float32)
        return y

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def _step(
        self, split: str, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        if split == "train":
            log_kwargs = {"on_step": True, "on_epoch": True, "sync_dist": True}
        else:
            log_kwargs = {"on_step": True, "on_epoch": True, "sync_dist": True}

        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, self._cast_type_to_loss(y))  # Shape mismatch for binary: need to 'y = y.unsqueeze(1)' (or use .reshape(2)) to cast from [2] to [2,1] and cast y to float with .float()
        self.log(f"loss/{split}", loss, **log_kwargs)

        for metric_name, metric_func in self.metrics.items():
            if isinstance(metric_func, dict):
                score = metric_func['callable'](y_hat, y, **metric_func['kwargs'])
            else:
                score = metric_func(y_hat, y)
            self.log(f"{metric_name}/{split}", score, **log_kwargs)

        return loss

    def training_step(
        self, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        return self._step("train", batch, batch_idx)

    def validation_step(
        self, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        return self._step("val", batch, batch_idx)

    def test_step(
        self, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        return self._step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def configure_optimizers(self) -> dict:
        res = []
        for i, opt in enumerate(self.optimizer):
            tmp = {'optimizer': opt, 'lr_scheduler': self.scheduler[i]}
            if tmp['lr_scheduler'] is None:
                tmp.pop('lr_scheduler')
            res.append(tmp)
        return res

    @staticmethod
    def state_dict_replace_key(
        state_dict: dict,
        replace: Optional[list[str]] = ['.', '']
    ):
        """Replace keys in a state_dict dictionnary.

        A state_dict usually is an OrderedDict where the keys are the model's
        module names. This method allows to change these names by replacing
        a given string, or token, by another.

        Parameters
        ----------
        state_dict : dict
            Model state_dict
        replace : Optional[list[str]], optional
            Tokens to replace in the
            state_dict module names. The first element is the token
            to look for while the second is the replacement value.
            By default ['.', ''].

        Returns
        -------
        dict
            State_dict with new keys.

        Examples
        -------
        I have loaded a Resnet18 model through a checkpoint after a training
        session. But the names of the model modules have been altered with a
        prefix "model.":

        >>> sd = model.state_dict()
        >>> print(list(sd)[:2])
        (model.conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (model.bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        To remove this prefix:

        >>> sd = GenericPredictionSystem.state_dict_replace_key(sd, ['model.', ''])
        >>> print(sd[:2])
        (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        """
        replace[0] += '.' if not replace[0].endswith('.') else ''
        for key in list(state_dict):
            state_dict[key.replace(replace[0], replace[1])] = state_dict.pop(key)
        print(f'Inference state_dict: replaced {len(state_dict)} keys from "{replace[0]}" to "{replace[1]}"')
        return state_dict

    def remove_state_dict_prefix(
        self,
        state_dict: dict,
        prefix: str = 'model.',
    ):
        """Remove a prefix from the keys of a state_dict.

        This method is intended to remove the ".model" prefix from the
        keys of a state_dict which is added by PyTorchLightning
        when saving a LightningModule's checkpoint. This is due to the fact
        that a LightningModule contains a model attribute which is referenced
        in the LightningModule state_dict as "model.<model_state_dict_key>".
        And the LightningModule state_dict is saved as a whole when calling
        the save_checkpoint method (enabling the saving of more
        hyperparameters).
        This is useful when loading a state_dict directly on a model object
        instead of a LightningModule.

        Parameters
        ----------
        state_dict : dict
            Model state_dict
        prefix : str
            Prefix to remove from the state_dict keys.

        Returns
        -------
        dict
            State_dict with new keys.
        """
        for key in list(state_dict):
            state_dict[key.replace(prefix, '')] = state_dict.pop(key)
        print(f'Inference state_dict: removed prefix "{prefix}" from {len(state_dict)} keys')
        return state_dict

    def predict(
        self,
        datamodule,
        trainer,
    ):
        """Predict a model's output on the test dataset.

        This method performs inference on the test dataset using only
        pytorchlightning tools.

        Parameters
        ----------
        datamodule : pl.LightningDataModule
            pytorchlightning datamodule handling all train/test/val datasets.
        trainer : pl.Trainer
            pytorchlightning trainer in charge of running the model on train
            and inference mode.

        Returns
        -------
        array
            Predicted tensor values.
        """
        datamodule.setup(stage="test")
        predictions = trainer.predict(dataloaders=datamodule.test_dataloader(), model=self)
        predictions = torch.cat(predictions)
        return predictions

    def predict_point(
        self,
        checkpoint_path: str,
        data: Union[Tensor, tuple[Any, Any]],
        state_dict_replace_key: Optional[list[str, str]] = None,
        ckpt_transform: Callable = None,
    ):
        """Predict a model's output on 1 data point.

        Performs as predict() but for a single data point and using native
        pytorch tools.

        Parameters
        ----------
        checkpoint_path : str
            path to the model's checkpoint to load.
        data : Union[Tensor, tuple[Any, Any]]
            data point to perform inference on.
        state_dict_replace_key : Optional[list[str, str]], optional
            list of values used to call the static method
            state_dict_replace_key(). Defaults to None.
        ckpt_transform : Callable, optional
            callable function applied to the loaded checkpoint object.
            Use this to modify the structure of the loaded model's checkpoint
            on the fly. Defaults to None.
        remove_model_prefix : bool, optional
            if True, removes the "model." prefix from the keys of the
            loaded checkpoint. Defaults

        Returns
        -------
        array
            Predicted tensor value.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        if state_dict_replace_key:
            ckpt['state_dict'] = self.state_dict_replace_key(ckpt['state_dict'],
                                                             state_dict_replace_key)
        if ckpt_transform:
            ckpt = ckpt_transform(ckpt)
        self.load_state_dict(ckpt['state_dict'])
        self.model.eval()
        with torch.no_grad():
            if isinstance(data, (tuple, list, set, dict)):
                for i, d in enumerate(data):
                    data[i] = d.to(device) if isinstance(d, torch.Tensor) else d
                prediction = self.model(*data)
            else:
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                prediction = self.model(data)
        return prediction


class ClassificationSystem(GenericPredictionSystem):
    """Classification task class."""
    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        optimizer: Union[torch.nn.Module, Mapping] = None,
        lr: float = 1e-2,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = True,
        metrics: Optional[dict[str, Callable]] = None,
        task: str = 'classification_binary',
        loss_kwargs: Optional[dict] = {},
        hparams_preprocess: bool = True,
        checkpoint_path: Optional[str] = None
    ):
        """Class constructor.

        Parameters
        ----------
        model : dict
            model to use
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
            created from the `malpolon.models.utils.FMETRICS_CALLABLES`
            constant or supplied, by the user directly.
        task : str, optional
            Machine learning task (used to format labels accordingly),
            by default 'classification_multiclass'. The value determines
            the loss to be selected. if 'multilabel' or 'binary' is
            in the task, the BCEWithLogitsLoss is selected, otherwise
            the CrossEntropyLoss is used.
        hparams_preprocess : bool, optional
            if True performs preprocessing operations on the hyperparameters,
            by default True
        """
        if hparams_preprocess:
            task = task.split('classification_')[1]
            metrics = check_metric(metrics)

        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.checkpoint_path = checkpoint_path
        model = check_model(model)

        if optimizer is None:
            print(f'[INFO] No optimizer provided: using SGD with lr={lr}, weight_decay={weight_decay}, momentum={momentum}, nesterov={nesterov}')
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )

        if 'binary' in task or 'multilabel' in task:
            loss = torch.nn.BCEWithLogitsLoss(**loss_kwargs)
        else:
            loss = torch.nn.CrossEntropyLoss(**loss_kwargs)

        if metrics is None:
            metrics = {
                "accuracy": {'callable': Fmetrics.classification.binary_accuracy,
                             'kwargs': {}}
            }

        super().__init__(model, loss, optimizer, metrics=metrics)
