"""This module provides a Multimodal Ensemble model for GeoLifeCLEF2024 data.

Author: Lukas Picek <lukas.picek@inria.fr>
        Theo Larcher <theo.larcher@inria.fr>

License: GPLv3
Python version: 3.10.6
"""
from typing import Any, Callable, Mapping, Optional, Union

import omegaconf
import torch
from omegaconf import OmegaConf
from torch import Tensor

from malpolon.models.standard_prediction_systems import ClassificationSystem


class ClassificationSystemGLC24(ClassificationSystem):
    """Classification task class for GLC24_pre-extracted.

    Inherits ClassificationSystem.
    """
    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        optimizer: Union[torch.nn.Module, Mapping] = None,
        metrics: Optional[dict[str, Callable]] = None,
        task: str = 'classification_multilabel',
        loss_kwargs: Optional[dict] = {},
        hparams_preprocess: bool = True,
        weights_dir: str = 'outputs/glc24_cnn_multimodal_ensemble/',
        checkpoint_path: Optional[str] = None,
        num_classes: int = None,
    ):
        """Class constructor.

        Parameters
        ----------
        model : Union[torch.nn.Module, Mapping]
            model to use, either a torch model object, or a mapping
            (dictionary from config file) used to load and build
            the model
        optimizer : Union[torch.nn.Module, Mapping], optional
            optimizer to use, either a torch optimizer object, or a mapping
            (dictionary from config file) used to load and build
            the optimizer and scheduler, by default None (SGD is used)
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
        loss_kwargs : Optional[dict], optional
            loss parameters, by default {}
        hparams_preprocess : bool, optional
            if True performs preprocessing operations on the hyperparameters,
            by default True
        weights_dir : str, optional
            directory where to download the model weights,
            by default 'outputs/glc24_cnn_multimodal_ensemble/'
        checkpoint_path : Optional[str], optional
            path to the model checkpoint to load either to resume
            a previous training, perform transfer learning or run in
            prediction mode (inference), by default None
        num_classes : int, optional
            number of classes for the classification task, by default None
        """
        loss_kwargs = {} if loss_kwargs is None else loss_kwargs
        if isinstance(loss_kwargs, omegaconf.dictconfig.DictConfig):
            loss_kwargs = OmegaConf.to_container(loss_kwargs, resolve=True)
        if 'pos_weight' in loss_kwargs.keys() and not isinstance(loss_kwargs['pos_weight'], Tensor):
            # Backwards compatibility for num_classes
            if num_classes is None:
                if 'multilabel' in task:
                    num_classes = metrics['multilabel_f1-score'].kwargs.num_labels
                elif 'multiclass' in task:
                    num_classes = metrics['multiclass_f1-score'].kwargs.num_classes
            loss_kwargs['pos_weight'] = Tensor([loss_kwargs['pos_weight']] * num_classes)
        super().__init__(model, optimizer=optimizer, metrics=metrics, task=task, loss_kwargs=loss_kwargs, hparams_preprocess=hparams_preprocess, checkpoint_path=checkpoint_path)
        if self.model.pretrained and not self.checkpoint_path:
            self.download_weights("https://lab.plantnet.org/seafile/f/d780d4ab7f6b419194f9/?dl=1",
                                  weights_dir,
                                  filename="pretrained.ckpt",
                                  md5="69111dd8013fcd8e8f4504def774f3a5")

    def forward(self, x, y, z):  # noqa: D102 pylint: disable=C0116
        return self.model(x, y, z)

    def _step(
        self, split: str, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        if split == "train":
            log_kwargs = {"on_step": True, "on_epoch": True, "sync_dist": True}
        else:
            log_kwargs = {"on_step": True, "on_epoch": True, "sync_dist": True}

        x_landsat, x_bioclim, x_sentinel, y, _ = batch  # x_landsat, x_bioclim, x_sentinel, y, survey_id
        y_hat = self(x_landsat, x_bioclim, x_sentinel)

        if 'pos_weight' in dir(self.loss):
            loss_pos_weight = self.loss.pos_weight.clone()  # save initial loss parameter value
            self.loss.pos_weight = y * torch.Tensor(self.loss.pos_weight).to(y)   # Proper way would be to forward pos_weight to loss instantiation via loss_kwargs, but pos_weight must be a tensor, i.e. have access to y -> Not possible in Malpolon as datamodule and optimizer instantiations are separate

        loss = self.loss(y_hat, self._cast_type_to_loss(y))  # Shape mismatch for binary: need to 'y = y.unsqueeze(1)' (or use .reshape(2)) to cast from [2] to [2,1] and cast y to float with .float()
        self.log(f"loss/{split}", loss, **log_kwargs)

        if 'pos_weight' in dir(self.loss):
            self.loss.pos_weight = loss_pos_weight  # restore initial loss parameter value to not alter lightning module state_dict

        for metric_name, metric_func in self.metrics.items():
            if isinstance(metric_func, dict):
                score = metric_func['callable'](y_hat, y, **metric_func['kwargs'])
            else:
                score = metric_func(y_hat, y)
            self.log(f"{metric_name}/{split}", score, **log_kwargs)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  # noqa: D102 pylint: disable=C0116
        x_landsat, x_bioclim, x_sentinel, _, _ = batch  # x_landsat, x_bioclim, x_sentinel, y, survey_id
        return self(x_landsat, x_bioclim, x_sentinel)
