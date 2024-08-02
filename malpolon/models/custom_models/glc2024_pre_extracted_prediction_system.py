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
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from malpolon.models.standard_prediction_systems import ClassificationSystem
from malpolon.models.utils import check_optimizer


class ClassificationSystemGLC24(ClassificationSystem):
    """Classification task class for GLC24_pre-extracted.

    Inherits ClassificationSystem.
    """
    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        lr: float = 1e-2,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = True,
        metrics: Optional[dict[str, Callable]] = None,
        task: str = 'classification_binary',
        loss_kwargs: Optional[dict] = {},
        hparams_preprocess: bool = True,
        download_weights: bool = None,
        weights_dir: str = 'outputs/glc24_cnn_multimodal_ensemble/',
    ):
        if isinstance(loss_kwargs, omegaconf.dictconfig.DictConfig):
            loss_kwargs = OmegaConf.to_container(loss_kwargs, resolve=True)
        if 'pos_weight' in loss_kwargs.keys():
            length = metrics['multilabel_f1-score'].kwargs.num_labels
            loss_kwargs['pos_weight'] = Tensor([loss_kwargs['pos_weight']] * length)
        super().__init__(model, lr, weight_decay, momentum, nesterov, metrics, task, loss_kwargs, hparams_preprocess)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.optimizer = check_optimizer(optimizer)
        if download_weights:
            self.download_weights("https://lab.plantnet.org/seafile/f/d780d4ab7f6b419194f9/?dl=1",
                                  weights_dir,
                                  filename="pretrained.ckpt",
                                  md5="69111dd8013fcd8e8f4504def774f3a5")

    def configure_optimizers(self):
        """Override default optimizer and scheduler.

        By default, SGD is selected and the scheduler is handled by
        PyTorch Lightning's default one.

        Returns
        -------
        (dict)
            dictionary containing keys for optimizer and scheduler,
            passed on to PyTorch Lightning
        """
        scheduler = CosineAnnealingLR(self.optimizer, T_max=25, verbose=True)
        res = {'optimizer': self.optimizer,
               'lr_scheduler': scheduler}
        return res

    def forward(self, x, y, z):  # noqa: D102 pylint: disable=C0116
        return self.model(x, y, z)

    def _step(
        self, split: str, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        if split == "train":
            log_kwargs = {"on_step": True, "on_epoch": True, "sync_dist": True}
        else:
            log_kwargs = {"on_step": True, "on_epoch": True, "sync_dist": True}

        x_landsat, x_bioclim, x_sentinel, y, survey_id = batch
        y_hat = self(x_landsat, x_bioclim, x_sentinel)

        loss_pos_weight = self.loss.pos_weight  # save initial loss parameter value
        self.loss.pos_weight = y * torch.Tensor([10.0]).to(y)   # Proper way would be to forward pos_weight to loss instantiation via loss_kwargs, but pos_weight must be a tensor, i.e. have access to y -> Not possible in Malpolon as datamodule and optimizer instantiations are separate
        loss = self.loss(y_hat, self._cast_type_to_loss(y))  # Shape mismatch for binary: need to 'y = y.unsqueeze(1)' (or use .reshape(2)) to cast from [2] to [2,1] and cast y to float with .float()
        self.log(f"loss/{split}", loss, **log_kwargs)
        self.loss.pos_weight = loss_pos_weight  # restore initial loss parameter value to not alter lightning module state_dict

        for metric_name, metric_func in self.metrics.items():
            if isinstance(metric_func, dict):
                score = metric_func['callable'](y_hat, y, **metric_func['kwargs'])
            else:
                score = metric_func(y_hat, y)
            self.log(f"{metric_name}/{split}", score, **log_kwargs)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  # noqa: D102 pylint: disable=C0116
        x_landsat, x_bioclim, x_sentinel, y, survey_id = batch
        return self(x_landsat, x_bioclim, x_sentinel)
