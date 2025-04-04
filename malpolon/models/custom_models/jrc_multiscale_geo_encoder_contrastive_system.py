"""This module provides a model to align features from multiscale geo-tagged data.

Author: Theo Larcher <theo.larcher@inria.fr>
        Alexis Joly <alexis.joly@inria.fr>

License: GPLv3
Python version: 3.12.9
"""
from typing import Any, Callable, Mapping, Optional, Union

import omegaconf
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn import functional as F

from malpolon.models.standard_prediction_systems import ClassificationSystem

def Info_nce_loss(self, batch, mode="train"):
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)  # (512, 512)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)  # (512, 512)
    cos_sim.masked_fill_(self_mask, -9e15)  # (512, 512)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)  # (512, 512)
    # InfoNCE loss
    cos_sim = cos_sim / self.hparams.temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)  # (512,)
    nll = nll.mean()  # scalar

    # Logging loss
    self.log(mode + "_loss", nll)
    # Get ranking position of positive example
    comb_sim = torch.cat(
        [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
        dim=-1,
    )
    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    # Logging ranking metrics
    self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
    self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
    self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

    return nll

class MultiScaleGeoContrastiveSystem(ClassificationSystem):
    def __init__(self,
        model: Union[torch.nn.Module, Mapping],
        optimizer: Union[torch.nn.Module, Mapping] = None,
        metrics: Optional[dict[str, Callable]] = None,
        task: str = 'classification_multilabel',
        loss_kwargs: Optional[dict] = {},
        hparams_preprocess: bool = True,
        weights_dir: str = 'outputs/glc24_cnn_multimodal_ensemble/',
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(model, optimizer=optimizer, metrics=metrics, task=task, loss_kwargs=loss_kwargs, hparams_preprocess=hparams_preprocess, checkpoint_path=checkpoint_path)
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x):
        pass

    def _step(
        self, split: str, batch: tuple[Any, Any], batch_idx: int
    ) -> Union[Tensor, dict[str, Any]]:
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)  # (512, 3, 96, 96)

        # Encode all images
        feats = self.convnet(imgs)  # (512, 128)
        
        
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass
