"""This module provides a model to align features from multiscale geo-tagged data.
Inspired from:
- https://github.com/sthalles/SimCLR/

Author: Theo Larcher <theo.larcher@inria.fr>
        Alexis Joly <alexis.joly@inria.fr>

License: GPLv3
Python version: 3.12.9
"""
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional

import timm
import torch
import torch.nn as nn
from geoclip import LocationEncoder
from omegaconf import OmegaConf
from torch import nn
from torchvision.datasets.utils import download_and_extract_archive, download_url

from malpolon.models.utils import check_model


def download_weights(
        url: str,
        checkpoint_path: str,
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
        def _check_integrity(fp: str) -> bool:
            return (fp).exists()
        path = checkpoint_path
        if Path(filename).suffix == '.zip':
            path = Path(out_path) / Path(filename).stem / 'pretrained.ckpt'
            if _check_integrity(path):
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
            if _check_integrity(path):
                print("Files already downloaded and verified")
                return
            download_url(
                url,
                out_path,
                filename=filename,
                md5=md5,
            )
        checkpoint_path = path

def drop_last_k_layers(model, k):
    # Get all layers of the model
    layers = list(model.children())
    # Drop the last k layers
    if k > 0:
        layers = layers[:-k]
    # Rebuild the model with the remaining layers
    return nn.Sequential(*layers)

def replace_last_k_layers_with_identity(model, k=0):
    # Get all layers of the model
    n_c = list(model.named_children())
    # Replace the last k layers with identity layers
    for i in range(0, min(len(n_c), k)):
        setattr(model, n_c[-(i+1)][0], torch.nn.Identity())

def remove_state_dict_prefix(
    state_dict: dict,
    sep: str = '.',
    n_prefix: int = 1,
):
    """Remove a prefix from the keys of a state_dict."""
    for key in list(state_dict):
        state_dict['.'.join(key.split(sep)[n_prefix:])] = state_dict.pop(key)
    print(f'State_dict: removed {n_prefix} prefix based on separator "{sep}" from {len(state_dict)} keys')
    return state_dict

def get_model_species():
    model_root_path_species = 'weights/scale_1_species/'
    ckpt_path = str(Path(model_root_path_species) / Path('vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all_best.pth.tar'))
    model_species = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=False,
        num_classes=7806,
        checkpoint_path=ckpt_path,
    )
    return model_species

def get_model_landscape(out_dim=512):
    # model_root_path_landscape = 'weights/scale_2_landscape/'
    model_landscape = timm.create_model(
        'resnet18',
        pretrained=True,
        num_classes=out_dim,
    )
    return model_landscape

def get_model_satellite():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_root_path_satellite = 'weights/scale_3_satellite/'
    model_satellite_config = str(Path(model_root_path_satellite) / Path('glc24_cnn_multimodal_ensemble.yaml'))
    ckpt_path = str(Path(model_root_path_satellite) / 'pretrained.ckpt')

    if ckpt_path:
        download_weights("https://lab.plantnet.org/seafile/f/eb90daeb510c44349fb5/?dl=1",
                         ckpt_path,
                         model_root_path_satellite,
                         filename="pretrained.ckpt",
                         md5="680a6a8f66480dff21ead28031ab1ca0")
    model_config = OmegaConf.load(model_satellite_config)
    model = check_model(model_config.model)
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)

    state_dict = remove_state_dict_prefix(checkpoint['state_dict'].copy())
    if 'pos_weight' in state_dict:
        _ = state_dict.pop('pos_weight')
    model.load_state_dict(state_dict)
    model_satellite = model.sentinel_model.to(device)
    print(model_satellite)
    return model_satellite

class BaseSimCLRException(Exception):
    """Base exception"""

class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""

class SelectTensor(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index  # Index of the tensor to select

    def forward(self, inputs):
        return inputs[:, self.index, :]

class ModelSimCLR(nn.Module):
    """My custom model for SimCLR using features from 2 different other models."""
    def __init__(self, base_model, out_dim=512, freeze_modality_backbone=False, freeze_gps_backbone=False):
        super().__init__()
        self.base_model = base_model
        self.freeze_modality_backbone = freeze_modality_backbone
        self.freeze_gps_backbone = freeze_gps_backbone
        modality_dict = {
            'gps': LocationEncoder,  # GeoCLIP. Selected by default.
            'species': get_model_species,  # DinoV2
            'landscape': get_model_landscape,  # ResNet18 or DinoV2
            'satellite': get_model_satellite,  # Swin_t (MME)
        }
        
        # GPS
        self.gps_encoder = modality_dict['gps']()
        dim_mlp = self.gps_encoder.LocEnc1.head[0].out_features
        self.gps_contrastive_head = torch.nn.Sequential(
            OrderedDict(
                [
                    ("relu", torch.nn.ReLU()),
                    ("head", torch.nn.Linear(dim_mlp, out_dim)),
                ]
            )
        )
        if self.freeze_gps_backbone:
            for param in self.gps_encoder.parameters():
                param.requires_grad = False
        
        # Modalities
        if base_model == 'species':
            self.modality_encoder = modality_dict[base_model]()
            dim_mlp = self.modality_encoder.head.out_features  # model_species.head.in_features
            self.modality_contrastive_head = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("fc", self.modality_encoder.head),
                        ("relu", torch.nn.ReLU()),
                        ("head", torch.nn.Linear(dim_mlp, out_dim)),
                    ]
                )
            )
            replace_last_k_layers_with_identity(self.modality_encoder, 4)
        elif base_model == 'landscape':
            self.modality_encoder = modality_dict[base_model](out_dim=out_dim) # assuming resnet18
            dim_mlp = self.modality_encoder.fc.out_features
            self.modality_contrastive_head = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("fc", torch.nn.Linear(dim_mlp, out_dim)),
                        ("relu", torch.nn.ReLU()),
                        ("head", torch.nn.Linear(out_dim, out_dim)),
                    ]
                )
            )
            replace_last_k_layers_with_identity(self.modality_encoder, 1)
        elif base_model == 'satellite':
            self.modality_encoder = modality_dict[base_model]()
            dim_mlp = self.modality_encoder.norm.normalized_shape[0]
            self.modality_contrastive_head = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("fc", torch.nn.Linear(dim_mlp, out_dim)),
                        ("relu", torch.nn.ReLU()),
                        ("head", torch.nn.Linear(out_dim, out_dim)),
                    ]
                )
            )
            replace_last_k_layers_with_identity(self.modality_encoder, 1)
        else:
            raise InvalidDatasetSelection(
                "Invalid dataset selection. Check the config file and pass one of: 'species', 'landscape' or 'satellite'")
        if self.freeze_modality_backbone:
            for param in self.modality_encoder.parameters():
                param.requires_grad = False

    def forward(self, img, gps):
        gps_h = self.gps_encoder(gps)
        gps_z = self.gps_contrastive_head(gps_h)

        if self.base_model == 'species':
            img_h = self.modality_encoder.forward_features(img)  # includes the (norm) layer
            img_h = self.modality_encoder.pool(img_h)
        elif self.base_model == 'landscape':
            img_h = self.modality_encoder(img)
        elif self.base_model == 'satellite':
            img_h = self.modality_encoder(img)

        img_z = self.modality_contrastive_head(img_h)
        return gps_z, img_z
