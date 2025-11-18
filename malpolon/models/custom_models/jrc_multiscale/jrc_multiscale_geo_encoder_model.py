"""This module provides a model to align features from multiscale geo-tagged data.
Inspired from:
- https://github.com/sthalles/SimCLR/

Author: Theo Larcher <theo.larcher@inria.fr>
        Alexis Joly <alexis.joly@inria.fr>

License: GPLv3
Python version: 3.12.9
"""
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn
from geoclip import LocationEncoder
from omegaconf import OmegaConf
from torchvision.datasets.utils import download_and_extract_archive, download_url

from malpolon.models.utils import check_model
from malpolon.models.model_builder import _find_module_of_type


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

def reinitialize_weights(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()

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

def set_dropout_p(model, new_p=0.1):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = new_p

def get_model_species():
    model_root_path_species = 'weights/scale_1_species/'
    ckpt_path = str(Path(model_root_path_species) / Path('vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all_best.pth.tar'))
    model_species = timm.create_model(
        'vit_base_patch14_reg4_dinov2.lvd142m',
        pretrained=False,
        num_classes=7806,
        checkpoint_path=ckpt_path,
    )
    print(f'Loaded species model with {sum(p.numel() for p in model_species.parameters() if p.requires_grad):,} trainable parameters')
    return model_species

def get_model_landscape(out_dim=512):
    model_root_path_landscape = 'weights/scale_2_landscape/'
    model_landscape = timm.create_model(
        'resnet18',
        pretrained=True,
        num_classes=out_dim,
    )
    # model_landscape = timm.create_model(
    #     # 'vit_base_patch14_reg4_dinov2.lvd142m',
    #     'vit_small_patch14_dinov2.lvd142m',
    #     pretrained=True,
    # )
    print(f'Loaded landscape model with {sum(p.numel() for p in model_landscape.parameters() if p.requires_grad):,} trainable parameters')
    return model_landscape

def get_model_satellite(ckpt: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_root_path_satellite = 'weights/scale_3_satellite/'
    model_satellite_config = str(Path(model_root_path_satellite) / Path('glc24_cnn_multimodal_ensemble.yaml'))
    model_config = OmegaConf.load(model_satellite_config)
    model = check_model(model_config.model)

    # Load MME's weights
    if ckpt:
        ckpt_path = str(Path(model_root_path_satellite) / 'pretrained.ckpt')
        download_weights("https://lab.plantnet.org/seafile/f/eb90daeb510c44349fb5/?dl=1",
                         ckpt_path,
                         model_root_path_satellite,
                         filename="pretrained.ckpt",
                         md5="680a6a8f66480dff21ead28031ab1ca0")
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)

        state_dict = remove_state_dict_prefix(checkpoint['state_dict'].copy())
        if 'pos_weight' in state_dict:
            _ = state_dict.pop('pos_weight')
        model.load_state_dict(state_dict)

    model_satellite = model.sentinel_model.to(device)
    print(f'Loaded satellite model with {sum(p.numel() for p in model_satellite.parameters() if p.requires_grad):,} trainable parameters')
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
    def __init__(self, base_model, out_dim=512, dropout=-1,
                 gps_encoder: nn.Module = None, gps_head: nn.Module = None,
                 freeze_modality_backbone=False, freeze_gps_backbone=False,
                 unfreeze_modality_backbone_last_layer=False, unfreeze_gps_backbone_last_layer=False, sat_ckpt='MME'):
        super().__init__()
        self.base_model = base_model
        self.freeze_modality_backbone = freeze_modality_backbone
        self.freeze_gps_backbone = freeze_gps_backbone
        self.dropout = dropout
        self.sat_ckpt = sat_ckpt
        modality_dict = {
            'gps': LocationEncoder,  # GeoCLIP. Selected by default.
            'species': get_model_species,  # DinoV2
            'landscape': get_model_landscape,  # ResNet18 or DinoV2
            'satellite': get_model_satellite,  # Swin_t (MME)
        }
        
        # GPS
        if gps_encoder is None:
            self.gps_encoder = modality_dict['gps']()
        else:
            self.gps_encoder = gps_encoder
        dim_mlp = self.gps_encoder.LocEnc1.head[0].out_features
        if gps_head is None:
            self.gps_contrastive_head = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("relu", torch.nn.ReLU()),
                        ("head", torch.nn.Linear(dim_mlp, out_dim)),
                    ]
                )
            )
        else:
            self.gps_contrastive_head = gps_head
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
            self.modality_encoder = modality_dict[base_model](out_dim=out_dim)
            # dim_mlp = self.modality_encoder.fc.out_features  # ResNet18
            dim_mlp = self.modality_encoder.num_features  # DinoV2
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
            self.modality_encoder = modality_dict[base_model](ckpt=self.sat_ckpt)  # assuming swin_t
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

        # Modifications
        if self.freeze_modality_backbone:
            for param in self.modality_encoder.parameters():
                param.requires_grad = False
        if dropout >= 0:
            set_dropout_p(self.gps_encoder, new_p=dropout)
            set_dropout_p(self.modality_encoder, new_p=dropout)

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


class MultiLabelClassifier(nn.Module):
    """
    Out features shapes:
      - Contrastive head: 512 for all
      - Encoders:
          - GPS: 512 -> model['satellite'].gps_encoder.LocEnc1.head[0].out_features
          - species: 768 -> model['species'].num_features
          - landscape: 384 -> model['landscape'].num_features
          - satellite: 768 -> model['satellite'].features[-1][-1].mlp[3].out_features
    """
    def __init__(self,
                 gps_encoder, gps_contrastive_head,
                 species_encoder, species_contrastive_head,
                 landscape_encoder, landscape_contrastive_head,
                 satellite_encoder, satelite_contrastive_head,
                 classifier_type='linear_probing', contrastive_head_out_dim=512,
                 num_labels=11255, skip_modalities=[]):
        super().__init__()
        modalities_name = ['species', 'landscape', 'satellite']
        self.modalities_to_process = [b for b in modalities_name if b not in skip_modalities]
        self.gps_encoder = gps_encoder
        self.species_encoder = species_encoder
        self.landscape_encoder = landscape_encoder
        self.satellite_encoder = satellite_encoder
        self.gps_contrastive_head = gps_contrastive_head
        self.species_contrastive_head = species_contrastive_head
        self.landscape_contrastive_head = landscape_contrastive_head
        self.satellite_contrastive_head = satelite_contrastive_head
        self.contrastive_head_out_dim = contrastive_head_out_dim
        self.gps_out_features = gps_encoder.LocEnc1.head[0].out_features
        self.species_out_features = species_encoder.num_features
        self.landscape_out_features = landscape_encoder.num_features
        self.satellite_out_features = satellite_encoder.features[-1][-1].mlp[3].out_features
        # self.modalities_out_features = sum([self.gps_out_features, self.species_out_features, self.landscape_out_features, self.satellite_out_features])
        self.modalities_out_features = sum([out_features for modality, out_features in {
            'species': self.species_out_features,
            'landscape': self.landscape_out_features,
            'satellite': self.satellite_out_features
        }.items() if modality in self.modalities_to_process])

        if classifier_type == 'fine_tuning':
            self.classifier = nn.Sequential(
                nn.Linear(self.contrastive_head_out_dim, 2048),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(2048, 4096),
                nn.Dropout(0.1),
                nn.Linear(4096, num_labels)
            )
        elif classifier_type == 'linear_probing':
            self.classifier = nn.Sequential(
                nn.Linear(self.contrastive_head_out_dim, 2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 4096),
                nn.Dropout(0.1),
                nn.Linear(4096, num_labels)
            )
            # Freeze encoders
            for encoder in [self.gps_encoder, self.gps_contrastive_head,
                            self.species_encoder, self.species_contrastive_head, 
                            self.landscape_encoder, self.landscape_contrastive_head,
                            self.satellite_encoder, self.satellite_contrastive_head]:
                for param in encoder.parameters():
                    param.requires_grad = False

    def forward(self, input, input_type):
        # torch.nn.Sequential(model['species'].gps_encoder, model['species'].gps_contrastive_head),
        with torch.no_grad():
            if '_gps' in input_type:
                features_z = self.gps_encoder(input)
                features_z = self.gps_contrastive_head(features_z)
            if 'species_img' in input_type and 'species' in self.modalities_to_process:
                _ = self.species_encoder.forward_features(input)  # includes the (norm) layer
                features_z = self.species_encoder.pool(_)
                features_z = self.species_contrastive_head(features_z)
            elif 'landscape_img' in input_type and 'landscape' in self.modalities_to_process:
                features_z = self.landscape_encoder(input)
                features_z = self.landscape_contrastive_head(features_z)
            elif 'satellite_img' in input_type and 'satellite' in self.modalities_to_process:
                features_z = self.satellite_encoder(input)
                features_z = self.satellite_contrastive_head(features_z)

        classif_head_logits = self.classifier(features_z)
        return classif_head_logits

    def predict(self, inputs: torch.tensor, input_type: list[str]):
        self.eval()
        with torch.no_grad():
            features = []
            if '_gps' in input_type:
                features_z = self.gps_encoder(inputs)
                features.append(self.gps_contrastive_head(features_z))
            if 'species_img' in input_type and 'species' in self.modalities_to_process:
                _ = self.species_encoder.forward_features(inputs)  # includes the (norm) layer
                features_z = self.species_encoder.pool(_)
                features.append(self.species_contrastive_head(features_z))
            elif 'landscape_img' in input_type and 'landscape' in self.modalities_to_process:
                features_z = self.landscape_encoder(inputs)
                features.append(self.landscape_contrastive_head(features_z))
            elif 'satellite_img' in input_type and 'satellite' in self.modalities_to_process:
                features_z = self.satellite_encoder(inputs)
                features.append(self.satellite_contrastive_head(features_z))
            features = torch.mean(torch.stack(features, dim=0), dim=0)
            logits = self.classifier(features)
        return logits


class ImgToGPS(nn.Module):
    def __init__(self, contrastive_model, modality, out_dim=2, freeze_encoder=True, ):
        super().__init__()
        modalities_name = ['species', 'landscape', 'satellite']
        self.modalities_to_process = [b for b in modalities_name if b not in skip_modalities]
        match modality:
            case 'species':
                species_encoder = _find_module_of_type(self.encoder, 'species')
                self.embed_dim = species_encoder.num_features
            case 'landscape':
                landscape_encoder = _find_module_of_type(self.encoder, 'landscape')
                self.landscape_out_features = landscape_encoder.num_features
            case 'satellite':
                satellite_encoder = _find_module_of_type(self.encoder, 'satellite')
                self.satellite_out_features = satellite_encoder.features[-1][-1].mlp[3].out_features
            case _:
                raise InvalidDatasetSelection(
                    "Invalid dataset selection. Check the config file and pass one of: 'species', 'landscape' or 'satellite'")
        if modality == 'species':
            self.embed_dim = species_encoder.num_features
        self.landscape_out_features = landscape_encoder.num_features
        self.satellite_out_features = satellite_encoder.features[-1][-1].mlp[3].out_features
        # self.modalities_out_features = sum([self.gps_out_features, self.species_out_features, self.landscape_out_features, self.satellite_out_features])
        self.modalities_out_features = sum([out_features for modality, out_features in {
            'species': self.species_out_features,
            'landscape': self.landscape_out_features,
            'satellite': self.satellite_out_features
        }.items() if modality in self.modalities_to_process])
        self.encoder = img_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)   # GPS coordinates (lat, lon)
        )

    def forward(self, x):
        z_img = self.encoder(x)  # get image embedding
        gps_pred = self.head(z_img)
        return gps_pred
