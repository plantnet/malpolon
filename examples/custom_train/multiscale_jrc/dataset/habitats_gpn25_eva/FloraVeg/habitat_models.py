import sys
import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from time import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb
import timm
from matplotlib import pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from dino_v2_large_PN22M.models import vit_large


class DinoV2ClassifierPN22M(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        # DINOv2 returns CLS token when using forward_features
        features = self.backbone.forward_features(x)        # (B, seq_len, embed_dim)
        # CLS token
        if isinstance(features, dict):
            cls_token = features["x_norm_clstoken"]
            # cls_token = torch.cat([features["x_norm_clstoken"], features["x_norm_regtokens"].mean(dim=1)], dim=-1)  # don't forget  to adjust head input size x2
        else:
            cls_token = features[:, 0]  # (B, embed_dim)
        x = self.head(cls_token)
        return x

def freeze_dinov2_backbone(model, N: int = 2):
    """Freeze up to the N last attention blocks of a DinoV2 model.

    Args:
        model (torchvision.model): a dinov2 model
        N (int, optional): Number of attention blocks to keep learnable. Defaults to 2.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head
    for param in model.head.parameters():
        param.requires_grad = True

    # Unfreeze the last N transformer blocks
    for block in model.blocks[-N:]:
        for param in block.parameters():
            param.requires_grad = True

    # Unfreeze the final norm layer
    for param in model.norm.parameters():
        param.requires_grad = True


def get_model(model_name, num_classes, **kwargs):
    match model_name:
        case 'resnet18':
            print("[INFO] Using ResNet18")
            model = models.resnet18(weights="IMAGENET1K_V1")
            model.fc = nn.Linear(
                model.fc.in_features,
                num_classes,
            )
        case 'resnet50':
            print("[INFO] Using ResNet50")
            model = models.resnet50(weights="IMAGENET1K_V2")
            model.fc = nn.Linear(
                model.fc.in_features,
                num_classes,
            )
        case 'dinov2_vits14':
            print("[INFO] Using DINOv2 ViT-S/14 (partial unfreezing: last 2 transformer blocks and head)")
            model = timm.create_model('timm/vit_small_patch14_dinov2.lvd142m',
                                    pretrained=True,
                                    num_classes=num_classes,
                                    **kwargs)
            freeze_dinov2_backbone(model, N=2)
        case 'dinov2_PN22M':    
            def load_state_dict(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                state_dict = state_dict["teacher"]
                state_dict = {k.removeprefix("backbone."): v for k, v in state_dict.items() if k.startswith("backbone.")}
                state_dict = {remap_key(k): v for k, v in state_dict.items()}
                return state_dict

            def remap_key(k):
                return re.sub(r"blocks\.\d+\.(?=\d+)", "blocks.", k)

            model = vit_large(patch_size=16, img_size=224, init_values=0.1, block_chunks=0, num_register_tokens=4)
            state_dict = load_state_dict('./dino_v2_large_PN22M/ssl_vitl_224.pth')
            model.load_state_dict(state_dict, strict=True)
            model = DinoV2ClassifierPN22M(model, num_classes)
            freeze_dinov2_backbone(model.backbone, N=2)
        case 'convnext':
            model = models.convnext_base(weights="IMAGENET1K_V1")
            model.classifier[2] = nn.Linear(
                model.classifier[2].in_features,
                num_classes,
            )
        case 'vgg16':
            model = models.vgg16(weights="IMAGENET1K_V1")
            model.classifier[6] = nn.Linear(
                model.classifier[6].in_features,
                num_classes,
            )
        case 'vitb32':
            model = models.vit_b_32(weights="IMAGENET1K_V1")
            model.heads.head = nn.Linear(
                model.heads.head.in_features,
                num_classes,
            )
        case 'mobilenet_v3':
            model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
            model.classifier[3] = nn.Linear(
                model.classifier[3].in_features,
                num_classes,
            )
        case 'inception_v3':
            model = models.inception_v3(weights="IMAGENET1K_V1")
            model.fc = nn.Linear(
                model.fc.in_features,
                num_classes,
            )
            # Also replace the auxiliary classifier head if training
            model.AuxLogits.fc = nn.Linear(
                model.AuxLogits.fc.in_features,
                num_classes,
            )
    return model


def get_in_features(model, model_name):
    match model_name:
        case 'resnet18' | 'resnet50':
            return model.fc.in_features
        case 'dinov2_vits14':
            return model.head.in_features
        case 'dinov2_PN22M':
            return model.head.in_features
        case 'convnext':
            return model.classifier[2].in_features
        case 'vgg16':
            return model.classifier[6].in_features
        case 'vitb32':
            return model.heads.head.in_features
        case 'mobilenet_v3':
            return model.classifier[3].in_features
        case 'inception_v3':
            return model.fc.in_features

class MultiHeadModel(nn.Module):
    def __init__(self, backbone, bb_model_name, eunis_lvl, n_classes1, n_classes2, n_classes3, n_classes4, n_classes3_4,
                 gradcam_head=None):
        super().__init__()

        self.backbone = backbone if not isinstance(backbone, DinoV2ClassifierPN22M) else backbone.backbone
        self.bb_model_name = bb_model_name
        self.in_features = get_in_features(backbone, bb_model_name)
        self.eunis_lvl = eunis_lvl
        self.gradcam_head = gradcam_head
        
        self.heads = nn.ModuleDict({
            '1': nn.Linear(self.in_features, n_classes1),
            '2': nn.Linear(self.in_features, n_classes2),
            '3': nn.Linear(self.in_features, n_classes3),
            '4': nn.Linear(self.in_features, n_classes4),
            '3_4': nn.Linear(self.in_features, n_classes3_4),
        })

    def forward(self, x):

        features = self.extract_features(x)

        if self.gradcam_head:
            return self.heads[self.gradcam_head](features)
        
        return {
            lvl: self.heads[lvl](features)
            for lvl in self.eunis_lvl
        }

    def extract_features(self, x):
        # DINOv2
        if ('dinov2' in self.bb_model_name):
            feat = self.backbone.forward_features(x)
            if isinstance(feat, dict):
                feat = feat["x_norm_clstoken"]
            else:
                feat = feat[:, 0]  # (B, embed_dim)
        # ResNet
        elif self.bb_model_name in ['resnet18', 'resnet50']:
            self.backbone.fc = nn.Identity()
            feat = self.backbone(x)
        else:
            raise NotImplementedError(f"Feature extraction not implemented for model {self.bb_model_name}")
        return feat