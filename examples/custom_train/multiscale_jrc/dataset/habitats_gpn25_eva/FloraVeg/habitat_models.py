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


def get_in_features(model, model_name, use_gps_encoder=False):
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
        case 'simclr_landscape_dinov2':
            return model.head.out_features
        case 'simclr_landscape_resnet':
            features = model.modality_contrastive_head.head.out_features
            if use_gps_encoder:
                features += model.gps_contrastive_head.head.out_features
            return features


class MultiHeadModel(nn.Module):
    def __init__(self, backbone, bb_model_name, eunis_lvl, n_classes1, n_classes2, n_classes3, n_classes4, n_classes3_4,
                 gradcam_head=None, use_gps_encoder=False, fusion_strategy=None):
        super().__init__()

        self.backbone = backbone if not isinstance(backbone, DinoV2ClassifierPN22M) else backbone.backbone
        self.use_gps_encoder = use_gps_encoder
        self.bb_model_name = bb_model_name
        self.eunis_lvl = eunis_lvl
        self.gradcam_head = gradcam_head
        self.fusion_strategy = fusion_strategy
        self.in_features = get_in_features(backbone, bb_model_name, self.use_gps_encoder)

        self.heads = nn.ModuleDict({
            '1': nn.Linear(self.in_features, n_classes1),
            '2': nn.Linear(self.in_features, n_classes2),
            '3': nn.Linear(self.in_features, n_classes3),
            '4': nn.Linear(self.in_features, n_classes4),
            '3_4': nn.Linear(self.in_features, n_classes3_4),
        })

        if self.fusion_strategy == 'mean_pooling_img_gps':
            img_encoder = nn.Sequential(self.backbone.modality_encoder, self.backbone.modality_contrastive_head)
            gps_encoder = None if not self.use_gps_encoder else nn.Sequential(self.backbone.gps_encoder, self.backbone.gps_contrastive_head)
            self.transfer_heads = nn.ModuleDict({
                '1': TransferModel(img_encoder, gps_encoder, embed_dim=get_in_features(backbone, bb_model_name), num_classes=n_classes1),
                '2': TransferModel(img_encoder, gps_encoder, embed_dim=get_in_features(backbone, bb_model_name), num_classes=n_classes2),
                '3': TransferModel(img_encoder, gps_encoder, embed_dim=get_in_features(backbone, bb_model_name), num_classes=n_classes3),
                '4': TransferModel(img_encoder, gps_encoder, embed_dim=get_in_features(backbone, bb_model_name), num_classes=n_classes4),
                '3_4': TransferModel(img_encoder, gps_encoder, embed_dim=get_in_features(backbone, bb_model_name), num_classes=n_classes3_4),
            })

    def fusion(self, fusion_strategy: str, embs: list[torch.Tensor]):
        match fusion_strategy:
            case 'concatenation':
                features = torch.concat(embs, dim=-1)
            case _:
                raise NotImplementedError(f"Fusion strategy '{fusion_strategy}' not implemented.")
        return features

    def forward(self, x, x_gps=None):
        # 🔽 Short-circuit forward if using Claude's fusing strategies 🔽
        if self.fusion_strategy == 'mean_pooling_img_gps':
            return {
                lvl: self.transfer_heads[lvl](x, x_gps)
                for lvl in self.eunis_lvl
            }

        # 🔽 Non-Claude fusing strategies 🔽
        features = self.extract_features_img(x)
        if self.use_gps_encoder:
            features_gps = self.extract_features_gps(x_gps)
            features = self.fusion(self.fusion_strategy, [features, features_gps])

        if self.gradcam_head:
            return self.heads[self.gradcam_head](features)

        return {
            lvl: self.heads[lvl](features)
            for lvl in self.eunis_lvl
        }

    def extract_features_gps(self, x):
        if self.bb_model_name in ['simclr_landscape_resnet']:
            # self.backbone.modality_contrastive_head.head = nn.Identity()
            feat = self.backbone.gps_encoder(x)
            feat = self.backbone.gps_contrastive_head(feat)
        else:
            raise NotImplementedError(f"Feature extraction not implemented for model {self.bb_model_name}")
        return feat

    def extract_features_img(self, x):
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
        elif self.bb_model_name in ['simclr_landscape_resnet']:
            # self.backbone.modality_contrastive_head.head = nn.Identity()
            feat = self.backbone.modality_encoder(x)
            feat = self.backbone.modality_contrastive_head(feat)
        else:
            raise NotImplementedError(f"Feature extraction not implemented for model {self.bb_model_name}")
        return feat


# ==== Claude (Sonnet 5) models ====

class MeanPoolFusionHead(nn.Module):
    """
    Mean-pooling fusion for embeddings that already live in a shared contrastive
    space (e.g. image and GPS encoders trained under the same InfoNCE loss).

    Requires image_embed_dim == gps_embed_dim, since you're averaging, not projecting.
    If dims differ, you need a linear projection first (see note below).
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256, num_classes: int = 10,
                 dropout: float = 0.3, skip_gps: bool = False):
        super().__init__()
        self.skip_gps = skip_gps
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, img_embed: torch.Tensor, gps_embed: torch.Tensor):
        # Normalize to match pretraining-time geometry (SimCLR embeddings are
        # typically L2-normalized before the InfoNCE loss is computed)
        if not self.skip_gps:
            img_embed = F.normalize(img_embed, dim=-1)
            gps_embed = F.normalize(gps_embed, dim=-1)

            # Mean pool in the shared space
            fused = (img_embed + gps_embed) / 2.0   # (B, D)

            # Optional: re-normalize after averaging, since averaging two unit
            # vectors does NOT generally produce a unit vector (it shrinks toward
            # the origin as the two vectors diverge, which is actually informative —
            # see note below — but re-normalize if your head expects unit-norm input)
            # fused = F.normalize(fused, dim=-1)

            return self.head(fused)
        return self.head(F.normalize(img_embed, dim=-1))

# --- Wiring it up with frozen pretrained encoders ---

class TransferModel(nn.Module):
    def __init__(self, image_encoder: nn.Module, gps_encoder: nn.Module,
                 embed_dim: int, num_classes: int, freeze_encoders: bool = True):
        super().__init__()
        self.image_encoder = image_encoder
        self.gps_encoder = gps_encoder
        self.skip_gps = gps_encoder is None

        if freeze_encoders:
            for enc in (self.image_encoder, self.gps_encoder):
                if enc:
                    for p in enc.parameters():
                        p.requires_grad = False

        self.fusion_head = MeanPoolFusionHead(embed_dim, num_classes=num_classes, skip_gps=self.skip_gps)

    def forward(self, image: torch.Tensor, gps: torch.Tensor):
        img_embed = self.image_encoder(image)   # (B, D)
        if not self.skip_gps:
            gps_embed = self.gps_encoder(gps)        # (B, D)
            return self.fusion_head(img_embed, gps_embed)
        return self.fusion_head(img_embed, None)
