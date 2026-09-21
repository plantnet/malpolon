"""Backbone adapters and the shared multi-head EUNIS classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models
from geoclip import LocationEncoder



def freeze_dinov2_blocks(model, N: int = 2):
    """Freeze up to the N last attention blocks of a DinoV2 model.

    DinoV2 architecture:
        ```
        <block 0>
        <block 1>
        ...
        <block N>
        (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (fc_norm): Identity()
        (head_drop): Dropout(p=0.0, inplace=False)
        (head): Linear(in_features=768, out_features=7806, bias=True)
        ```

    Args:
        model (torchvision.model): a dinov2 model
        N (int, optional): Number of attention blocks to keep learnable. Defaults to 2.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last N transformer blocks
    for block in model.blocks[-N:]:
        for param in block.parameters():
            param.requires_grad = True

    # Unfreeze the final norm layer
    for param in model.norm.parameters():
        param.requires_grad = True
    
    # Unfreeze the classifier head
    for param in model.head.parameters():
        param.requires_grad = True


def freeze_resnet_blocks(model: nn.Module, n: int) -> None:
    """
    Freeze the first N residual blocks of a ResNet.

    Residual blocks are counted sequentially:
        layer 1
          block 1
          block 2
          ...
        layer  2
        ...
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Linear(in_features=512, out_features=1000, bias=True)

    Everything before/inside those N blocks is frozen.
    Everything after those N blocks is trainable.

    The classifier (model.fc) is always left trainable.

    Args:
        model: torchvision-style ResNet.
        n: Number of residual blocks to freeze. n=0 freezes no
           residual blocks.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    # First freeze everything.
    for param in model.parameters():
        param.requires_grad = False

    # Residual blocks in execution order.
    blocks = []
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name)
        blocks.extend(layer)

    if n > len(blocks):
        raise ValueError(
            f"Requested n={n}, but model only has "
            f"{len(blocks)} residual blocks."
        )

    # Unfreeze blocks after the first N.
    for block in blocks[n:]:
        for param in block.parameters():
            param.requires_grad = True

    # Keep classifier trainable.
    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True


def freeze_geoclip_blocks(model: nn.Module, n: int) -> None:
    """
    Freeze the first N GeoCLIP LocationEncoder capsules.

    3 Blocks ordered:
        ...
        (LocEnc2): LocationEncoderCapsule(
            (capsule): Sequential(
            (0): GaussianEncoding()
            (1): Linear(in_features=512, out_features=1024, bias=True)
            (2): ReLU()
            (3): Linear(in_features=1024, out_features=1024, bias=True)
            (4): ReLU()
            (5): Linear(in_features=1024, out_features=1024, bias=True)
            (6): ReLU()
            )
            (head): Sequential(
            (0): Linear(in_features=1024, out_features=512, bias=True)
            )
        )
    (nothing after the 3 blocks, the heads are self contained)
    

    Blocks after N remain trainable.

    The rest of GeoCLIP (image encoder, logit_scale, etc.) is frozen.

    Args:
        model: GeoCLIP model.
        n: Number of location-encoder blocks to freeze.
    """
    if n < 0 or n > 3:
        raise ValueError(f"n must be between 0 and 3, got {n}")

    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad = False

    # GeoCLIP's three hierarchical location encoder blocks.
    blocks = [
        model.location_encoder.LocEnc0,
        model.location_encoder.LocEnc1,
        model.location_encoder.LocEnc2,
    ]

    # Unfreeze blocks after the first N.
    for block in blocks[n:]:
        for param in block.parameters():
            param.requires_grad = True
         

def try_freeze_backbone(model: nn.Module, name: str, granularity: str, N: int) -> None:
    """Try to freeze the first N blocks of a supported backbone.
    

    Args:
        model: The model to freeze.
        name: The name of the backbone. One of 'dinov2', 'resnet', or 'geoclip'.
        granularity: The granularity of freezing. One of 'block' or 'layer'.
        N: The number of blocks to freeze.
    """
    if granularity == "blocks":
        if N>0:
            if name.startswith("dinov2"):
                freeze_dinov2_blocks(model, N)
            elif name.startswith("resnet"):
                freeze_resnet_blocks(model, N)
            elif name.startswith("geoclip"):
                freeze_geoclip_blocks(model, N)
            else:
                raise ValueError(f"Unsupported backbone '{name}' for freezing")
        else:
            for param in model.parameters():
                param.requires_grad = False
            print(f"[INFO] All parameters frozen.", flush=True)
    elif granularity == "layers":
        raise ValueError(f"Unsupported granularity '{granularity}' for freezing")

class GPSEncoder(nn.Module):
    """Project latitude/longitude coordinates into a learned feature vector.
    
    Backbone takes values in: [mlp, geoclip] or be None.
    If neither of those: raises an error.
    If None: the model won't use GPS coordinates.
    """
    def __init__(self, name: str, output_dim: int = 32):
        super().__init__()
        backbones = {'mlp': nn.Sequential(nn.Linear(2, output_dim), nn.LayerNorm(output_dim), nn.GELU()),
                     'geoclip': LocationEncoder()}
        if isinstance(name, str) and name not in backbones.keys():
            raise ValueError(f'[ERROR]: build_gps_encoder: GPS backbone {name} not supported.')
        self.network = backbones.get(name, None)
        
        out_dims = {'mlp': output_dim,
                    'geoclip': self.network.LocEnc1.head[0].out_features}
        self.output_dim = out_dims[name]

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Normalise degrees to an approximately unit range and encode them."""
        if not torch.isfinite(coordinates).all():
            raise ValueError("GPS was requested but a batch contains missing/non-numeric coordinates")
        # Geographic degrees have very different magnitude from image embeddings.
        return self.network(coordinates / coordinates.new_tensor([90.0, 180.0]))


class TorchvisionFeatureExtractor(nn.Module):
    """Expose image embeddings from supported torchvision architectures.

    The original classification layer is replaced once during construction with
    ``Identity``. This avoids the unsafe pattern of mutating a model during each
    forward pass and lets all EUNIS heads consume the same embedding.
    """
    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        weights = {'resnet18': 'IMAGENET1K_V1',
                   'resnet50': 'IMAGENET1K_V2',
                   'convnext': 'IMAGENET1K_V1',
                   'vgg16': 'IMAGENET1K_V1',
                   'vitb32': 'IMAGENET1K_V1',
                   'mobilenet_v3': 'IMAGENET1K_V1',
                   'inception_v3': 'IMAGENET1K_V1'}
        if pretrained is None:
            # An explicit ``None`` prevents torchvision from trying to fetch weights.
            weights[name] = None
            raise ValueError("Torchvision pretrained downloads are disabled for this offline pipeline. Use --pretrained false, or provide a local PN22M checkpoint.")
        if name in {"resnet18", "resnet50"}:
            constructor = getattr(models, name)
            self.backbone = constructor(weights=weights[name])
            self.output_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif name == "convnext":
            self.backbone = models.convnext_base(weights=weights[name])
            self.output_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
        elif name == "vgg16":
            self.backbone = models.vgg16(weights=weights[name])
            self.output_dim = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Identity()
        elif name == "vitb32":
            self.backbone = models.vit_b_32(weights=weights[name])
            self.output_dim = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
        elif name == "mobilenet_v3":
            self.backbone = models.mobilenet_v3_large(weights=weights[name])
            self.output_dim = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Identity()
        elif name == "inception_v3":
            self.backbone = models.inception_v3(weights=weights[name], aux_logits=False)
            self.output_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported torchvision backbone '{name}'")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return a batch of unclassified image embeddings."""
        return self.backbone(images)


class TimmDinoExtractor(nn.Module):
    """Expose CLS embeddings from the locally installed DINOv2-small model."""
    def __init__(self, pretrained: bool):
        super().__init__()
        if pretrained:
            raise ValueError("timm pretrained downloads are disabled for this offline pipeline. Use a locally supplied checkpoint instead.")
        import timm
        self.backbone = timm.create_model("timm/vit_small_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0)
        self.output_dim = self.backbone.num_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class PN22MDinoExtractor(nn.Module):
    """Expose CLS embeddings from the local PN22M DINOv2-large checkpoint."""
    def __init__(self, checkpoint: Path, pretrained: bool):
        super().__init__()
        if pretrained and not checkpoint.is_file():
            raise FileNotFoundError(f"PN22M checkpoint not found: {checkpoint}")
        from .dino_v2_large_PN22M.models import vit_large
        import re
        self.backbone = vit_large(patch_size=16, img_size=224, init_values=0.1, block_chunks=0, num_register_tokens=4)
        if pretrained:
            # The project checkpoint stores the teacher backbone with a prefix
            # and chunked block names that must be normalised before loading.
            state = torch.load(checkpoint, map_location="cpu")["teacher"]
            state = {re.sub(r"blocks\.\d+\.(?=\d+)", "blocks.", key.removeprefix("backbone.")): value for key, value in state.items() if key.startswith("backbone.")}
            self.backbone.load_state_dict(state, strict=True)
        self.output_dim = self.backbone.embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(images)["x_norm_clstoken"]


def build_image_encoder(name: str, pretrained: bool, pn22m_checkpoint: Path = Path("eunis_pipeline/dino_v2_large_PN22M/ssl_vitl_224.pth")) -> nn.Module:
    """Instantiate an image encoder while preserving the offline-only policy.

    ``pretrained=True`` is only meaningful for PN22M, which reads its explicitly
    local checkpoint; other adapters reject it rather than downloading weights.
    """
    if name == "dinov2_vits14":
        return TimmDinoExtractor(pretrained)
    if name == "dinov2_PN22M":
        return PN22MDinoExtractor(pn22m_checkpoint, pretrained)
    return TorchvisionFeatureExtractor(name, pretrained)


def build_gps_encoder(name: str, output_dim: int = 32) -> nn.Module:
    """Instantiate a GPS encoder.
    
    Currently supports MLP and GeoClip.
    """
    return GPSEncoder(name, output_dim=output_dim)


class MultiHeadEunisClassifier(nn.Module):
    """Shared image encoder, optional GPS encoder, and one classifier per EUNIS level.

    GPS fusion is either concatenation (preserves all image and coordinate
    features) or mean pooling after projecting GPS features into image-embedding
    space. Each level has independent linear logits but shares the encoders.
    """
    def __init__(self,
                 image_encoder: nn.Module,
                 gps_encoder: nn.Module,
                 image_encoder_name: str,
                 gps_encoder_name: str,
                 class_counts: dict[str, int],
                 levels: tuple[str, ...],
                 use_gps: bool = False,
                 fusion: str = "concat",
                 freeze_img_backbone_dict: bool = False,
                 freeze_gps_backbone_dict: bool = False,
    ):
        super().__init__()
        self.image_encoder, self.gps_encoder, self.use_gps, self.fusion = image_encoder, gps_encoder, use_gps, fusion
        image_dim = image_encoder.output_dim
        gps_embedding_dim = gps_encoder.output_dim if use_gps else 0
        print(freeze_img_backbone_dict)
        if freeze_img_backbone_dict['freeze']:
            try_freeze_backbone(self.image_encoder, image_encoder_name, freeze_img_backbone_dict['granularity'], freeze_img_backbone_dict['unfreeze_N'])
        if freeze_gps_backbone_dict['freeze'] and self.gps_encoder is not None:
            try_freeze_backbone(self.gps_encoder, gps_encoder_name, freeze_gps_backbone_dict['granularity'], freeze_gps_backbone_dict['unfreeze_N'])

        if use_gps and fusion == "mean":
            self.gps_projection = nn.Linear(gps_embedding_dim, image_dim)
            feature_dim = image_dim
        elif use_gps and fusion == "concat":
            self.gps_projection, feature_dim = nn.Identity(), image_dim + gps_embedding_dim
        elif use_gps:
            raise ValueError("fusion must be 'concat' or 'mean'")
        else:
            self.gps_projection, feature_dim = None, image_dim
        self.heads = nn.ModuleDict(
            {level: nn.Linear(feature_dim, class_counts[level]) for level in levels}
        )

    # def forward(self, images: torch.Tensor, gps: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    #     """Return unnormalised class logits keyed by EUNIS level."""
    #     features = self.image_encoder(images)
    #     if self.use_gps:
    #         if gps is None:
    #             raise ValueError("This model was configured with GPS, but no GPS tensor was supplied")
    #         gps_features = self.gps_projection(self.gps_encoder(gps))
    #         features = (
    #             torch.cat((features, gps_features), dim=-1)
    #             if self.fusion == "concat"
    #             else (features + gps_features) / 2
    #         )
    #     return {level: head(features) for level, head in self.heads.items()}

    def forward(self, images: torch.Tensor, gps: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return unnormalised class logits keyed by EUNIS level.

        ``gps`` may contain rows with NaN latitude/longitude for sites where no
        coordinate was recorded (see ``EunisMultiLabelDataset``/``_coordinate``).
        Those rows fall back to image-only features instead of raising, while
        rows with a real coordinate keep the exact original fused behaviour.
        """
        features = self.image_encoder(images)
        if self.use_gps:
            if gps is None:
                raise ValueError("This model was configured with GPS, but no GPS tensor was supplied")
            # A site is usable only if BOTH coordinates were recorded and finite.
            gps_present = torch.isfinite(gps).all(dim=-1)
            # GPSEncoder requires an entirely finite batch (see its docstring);
            # substitute a finite placeholder for missing rows just so it can
            # run, then zero out exactly those rows' contribution below.
            safe_gps = torch.where(gps_present.unsqueeze(-1), gps, torch.zeros_like(gps))
            gps_features = self.gps_projection(self.gps_encoder(safe_gps))
            gps_features = gps_features * gps_present.unsqueeze(-1).to(gps_features.dtype)
            if self.fusion == "concat":
                # Missing-GPS rows get a zero-filled GPS segment; rows with a
                # real coordinate are fused exactly as before.
                features = torch.cat((features, gps_features), dim=-1)
            else:
                fused = (features + gps_features) / 2
                features = torch.where(gps_present.unsqueeze(-1), fused, features)
        return {level: head(features) for level, head in self.heads.items()}
