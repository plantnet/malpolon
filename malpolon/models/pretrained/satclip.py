from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

import timm
import torchgeo.models
from torchgeo.models import ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights
from location_encoder import get_positional_encoding, get_neural_network, LocationEncoder
from datamodules.s2geo_dataset import S2Geo


def get_positional_encoding(name, legendre_polys=10, harmonics_calculation='analytic', min_radius=1, max_radius=360, frequency_num=10):
    if name == "direct":
        return PE.Direct()
    elif name == "cartesian3d":
        return PE.Cartesian3D()
    elif name == "sphericalharmonics":
        if harmonics_calculation == 'discretized':
            return PE.DiscretizedSphericalHarmonics(legendre_polys=legendre_polys)
        else:
            return PE.SphericalHarmonics(legendre_polys=legendre_polys,
                                         harmonics_calculation=harmonics_calculation)
    elif name == "theory":
        return PE.Theory(min_radius=min_radius,
                         max_radius=max_radius,
                         frequency_num=frequency_num)
    elif name == "wrap":
        return PE.Wrap()
    elif name in ["grid", "spherec", "spherecplus", "spherem", "spheremplus"]:
        return PE.GridAndSphere(min_radius=min_radius,
                       max_radius=max_radius,
                       frequency_num=frequency_num,
                       name=name)
    else:
        raise ValueError(f"{name} not a known positional encoding.")

def get_neural_network(name, input_dim, num_classes=256, dim_hidden=256, num_layers=2):
    if name == "linear":
        return nn.Linear(input_dim, num_classes)
    elif name ==  "mlp":
        return MLP(
                input_dim=input_dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers,
                out_dims=num_classes
        )
    elif name ==  "siren":
        return SirenNet(
                dim_in=input_dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers,
                dim_out=num_classes
            )
    elif name ==  "fcnet":
        return FCNet(
                num_inputs=input_dim,
                num_classes=num_classes,
                dim_hidden=dim_hidden
            )
    else:
        raise ValueError(f"{name} not a known neural networks.")

class LocationEncoder(nn.Module):
    def __init__(self, posenc, nnet):
        super().__init__()
        self.posenc = posenc
        self.nnet = nnet

    def forward(self, x):
        x = self.posenc(x)
        return self.nnet(x)


class SatCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int, str],
                 vision_width: int,
                 vision_patch_size: int,
                 in_channels: int,
                 # location
                 le_type: str,
                 pe_type: str,
                 frequency_num: int,
                 max_radius: int,
                 min_radius: int,
                 harmonics_calculation: str,
                 legendre_polys: int = 10,
                 sh_embedding_dims: int = 16,
                 ffn: bool = True,
                 num_hidden_layers: int = 2,
                 capacity: int = 256,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            print('using modified resnet')
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                in_channels=in_channels
            )

        elif vision_layers == 'moco_resnet18':
            print('using pretrained moco resnet18')
            weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.visual = timm.create_model("resnet18", in_chans=in_chans, num_classes=embed_dim)
            self.visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.visual.requires_grad_(False)
            self.visual.fc.requires_grad_(True)

        elif vision_layers == 'moco_resnet50':
            print('using pretrained moco resnet50')
            weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.visual = timm.create_model("resnet50", in_chans=in_chans, num_classes=embed_dim)
            self.visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.visual.requires_grad_(False)
            self.visual.fc.requires_grad_(True)

        elif vision_layers == 'moco_vit16':
            print('using pretrained moco vit16')
            weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.visual = timm.create_model("vit_small_patch16_224", in_chans=in_chans,
                                            num_classes=embed_dim)
            self.visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.visual.requires_grad_(False)
            self.visual.head.requires_grad_(True)

        else:
            print('using vision transformer')
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                in_channels=in_channels
            )

        self.posenc = get_positional_encoding(name=le_type,
                                              harmonics_calculation=harmonics_calculation,
                                              legendre_polys=legendre_polys, min_radius=min_radius,
                                              max_radius=max_radius,
                                              frequency_num=frequency_num).double()
        self.nnet = get_neural_network(name=pe_type, input_dim=self.posenc.embedding_dim,
                                       num_classes=embed_dim, dim_hidden=capacity,
                                       num_layers=num_hidden_layers).double()
        self.location = LocationEncoder(self.posenc,
                                        self.nnet
                                        ).double()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3,
                                 self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        if isinstance(self.visual, timm.models.vision_transformer.VisionTransformer):
            return self.visual.patch_embed.proj.weight.dtype
        else:
            return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_location(self, coords):
        return self.location(coords.double())

    def forward(self, image, coords):

        image_features = self.encode_image(image)
        location_features = self.encode_location(coords).float()
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features = location_features / location_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ location_features.t()
        logits_per_location = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_location


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias",
                         "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)