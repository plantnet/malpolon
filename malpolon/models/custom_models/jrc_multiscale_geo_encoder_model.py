"""This module provides a model to align features from multiscale geo-tagged data.

Author: Theo Larcher <theo.larcher@inria.fr>
        Alexis Joly <alexis.joly@inria.fr>

License: GPLv3
Python version: 3.12.9
"""
import timm
from torch import nn
from torchvision import models
from malpolon.models.custom_models.glc2024_multimodal_ensemble_model import \
    MultimodalEnsemble
import pandas as pd
import timm
import torch
import torch.nn.functional as F


def info_nce_loss_1_to_k(query, positives, negatives, temperature=0.07):
    """
    Computes 1-to-K InfoNCE contrastive loss.

    query: (batch_size, dim) - Query embeddings
    positives: (batch_size, K, dim) - Multiple positive embeddings per query
    negatives: (batch_size, N, dim) - Negative embeddings
    temperature: Softmax temperature scaling
    """
    query = F.normalize(query, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Compute similarities
    pos_sim = torch.matmul(query.unsqueeze(1), positives.transpose(1, 2)).squeeze(1)  # (batch_size, K)
    neg_sim = torch.matmul(query, negatives.transpose(1, 2))  # (batch_size, N)

    # Combine all similarities
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature  # (batch_size, K+N)
    
    # Labels: positives are at indices [0:K]
    labels = torch.arange(query.size(0), device=query.device).repeat_interleave(K)

    # Use CrossEntropyLoss where multiple indices are considered positive
    return F.cross_entropy(logits, labels)



def init_pn_model(class_mapping="class_mapping.txt",
                  species_mapping="species_id_to_name.txt",
                  pretrained_path="model_best.pth.tar",):
    root = "jrc_multicale/PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/"
    def load_class_mapping(class_list_file):
        with open(class_list_file) as f:
            class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
        return class_index_to_class_name


    def load_species_mapping(species_map_file):
        df = pd.read_csv(species_map_file, sep=";", quoting=1, dtype={"species_id": str})
        df = df.set_index("species_id")
        return df["species"].to_dict()

    cid_to_spid = load_class_mapping(class_mapping)
    spid_to_sp = load_species_mapping(species_mapping)

    device = torch.device(device)

    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        num_classes=len(cid_to_spid),
        checkpoint_path=pretrained_path,
    )
    model = model.to(device)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

class MultiScaleGeoEncoder(nn.Module):
    # Need to add 4 encoders:
    # 1. Pl@ntNet encoder (species scale)
    # 2. MME encoder (satellite scale)
    # 3. Any ResNet encoder (LUCAS scale)
    # 4. GeoCLIP encoder (GPS encoding)
    def __init__(self):
        self.scale_1_species = init_pn_model()
        self.scale_1_species.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        
        self.scale_2_landscape = models.resnet18(weights=None)
        
        # TODO: For satellite, keep the satellite branch, cut the last 2 fc layers
        self.scale_3_satellite_encoder = MultimodalEnsemble()

    def forward(self, img_species, img_landscape, img_satellite):
        feat_species = self.scale_1_species(img_species)
        feat_landscape = self.scale_2_landscape(img_landscape)
        feat_satellite = self.scale_3_satellite_encoder(img_satellite)
        out = torch.cat([feat_species, feat_landscape, feat_satellite], dim=1)
        return out