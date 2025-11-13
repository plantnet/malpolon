
import logging
import os
from abc import abstractmethod
from copy import deepcopy
from typing import Any, List
import shutil
import yaml
import wandb
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt


def get_rolled_eye_mask(sim_matrix: torch.tensor, remove_main_diag: bool = True):
    n, m = sim_matrix.shape  # (batch_size, batch_size-1)
    main_diag_mask = torch.eye(n, dtype=torch.bool).to(sim_matrix.device)
    pos_mask = torch.roll(torch.eye(n), n//2, dims=1).to(sim_matrix.device)  # not the main diagonal
    if remove_main_diag:
        pos_mask = pos_mask[~main_diag_mask.bool()].view(pos_mask.shape[0], -1)  # remove main diagonal from the mask
    return pos_mask.bool()


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self, mask='double_diag'):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-4)
        self.mask = mask
    
    def add_rolled_eye(self, x):
        n = x.shape[0]
        rolled_diag_mask = torch.roll(torch.eye(n), n//2, dims=1).to(x.device)
        return torch.where(rolled_diag_mask.bool().to(x.device), torch.tensor([-1]).to(x.device), x)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        ## save_heatmap(dots, filename="Heatmap_dots.png", title="Dots")
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        if self.mask == 'double_diag':
            dots = self.add_rolled_eye(dots)
        elif self.mask == 'main_diag':
            main_diag = torch.eye(n)
            dots = torch.where(main_diag.bool().to(x.device), torch.tensor([-1]).to(x.device), dots)
        ## save_heatmap(dots, filename="Heatmap_dots.png", title="Dots")
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        ## save_heatmap_with_max_neg_sim(dots, highlight_indices=I, filename="Dots_diag_max_neg_sim.png", title="Dots with diag to -1 and max negative similarity")
        ## save_1d_tensor_as_vertical_image(I, filename="I.png")
        return I

    def forward(self, output, eps=1e-8):
        """
        Args:
            output (BxD): backbone output of student
        """
        with torch.amp.autocast('cuda', enabled=False):
            ## save_heatmap(output, filename="Heatmap_output.png", title="Output")
            output = F.normalize(output, eps=eps, p=2, dim=-1)
            ## save_heatmap(output, filename="Heatmap_output_norm.png", title="Output norm")
            I = self.pairwise_NNs_inner(output)  # noqa: E741
            distances = self.pdist(output, output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss

# Does not converge
def crisp_loss_manual(sim_raw, temperature=2.659,
                      return_sim_matrix_and_targets=False):
    """Computes the CRISP loss from Huynh et al., ECCV 2024.
    
    Same as crisp_loss but manual implementation of Softmax with log-sum-exp trick.
    """
    sim = torch.from_numpy(sim_raw) if isinstance(sim_raw, np.ndarray) else sim_raw
    sim /= temperature
    N = sim.shape[0]
    positive_mask = torch.eye(N).bool()

    sim_max_col = sim.max(dim=1, keepdim=True).values
    sim_max_row = sim.max(dim=0, keepdim=True).values

    log_sum_exp_col = sim_max_col + torch.log(torch.sum(torch.exp(sim - sim_max_col), dim=1, keepdim=True))
    log_sum_exp_row = sim_max_row + torch.log(torch.sum(torch.exp(sim - sim_max_row), dim=0, keepdim=True))
    positive_matches = sim[positive_mask]

    nll_row = -(positive_matches - log_sum_exp_row)  # -x_n + lse(x)
    nll_col = -(positive_matches - log_sum_exp_col)
    nll = torch.tensor([nll_row.mean(), nll_col.mean()]).mean()
    nll = nll.to(sim.device)
    if return_sim_matrix_and_targets:
        return nll, sim, torch.where(positive_mask.bool())[0].to(sim.device)
    return nll

# Converges
def crisp_loss(sim, temperature=2.659, return_sim_matrix_and_targets=False):
    """
    Manual implementation of the symmetric CRISP / InfoNCE loss.
    sim: (N, N) similarity matrix between two sets of embeddings.
    Positive pairs are on the diagonal.
    """
    N = sim.shape[0]
    sim = sim / temperature
    labels = torch.arange(N, device=sim.device)
    
    # Row-wise and column-wise cross entropy
    loss_i = F.cross_entropy(sim, labels)
    loss_t = F.cross_entropy(sim.T, labels)
    loss = (loss_i + loss_t)
    if return_sim_matrix_and_targets:
        return loss, sim, labels.to(sim.device)
    return loss

def cosine_similarity_mean(cos_sim, lambd=0.8, mask='double_diag'):
#     """Computes a loss directly from a cosine similarity matrix.

#     Takes a cosine similarity matrix and computes a loss that maximizes the mean of the positive
#     pairs similarity while minimizing the mean of the negative pairs similarity.
#     """
#     n = cos_sim.shape[0]
#     if isinstance(cos_sim, np.ndarray):
#         cos_sim = torch.from_numpy(cos_sim)
#     if mask == 'double_diag':
#         main_diag_mask = torch.eye(n, dtype=torch.bool).to(cos_sim.device)
#         diag_mask = torch.roll(main_diag_mask, n//2, dims=1)
#         diag_mask = diag_mask[~main_diag_mask].view(cos_sim.shape[0], -1)  # remove main diagonal from the mask
#     else:
#         diag_mask = torch.eye(n, dtype=torch.bool).to(cos_sim.device)
#     cos_sim_pos = cos_sim[diag_mask].view(cos_sim.shape[0], -1)
#     cos_sim_neg = cos_sim[~diag_mask].view(cos_sim.shape[0], -1)
#     cos_sim_loss = -(lambd * cos_sim_pos).mean() + ((1 - lambd) * cos_sim_neg).mean()
#     return cos_sim_loss
    pass

def cosine_loss_pytorch_like(cos_sim, lambd=0.8, margin=0.2, mask='double_diag'):
#     """Computes a loss directly from a cosine similarity matrix.

#     Manual re-implementation of the PyTorch CosineEmbeddingLoss behavior.
#     """
#     n = cos_sim.shape[0]
#     if isinstance(cos_sim, np.ndarray):
#         cos_sim = torch.from_numpy(cos_sim)
#     if mask == 'double_diag':
#         main_diag_mask = torch.eye(n, dtype=torch.bool).to(cos_sim.device)
#         diag_mask = torch.roll(main_diag_mask, n//2, dims=1)
#         diag_mask = diag_mask[~main_diag_mask].view(cos_sim.shape[0], -1)  # remove main diagonal from the mask
#     elif mask == 'main_diag':
#         diag_mask = torch.eye(n, dtype=torch.bool).to(cos_sim.device)
#     cos_sim_pos = cos_sim[diag_mask].view(cos_sim.shape[0], -1)
#     cos_sim_neg = cos_sim[~diag_mask].view(cos_sim.shape[0], -1)
#     cos_sim_loss = lambd*(1 - cos_sim_pos.mean()) + (1 - lambd) * (max(0, cos_sim_neg.mean()-margin))
#     return cos_sim_loss
    pass

# Converges (single & double diag)
def cosine_embedding_from_sim(sim_matrix, pos_weight=0.8, margin=0.3, mask='double_diag'):
    """Computes a loss directly from a cosine similarity matrix.
    
    Compute the cosine similarity from pre-computed sim matrix, manually, like PyTorch does
    with CosineEmbeddingLoss.
    """
    if isinstance(sim_matrix, np.ndarray):
        sim_matrix = torch.from_numpy(sim_matrix)
    if mask == 'double_diag':
        pos_mask = get_rolled_eye_mask(sim_matrix)
    else:
        n, _ = sim_matrix.shape  # (batch_size, batch_size-1)
        pos_mask = torch.eye(n, dtype=torch.bool).to(sim_matrix.device)
        
    pos = sim_matrix[pos_mask]
    neg = sim_matrix[~pos_mask]
    # Same idea as CosineEmbeddingLoss
    loss_pos = (1 - pos).mean()
    loss_neg = F.relu(neg - margin).mean()  # ReLu because the CosineEmbeddingLoss does max(0, cos(x1,x2) - margin)
    return pos_weight * loss_pos + (1 - pos_weight) * loss_neg

def cosine_embedding_loss(img_emb, gps_emb, pos_weight=0.8, margin=0.3):
    """Compute the CosineEmbeddingLoss from raw embeddings.
    
    Uses the PyTorch implementation. Only works with even batch sizes.
    """
    batch_size, dim = img_emb.shape
    assert batch_size % 2 == 0, "Batch size must be even."
    
    def construct_mask(bs):
        N = bs * bs
        mask = torch.zeros(N, dtype=torch.int) -1
        mod_values = [(N + i) % bs for i in range(bs)]
        for i, v in enumerate(mod_values):
            block_start = v*bs
            block_end = block_start + bs
            mask[block_start+i] = 1
        return mask

    criterion = nn.CosineEmbeddingLoss(margin)

    # Create labels: +1 for diag (pos), -1 for off-diagonal (neg)
    pos_mask = construct_mask(batch_size)

    # Build all pair combinations
    ## Gives the same features pairs as those used to construct a sim_matrix from img_emb @ gps_emb.T
    img_i = img_emb.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, dim)       # [[A1, A2...], [B2, B2...]] -> [[A1, A2...], [A1, A2...], [B1, B2...], [B1, B2...]] | shape: (BS, dim) -> (BS*BS, dim)
    gps_j = gps_emb.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, dim)       # [[C1, C2...], [D2, D2...]] -> [[C1, C2...], [D1, D2...], [C1, C2...], [D1, D2...]] | shape: (BS, dim) -> (BS*BS, dim)
    y = pos_mask.to(img_emb.device)  # [1,-1,-1, -1,1,-1, -1,-1,1]

    # Compute loss
    loss1 = criterion(img_i, gps_j, y)
    loss2 = criterion(gps_j, img_i, y)
    loss = torch.tensor([loss1, loss2]).mean()
    return loss

# === Minimum Covariance Regularizer ===
def half_logdet(X):
    return torch.linalg.cholesky_ex(X)[0].diagonal().log().sum()

class MCR(torch.nn.Module):
    """Maximum Coding Rate"""
    def __init__(self, eps=0.05):
        super(MCR, self).__init__()
        self.eps = eps
    
    def forward(self, X):
        m, p = X.shape
        # X = F.normalize(X, dim=-1, p=2)
        cov = X.T @ X  # [p, p]
        scalar = p / (m * self.eps)
        I = torch.eye(p, device=X.device)
        loss = -half_logdet(I + scalar * cov)
        loss *= (p + m) / (p * m)  # balancing factor
        return loss

def cosine_mcr(img_emb, gps_emb, weight_mcr, eps_mcr=0.05):
    mcr = MCR(eps=eps_mcr)
    img_emb = F.normalize(img_emb, dim=-1, p=2)
    gps_emb = F.normalize(gps_emb, dim=-1, p=2)
    # cosine = F.mse_loss(img_emb, gps_emb)
    cosine = (1 - F.cosine_similarity(img_emb, gps_emb)).mean()
    mcr_img, mcr_gps = mcr(img_emb), mcr(gps_emb)
    print(f'Cosine: {cosine.item()}, MCR_weighted: {weight_mcr * (mcr_img + mcr_gps).mean()}, MCR_img: {mcr_img.item()}, MCR_gps: {mcr_gps.item()}')
    loss = cosine + weight_mcr * (mcr_img + mcr_gps).mean()  # over-parametrization between weight_mcr & eps
    return loss