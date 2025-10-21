"""This module provides a model to align features from multiscale geo-tagged data.

Author: Theo Larcher <theo.larcher@inria.fr>
        Alexis Joly <alexis.joly@inria.fr>

License: GPLv3
Python version: 3.12.9
"""
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

from malpolon.models.custom_models.jrc_multiscale.jrc_contrastive_losses import (
    KoLeoLoss, cosine_embedding_loss, cosine_similarity_mean, cosine_loss_pytorch_like,
    cosine_embedding_from_sim
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_1d_tensor_as_vertical_image(tensor, filename="tensor_vertical_image.png"):
    """
    Saves a 1D tensor as an image, displaying the values of the tensor's elements
    with their indices shown vertically.

    Args:
        tensor (torch.Tensor): The 1D tensor to visualize.
        filename (str): The name of the file to save the image.
    """
    # Ensure the tensor is on the CPU and convert to a NumPy array
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # Create a figure
    fig, ax = plt.subplots(figsize=(2, len(tensor) * 0.5))
    ax.axis("off")  # Turn off the axis

    # Create a table-like visualization
    for i, value in enumerate(tensor):
        # Display the value
        ax.text(
            1, len(tensor) - i - 0.5, str(value),  # Position and value
            ha="center", va="center",  # Center alignment
            fontsize=12, color="black", bbox=dict(boxstyle="square", facecolor="white")
        )
        # Display the index next to the value
        ax.text(
            0, len(tensor) - i - 0.5, str(i),  # Position and index
            ha="center", va="center",  # Center alignment
            fontsize=10, color="gray"
        )

    # Set limits to fit the tensor
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, len(tensor))

    # Save the image
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def save_heatmap(output, filename="heatmap.png", title=""):
    """
    Saves the given tensor `output` as a heatmap image.

    Args:
        output (torch.Tensor): The tensor to visualize as a heatmap.
        filename (str): The name of the file to save the heatmap.
    """
    # Ensure the tensor is on the CPU and convert to NumPy
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(output, cmap="viridis", aspect="auto")
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Samples")

    # Save the heatmap
    plt.savefig(filename)
    plt.close()

def save_heatmap_with_max_neg_sim(output, highlight_indices=None, filename="heatmap.png", title=""):
    """
    Saves the given tensor `output` as a heatmap image and highlights specific indices in red.

    Args:
        output (torch.Tensor): The tensor to visualize as a heatmap.
        filename (str): The name of the file to save the heatmap.
        highlight_indices (torch.Tensor): A 1D tensor containing column indices to highlight for each row.
    """
    # Ensure the tensor is on the CPU and convert to NumPy
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(output, cmap="viridis", aspect="auto")
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Samples")

    # Highlight specific indices in red
    if highlight_indices is not None:
        if isinstance(highlight_indices, torch.Tensor):
            highlight_indices = highlight_indices.detach().cpu().numpy()
        rows = np.arange(len(highlight_indices))
        plt.scatter(highlight_indices, rows, color="red", label="Highlighted Indices", s=10)

    # Add legend if highlights exist
    if highlight_indices is not None:
        plt.legend(loc="upper right")

    # Save the heatmap
    plt.savefig(filename)
    plt.close()

def save_checkpoint(state, is_best, dirpath='./wandb/'):
    torch.save(state, os.path.join(dirpath, 'last.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(dirpath, 'last.pth.tar'), os.path.join(dirpath, 'best.pth.tar'))

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def update_ema(model, ema_model, tau):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(tau).add_(param.data, alpha=1 - tau)


def mean_sim_matrices_over_modalities(sim_matrices: list, n_modalities: int = 3):
    """Reshape similarity matrices to have 3 modalities."""
    sim_matrices = np.array(sim_matrices)
    B, H, W = sim_matrices.shape
    n_steps = B // n_modalities
    sim_matrix_mean = np.mean(sim_matrices.reshape(n_steps, n_modalities, H, W), axis=0)
    return sim_matrix_mean

def log_input_imgs(images, inds, survey_ids, step, epoch_counter,
                   log_images=True, mode='train'):
    if not log_images:
        return
    # Log input batch images
    fig, axes = plt.subplots(4, 8, figsize=(16, 14))  # 4 rows, 8 columns
    axes = np.array(axes).flatten()
    for idx, (img, ind, sid, ax) in enumerate(zip(images, inds, survey_ids, axes)):
        img = img[:3, :, :] if img.shape[0] >= 3 else img[0, :, :]
        img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # Cutting multi-band images to the first 3
        ax.imshow(img, cmap='gray' if img.ndim < 3 else None)
        ax.set_title(f"Image {idx}, iter_idx {ind[0]}, \nsurveyId {sid}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    wandb.log({f'Input_imgs_{mode}/e_{epoch_counter:03d}_s_{step:03d}': wandb.Image(fig)})
    plt.close()

def log_input_imgs_multimodalities(images, inds, survey_ids, step, epoch_counter,
                                   n_samples=32, n_modalities=3, log_images=True, mode='train'):
    if (not log_images) or (step > 0 and epoch_counter > 1):  # Only log at the first step of the first 2 epochs for storage reasons
        return
    # Log input batch images
    px = 1/plt.rcParams['figure.dpi'] 
    fig, axes = plt.subplots(n_modalities, n_samples, figsize=(n_samples*200*px, n_modalities*256*px), constrained_layout=True)
    axes = np.array(axes).flatten()
    axi = 0
    rand_inds = torch.arange(n_samples)
    if n_samples < images[0].shape[0]:
        rand_inds = torch.randperm(images[0].shape[0])[:n_samples]
        fig.suptitle(f"Modalities randomly sampled to display {n_samples} samples each", fontsize=16, style='italic')
    for (image, ind, survey_id) in zip(images, inds, survey_ids):
        image, ind, survey_id = image[rand_inds], ind[rand_inds], survey_id[rand_inds]
        for idx, (img, ind, sid) in enumerate(zip(image, ind, survey_id)):
            img = img[:3, :, :] if img.shape[0] >= 3 else img[0, :, :]
            img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # Cutting multi-band images to the first 3
            axes[axi].imshow(img, cmap='gray' if img.ndim < 3 else None)
            axes[axi].set_title(f"Image {idx}, iter_idx {ind[0]}, \nsurveyId {sid}", fontsize=8)
            axes[axi].axis('off')
            axi += 1
    plt.tight_layout()
    wandb.log({f'Input_imgs_{mode}/e_{epoch_counter:03d}_s_{step:03d}': wandb.Image(fig)})
    plt.close()

def log_loss_scheduler(loss, scheduler, mode='train'):
    wandb.log({f"Loss_step/{mode}": loss})
    if scheduler:
        wandb.log({f"learning_rate": scheduler.get_last_lr()[0]})

def log_moments(norm_img, norm_gps, std_mean_img, std_mean_gps, std_mean_diff, mode='train'):
    # wandb.log({f"std-avg_img/{mode}": std_mean_img[0].mean(), f"mean-avg_img/{mode}": std_mean_img[1].mean(),
    #            f"std-avg_gps/{mode}": std_mean_gps[0].mean(), f"mean-avg_gps/{mode}": std_mean_gps[1].mean(),
    #            f"std-avg_diff/{mode}": std_mean_diff[0].mean(), f"mean-avg_diff/{mode}": std_mean_diff[1].mean()})
    wandb.log({f"norm_img_avg/{mode}": norm_img.mean(),
               f"norm_gps_avg/{mode}": norm_gps.mean(),
               f"norm_avg_diff/{mode}": abs(norm_img.mean()-norm_gps.mean())})

def log_similarity_matrix_step(sim_matrices, epoch_counter, step,
                               n_modalities=3, modalities_name=['species', 'landscape', 'satellite'], log_images=True, mode='train'):
    if not log_images:
        return
    fig, axes = plt.subplots(1, n_modalities, figsize=(12, 4))
    axes = np.array(axes).flatten()
    fig.suptitle("Similarity matrix (left to right: species, landscape, satellite)", fontsize=16)
    for i in range(n_modalities, 0, -1):
        hm = sns.heatmap(sim_matrices[-i], ax=axes[n_modalities-i], cmap='viridis', cbar=True, annot=False)
        axes[n_modalities-i].set_title(f"Sim matrix {modalities_name[n_modalities-i]}")
    wandb.log({f'SimMatrix_{mode}/e_{epoch_counter:03d}_s_{step:03d}': wandb.Image(hm)})
    plt.close()

def log_similarity_matrix_mean(sim_matrices, epoch_counter, step,
                               n_modalities=3, log_every_n_steps=None, modalities_name=['species', 'landscape', 'satellite'], log_images=True, mode='train'):
    if not log_images:
        return
    sim_matrix_mean = mean_sim_matrices_over_modalities(sim_matrices, n_modalities=n_modalities)
    fig, axes = plt.subplots(1, n_modalities, figsize=(12, 4))
    axes = np.array(axes).flatten()
    fig.suptitle("Similarity matrix (left to right: species, landscape, satellite)", fontsize=16)
    for i in range(n_modalities):
        hm = sns.heatmap(sim_matrix_mean[i], ax=axes[i], cmap='viridis', cbar=True, annot=False)
        axes[i].set_title(f"Sim matrix {modalities_name[i]}")
    if mode == 'val':
        wandb.log({f'SimMatrix_mean-epoch_{mode}/e{epoch_counter:03d}': wandb.Image(hm)})
    else:
        wandb.log({f'SimMatrix_mean-{log_every_n_steps}-steps_{mode}/{epoch_counter:03d}_s_{step:03d}': wandb.Image(hm)})
    plt.close()
    sim_matrices = []

def log_acc_topk_step(logits, labels, modality_name, topk=(1, 5), mode='train'):
    if logits[0].shape[0] >= max(topk):
        acc_topks = accuracy(logits, labels, topk=topk)
        if mode != 'test':
            for acc_topk, topk in zip(acc_topks, topk):
                wandb.log({f"acc/{mode}/top{topk}_{modality_name}": acc_topk[0]})
    else:
        print("Batch size is too small for accuracy calculation.")
        return None, None
    return tuple(map(lambda x: x[0].item(), acc_topks))

def log_tsne(features_img, features_gps, epoch_counter, modality_name, 
             log_images=True, mode='train'):
    if not log_images:
        return
    n = features_img.shape[0]
    embeddings = torch.cat([features_img, features_gps], dim=0)
    if sum(torch.isnan(torch.flatten(embeddings))) > 0:
        print("NaN detected in embeddings, skipping t-SNE log.")
        return
    tsne = TSNE(n_components=2, perplexity=min(30, n-1), learning_rate=200, metric='cosine', init='pca', random_state=42)
    proj = tsne.fit_transform(embeddings.detach().to('cpu').numpy())
    proj_a, proj_b = proj[:n], proj[n:]
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(proj_a[:, 0], proj_a[:, 1], c='red', label='Modality image', alpha=0.7)
    plt.scatter(proj_b[:, 0], proj_b[:, 1], c='blue', label='Modality GPS', alpha=0.7)
    for i in range(n):
        plt.plot([proj_a[i, 0], proj_b[i, 0]], [proj_a[i, 1], proj_b[i, 1]], 'gray', alpha=0.3)
    plt.title(f"t-SNE projection of contrastive embeddings GPS vs {modality_name} ({mode})"); plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2"); plt.grid(True); plt.legend()
    wandb.log({f"t-sne_{mode}/{mode}/e_{epoch_counter:03d}_{modality_name}": wandb.Image(fig)})
    plt.close()

def wandb_init():
    # Iterations metrics
    wandb.define_metric("epoch")
    wandb.define_metric("train_steps")
    wandb.define_metric("val_steps")
    
    wandb.define_metric("acc_epoch (batch avg)/*", step_metric="epoch")

    # Train metrics
    wandb.define_metric("Loss_step/train", step_metric="train_steps")
    wandb.define_metric("norm_img_avg/train", step_metric="train_steps")
    wandb.define_metric("norm_gps_avg/train", step_metric="train_steps")
    wandb.define_metric("norm_avg_diff/train", step_metric="train_steps")
    wandb.define_metric("acc/train/*", step_metric="train_steps")
    wandb.define_metric("Input_imgs_train/*", step_metric='train_steps')
    wandb.define_metric("SimMatrix_train/*", step_metric='train_steps')
    wandb.define_metric("Loss_epoch (batch avg)/train", step_metric="epoch")
    wandb.define_metric("acc_epoch (batch avg)/train/top1", step_metric="epoch")
    wandb.define_metric("acc_epoch (batch avg)/train/top5", step_metric="epoch")
    wandb.define_metric("t-sne/train/*", step_metric='epoch')

    # Validation metrics
    wandb.define_metric("Loss_step/val", step_metric="val_steps")
    wandb.define_metric("acc/val/*", step_metric="val_steps")
    wandb.define_metric("Input_imgs_val/*", step_metric='val_steps')
    wandb.define_metric("SimMatrix_val/*", step_metric='val_steps')
    wandb.define_metric("Loss_epoch (batch avg)/val", step_metric="epoch")
    wandb.define_metric("Input_imgs_val/*", step_metric='val_steps')
    wandb.define_metric("acc_epoch (batch avg)/val/top1", step_metric="epoch")
    wandb.define_metric("acc_epoch (batch avg)/val/top5", step_metric="epoch")
    wandb.define_metric("SimMatrix_mean-epoch_val/*", step_metric="epoch")
    wandb.define_metric("t-sne/val/*", step_metric='epoch')

class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.args.last_epoch = getattr(self.args, 'last_epoch', 0)
        self.model = kwargs['model'].to(self.args.device)
        self.koleo_weights = self.args.koleo_weights if hasattr(self.args, 'koleo_weights') else 0.0
        self.koleo_eps = self.args.koleo_eps if hasattr(self.args, 'koleo_eps') else 1e-4
        self.koleo_modalities = self.args.koleo_modalities if hasattr(self.args, 'koleo_modalities') else ['species', 'landscape', 'satellite']
        self.ema_model = kwargs['model'] if getattr(self.args, 'use_ema', False) else None
        self.ema_decay = getattr(self.args, 'ema_decay', 0.99)
        self.ema_update_step = getattr(self.args, 'ema_update_step', 1)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.resume_wandb_run = getattr(self.args, 'resume_wandb_run', False)
        self.log_images = getattr(self.args, 'log_images', True)
        self.skip_modalities = getattr(self.args, 'skip_modalities', [])
        self.inference = bool(getattr(self.args, 'predict', False))
        self.writer = wandb.init(
            entity="tlarcher-phd-jrc",
            id=self.args.ckpt_path.split('/')[-2].split('-')[2] if (self.args.ckpt_path and self.resume_wandb_run) else None,
            project=self.args.wandb_project,
            name=self.args.name,  #'Unique surveyId spatial split 0.06min, dropout',
            notes=f"Shuffle train ON, val OFF. Info_nce_loss symmetrical. "\
                  f"All unique surveyId obs."\
                  f"Modality backbone: {'frozen' if self.args.freeze_modality_backbone else 'hot'}"\
                  f"GPS backbone: {'frozen' if self.args.freeze_gps_backbone else 'hot'}"\
                  f"LR cosine annealing {self.args.learning_rate}. "\
                  f"Temp {self.args.temperature}. "\
                  f"Dropout {self.args.dropout}. "\
                  f"Weight_decay {self.args.weight_decay}. ",
            group="SimCLR: satellite VS GPS",
            config=kwargs['args'],
            job_type='inference' if self.inference else 'train',
        )
        print(f"[INFO] Wandb ID: {self.writer.id}")
        print(f"[INFO] Wandb output directory: {self.writer.dir}")
        logging.basicConfig(filename=os.path.join(self.writer.dir, 'training.log'), level=logging.DEBUG)
        wandb_init()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features, dataset_type: str = 'species'):
        # Flexible bastch_size strategy on hold
        # batch_size = self.args.batch_size
        # if dataset_type == 'landscape':
        #     batch_size = features.shape[0] // self.args.n_views  # LUCAS image views stacked along the batch dim
        labels = torch.cat([torch.arange(features.shape[0]//self.args.n_views) for i in range(self.args.n_views)], dim=0)
        # Labels is a vector of size 64 with values 0 to 31 concatenated n_views times. E.g. if n_views==2: [0, 1, 2, ..., 31, 0, 1, 2, ..., 31]
        labels = (torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)).float()
        # Labels is transformed to one-hot and is of shape (64, 64).
        # There are 2 diagonals of ones: the 64x64 main diagonal, and a shifted diagonal (of the 2nd [0:31] vector originlly concatenated) which warps at the end of the columns to continue at the start of them on the next rows.
        labels = labels.to(self.args.device)
        
        # Features are the output of the MLP head. Shape (batch_size, 512)
        features = F.normalize(features, dim=1)
        # Features are L2-normalized to unit length. Shape (batch_size, 512)
        similarity_matrix = torch.matmul(features, features.T)

        # Discard the main diagonal from both labels and similarities matrix.
        # For the rows 1 to 31, it shifts the ones index by -1 since the main diagonal comes "before" them.
        # For the remaining rows, it doesn't change their indexs since the main diagonal comes after these columns.
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)  # mask is of shape (64, 64) with main diagonal at 1
        labels = labels[~mask].view(labels.shape[0], -1)  # labels is of shape (64, 63). By preventing a feature to be matched with itself using the mask, there is one less possible matching per row
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # similarity_matrix is of shape (64, 63).

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # shape (64, 1)
        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # shape (64, 62)
        logits = torch.cat([positives, negatives], dim=1)  # positives are at index 0, negatives at index 1 to 62
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)  # labels at index 0 are the positives
        logits = logits / self.args.temperature
        return logits, labels, similarity_matrix.detach().to('cpu').numpy()

    def info_nce_loss_single_diag(self, features_img, features_gps, dataset_type: str = 'species'):
        # Not handling the landscape case with more than 2 views !
        labels = torch.eye(features_img.shape[0]).to(self.args.device)  # mask is of shape (batch_size, batch_size) with main diagonal at 1. Except when batch_size if lower than the nb of samples in val_loader, in which case the mask is of shape (n_samples, n_samples)

        # Features are the output of the MLP head. Shape (batch_size, 512)
        features_img = F.normalize(features_img, dim=1)
        features_gps = F.normalize(features_gps, dim=1)
        # Features are L2-normalized to unit length. Shape (batch_size, 512)

        similarity_matrix = torch.matmul(features_gps, features_img.T)  # rows are gps, columns are images

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # shape (64, 1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # shape (64, 62)

        logits = torch.cat([positives, negatives], dim=1)  # positives are at index 0, negatives at index 1 to 62
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)  # labels at index 0 are the positives

        logits = logits / self.args.temperature
        return logits, labels, similarity_matrix.detach().to('cpu').numpy()

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        max_iter: int = torch.inf
    ):
        """Train the model using SimCLR.

        Args:
            train_loader (torch.utils.data.DataLoader): pytorch dataloader for training data
            val_loader (torch.utils.data.DataLoader): pytorch dataloader for validation data
            max_iter (int, optional): Max iter nb over both train and val dataloaders. Defaults to torch.inf.
        """
        modalities_name = ['species', 'landscape', 'satellite']
        modalities_to_process = [b for b in modalities_name if b not in self.skip_modalities]
        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.dir, self.args)

        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        best_train_loss, best_val_loss = torch.inf, torch.inf
        train_steps, val_steps = 0, 0

        for epoch_counter in range(self.args.last_epoch, self.args.epochs + self.args.last_epoch):
            running_loss, sim_matrices, top1s, top5s = [], [], [], []
            running_criterion, running_koleo = [], []
            wandb.log({"epoch": epoch_counter})
            print("Training the model...")
            print(f"> Starting epoch {epoch_counter}...")
            ### Debug
            # lmin, lmax, lmean, lstd = [], [], [], []
            # lgpsmin, lgpsmax, lgpsmean, lgpsstd = [], [], [], []
            # stats_name = ['min','max','mean','std']
            ###
            self.model.train()
            for step, train_dict in enumerate(tqdm(train_loader)):
                batch_inds = train_dict['indices']
                # species_img = train_dict['species'][0]
                # species_coords = train_dict['species'][1]
                # species_idx = train_dict['species'][2]
                # species_id = train_dict['species'][3]
                # landscape_img = train_dict['landscape'][0]
                # landscape_coords = train_dict['landscape'][1]
                # landscape_idx = train_dict['landscape'][2]
                # landscape_id = train_dict['landscape'][3]
                # satellite_img = train_dict['satellite'][0]
                # satellite_coords = train_dict['satellite'][1]
                # satellite_idx = train_dict['satellite'][2]
                # satellite_id = train_dict['satellite'][3]
                # imgs = [species_img, landscape_img, satellite_img]
                # coords = [species_coords, landscape_coords, satellite_coords]
                # idxs = [species_idx, landscape_idx, satellite_idx]
                # ids = [species_id, landscape_id, satellite_id]

                train_dict_items = train_dict.items()
                train_dict_items = [(k, v) for k, v in train_dict_items if k in modalities_to_process]
                wandb.log({"train_steps": train_steps})

                with autocast(device_type=str(self.args.device), enabled=self.args.fp16_precision):
                    loss, all_logits, all_features_img, all_features_gps, all_images, idxs, ids = 0, [], [], [], [], [], []
                    self.optimizer.zero_grad()
                    for i, v in enumerate(train_dict_items):
                        mod_name, (images, gps, inds, survey_ids) = v[0], v[1]
                    # for i, (images, gps, inds, survey_ids) in enumerate(zip(imgs, coords, idxs, ids)):
                    #   if i in skip_modalities:
                    #     continue  # careful about checking if logging functions are compatible with other than 3 modalities
                        images = images.to(self.args.device)
                        gps = gps.to(self.args.device)
                        features_img, features_gps = self.model[mod_name](images, gps)
                        features = torch.cat([features_img, features_gps], dim=0)
                        if torch.isnan(features_img).sum() > 0:
                            print("NaN detected in image features.")
                        if self.args.symmetric_loss:
                            logits, labels, sim_matrix = self.info_nce_loss(features, dataset_type=self.args.arch)
                            koleo = KoLeoLoss()
                            # criterion = self.criterion(logits, labels)
                            criterion = cosine_embedding_loss(features_img, features_gps).item()
                            if mod_name in self.koleo_modalities:
                                koleo_train = koleo(features, eps=self.koleo_eps[i])
                                loss += (self.koleo_weights[i] * koleo_train)/len(modalities_to_process)
                                print(f'KoLeo loss {mod_name}: {self.koleo_weights[i] * koleo_train.item()}')
                                print(f'Criterion loss {mod_name}: {criterion}')
                                running_koleo.append(koleo_train.item())
                            running_criterion.append(criterion)
                        else:
                            logits, labels, sim_matrix = self.info_nce_loss_single_diag(features_img, features_gps, dataset_type=self.args.arch)
                        all_logits.append(logits)
                        sim_matrices.append(sim_matrix)
                        loss += criterion/len(modalities_to_process)  # Average loss over the 3 modalities + GPS
                        # loss = loss/num_steps_par_batch
                        # loss = loss/batch_size
                        std_mean_img, std_mean_gps = torch.std_mean(features_img, dim=0), torch.std_mean(features_gps, dim=0)
                        std_mean_diff = (std_mean_img[0] - std_mean_gps[0], std_mean_img[1] - std_mean_gps[1])
                        norm_img, norm_gps = torch.norm(features_img, dim=1), torch.norm(features_gps, dim=1)

                        all_images.append(images)  # Randomly select 1/3 of a batch of images from the current modality to later display everything in a batch_size plt image
                        all_features_img.append(features_img)
                        all_features_gps.append(features_gps)
                        idxs.append(inds)
                        ids.append(survey_ids)
                    
                    print(f"Epoch {epoch_counter} loss: {loss.item():.4f}")

                    scaler.scale(loss).backward()  # Gradients are accumulated. Calling backward after each modality loss equals calling backward once after sum + average of losses
                    running_loss.append(loss.item())
                    best_train_loss = min(best_train_loss, loss.item())

                scaler.step(self.optimizer)
                scaler.update()

                if step % self.args.log_every_n_steps_train == 0:       
                    log_input_imgs_multimodalities(all_images, idxs, ids, step, epoch_counter, n_samples=8, n_modalities=len(modalities_to_process), mode='train', log_images=self.log_images)
                    
                    # Log loss and moments step wise
                    log_loss_scheduler(loss, self.scheduler)
                    # log_moments(norm_img, norm_gps, std_mean_img, std_mean_gps, std_mean_diff)
                    
                    # Log similarity matrix step wise
                    log_similarity_matrix_step(sim_matrices, epoch_counter, step, n_modalities=len(modalities_to_process), modalities_name=modalities_to_process, log_images=self.log_images, mode='train')
                    
                    # Log mean of similarity matrices computed over self.args.log_every_n_steps_train steps
                    sim_matrix_mean = mean_sim_matrices_over_modalities(sim_matrices, n_modalities=len(modalities_to_process))
                    log_similarity_matrix_mean(sim_matrix_mean, epoch_counter, step, n_modalities=len(modalities_to_process), log_every_n_steps=self.args.log_every_n_steps_train, log_images=self.log_images)
                    
                    # Log accuracy step wise
                    for logits, modality_name in zip(all_logits, modalities_to_process):
                        top1, top5 = log_acc_topk_step(logits, labels, modality_name, topk=(1, 5), mode='train')
                        # Every step wise top-k is stored in a list where modalities are interleaved. E.g. [topk_modality1, topk_modality2, topk_modality3]
                        if all(topk is not None for topk in [top1, top5]):
                            top1s.append(top1)
                            top5s.append(top5)
                train_steps += 1
                if step >= max_iter:  # Debug purposes
                    break
            wandb.log({"Loss_epoch (batch avg)/train": np.array(running_loss).mean()})
            for i, modality_name in enumerate(modalities_to_process):
                wandb.log({f"acc_epoch (batch avg)/train/top1_{modality_name}": np.array(top1s[i::len(modalities_to_process)]).mean(),
                           f"acc_epoch (batch avg)/train/top5_{modality_name}": np.array(top5s[i::len(modalities_to_process)]).mean()})
            wandb.log({"acc_epoch (batch avg)/train/top1": np.array(top1s).mean(),
                       "acc_epoch (batch avg)/train/top5": np.array(top5s).mean()})
            
            # Log t-sne projection
            for features_img, features_gps, modality_name in zip(all_features_img, all_features_gps, modalities_to_process):
                log_tsne(features_img, features_gps, epoch_counter, modality_name, log_images=self.log_images, mode='train')

            # Evaluation
            self.model.eval()
            print("Evaluating the model...")
            with torch.no_grad():
                running_vloss, vsim_matrices, vtop1s, vtop5s = [], [], [], []
                # for vstep, (vimages, vgps, vinds, vsurvey_ids) in enumerate(tqdm(val_loader)):
                for vstep, val_dict in enumerate(tqdm(val_loader)):
                    vbatch_inds = val_dict ['indices']
                    # vspecies_img = val_dict['species'][0]
                    # vspecies_coords = val_dict['species'][1]
                    # vspecies_idx = val_dict['species'][2]
                    # vspecies_id = val_dict['species'][3]
                    # vlandscape_img = val_dict['landscape'][0]
                    # vlandscape_coords = val_dict['landscape'][1]
                    # vlandscape_idx = val_dict['landscape'][2]
                    # vlandscape_id = val_dict['landscape'][3]
                    # vsatellite_img = val_dict['satellite'][0]
                    # vsatellite_coords = val_dict['satellite'][1]
                    # vsatellite_idx = val_dict['satellite'][2]
                    # vsatellite_id = val_dict['satellite'][3]

                    val_dict.pop('indices', None)
                    val_dict_items = val_dict.items()  # Useless if keeping all modalities
                    val_dict_items = [(k, v) for k, v in val_dict_items]  # Useless if keeping all modalities
                    wandb.log({"val_steps": val_steps})
                    vloss, vall_logits, vall_features_img, vall_features_gps, vall_images, vidxs, vids  = 0, [], [], [], [], [], []
                    
                    # for i, (vimages, vgps, vinds, vsurvey_ids) in enumerate(zip([vspecies_img, vlandscape_img, vsatellite_img],
                    #                                                             [vspecies_coords, vlandscape_coords, vsatellite_coords],
                    #                                                             [vspecies_idx, vlandscape_idx, vsatellite_idx],
                    #                                                             [vspecies_id, vlandscape_id, vsatellite_id])):
                    for i, v in enumerate(val_dict_items):
                        mod_name, (vimages, vgps, vinds, vsurvey_ids) = v[0], v[1]

                        vimages = vimages.to(self.args.device)
                        vgps = vgps.to(self.args.device)
                        vfeatures_img, vfeatures_gps = self.model[mod_name](vimages, vgps)
                        vfeatures = torch.cat([vfeatures_img, vfeatures_gps], dim=0)
                        if self.args.symmetric_loss:
                            vlogits, vlabels, vsim_matrix = self.info_nce_loss(vfeatures, dataset_type=self.args.arch)
                        else:
                            vlogits, vlabels, vsim_matrix = self.info_nce_loss_single_diag(vfeatures_img, vfeatures_gps, dataset_type=self.args.arch)
                        # vloss = self.criterion(vlogits, vlabels)
                        # vloss += self.criterion(vlogits, vlabels)/3  # Average loss over the 3 modalities + GPS
                        vloss += cosine_embedding_loss(vfeatures_img, vfeatures_gps)/3
                        print(f'Criterion vloss {mod_name}: {vloss}')

                        vall_images.append(vimages)
                        vall_features_img.append(vfeatures_img)
                        vall_features_gps.append(vfeatures_gps)
                        vall_logits.append(vlogits)
                        vsim_matrices.append(vsim_matrix)
                        vids.append(vsurvey_ids)
                        vidxs.append(vinds)
                    running_vloss.append(vloss.item())

                    # Save best checkpoint
                    if vloss.item() <= best_val_loss:
                        logging.info(f"Saving new best model at epoch {epoch_counter}, step {vstep} with loss {vloss.item()}.")
                        save_checkpoint({
                            'epoch': epoch_counter,
                            'arch': self.args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, is_best=True, dirpath=self.writer.dir)
                    best_val_loss = min(best_val_loss, vloss.item())

                    # Log accuracy step wise
                    for vlogits, modality_name in zip(vall_logits, modalities_name):
                        vtop1, vtop5 = log_acc_topk_step(vlogits, vlabels, modality_name, topk=(1, 5), mode='val')
                        if all(topk is not None for topk in [vtop1, vtop5]):
                            vtop1s.append(vtop1)
                            vtop5s.append(vtop5)

                    if vstep % self.args.log_every_n_steps_val == 0:
                        # Log input batch images
                        log_input_imgs_multimodalities(vall_images, vidxs, vids, vstep, epoch_counter, n_samples=8, n_modalities=3, mode='val', log_images=self.log_images)
                        
                        # Log loss and moments step wise
                        log_loss_scheduler(vloss, self.scheduler, mode='val')
                    
                        # Log similarity matrix step wise
                        log_similarity_matrix_step(vsim_matrices, epoch_counter, vstep, log_images=self.log_images, mode='val')

                    val_steps += 1
                    if vstep >= max_iter:
                        break
                wandb.log({"Loss_epoch (batch avg)/val": np.array(running_vloss).mean()})
                for vtop1, vtop5, modality_name in zip(vtop1s, vtop5s, modalities_name):
                    wandb.log({f"acc_epoch (batch avg)/val/top1_{modality_name}": vtop1,
                               f"acc_epoch (batch avg)/val/top5_{modality_name}": vtop5})
                wandb.log({"acc_epoch (batch avg)/val/top1": np.array(vtop1s).mean(),
                           "acc_epoch (batch avg)/val/top5": np.array(vtop5s).mean()})
                
                # Log similarity matrix epoch wise
                while (vsim_matrices[-1].shape[0] != self.args.batch_size//2) and (len(vsim_matrices) > 3):  # True if the last matrix was computed on a smaller batch AND if there are more than 1 batch matrix
                    vsim_matrices = vsim_matrices[:-1]  # Remove the last matrix if it has a different shape than the others (e.g. if the last batch is smaller than the others)
                vsim_matrix_mean = mean_sim_matrices_over_modalities(vsim_matrices)
                log_similarity_matrix_mean(vsim_matrix_mean, epoch_counter, vstep, log_images=self.log_images, mode='val')
                
                # Log t-sne projection
                for vfeatures_img, vfeatures_gps, modality_name in zip(vall_features_img, vall_features_gps, modalities_name):
                    log_tsne(vfeatures_img, vfeatures_gps, epoch_counter, modality_name, log_images=self.log_images, mode='val')

            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")

            # Save model checkpoints
            save_checkpoint({
                'epoch': epoch_counter,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=(vloss < best_val_loss), dirpath=self.writer.dir)
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.dir}.")
        logging.info("Training has finished.")

    def predict(self, test_dataloader: torch.utils.data.DataLoader):
        """Predict the model using SimCLR.

        Args:
            dataloader (torch.utils.data.DataLoader): pytorch dataloader for validation data
        """
        modalities_name = ['species', 'landscape', 'satellite']
        modalities_to_process = [b for b in modalities_name if b not in self.skip_modalities]
        metrics = {}
        self.model.eval()
        features_img, features_gps = [], []
        sim_matrices, top1s, top5s = [], [], []
        print("Running inference...")
        with torch.no_grad():
            for step, test_dict in enumerate(tqdm(test_dataloader)):
                all_logits, all_features_img, all_features_gps, idxs, ids  = [], [], [], [], []
                batch_inds = test_dict['indices']
                test_dict.pop('indices', None)
                for i, (mod_name, (images, gps, inds, survey_ids)) in enumerate(test_dict.items()):
                    images = images.to(self.args.device)
                    gps = gps.to(self.args.device)
                    features_img, features_gps = self.model[mod_name](images, gps) if isinstance(self.model, torch.nn.ModuleDict) else self.model[i](images, gps)
                    features = torch.cat([features_img, features_gps], dim=0)
                    if self.args.symmetric_loss:
                        logits, labels, sim_matrix = self.info_nce_loss(features, dataset_type=self.args.arch)
                    else:
                        logits, labels, sim_matrix = self.info_nce_loss_single_diag(features_img, features_gps, dataset_type=self.args.arch)

                    all_features_img.append(features_img)
                    all_features_gps.append(features_gps)
                    all_logits.append(logits)
                    sim_matrices.append(sim_matrix)
                    ids.append(survey_ids)
                    idxs.append(inds)
                    
                # Log accuracy step wise
                for logits, modality_name in zip(all_logits, modalities_name):
                    vtop1, vtop5 = log_acc_topk_step(logits, labels, modality_name, topk=(1, 5), mode='test')
                    if all(topk is not None for topk in [vtop1, vtop5]):
                        top1s.append(vtop1)
                        top5s.append(vtop5)
                        
            # Log similarity matrix epoch wise
            while (sim_matrices[-1].shape[0] != self.args.batch_size//2) and (len(sim_matrices) > 3):  # True if the last matrix was computed on a smaller batch AND if there are more than 1 batch matrix
                sim_matrices = sim_matrices[:-1]
            sim_matrix_mean = mean_sim_matrices_over_modalities(sim_matrices)
            log_similarity_matrix_mean(sim_matrix_mean, 0, step, log_images=self.log_images, mode='test')
            for vtop1, vtop5, modality_name in zip(top1s, top5s, modalities_name):
                wandb.log({f"acc_epoch (batch avg)/test/top1_{modality_name}": vtop1,
                           f"acc_epoch (batch avg)/test/top5_{modality_name}": vtop5})
                print(f'Test accuracy for {modality_name} - Top-1: {vtop1:.4f}, Top-5: {vtop5:.4f}')
                metrics[f"acc_epoch (batch avg)/test/top1_{modality_name}"] = vtop1
                metrics[f"acc_epoch (batch avg)/test/top5_{modality_name}"] = vtop5
            wandb.log({"acc_epoch (batch avg)/test/top1": np.array(top1s).mean(),
                        "acc_epoch (batch avg)/test/top5": np.array(top5s).mean()})
            metrics["acc_epoch (batch avg)/test/top1"] = np.array(top1s).mean()
            metrics["acc_epoch (batch avg)/test/top5"] = np.array(top5s).mean()
            print(f'Test accuracy (mean) - Top-1: {np.array(top1s).mean():.4f}, Top-5: {np.array(top5s).mean():.4f}')
            # Log t-sne projection
            for vfeatures_img, vfeatures_gps, modality_name in zip(all_features_img, all_features_gps, modalities_name):
                log_tsne(vfeatures_img, vfeatures_gps, 0, modality_name, log_images=self.log_images, mode='test')
            df_metrics = pd.DataFrame(metrics, index=[0])
            df_metrics.to_csv(os.path.join(self.writer.dir, 'test_metrics.csv'), index=False)
