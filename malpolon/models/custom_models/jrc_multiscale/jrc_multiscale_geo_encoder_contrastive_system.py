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
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.retrieval import retrieval_recall
from torchmetrics.functional.classification import multilabel_auroc, multilabel_average_precision
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn

from malpolon.models.custom_models.jrc_multiscale.jrc_contrastive_losses import (
    KoLeoLoss, cosine_embedding_loss, cosine_similarity_mean, cosine_loss_pytorch_like,
    cosine_embedding_from_sim, crisp_loss, crisp_loss_manual, cosine_mcr
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
        pred = pred.t()  # Rows: top-k predictions for each sample, Cols: batch samples
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # Targets are indices 0, always

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Scalar
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# EMA update function
def update_ema(model, ema_model, alpha=0.99):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data = alpha * ema_p.data + (1. - alpha) * p.data

def wandb_init():
    # Iterations metricsx
    wandb.define_metric("epoch")
    wandb.define_metric("train_steps")
    wandb.define_metric("val_steps")

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
    wandb.define_metric("recall_step/train", step_metric="train_steps")
    wandb.define_metric("recall@1_step/train", step_metric="train_steps")
    wandb.define_metric("recall@20_step/train", step_metric="train_steps")
    wandb.define_metric("recall@100_step/train", step_metric="train_steps")
    wandb.define_metric("MultilabelAUROC_micro_step/train", step_metric="train_steps")    
    wandb.define_metric("MultilabelAUROC_macro_step/train", step_metric="train_steps")    
    wandb.define_metric("MultilabelAveragePrecision_micro_step/train", step_metric="train_steps")
    wandb.define_metric("MultilabelAveragePrecision_macro_step/train", step_metric="train_steps")

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
    wandb.define_metric("recall_step/val", step_metric="val_steps")
    wandb.define_metric("recall@1_step/val", step_metric="val_steps")
    wandb.define_metric("recall@20_step/val", step_metric="val_steps")
    wandb.define_metric("recall@100_step/val", step_metric="val_steps")
    wandb.define_metric("MultilabelAUROC_micro_step/val", step_metric="val_steps")
    wandb.define_metric("MultilabelAUROC_macro_step/val", step_metric="val_steps")
    wandb.define_metric("MultilabelAveragePrecision_micro_step/val", step_metric="val_steps")
    wandb.define_metric("MultilabelAveragePrecision_macro_step/val", step_metric="val_steps")

class SimCLR(object):
    def __init__(self, writer=SummaryWriter(), *args, **kwargs):
        self.writer = writer
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)

        self.koleo_weight = self.args.koleo_weight if hasattr(self.args, 'koleo_weight') else 0.0
        self.koleo_eps = self.args.koleo_eps if hasattr(self.args, 'koleo_eps') else 1e-4

        self.ema_model = kwargs['model'] if getattr(self.args, 'use_ema', False) else None
        self.ema_decay = getattr(self.args, 'ema_decay', 0.99)
        self.ema_update_step = getattr(self.args, 'ema_update_step', 1)

        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.inference = bool(getattr(self.args, 'predict', False))
        self.skip_modalities = getattr(self.args, 'skip_modalities', [])
        self.log_images = getattr(self.args, 'log_images', True)
        self.resume_wandb_run = getattr(self.args, 'resume_wandb_run', False)
        self.args.last_epoch = getattr(self.args, 'last_epoch', 0)
        self.jobid = getattr(self.args, 'OAR_job_id', 'no_jobid')
        print(f"[INFO] Job ID: {self.jobid}")
        print(f"[INFO] Wandb ID: {self.writer.id}")
        print(f"[INFO] Wandb output directory: {self.writer.dir}")
        logging.basicConfig(filename=os.path.join(self.writer.dir, 'training.log'), level=logging.DEBUG)
        wandb_init()
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.criterion_name = getattr(self.args, 'loss_criterion', 'cross_entropy')
    
    def get_criterion(self, criterion_name, features_img, features_gps, sim_matrix, logits, labels, temperature=0.07):
        """Retrieves the right criterion with correct inputs.
        
        Possible values of criterion_name: 'cross_entropy', 'cosine_embedding', 'cosine_embedding_from_sim',
        'cosine_similarity_mean', 'cosine_loss_pytorch_like'.

        """
        mask = 'double_diag' if self.args.symmetric_loss else 'main_diag'
        if criterion_name == 'cross_entropy':
            criterion = torch.nn.CrossEntropyLoss().to(features_img.device)
            loss = criterion(logits, labels)
        elif criterion_name == 'crisp':
            loss, sim_matrix, labels = crisp_loss(sim_matrix, temperature=temperature, return_sim_matrix_and_targets=True)
        elif criterion_name == 'crisp_manual':
            loss, sim_matrix, labels = crisp_loss_manual(sim_matrix, temperature=temperature, return_sim_matrix_and_targets=True)
        elif criterion_name == 'cosine_embedding_loss':
            loss2 = cosine_embedding_loss(features_img, features_gps, pos_weight=0.8, margin=0.3)
            criterion = torch.nn.CrossEntropyLoss().to(features_img.device)
            loss = criterion(logits, labels)
            print(f'Cosine embedding loss: {loss2}, cross-entropy loss: {loss}')
        elif criterion_name == 'cosine_embedding_from_sim':
            loss = cosine_embedding_from_sim(sim_matrix, pos_weight=0.8, margin=0.3, mask=mask)
        elif criterion_name == 'cosine_similarity_mean':
            loss = cosine_similarity_mean(features_img, features_gps, lambd=0.8, mask=mask)
        elif criterion_name == 'cosine_loss_pytorch_like':
            loss = cosine_loss_pytorch_like(sim_matrix, lambd=0.8, margin=0.2, mask=mask) 
        elif criterion_name == 'cosine_mcr':
            loss = cosine_mcr(features_img, features_gps, weight_mcr=0.1, eps_mcr=self.args.mcr_eps)
        else:
            raise NotImplementedError(f"Loss criterion {criterion_name} not implemented.")
        return loss, sim_matrix, labels
    
    def get_regularizer(self, features, regularizer_name: str):
        """Retrieves the right regularizer.

        Possible values of regularizer_name: 'koleo', 'mcr'.
        """
        if regularizer_name == 'koleo':
            regularizer = KoLeoLoss(eps=self.koleo_eps)
            reg_term = regularizer(features, eps=self.koleo_eps)
        elif regularizer_name == 'mcr':
            regularizer = MCR(eps=self.mcr_eps)
            reg_term = regularizer(features)
        else:
            raise NotImplementedError(f"Regularizer {regularizer_name} not implemented.")
        return reg_term

    def info_nce_loss(self, features, dataset_type: str = 'species'): # [features_img, features_gps]
        labels = torch.cat([torch.arange(features.shape[0]//self.args.n_views) for i in range(self.args.n_views)], dim=0)
        # Labels is a vector of size 64 with values 0 to 31 concatenated n_views times. E.g. if n_views==2: [0, 1, 2, ..., 31, 0, 1, 2, ..., 31]
        labels = (torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)).float()
        # Labels is transformed to one-hot and is of shape (64, 64).
        # There are 2 diagonals of ones: the 64x64 main diagonal, and a shifted diagonal (of the 2nd [0:31] vector originlly concatenated) which warps at the end of the columns to continue at the start of them on the next rows.
        labels = labels.to(self.args.device)
        
        # Features are the output of the MLP head and L2-normalized to unit length. Shape (batch_size, 512)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # Discard the main diagonal from both labels and similarities matrix.
        # For the rows 1 to 31, it shifts the ones index by -1 since the main diagonal comes "before" them.
        # For the remaining rows, it doesn't change their indexs since the main diagonal comes after these columns.
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)  # mask is of shape (64, 64) with main diagonal at 1
        labels = labels[~mask].view(labels.shape[0], -1)  # labels is of shape (64, 63). By preventing a feature to be matched with itself using the mask, there is one less possible matching per row
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # similarity_matrix is of shape (64, 63).

        # Re-arranging the similarity matrix and labels to move positive matches to the 1st column
        # # select and combine multiple positives
        # positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # shape (64, 1)
        # # select only the negatives
        # negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # shape (64, 62)
        # logits = torch.cat([positives, negatives], dim=1)  # positives are at index 0, negatives at index 1 to 62
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)  # labels at index 0 are the positives
        logits = similarity_matrix
        labels = torch.where(labels.bool())[1].to(self.args.device)  # each feature at row i should match the feature at column i+n_views or i-n_views

        logits = logits / self.args.temperature
        return logits, labels, similarity_matrix

    def info_nce_loss_single_diag(self, features_img, features_gps, dataset_type: str = 'species'):
        # Not handling the landscape case with more than 2 views !
        labels = torch.eye(features_img.shape[0]).to(self.args.device)  # mask is of shape (batch_size, batch_size) with main diagonal at 1. Except when batch_size if lower than the nb of samples in val_loader, in which case the mask is of shape (n_samples, n_samples)

        # Features are the output of the MLP head and L2-normalized to unit length. Shape (batch_size, 512)
        features_img = F.normalize(features_img, dim=1)
        features_gps = F.normalize(features_gps, dim=1)
        similarity_matrix = torch.matmul(features_gps, features_img.T)  # rows are gps, columns are images

        logits = similarity_matrix
        labels = torch.arange(logits.shape[0]).to(self.args.device)  # each gps feature at row i should match image feature at column i

        logits = logits / self.args.temperature
        return logits, labels, similarity_matrix

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
        self.model.train()
        wandb.watch(self.model.gps_contrastive_head, log="gradients", log_freq=self.args.log_every_n_steps_train)
        wandb.watch(self.model.modality_contrastive_head, log="gradients", log_freq=self.args.log_every_n_steps_train)
        save_config_file(self.writer.dir, self.args)
        scaler = GradScaler(enabled=self.args.fp16_precision)

        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        best_train_loss, best_val_loss = torch.inf, torch.inf
        train_steps, val_steps = 0, 0

        for epoch_counter in range(self.args.last_epoch, self.args.epochs + self.args.last_epoch):
            print("Training the model...")
            print(f"> Starting epoch {epoch_counter}...")
            running_loss, sim_matrices, top1s, top5s = [], [], [], []
            running_criterion, running_koleo = [], []
            wandb.log({"epoch": epoch_counter})
            self.model.train()
            if self.ema_model is not None:
                self.ema_model.train()
            for step, (images, gps, inds, survey_ids) in enumerate(tqdm(train_loader)):
                wandb.log({"train_steps": train_steps})
                images = images.to(self.args.device)
                gps = gps.to(self.args.device)

                with autocast(device_type=str(self.args.device), enabled=self.args.fp16_precision):
                    features_img, features_gps = self.model(images, gps)
                    std_mean_img, std_mean_gps = torch.std_mean(features_img, dim=0), torch.std_mean(features_gps, dim=0)
                    std_mean_diff = (std_mean_img[0] - std_mean_gps[0], std_mean_img[1] - std_mean_gps[1])
                    norm_img, norm_gps = torch.norm(features_img, dim=1), torch.norm(features_gps, dim=1)
                    features = torch.cat([features_img, features_gps], dim=0)
                    if self.args.symmetric_loss:
                        logits, labels, sim_matrix = self.info_nce_loss(features, dataset_type=self.args.arch)
                        koleo = KoLeoLoss()
                    else:
                        logits, labels, sim_matrix = self.info_nce_loss_single_diag(features_img, features_gps, dataset_type=self.args.arch)
                        koleo = KoLeoLoss(mask='main_diag')
                    criterion, sim_matrix, labels = self.get_criterion(self.criterion_name, features_img, features_gps,
                                                                       sim_matrix.clone(), logits.clone(), labels.clone(),
                                                                       temperature=self.args.temperature)
                    koleo_train = koleo(features, eps=self.koleo_eps)
                    loss = criterion + self.koleo_weight * koleo_train
                    running_criterion.append(criterion.item())
                    running_koleo.append(koleo_train.item())
                    running_loss.append(loss.item())
                    best_train_loss = min(best_train_loss, loss.item())
                    sim_matrix = sim_matrix.detach().to('cpu').numpy()
                    sim_matrices.append(sim_matrix)
                    print(f"Epoch {epoch_counter} loss: {loss.item():.4f} criterion: {criterion:.4f} Koleo: {(self.koleo_weight * koleo_train.item()):.4f}")

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if self.ema_model is not None and step % self.ema_update_step == 0:
                    update_ema(self.model, self.ema_model, self.ema_decay)

                if step % self.args.log_every_n_steps_train == 0:            
                    # # Log input batch images
                    # fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # 4 rows, 8 columns
                    # axes = axes.flatten()
                    # for idx, (img, ind, sid, ax) in enumerate(zip(images, inds, survey_ids, axes)):
                    #     img = img[:3, :, :] if img.shape[0] >= 3 else img[0, :, :]
                    #     img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # Cutting multi-band images to the first 3
                    #     ax.imshow(img, cmap='gray' if img.ndim < 3 else None)
                    #     ax.set_title(f"Image {idx}, iter_idx {ind[0]}, \nsurveyId {sid}", fontsize=8)
                    #     ax.axis('off')
                    # plt.tight_layout()
                    # wandb.log({f'Input_imgs_train/e_{epoch_counter:03d}_s_{step:03d}': wandb.Image(fig)})
                    # plt.close()
                    
                    # Log loss and moments step wise
                    wandb.log({"Loss_step/train": loss, "learning_rate": self.scheduler.get_last_lr()[0]})
                    # wandb.log({"std-avg_img/train": std_mean_img[0].mean(), "mean-avg_img/train": std_mean_img[1].mean(),
                    #            "std-avg_gps/train": std_mean_gps[0].mean(), "mean-avg_gps/train": std_mean_gps[1].mean(),
                    #            "std-avg_diff/train": std_mean_diff[0].mean(), "mean-avg_diff/train": std_mean_diff[1].mean()})
                    wandb.log({"norm_img_avg/train": norm_img.mean(),
                               "norm_gps_avg/train": norm_gps.mean(),
                               "norm_avg_diff/train": abs(norm_img.mean()-norm_gps.mean())})
                    
                    # # Log similarity matrix step wise
                    # plt.figure(figsize=(12, 10))
                    # plt.title("Similarity Matrix")
                    # hm = sns.heatmap(sim_matrix, cmap="viridis", annot=False)
                    # wandb.log({f'SimMatrix_train/e_{epoch_counter:03d}_s_{step:03d}': wandb.Image(hm)})
                    # plt.close()
                    
                    # Log mean of similarity matrices computed over self.args.log_every_n_steps_train steps
                    # sim_matrix_mean = torch.Tensor(np.array(sim_matrices)).mean(dim=0)
                    # plt.figure(figsize=(12, 10))
                    # plt.title("Similarity Matrix")
                    # hm = sns.heatmap(sim_matrix_mean, cmap="viridis", annot=False)
                    # wandb.log({f'SimMatrix_mean-{self.args.log_every_n_steps_train}-steps_train/{epoch_counter:03d}_s_{step:03d}': wandb.Image(hm)})
                    # plt.close()
                    # sim_matrices = []
                    
                    # Log accuracy step wise
                    if logits.shape[0] >= 5:
                        top1, top5 = accuracy(logits, labels, topk=(1, 5))
                        top1s.append(top1[0].item())
                        top5s.append(top5[0].item())
                        wandb.log({"acc/train/top1": top1[0],
                                   "acc/train/top5": top5[0]})
                        print(f"acc/train/top1: {top1[0]:.4f}, acc/train/top5: {top5[0]:.4f}")
                    else:
                        print("Batch size is too small for accuracy calculation.")
                        
                    # Log recall@K
                    n_cls = logits.shape[1]
                    labels_oh = F.one_hot(labels, num_classes=n_cls)
                    wandb.log({"recall_step/train": retrieval_recall(logits, labels_oh)})
                    wandb.log({"recall@1_step/train": retrieval_recall(logits, labels_oh, top_k=1)})
                    wandb.log({"recall@20_step/train": retrieval_recall(logits, labels_oh, top_k=5)})
                    wandb.log({"recall@100_step/train": retrieval_recall(logits, labels_oh, top_k=100)})

                    # Log AUROC
                    wandb.log({"MultilabelAUROC_micro_step/train": multilabel_auroc(logits, labels_oh, n_cls, average='micro')})
                    wandb.log({"MultilabelAUROC_macro_step/train": multilabel_auroc(logits, labels_oh, n_cls, average='macro')})

                    # Log mAP
                    wandb.log({"MultilabelAveragePrecision_micro_step/train": multilabel_average_precision(logits, labels_oh, n_cls, average='micro')})
                    wandb.log({"MultilabelAveragePrecision_macro_step/train": multilabel_average_precision(logits, labels_oh, n_cls, average='macro')})

                train_steps += 1
                if step >= max_iter:  # Debug purposes
                    break
            wandb.log({"Loss_epoch (batch avg)/train": np.array(running_loss).mean()})
            wandb.log({"acc_epoch (batch avg)/train/top1": np.array(top1s).mean(),
                       "acc_epoch (batch avg)/train/top5": np.array(top5s).mean()})
            print(f"acc_epoch (batch avg)/train/top1: {np.array(top1s).mean():.4f}, acc_epoch (batch avg)/train/top5: {np.array(top5s).mean():.4f}")
            
            # # Log t-sne projection
            # n = features_img.shape[0]
            # embeddings = torch.cat([features_img, features_gps], dim=0)
            # tsne = TSNE(n_components=2, perplexity=min(30, n-1), learning_rate=200, metric='cosine', init='pca', random_state=42)
            # proj = tsne.fit_transform(embeddings.detach().to('cpu').numpy())
            # proj_a, proj_b = proj[:n], proj[n:]
            # fig = plt.figure(figsize=(8, 6))
            # plt.scatter(proj_a[:, 0], proj_a[:, 1], c='red', label='Modality image', alpha=0.7)
            # plt.scatter(proj_b[:, 0], proj_b[:, 1], c='blue', label='Modality GPS', alpha=0.7)
            # for i in range(n):
            #     plt.plot([proj_a[i, 0], proj_b[i, 0]], [proj_a[i, 1], proj_b[i, 1]], 'gray', alpha=0.3)
            # plt.title("t-SNE projection of contrastive embeddings (Train)"); plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2"); plt.grid(True); plt.legend()
            # wandb.log({f"t-sne_train/train/e_{epoch_counter:03d}": wandb.Image(fig)})
            # plt.close()

            # Evaluation
            if self.ema_model is not None:
                # self.model.to('cpu')
                # self.ema_model.to(self.args.device)
                self.ema_model.eval()
                eval_model = self.ema_model
            else:
                self.model.eval()
                eval_model = self.model
            print("Evaluating the model...")
            with torch.no_grad():
                running_vloss, vsim_matrices, vtop1s, vtop5s = [], [], [], []
                for vstep, (vimages, vgps, vinds, vsurvey_ids) in enumerate(tqdm(val_loader)):
                    wandb.log({"val_steps": val_steps})
                    vimages = vimages.to(self.args.device)
                    vgps = vgps.to(self.args.device)

                    with autocast(device_type=str(self.args.device), enabled=self.args.fp16_precision):
                        vfeatures_img, vfeatures_gps = eval_model(vimages, vgps)
                        vfeatures = torch.cat([vfeatures_img, vfeatures_gps], dim=0)
                        if self.args.symmetric_loss:
                            vlogits, vlabels, vsim_matrix = self.info_nce_loss(vfeatures, dataset_type=self.args.arch)
                        else:
                            vlogits, vlabels, vsim_matrix = self.info_nce_loss_single_diag(vfeatures_img, vfeatures_gps, dataset_type=self.args.arch)
                        vcriterion, vsim_matrix, vlabels = self.get_criterion(self.criterion_name, vfeatures_img, vfeatures_gps, vsim_matrix, vlogits, vlabels, temperature=self.args.temperature)
                        vloss = vcriterion
                        running_vloss.append(vloss.item())
                        vsim_matrix = vsim_matrix.detach().to('cpu').numpy()
                        vsim_matrices.append(vsim_matrix)
                        print(f"Epoch {epoch_counter} val step {vstep:.4f} loss: {vloss.item():.4f}")

                    # Log accuracy step wise
                    if vlogits.shape[0] >= 5:
                        vtop1, vtop5 = accuracy(vlogits, vlabels, topk=(1, 5))
                        vtop1s.append(vtop1[0].item())
                        vtop5s.append(vtop5[0].item())
                        wandb.log({"acc/val/top1": vtop1[0],
                                   "acc/val/top5": vtop5[0]})
                    else:
                        print("Batch size (val) is too small for accuracy calculation.")


                    if vstep % self.args.log_every_n_steps_val == 0:
                        # # Log input batch images
                        # fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # 4 rows, 8 columns
                        # axes = axes.flatten()
                        # for idx, (img, ind, sid, ax) in enumerate(zip(vimages, vinds, vsurvey_ids, axes)):
                        #     img = img[:3, :, :] if img.shape[0] >= 3 else img[0, :, :]
                        #     img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        #     ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                        #     ax.set_title(f"Image {idx}, iter_idx {ind[0]}, \nsurveyId {sid}", fontsize=8)
                        #     ax.axis('off')
                        # plt.tight_layout()
                        # wandb.log({f'Input_imgs_val/e_{epoch_counter:03d}_s_{vstep:03d}': wandb.Image(fig)})
                        # plt.close()
                        
                        # Log loss and moments step wise
                        wandb.log({"Loss_step/val": vloss})
                    
                        # Log recall@K
                        n_cls = vlogits.shape[1]
                        vlabels_oh = F.one_hot(vlabels, num_classes=n_cls)
                        wandb.log({"recall_step/val": retrieval_recall(vlogits, vlabels_oh)})
                        wandb.log({"recall@1_step/val": retrieval_recall(vlogits, vlabels_oh, top_k=1)})
                        wandb.log({"recall@20_step/val": retrieval_recall(vlogits, vlabels_oh, top_k=5)})
                        wandb.log({"recall@100_step/val": retrieval_recall(vlogits, vlabels_oh, top_k=100)})

                        # Log AUROC
                        wandb.log({"MultilabelAUROC_micro_step/val": multilabel_auroc(vlogits, vlabels_oh, n_cls, average='micro')})
                        wandb.log({"MultilabelAUROC_macro_step/val": multilabel_auroc(vlogits, vlabels_oh, n_cls, average='macro')})

                        # Log mAP
                        wandb.log({"MultilabelAveragePrecision_micro_step/val": multilabel_average_precision(vlogits, vlabels_oh, n_cls, average='micro')})
                        wandb.log({"MultilabelAveragePrecision_macro_step/val": multilabel_average_precision(vlogits, vlabels_oh, n_cls, average='macro')})

                        # # Log similarity matrix step wise
                        # plt.figure(figsize=(12, 10))
                        # plt.title("Similarity Matrix val")
                        # hm = sns.heatmap(vsim_matrix, cmap="viridis", annot=False)
                        # wandb.log({f'SimMatrix_val/e_{epoch_counter:03d}_vstep_{vstep:03d}': wandb.Image(hm)})
                        # plt.close()

                    val_steps += 1
                    if vstep >= max_iter:
                        break
                wandb.log({"Loss_epoch (batch avg)/val": np.array(running_vloss).mean()})
                wandb.log({"acc_epoch (batch avg)/val/top1": np.array(vtop1s).mean(),
                           "acc_epoch (batch avg)/val/top5": np.array(vtop5s).mean()})
                print(f"acc_epoch (batch avg)/val/top1: {np.array(vtop1s).mean():4f}, acc_epoch (batch avg)/val/top5: {np.array(vtop5s).mean():4f}")
                
                # Log similarity matrix epoch wise
                vsim_matrices = vsim_matrices[:-1] if (vsim_matrices[-1].shape[0] != self.args.batch_size) and (len(vsim_matrices) > 1) else vsim_matrices  # Remove the last matrix if it has a different shape than the others (e.g. if the last batch is smaller than the others)
                vsim_matrix_mean = torch.Tensor(np.array(vsim_matrices)).mean(dim=0)
                plt.figure(figsize=(12, 10))
                plt.title("Similarity Matrix")
                vhm = sns.heatmap(vsim_matrix_mean, cmap="viridis", annot=False)
                wandb.log({f'SimMatrix_mean-epoch_val/e{epoch_counter:03d}': wandb.Image(vhm)})
                plt.close()
                
                # # Log t-sne projection
                # n = vfeatures_img.shape[0]
                # if n > 1:
                #     embeddings = torch.cat([vfeatures_img, vfeatures_gps], dim=0)
                #     tsne = TSNE(n_components=2, perplexity=min(30, n-1), learning_rate=200, metric='cosine', init='pca', random_state=42)
                #     proj = tsne.fit_transform(embeddings.detach().to('cpu').numpy())
                #     proj_a, proj_b = proj[:n], proj[n:]
                #     fig = plt.figure(figsize=(8, 6))
                #     plt.scatter(proj_a[:, 0], proj_a[:, 1], c='red', label='Modality image', alpha=0.7)
                #     plt.scatter(proj_b[:, 0], proj_b[:, 1], c='blue', label='Modality GPS', alpha=0.7)
                #     for i in range(n):
                #         plt.plot([proj_a[i, 0], proj_b[i, 0]], [proj_a[i, 1], proj_b[i, 1]], 'gray', alpha=0.3)
                #     plt.title("t-SNE projection of contrastive embeddings (validation)"); plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2"); plt.grid(True); plt.legend()
                #     wandb.log({f"t-sne_val/val/e_{epoch_counter:03d}": wandb.Image(fig)})
                #     plt.close()
                # else:
                #     logging.warning(f"Skipping t-SNE projection for epoch {epoch_counter} as the number of samples is too low ({n}).")
            
                sim_matrices = []
                
                # Save best checkpoint
                if vloss.item() <= best_val_loss:
                    logging.info(f"Saving new best model at epoch {epoch_counter}, step {vstep} with loss {vloss.item()}.")
                    save_checkpoint({
                        'epoch': epoch_counter,
                        'arch': self.args.arch,
                        'state_dict': eval_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=True, dirpath=self.writer.dir)
                best_val_loss = min(best_val_loss, vloss.item())

            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")

            # Save model checkpoints
            save_checkpoint({
                'epoch': epoch_counter,
                'arch': self.args.arch,
                'state_dict': eval_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=(vloss < best_val_loss), dirpath=self.writer.dir)
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.dir}.")
        logging.info("Training has finished.")
