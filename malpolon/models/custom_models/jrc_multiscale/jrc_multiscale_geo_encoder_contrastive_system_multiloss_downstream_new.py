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
import torchmetrics.functional as Fmetrics
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn

from malpolon.models.custom_models.jrc_multiscale.jrc_contrastive_losses import (
    KoLeoLoss, MCR, cosine_embedding_loss, cosine_similarity_mean, cosine_loss_pytorch_like,
    cosine_embedding_from_sim, crisp_loss, crisp_loss_manual
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

# Metrics
def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # Rows: top-k predictions for each sample, Cols: batch samples
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # Scalar
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_multilabel_topk(scores, labels, topk=(1,), average='samples'):
    """Compute the multilabel accuracy within the top-k predictions.
    
    This function computes the number of true positives within the top-k predicted labels for each
    sample, averages over k then over samples.
    """
    res = []
    for k in topk:
        topk_values, topk_indices = torch.topk(scores, k)
        label_indices = (labels == 1).nonzero(as_tuple=True)[0]  # Get indices where labels == 1
        if average == 'samples':
            count_sample = torch.isin(topk_indices, label_indices).sum(dim=1)
            acc_topk_mean = (count_sample / k).mean()
        else:
            raise NotImplementedError(f"Average method {average} not implemented.")
        res.append(acc_topk_mean)
    return res

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

def log_loss_scheduler(loss, scheduler=None, mode='train'):
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

def log_acc_topk_step(logits, labels, modality_name, topk=(1, 5), mode='train', acc_type='multilabel', average='samples'):
    if logits[0].shape[0] >= max(topk):
        if acc_type == 'multilabel':
            acc_topks = accuracy_multilabel_topk(logits, labels, topk=topk, average=average)
        elif acc_type in ['multiclass', '']:
            acc_topks = accuracy(logits, labels, topk=topk)
        else:
            raise NotImplementedError(f"Accuracy type {acc_type} not implemented.")
        if mode != 'test':
            for acc_topk, topk in zip(acc_topks, topk):
                wandb.log({f"acc{'_'+acc_type}{'_'+average}/{mode}/top{topk}_{modality_name}": acc_topk.item()})
    else:
        print("Batch size is too small for accuracy calculation.")
        return None, None
    return tuple(map(lambda x: x.item(), acc_topks))

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

def find_best_threshold(y_true, y_probs, num_labels):
    y_true = torch.from_numpy(y_true).cpu() if isinstance(y_true, np.ndarray) else y_true
    y_probs = torch.from_numpy(y_probs).cpu() if isinstance(y_probs, np.ndarray) else y_probs
    thresholds = np.linspace(0, 1, 101)  # test thresholds from 0.0 to 1.0
    best_thresh, best_f1 = 0.5, 0
    for t in thresholds:
        y_pred = (y_probs >= t).to(int)
        f1 = Fmetrics.classification.multilabel_f1_score(y_true, y_pred, num_labels=num_labels, average="macro")  # or "micro"/"weighted"
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1

def wandb_init():
    # Iterations metricsx
    wandb.define_metric("epoch")
    wandb.define_metric("train_steps")
    wandb.define_metric("val_steps")

    # Train metrics
    ## By epoch
    wandb.define_metric("Loss_epoch (batch avg)/train", step_metric="epoch")
    wandb.define_metric("acc_epoch (batch avg)/train/*", step_metric="epoch")
    wandb.define_metric("acc_multilabel_samples_epoch (batch avg)/train/*", step_metric="epoch")
    wandb.define_metric("acc_multilabel_micro_epoch (batch avg)/train/*", step_metric="epoch")
    wandb.define_metric("acc_multilabel_macro_epoch (batch avg)/train/*", step_metric="epoch")
    wandb.define_metric("t-sne/train/*", step_metric='epoch')
    ## By step
    wandb.define_metric("Loss_step/train", step_metric="train_steps")
    wandb.define_metric("norm_img_avg/train", step_metric="train_steps")
    wandb.define_metric("norm_gps_avg/train", step_metric="train_steps")
    wandb.define_metric("norm_avg_diff/train", step_metric="train_steps")
    wandb.define_metric("acc/train/*", step_metric="train_steps")
    wandb.define_metric("acc_multilabel_samples/train/*", step_metric="train_steps")
    wandb.define_metric("acc_multilabel_micro_step/train/", step_metric="train_steps")
    wandb.define_metric("acc_multilabel_macro_step/train/", step_metric="train_steps")
    wandb.define_metric("f1_micro_step/train/", step_metric="train_steps")
    wandb.define_metric("Input_imgs_train/*", step_metric='train_steps')
    wandb.define_metric("SimMatrix_train/*", step_metric='train_steps')
    wandb.define_metric("recall_step/train", step_metric="train_steps")
    wandb.define_metric("recall@1_step/train", step_metric="train_steps")
    wandb.define_metric("recall@20_step/train", step_metric="train_steps")
    wandb.define_metric("recall@100_step/train", step_metric="train_steps")
    wandb.define_metric("MultilabelAUROC_micro_step/train", step_metric="train_steps")    
    wandb.define_metric("MultilabelAUROC_macro_step/train", step_metric="train_steps")    
    wandb.define_metric("MultilabelAveragePrecision_micro_step/train", step_metric="train_steps")
    wandb.define_metric("MultilabelAveragePrecision_macro_step/train", step_metric="train_steps")

    # Validation metrics
    ## By epoch
    wandb.define_metric("Loss_epoch (batch avg)/val", step_metric="epoch")
    wandb.define_metric("Input_imgs_val/*", step_metric='val_steps')
    wandb.define_metric("acc_epoch (batch avg)/val/*", step_metric="epoch")
    wandb.define_metric("acc_multilabel_samples_epoch (batch avg)/val/*", step_metric="epoch")
    wandb.define_metric("acc_multilabel_micro_epoch (batch avg)/val/*", step_metric="epoch")
    wandb.define_metric("acc_multilabel_macro_epoch (batch avg)/val/*", step_metric="epoch")
    wandb.define_metric("SimMatrix_mean-epoch_val/*", step_metric="epoch")
    wandb.define_metric("t-sne/val/*", step_metric='epoch')
    ## By step
    wandb.define_metric("Loss_step/val", step_metric="val_steps")
    wandb.define_metric("acc/val/*", step_metric="val_steps")
    wandb.define_metric("acc_multilabel_samples/val/*", step_metric="val_steps")
    wandb.define_metric("acc_multilabel_micro_step/val/", step_metric="val_steps")
    wandb.define_metric("acc_multilabel_macro_step/val/", step_metric="val_steps")
    wandb.define_metric("f1_micro_step/val/", step_metric="val_steps")
    wandb.define_metric("Input_imgs_val/*", step_metric='val_steps')
    wandb.define_metric("SimMatrix_val/*", step_metric='val_steps')
    wandb.define_metric("recall_step/val", step_metric="val_steps")
    wandb.define_metric("recall@1_step/val", step_metric="val_steps")
    wandb.define_metric("recall@20_step/val", step_metric="val_steps")
    wandb.define_metric("recall@100_step/val", step_metric="val_steps")
    wandb.define_metric("MultilabelAUROC_micro_step/val", step_metric="val_steps")
    wandb.define_metric("MultilabelAUROC_macro_step/val", step_metric="val_steps")
    wandb.define_metric("MultilabelAveragePrecision_micro_step/val", step_metric="val_steps")
    wandb.define_metric("MultilabelAveragePrecision_macro_step/val", step_metric="val_steps")

class SimCLR_downstream(object):
    def __init__(self, writer=SummaryWriter(), *args, **kwargs):
        self.writer = writer
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.skip_modalities = getattr(self.args, 'skip_modalities', [])
        self.downstream_modalities_to_process = getattr(self.args, 'downstream_modalities_to_process', ['satellite_img'])

        self.koleo_weight = self.args.koleo_weight if hasattr(self.args, 'koleo_weight') else 0.0
        self.koleo_eps = self.args.koleo_eps if hasattr(self.args, 'koleo_eps') else 1e-4

        self.ema_model = kwargs['model'] if getattr(self.args, 'use_ema', False) else None
        self.ema_decay = getattr(self.args, 'ema_decay', 0.99)
        self.ema_update_step = getattr(self.args, 'ema_update_step', 1)

        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.inference = bool(getattr(self.args, 'predict', False))
        self.criterion_name = getattr(self.args, 'loss_criterion', 'cross_entropy')
        self.criterion = self.get_criterion(self.criterion_name)

        self.resume_wandb_run = getattr(self.args, 'resume_wandb_run', False)
        self.jobid = getattr(self.args, 'OAR_job_id', 'no_jobid')
        self.args.last_epoch = getattr(self.args, 'last_epoch', 0)
        print(f"[INFO] Job ID: {self.jobid}")
        print(f"[INFO] Wandb ID: {self.writer.id}")
        print(f"[INFO] Wandb output directory: {self.writer.dir}")
        logging.basicConfig(filename=os.path.join(self.writer.dir, 'training.log'), level=logging.DEBUG)
        wandb_init()

        self.log_images = getattr(self.args, 'log_images', True)
        self.best_f1_thresh = 0.3  # Initial value, then updated after each val step
        self.num_labels = getattr(self.args, 'num_labels', 1)
    
    def get_criterion(self, criterion_name):
        """Retrieves the right criterion with correct inputs.
        
        Possible values of criterion_name: 'cross_entropy', 'cosine_embedding', 'cosine_embedding_from_sim',
        'cosine_similarity_mean', 'cosine_loss_pytorch_like'.

        """
        if criterion_name.lower() == 'cross_entropy':
            criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        elif criterion_name.lower() == 'bce':
            criterion = torch.nn.BCEWithLogitsLoss().to(self.args.device)
        else:
            raise NotImplementedError(f"Loss criterion {criterion_name} not implemented.")
        return criterion

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

    def forward_accumulate(self, input: torch.tensor, input_type: str, loss: torch.tensor, labels: torch.tensor,
                           scaler: GradScaler = None, regularizer: str = '') -> tuple:
        # Calling backward multiple times is the same as accumulating the gradients (summing the loss) and calling backward once after.
        logits = self.model(input, input_type)
        loss += self.criterion(logits, labels)
        if regularizer:
            loss += self.get_regularizer(logits, regularizer)

        if scaler:
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

        return logits, loss

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
        # wandb.watch(self.model.gps_contrastive_head, log="gradients", log_freq=self.args.log_every_n_steps_train)
        # wandb.watch(self.model.modality_contrastive_head, log="gradients", log_freq=self.args.log_every_n_steps_train)
        save_config_file(self.writer.dir, self.args)
        scaler = GradScaler(enabled=self.args.fp16_precision)

        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        best_train_loss, best_val_loss = torch.inf, torch.inf
        train_steps, val_steps = 0, 0

        for epoch_counter in range(self.args.last_epoch, self.args.epochs + self.args.last_epoch):
            print("Training the model...")
            print(f"> Starting epoch {epoch_counter}...")
            wandb.log({"epoch": epoch_counter})
            topks = {f'top{k}': [] for k in self.args.metrics['accuracy_topks']}
            metrics = {'multilabel_accuracy_micro': [],
                       'multilabel_accuracy_macro': [],
                       'multilabel_f1_micro': []}
            running_loss = []

            for step, train_dict in enumerate(tqdm(train_loader)):
                # batch_inds = train_dict.pop('indices', None)
                wandb.log({"train_steps": train_steps})

                with autocast(device_type=str(self.args.device), enabled=self.args.fp16_precision):
                    loss, all_logits, all_gps, all_images, all_inds, all_ids = 0, [], [], [], [], []
                    for downstream_modality in self.downstream_modalities_to_process:
                        modality = downstream_modality.split('_')[0]
                        if modality not in self.skip_modalities:
                            if '_img' in downstream_modality:
                                input = train_dict[modality][0].to(self.args.device)
                            if '_gps' in downstream_modality:
                                input = train_dict[modality][1].to(self.args.device)
                            labels = train_dict[modality][-1].to(self.args.device)
                            logits, loss = self.forward_accumulate(input, downstream_modality, loss, labels, scaler=scaler)
                            all_images.append(train_dict[modality][0])
                            all_gps.append(train_dict[modality][1])
                            all_inds.append(train_dict[modality][2])
                            all_ids.append(train_dict[modality][3])
                    # labels = train_dict[-2].to(self.args.device)
                    # input = train_dict[2].to(self.args.device)
                    # logits, loss = self.forward_accumulate(input, 'satellite_img', loss, labels, scaler=scaler)
                    # all_images.append(train_dict[2])
                    # all_gps.append(-1)
                    # all_inds.append(-1)
                    # all_ids.append(train_dict[-1])
                    
                    # print(f"[TRAIN] N_pos_labels: {labels.sum(dim=1)}")
                    # print(f"[TRAIN] Positive label at class: {torch.where(labels==1)}")
                    # print(f"[TRAIN] Argmax logits: {torch.argmax(logits, dim=1)}")
                    # self.optimizer.zero_grad()
                    # scaler.scale(loss).backward()
                    # # Print gradients for debugging
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"Gradient for {name}: {param.grad.norm().item():.6f}")
                    #     else:
                    #         print(f"No gradient for {name}")
                    # scaler.step(self.optimizer)
                    # scaler.update()
                    
                    logits, labels = logits.to('cpu'), labels.to('cpu')
                    all_logits.append(logits)
                    running_loss.append(loss.item())
                    best_train_loss = min(best_train_loss, loss.item())
                    print(f"Epoch {epoch_counter} loss: {loss.item():.4f}")

                # self.optimizer.zero_grad()
                # scaler.scale(loss).backward()
                # scaler.step(self.optimizer)
                # scaler.update()

                if round(step % self.args.log_every_n_steps_train) == 0:
                    labels_oh = labels.int()
         
                    # log_input_imgs_multimodalities(all_images, all_inds, all_ids, step, epoch_counter, n_samples=8, n_modalities=len(self.downstream_modalities_to_process), mode='train', log_images=self.log_images)
                    
                    # Log loss and moments step wise
                    print('Logging scheduler & moments...')
                    log_loss_scheduler(loss, self.scheduler)
                    # log_moments(norm_img, norm_gps, std_mean_img, std_mean_gps, std_mean_diff)

                    # Log accuracy step wise
                    print('Logging top-k accuracy...')
                    for logits, modality_name in zip(all_logits, self.downstream_modalities_to_process):
                        topk_all = log_acc_topk_step(logits, labels, modality_name.split('_')[0], topk=self.args.metrics['accuracy_topks'], mode='train', acc_type=self.args.metrics['accuracy_type'], average=self.args.metrics['accuracy_average'])
                        # Every step wise top-k is stored in a list where modalities are interleaved. E.g. [topk_modality1, topk_modality2, topk_modality3]
                        if all(topk is not None for topk in topk_all):
                            for k, v in zip(self.args.metrics['accuracy_topks'], topk_all):
                                topks[f'top{k}'].append(v)
                    metrics['multilabel_accuracy_micro'].append(Fmetrics.classification.multilabel_accuracy(logits, labels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='micro'))
                    metrics['multilabel_accuracy_macro'].append(Fmetrics.classification.multilabel_accuracy(logits, labels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='macro'))
                    wandb.log({"acc_multilabel_micro_step/train/": metrics['multilabel_accuracy_micro'][-1],
                               "acc_multilabel_macro_step/train/": metrics['multilabel_accuracy_macro'][-1]})
                    print(f'Multilabel accuracy micro (step) mean / train: {np.mean(metrics["multilabel_accuracy_micro"]):.4f}, macro (step): {np.mean(metrics["multilabel_accuracy_macro"]):.4f}')
                    
                    # Log f1-score step wise
                    print('Logging f1-score...')
                    metrics['multilabel_f1_micro'].append(Fmetrics.classification.multilabel_f1_score(logits, labels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='micro'))
                    wandb.log({"f1_micro_step/train/": metrics['multilabel_f1_micro'][-1]})
                    print(f'Multilabel f1-score micro (step) mean / train: {np.mean(metrics["multilabel_f1_micro"]):.4f}')

                    # Log recall@K
                    print('Logging recall@K...')
                    wandb.log({"recall_step/train": retrieval_recall(logits, labels_oh)})
                    wandb.log({"recall@1_step/train": retrieval_recall(logits, labels_oh, top_k=1)})
                    wandb.log({"recall@20_step/train": retrieval_recall(logits, labels_oh, top_k=5)})
                    wandb.log({"recall@100_step/train": retrieval_recall(logits, labels_oh, top_k=100)})
                    print(f'Retrieval recall@K (step) / train: recall@1: {retrieval_recall(logits, labels_oh, top_k=1):.4f}, recall@20: {retrieval_recall(logits, labels_oh, top_k=20):.4f}, recall@100: {retrieval_recall(logits, labels_oh, top_k=100):.4f}')

                    # Log AUROC
                    # print('Logging AUROC...')
                    # wandb.log({"MultilabelAUROC_micro_step/train": multilabel_auroc(logits, labels_oh, self.num_labels, average='micro')})
                    # wandb.log({"MultilabelAUROC_macro_step/train": multilabel_auroc(logits, labels_oh, self.num_labels, average='macro')})

                    # Log mAP
                    # wandb.log({"MultilabelAveragePrecision_micro_step/train": multilabel_average_precision(logits, labels_oh, n_cls, average='micro')})
                    # wandb.log({"MultilabelAveragePrecision_macro_step/train": multilabel_average_precision(logits, labels_oh, n_cls, average='macro')})
                train_steps += 1
                if step >= max_iter:  # Debug purposes
                    break

            wandb.log({"Loss_epoch (batch avg)/train": np.array(running_loss).mean()})
            
            # Log accuracy epoch wise
            print('Logging top-k accuracy epoch wise...')
            for k, v in topks.items():
                wandb.log({f"acc_{self.args.metrics['accuracy_type']}_{self.args.metrics['accuracy_average']}_epoch (batch avg)/train/{k}": np.array(v).mean()})
                print(f"acc_{self.args.metrics['accuracy_type']}_{self.args.metrics['accuracy_average']}_epoch (batch avg)/train/{k}: {np.array(v).mean():.4f}")
            
            wandb.log({"acc_multilabel_micro_epoch (batch_avg)/train/": np.array(metrics['multilabel_accuracy_micro']).mean(),
                       "acc_multilabel_macro_epoch (batch_avg)/train/": np.array(metrics['multilabel_accuracy_macro']).mean()})
            
            # Log f1-score epoch wise
            print('Logging f1-score epoch wise...')
            wandb.log({"f1_micro_epoch (batch_avg)/train/": np.array(metrics['multilabel_f1_micro']).mean()})
            print(f"f1_micro_epoch (batch_avg)/train/: {np.array(metrics['multilabel_f1_micro']).mean():.4f}")

            # Log t-sne projection
            # for features_img, features_gps, modality_name in zip(all_features_img, all_features_gps, self.modalities_to_process):
            #     log_tsne(features_img, features_gps, epoch_counter, modality_name, log_images=self.log_images, mode='train')

            # Evaluation
            with torch.no_grad():
                print("Evaluating the model...")
                vtopks = {f'top{k}': [] for k in self.args.metrics['accuracy_topks']}
                vmetrics = {'multilabel_accuracy_micro': [],
                            'multilabel_accuracy_macro': [],
                            'multilabel_f1_micro': []}
                running_vloss = []

                for vstep, val_dict in enumerate(tqdm(val_loader)):
                    vbatch_inds = val_dict.pop('indices', None)
                    wandb.log({"val_steps": val_steps})

                    with autocast(device_type=str(self.args.device), enabled=self.args.fp16_precision):
                        vloss, vall_logits, vall_gps, vall_images, vall_inds, vall_ids = 0, [], [], [], [], []
                        for downstream_modality in self.downstream_modalities_to_process:
                            modality = downstream_modality.split('_')[0]
                            if modality not in self.skip_modalities:
                                if '_img' in downstream_modality:
                                    input = val_dict[modality][0].to(self.args.device)
                                if '_gps' in downstream_modality:
                                    input = val_dict[modality][1].to(self.args.device)
                                vlabels = val_dict[modality][-1].to(self.args.device)
                                vlogits, vloss = self.forward_accumulate(input, downstream_modality, vloss, vlabels)
                                vall_images.append(val_dict[modality][0])
                                vall_gps.append(val_dict[modality][1])
                                vall_inds.append(val_dict[modality][2])
                                vall_ids.append(val_dict[modality][3])
                        vlogits, vlabels = vlogits.to('cpu'), vlabels.to('cpu')
                        vall_logits.append(vlogits)
                        running_vloss.append(vloss.item())
                        best_val_loss = min(best_val_loss, vloss.item())
                        print(f"Epoch {epoch_counter} vloss: {vloss.item():.4f}")
                        # print(f"[VAL] N_pos_labels: {vlabels.sum(dim=1)}")
                        # print(f"[VAL] Positive label at class: {torch.where(vlabels==1)}")
                        # print(f"[VAL] Argmax logits: {torch.argmax(vlogits, dim=1)}")

                    if round(vstep % self.args.log_every_n_steps_val) == 0:
                        vlabels_oh = vlabels.int()

                        # log_input_imgs_multimodalities(vall_images, vall_inds, vall_ids, vstep, epoch_counter, n_samples=8, n_modalities=len(self.downstream_modalities_to_process), mode='val', log_images=self.log_images)
                        
                        # Log loss and moments step wise
                        print('Logging scheduler & moments...')
                        log_loss_scheduler(vloss, mode='val')

                        # Log accuracy step wise
                        if vlogits.shape[0] >= 5:
                            # Log accuracy step wise
                            print('Logging accuracy...')
                            for vlogits, modality_name in zip(vall_logits, self.downstream_modalities_to_process):
                                vtopk_all = log_acc_topk_step(vlogits, vlabels, modality_name.split('_')[0], topk=self.args.metrics['accuracy_topks'], mode='val', acc_type=self.args.metrics['accuracy_type'], average=self.args.metrics['accuracy_average'])
                                # Every step wise top-k is stored in a list where modalities are interleaved. E.g. [topk_modality1, topk_modality2, topk_modality3]
                                if all(vtopk is not None for vtopk in vtopk_all):
                                    for k, v in zip(self.args.metrics['accuracy_topks'], vtopk_all):
                                        vtopks[f'top{k}'].append(v)
                            vmetrics['multilabel_accuracy_micro'].append(Fmetrics.classification.multilabel_accuracy(vlogits, vlabels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='micro'))
                            vmetrics['multilabel_accuracy_macro'].append(Fmetrics.classification.multilabel_accuracy(vlogits, vlabels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='macro'))
                            wandb.log({"acc_multilabel_micro_step/val/": vmetrics['multilabel_accuracy_micro'][-1],
                                       "acc_multilabel_macro_step/val/": vmetrics['multilabel_accuracy_macro'][-1]})
                        else:
                            print("Batch size (val) is too small for accuracy calculation.")

                        # Log F1-score step wise
                        print('Logging f1-score...')
                        best_thresh, best_f1 = find_best_threshold(vlabels, vlogits, num_labels=self.num_labels)
                        self.best_f1_thresh = best_thresh
                        vmetrics['multilabel_f1_micro'].append(Fmetrics.classification.multilabel_f1_score(vlogits, vlabels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='micro'))
                        wandb.log({"f1_micro_step/val/": vmetrics['multilabel_f1_micro'][-1]})

                        # Log recall@K
                        print('Logging recall@K...')
                        wandb.log({"recall_step/val": retrieval_recall(vlogits, vlabels_oh)})
                        wandb.log({"recall@1_step/val": retrieval_recall(vlogits, vlabels_oh, top_k=1)})
                        wandb.log({"recall@20_step/val": retrieval_recall(vlogits, vlabels_oh, top_k=5)})
                        wandb.log({"recall@100_step/val": retrieval_recall(vlogits, vlabels_oh, top_k=100)})

                        # Log AUROC
                        # print('Logging AUROC...')
                        # wandb.log({"MultilabelAUROC_micro_step/val": multilabel_auroc(vlogits, vlabels_oh, self.num_labels, average='micro')})
                        # wandb.log({"MultilabelAUROC_macro_step/val": multilabel_auroc(vlogits, vlabels_oh, self.num_labels, average='macro')})

                        # Log mAP
                        # wandb.log({"MultilabelAveragePrecision_micro_step/val": multilabel_average_precision(vlogits, vlabels_oh, n_cls, average='micro')})
                        # wandb.log({"MultilabelAveragePrecision_macro_step/val": multilabel_average_precision(vlogits, vlabels_oh, n_cls, average='macro')})

                    val_steps += 1
                    if vstep >= max_iter:
                        break

                wandb.log({"Loss_epoch (batch avg)/val": np.array(running_vloss).mean()})

                # Log accuracy epoch wise
                print('Logging top-k accuracy epoch wise...')
                for k, v in vtopks.items():
                    wandb.log({f"acc_{self.args.metrics['accuracy_type']}_{self.args.metrics['accuracy_average']}_epoch (batch avg)/val/{k}": np.array(v).mean()})
                    print(f"acc_{self.args.metrics['accuracy_type']}_{self.args.metrics['accuracy_average']}_epoch (batch avg)/val/{k}: {np.array(v).mean():.4f}")

                wandb.log({"acc_multilabel_micro_epoch (batch_avg)/val/": np.array(vmetrics['multilabel_accuracy_micro']).mean(),
                           "acc_multilabel_macro_epoch (batch_avg)/val/": np.array(vmetrics['multilabel_accuracy_macro']).mean()})
                
                # Log f1-score epoch wise
                print('Logging f1-score epoch wise...')
                wandb.log({"f1_micro_epoch (batch_avg)/val/": np.array(vmetrics['multilabel_f1_micro']).mean()})
                print(f"f1_micro_epoch (batch_avg)/val/: {np.array(vmetrics['multilabel_f1_micro']).mean():.4f}")
            
                # Log t-sne projection
                # for vfeatures_img, vfeatures_gps, modality_name in zip(vall_features_img, vall_features_gps, self.modalities_to_process):
                #     log_tsne(vfeatures_img, vfeatures_gps, epoch_counter, modality_name, log_images=self.log_images, mode='val')

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
