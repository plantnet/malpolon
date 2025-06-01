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
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.args.last_epoch = getattr(self.args, 'last_epoch', 0)
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.writer = wandb.init(
            entity="tlarcher-phd-jrc",
            id=self.args.ckpt_path.split('/')[1].split('-')[2] if self.args.ckpt_path else None,
            project=self.args.wandb_project,
            name=self.args.name,#'Unique surveyId spatial split 0.06min, dropout',
            notes=f"Shuffle train ON, val OFF. Info_nce_loss operating only with the main diagonal (no features concatenation). "\
                  f"All unique surveyId obs."\
                  f"Modality backbone: {'frozen' if self.args.freeze_modality_backbone else 'hot'}"\
                  f"GPS backbone: {'frozen' if self.args.freeze_gps_backbone else 'hot'}"\
                  f"LR cosine annealing {self.args.learning_rate}. "\
                  f"Temp {self.args.temperature}. "\
                  f"Dropout {self.args.dropout}. "\
                  f"Weight_decay {self.args.weight_decay}. ",
            group="SimCLR: satellite VS GPS",
            config=kwargs['args'],
        )
        logging.basicConfig(filename=os.path.join(self.writer.dir, 'training.log'), level=logging.DEBUG)
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

        # Validation metrics
        wandb.define_metric("acc/val/*", step_metric="val_steps")
        wandb.define_metric("Input_imgs_val/*", step_metric='val_steps')
        wandb.define_metric("SimMatrix_val/*", step_metric='val_steps')
        wandb.define_metric("Loss_epoch (batch avg)/val", step_metric="epoch")
        wandb.define_metric("Input_imgs_val/*", step_metric='val_steps')
        wandb.define_metric("acc_epoch (batch avg)/val/top1", step_metric="epoch")
        wandb.define_metric("acc_epoch (batch avg)/val/top5", step_metric="epoch")
        wandb.define_metric("SimMatrix_mean-epoch_val/*", step_metric="epoch")
        wandb.define_metric("t-sne/val/*", step_metric='epoch')

    def info_nce_loss(self, features, dataset_type: str = 'species'):
        batch_size = self.args.batch_size
        if dataset_type == 'landscape':
            batch_size = features.shape[0] // self.args.n_views  # LUCAS image views stacked along the batch dim
        labels = torch.cat([torch.arange(batch_size) for i in range(self.args.n_views)], dim=0)
        # Labels is a vector of size 64 with values 0 to 31 concatenated n_views times. E.g. if n_views==2: [0, 1, 2, ..., 31, 0, 1, 2, ..., 31]
        labels = (torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)).float()
        # Labels is transformed to one-hot and is of shape (64, 64).
        # There are 2 diagonals of ones: the 64x64 main diagonal, and a shifted diagonal (of the 2nd [0:31] vector originlly concatenated) which warps at the end of the columns to continue at the start of them on the next rows.
        labels = labels.to(self.args.device)

        # Features are the output of the MLP head. Shape (batch_size, 512)
        features = F.normalize(features, dim=1)
        # Features are L2-normalized to unit length. Shape (batch_size, 512)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * batch_size, self.args.n_views * batch_size)
        # assert similarity_matrix.shape == labels.shape

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
        labels = torch.eye(self.args.batch_size).to(self.args.device)  # mask is of shape (64, 64) with main diagonal at 1

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

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.dir, self.args)

        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        sim_matrices = []
        best_train_loss = torch.inf
        best_val_loss = torch.inf
        train_steps = 0
        val_steps = 0

        for epoch_counter in range(self.args.last_epoch, self.args.epochs + self.args.last_epoch):
            running_loss = 0.
            top1s, top5s = 0, 0
            wandb.log({"epoch": epoch_counter})
            print("Training the model...")
            print(f"> Starting epoch {epoch_counter}...")
            for step, (images, gps, inds, survey_ids) in enumerate(tqdm(train_loader)):
                wandb.log({"train_steps": train_steps})
                images = images.to(self.args.device)
                gps = gps.to(self.args.device)

                with autocast(device_type=str(self.args.device), enabled=self.args.fp16_precision):
                    features_img, features_gps = self.model(images, gps)
                    # features_gps = features_img.clone() # SANITY CHECK TO REMOVE
                    std_mean_img, std_mean_gps = torch.std_mean(features_img, dim=0), torch.std_mean(features_gps, dim=0)
                    std_mean_diff = (std_mean_img[0] - std_mean_gps[0], std_mean_img[1] - std_mean_gps[1])
                    norm_img, norm_gps = torch.norm(features_img, dim=1), torch.norm(features_gps, dim=1)
                    features = torch.cat([features_img, features_gps], dim=0)
                    # logits, labels, sim_matrix = self.info_nce_loss(features, dataset_type=self.args.arch)
                    logits, labels, sim_matrix = self.info_nce_loss_single_diag(features_img, features_gps, dataset_type=self.args.arch)
                    sim_matrices.append(sim_matrix)
                    loss = self.criterion(logits, labels)
                    running_loss += loss
                    best_train_loss = min(best_train_loss, loss.item())

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if step % self.args.log_every_n_steps == 0:            
                    # Log input batch images
                    fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # 4 rows, 8 columns
                    axes = axes.flatten()
                    for idx, (img, ind, sid, ax) in enumerate(zip(images, inds, survey_ids, axes)):
                        img = img[:-1, :, :].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                        ax.set_title(f"Image {idx}, iter_idx {ind[0]}, \nsurveyId {sid}", fontsize=8)
                        ax.axis('off')
                    plt.tight_layout()
                    wandb.log({f'Input_imgs_train/e_{epoch_counter:03d}_s_{step:03d}': wandb.Image(fig)})
                    plt.close()
                    
                    # Log loss and moments step wise
                    wandb.log({"Loss_step/train": loss, "learning_rate": self.scheduler.get_last_lr()[0]})
                    # wandb.log({"std-avg_img/train": std_mean_img[0].mean(), "mean-avg_img/train": std_mean_img[1].mean(),
                    #            "std-avg_gps/train": std_mean_gps[0].mean(), "mean-avg_gps/train": std_mean_gps[1].mean(),
                    #            "std-avg_diff/train": std_mean_diff[0].mean(), "mean-avg_diff/train": std_mean_diff[1].mean()})
                    wandb.log({"norm_img_avg/train": norm_img.mean(),
                               "norm_gps_avg/train": norm_gps.mean(),
                               "norm_avg_diff/train": abs(norm_img.mean()-norm_gps.mean())})
                    
                    # Log similarity matrix step wise
                    plt.figure(figsize=(12, 10))
                    plt.title("Similarity Matrix")
                    hm = sns.heatmap(sim_matrix, cmap="viridis", annot=False)
                    wandb.log({f'SimMatrix_train/e_{epoch_counter:03d}_s_{step:03d}': wandb.Image(hm)})
                    plt.close()
                    
                    # Log mean of similarity matrices computed over self.args.log_every_n_steps steps
                    sim_matrix_mean = torch.Tensor(np.array(sim_matrices)).mean(dim=0)
                    plt.figure(figsize=(12, 10))
                    plt.title("Similarity Matrix")
                    hm = sns.heatmap(sim_matrix_mean, cmap="viridis", annot=False)
                    wandb.log({f'SimMatrix_mean-{self.args.log_every_n_steps}-steps_train/{epoch_counter:03d}_s_{step:03d}': wandb.Image(hm)})
                    plt.close()
                    sim_matrices = []
                    
                    # Log accuracy step wise
                    if logits.shape[0] >= 5:
                        top1, top5 = accuracy(logits, labels, topk=(1, 5))
                        top1s += top1[0]
                        top5s += top5[0]
                        wandb.log({"acc/train/top1": top1[0],
                                   "acc/train/top5": top5[0]})
                    else:
                        print("Batch size is too small for accuracy calculation.")
                train_steps += 1
                if step >= max_iter:  # Debug purposes
                    break
            wandb.log({"Loss_epoch (batch avg)/train": running_loss / (step + 1)})
            wandb.log({"acc_epoch (batch avg)/train/top1": top1s / (step + 1),
                       "acc_epoch (batch avg)/train/top5": top5s / (step + 1)})
            
            # Log t-sne projection
            embeddings = torch.cat([features_img, features_gps], dim=0)
            labels = np.array([0]*len(features_img) + [1]*len(features_gps))  # 0=head A, 1=head B
            tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, metric='cosine', init='pca', random_state=42)
            embedding_2d = tsne.fit_transform(embeddings.detach().to('cpu').numpy())
            fig = plt.figure(figsize=(8, 6))
            colors = ['red' if l == 0 else 'blue' for l in labels]
            plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, alpha=0.7)
            plt.title("t-SNE projection of contrastive embeddings")
            plt.xlabel("t-SNE-1")
            plt.ylabel("t-SNE-2")
            plt.grid(True)
            wandb.log({f"t-sne_train/train/e_{epoch_counter:03d}": wandb.Image(fig)})
            plt.close()

            # Evaluation
            self.model.eval()
            print("Evaluating the model...")
            with torch.no_grad():
                running_vloss = 0.0
                vtop1s, vtop5s = 0, 0
                vsim_matrices = []
                for vstep, (vimages, vgps, vinds, vsurvey_ids) in enumerate(tqdm(val_loader)):
                    val_steps += vstep
                    wandb.log({"val_steps": val_steps})
                    vimages = vimages.to(self.args.device)
                    vgps = vgps.to(self.args.device)

                    vfeatures_img, vfeatures_gps = self.model(vimages, vgps)
                    vfeatures = torch.cat([vfeatures_img, vfeatures_gps], dim=0)
                    # vlogits, vlabels, vsim_matrix = self.info_nce_loss(vfeatures, dataset_type=self.args.arch)
                    vlogits, vlabels, vsim_matrix = self.info_nce_loss_single_diag(vfeatures_img, vfeatures_gps, dataset_type=self.args.arch)
                    vsim_matrices.append(vsim_matrix)
                    vloss = self.criterion(vlogits, vlabels)
                    running_vloss += vloss

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

                    if vlogits.shape[0] >= 5:
                        vtop1, vtop5 = accuracy(vlogits, vlabels, topk=(1, 5))
                        vtop1s += vtop1[0]
                        vtop5s += vtop5[0]
                        wandb.log({"acc/val/top1": vtop1[0],
                                   "acc/val/top5": vtop5[0]})

                    if (vstep) % self.args.log_every_n_steps == 0:
                        # Log input batch images
                        fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # 4 rows, 8 columns
                        axes = axes.flatten()
                        for idx, (img, ind, sid, ax) in enumerate(zip(vimages, vinds, vsurvey_ids, axes)):
                            img = img[:-1, :, :].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                            ax.set_title(f"Image {idx}, iter_idx {ind[0]}, \nsurveyId {sid}", fontsize=8)
                            ax.axis('off')
                        plt.tight_layout()
                        wandb.log({f'Input_imgs_val/e_{epoch_counter:03d}_s_{vstep:03d}': wandb.Image(fig)})
                        plt.close()
                    
                        plt.figure(figsize=(12, 10))
                        plt.title("Similarity Matrix val")
                        hm = sns.heatmap(vsim_matrix, cmap="viridis", annot=False)
                        wandb.log({f'SimMatrix_val/e_{epoch_counter:03d}_vstep_{vstep:03d}': wandb.Image(hm)})
                        plt.close()

                    val_steps += 1
                    if vstep >= max_iter:
                        break
                wandb.log({"Loss_epoch (batch avg)/val": running_vloss / (val_steps + 1)})
                wandb.log({"acc_epoch (batch avg)/val/top1": vtop1s / (val_steps + 1),
                           "acc_epoch (batch avg)/val/top5": vtop5s / (val_steps + 1)})
                vsim_matrix_mean = torch.Tensor(np.array(vsim_matrices)).mean(dim=0)
                plt.figure(figsize=(12, 10))
                plt.title("Similarity Matrix")
                vhm = sns.heatmap(vsim_matrix_mean, cmap="viridis", annot=False)
                wandb.log({f'SimMatrix_mean-epoch_val/e{epoch_counter:03d}': wandb.Image(vhm)})
                plt.close()
                
                # Log t-sne projection
                embeddings = torch.cat([vfeatures_img, vfeatures_gps], dim=0)
                labels = np.array([0]*len(vfeatures_img) + [1]*len(vfeatures_gps))  # 0=head A, 1=head B
                tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, metric='cosine', init='pca', random_state=42)
                embedding_2d = tsne.fit_transform(embeddings.detach().to('cpu').numpy())
                fig = plt.figure(figsize=(8, 6))
                colors = ['red' if l == 0 else 'blue' for l in labels]
                plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, alpha=0.7)
                plt.title("t-SNE projection of contrastive embeddings")
                plt.xlabel("t-SNE-1")
                plt.ylabel("t-SNE-2")
                plt.grid(True)
                wandb.log({f"t-sne_val/val/e_{epoch_counter:03d}": wandb.Image(fig)})
                plt.close()
            
                sim_matrices = []

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
