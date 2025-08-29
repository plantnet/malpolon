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

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multilabel_auroc, multilabel_average_precision
from torchmetrics.functional.retrieval import retrieval_recall

from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchmetrics.functional as Fmetrics
from tqdm import tqdm
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    wandb.define_metric("acc_micro_step/train/*", step_metric="train_steps")
    wandb.define_metric("acc_macro_step/train/*", step_metric="train_steps")
    wandb.define_metric("f1_micro_step/train/*", step_metric="train_steps")
    wandb.define_metric("Input_imgs_train/*", step_metric='train_steps')
    wandb.define_metric("SimMatrix_train/*", step_metric='train_steps')
    wandb.define_metric("Loss_epoch (batch avg)/train", step_metric="epoch")
    wandb.define_metric("acc_epoch (batch avg)/train/top1", step_metric="epoch")
    wandb.define_metric("acc_epoch (batch avg)/train/top5", step_metric="epoch")
    wandb.define_metric("acc_micro_epoch (batch_avg)/train/", step_metric="epoch")
    wandb.define_metric("acc_macro_epoch (batch_avg)/train/", step_metric="epoch")
    wandb.define_metric("f1_micro_epoch (batch_avg)/train/", step_metric="epoch")
    wandb.define_metric("t-sne/train/*", step_metric='epoch')

    # Validation metrics
    wandb.define_metric("Loss_step/val", step_metric="val_steps")
    wandb.define_metric("acc/val/*", step_metric="val_steps")
    wandb.define_metric("acc_micro_step/val/*", step_metric="val_steps")
    wandb.define_metric("acc_macro_step/val/*", step_metric="val_steps")
    wandb.define_metric("f1_micro_step/val/*", step_metric="val_steps")
    wandb.define_metric("Input_imgs_val/*", step_metric='val_steps')
    wandb.define_metric("SimMatrix_val/*", step_metric='val_steps')
    wandb.define_metric("Loss_epoch (batch avg)/val", step_metric="epoch")
    wandb.define_metric("Input_imgs_val/*", step_metric='val_steps')
    wandb.define_metric("acc_epoch (batch avg)/val/top1", step_metric="epoch")
    wandb.define_metric("acc_epoch (batch avg)/val/top5", step_metric="epoch")
    wandb.define_metric("SimMatrix_mean-epoch_val/*", step_metric="epoch")
    wandb.define_metric("acc_micro_epoch (batch_avg)/val/", step_metric="epoch")
    wandb.define_metric("acc_macro_epoch (batch_avg)/val/", step_metric="epoch")
    wandb.define_metric("f1_micro_epoch (batch_avg)/val/", step_metric="epoch")
    wandb.define_metric("t-sne/val/*", step_metric='epoch')

def find_best_threshold(y_true, y_probs):
    thresholds = np.linspace(0, 1, 101)  # test thresholds from 0.0 to 1.0
    best_thresh, best_f1 = 0.5, 0
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = Fmetrics.classification.multilabel_f1_score(y_true, y_pred, average="macro")  # or "micro"/"weighted"
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1

class SimCLRToMultilabelClassification(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.args.last_epoch = getattr(self.args, 'last_epoch', 0)
        self.num_labels = getattr(self.args, 'num_labels', 1)
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.resume_wandb_run = getattr(self.args, 'resume_wandb_run', False)
        self.log_images = getattr(self.args, 'log_images', True)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.args.device)
        self.skip_modalities = getattr(self.args, 'skip_modalities', [])
        self.inference = bool(getattr(self.args, 'predict', False))
        # Wandb logger
        self.writer = wandb.init(
            entity="tlarcher-phd-jrc",
            id=self.args.ckpt_path.split('/')[-2].split('-')[2] if (self.args.ckpt_path and self.resume_wandb_run) else None,
            project=self.args.wandb_project,
            name=self.args.name,#'Unique surveyId spatial split 0.06min, dropout',
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
        logging.basicConfig(filename=os.path.join(self.writer.dir, 'training.log'), level=logging.DEBUG)
        wandb_init()
        # Tensorboard logger
        self.tensorboard_writer = SummaryWriter()
        self.best_f1_thresh = 0.3  # Initial value, then updated after each val step
        

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

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        max_iter: int = torch.inf,
        verbose: bool = False,
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

        for epoch_counter in range(self.args.epochs):
            metrics = {'multilabel_accuracy_micro': [],
                    'multilabel_accuracy_macro': [],
                    'multilabel_f1_micro': []}
            running_loss, top1s, top5s = [], [], []
            wandb.log({"epoch": epoch_counter})
            print("Training the model...")
            print(f"> Starting epoch {epoch_counter}...")
            for step, train_dict in enumerate(tqdm(train_loader)):
                batch_inds = train_dict['indices']
                train_dict.pop('indices', None)

                train_dict_items = train_dict.items()
                train_dict_items = [(k, v) for k, v in train_dict_items if k in modalities_to_process]
                wandb.log({"train_steps": train_steps})
                self.optimizer.zero_grad()
                loss, all_logits, all_images, idxs, ids = 0, [], [], [], []

                with autocast(device_type=str(self.args.device), enabled=self.args.fp16_precision):
                    labels = train_dict[modalities_to_process[0]][-1]
                    logits = self.model(train_dict['species'][0].to(self.args.device),
                                        train_dict['landscape'][0].to(self.args.device),
                                        train_dict['satellite'][0].to(self.args.device),
                                        train_dict[modalities_to_process[0]][1].to(self.args.device))
                    loss = self.criterion(logits, labels.to(self.args.device))
                logits = logits.to('cpu')
                all_images.extend((train_dict[mod_name][0] for mod_name in modalities_to_process))

                scaler.scale(loss).backward()  # Gradients are accumulated. Calling backward after each modality loss equals calling backward once after sum + average of losses
                scaler.step(self.optimizer)
                scaler.update()
                running_loss.append(loss.to('cpu').item())

                if step % self.args.log_every_n_steps_train == 0:   
                    if verbose:
                        print("\n")
                        print(f"Step {step}, loss {loss.item()}.")
                        print(f"Labels min {labels.min().item()}, max {labels.max().item()}.")
                        print(f"Labels positive indices: {[(labels[i]==1).nonzero().tolist() for i in range(labels.shape[0])]}.")
                        print(f"Labels sample (5 first rows, 25 first cols): {labels[:5, :25]}.")
                        print(f"Logits min {logits.min().item()}, max {logits.max().item()}.")
                        print(f"Logits max positive indices: {torch.argmax(logits, dim=1)}.")
                        print(f"Logits sample (5 first rows, 25 first cols): {logits[:5, :25]}.")
                    log_input_imgs_multimodalities(all_images, idxs, ids, step, epoch_counter, n_samples=8, n_modalities=len(modalities_to_process), mode='train', log_images=self.log_images)
                    
                    # Log loss and moments step wise
                    log_loss_scheduler(loss, self.scheduler)
                    self.tensorboard_writer.add_scalar("Loss/train", loss, train_steps)
                    # log_moments(norm_img, norm_gps, std_mean_img, std_mean_gps, std_mean_diff)

                    # Log accuracy step wise           
                    from sklearn.metrics import precision_recall_fscore_support         
                    metrics['multilabel_accuracy_micro'].append(Fmetrics.classification.multilabel_accuracy(logits, labels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='micro'))
                    metrics['multilabel_accuracy_macro'].append(Fmetrics.classification.multilabel_accuracy(logits, labels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='macro'))
                    metrics['multilabel_f1_micro'].append(Fmetrics.classification.multilabel_f1_score(logits, labels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='micro'))
                    wandb.log({f"acc_micro_step/train/": metrics['multilabel_accuracy_micro'][-1],
                               f"acc_macro_step/train/": metrics['multilabel_accuracy_macro'][-1],
                               f"f1_micro_step/train/": metrics['multilabel_f1_micro'][-1]})
                    self.tensorboard_writer.add_scalar("acc_micro_step/train", metrics['multilabel_accuracy_micro'][-1], train_steps)
                    self.tensorboard_writer.add_scalar("acc_macro_step/train", metrics['multilabel_accuracy_macro'][-1], train_steps)
                    self.tensorboard_writer.add_scalar("f1_micro_step/train", metrics['multilabel_f1_micro'][-1], train_steps)

                    # Log recall@K
                    self.tensorboard_writer.add_scalar("recall/train", retrieval_recall(logits, labels.to(int)), train_steps)
                    self.tensorboard_writer.add_scalar("recall@1/train", retrieval_recall(logits, labels.to(int), top_k=1), train_steps)
                    self.tensorboard_writer.add_scalar("recall@20/train", retrieval_recall(logits, labels.to(int), top_k=5), train_steps)
                    self.tensorboard_writer.add_scalar("recall@100/train", retrieval_recall(logits, labels.to(int), top_k=100), train_steps)

                    # Log AUROC
                    self.tensorboard_writer.add_scalar("MultilabelAUROC_micro/train", multilabel_auroc(logits, labels.to(int), self.args.num_labels, average='micro'), train_steps)
                    self.tensorboard_writer.add_scalar("MultilabelAUROC_macro/train", multilabel_auroc(logits, labels.to(int), self.args.num_labels, average='macro'), train_steps)

                    # Log mAP
                    self.tensorboard_writer.add_scalar("MultilabelAveragePrecision_micro/train", multilabel_average_precision(logits, labels.to(int), self.args.num_labels, average='micro'), train_steps)
                    self.tensorboard_writer.add_scalar("MultilabelAveragePrecision_macro/train", multilabel_average_precision(logits, labels.to(int), self.args.num_labels, average='macro'), train_steps)

                train_steps += 1
                if step >= max_iter:  # Debug purposes
                    break
            wandb.log({"Loss_epoch (batch avg)/train": np.array(running_loss).mean()})
            wandb.log({f"acc_micro_epoch (batch_avg)/train/": np.array(metrics['multilabel_accuracy_micro']).mean(),
                       f"acc_macro_epoch (batch_avg)/train/": np.array(metrics['multilabel_accuracy_macro']).mean(),
                       f"f1_micro_epoch (batch_avg)/train/": np.array(metrics['multilabel_f1_micro']).mean()})
            self.tensorboard_writer.add_scalar("Loss_epoch (batch avg)/train", np.array(running_loss).mean(), epoch_counter)
            self.tensorboard_writer.add_scalar("acc_micro_epoch (batch_avg)/train", np.array(metrics['multilabel_accuracy_micro']).mean(), epoch_counter)
            self.tensorboard_writer.add_scalar("acc_macro_epoch (batch_avg)/train", np.array(metrics['multilabel_accuracy_macro']).mean(), epoch_counter)
            self.tensorboard_writer.add_scalar("f1_micro_epoch (batch_avg)/train", np.array(metrics['multilabel_f1_micro']).mean(), epoch_counter)
            
            # Evaluation
            self.model.eval()
            vmetrics = {'multilabel_accuracy_micro': [],
                        'multilabel_accuracy_macro': [],
                        'multilabel_f1_micro': []}
            print("Evaluating the model...")
            with torch.no_grad():
                running_vloss, vsim_matrices, vtop1s, vtop5s = [], [], [], []
                for vstep, val_dict in enumerate(tqdm(val_loader)):
                    vbatch_inds = val_dict ['indices']
                    val_dict.pop('indices', None)

                    val_dict_items = val_dict.items()  # Useless if keeping all modalities
                    val_dict_items = [(k, v) for k, v in val_dict_items]  # Useless if keeping all modalities
                    wandb.log({"val_steps": val_steps})
                    vloss, vall_logits, vall_features_img, vall_features_gps, vall_images, vidxs, vids  = 0, [], [], [], [], [], []
                    
                    vlabels = val_dict[modalities_to_process[0]][-1]
                    vlogits = self.model(val_dict['species'][0].to(self.args.device),
                                         val_dict['landscape'][0].to(self.args.device),
                                         val_dict['satellite'][0].to(self.args.device),
                                         val_dict[modalities_to_process[0]][1].to(self.args.device))
                    vloss = self.criterion(logits.to(self.args.device), labels.to(self.args.device))
                    vlogits = logits.to('cpu')
                    all_images.extend((train_dict[mod_name][0] for mod_name in modalities_to_process))

                    running_vloss.append(vloss.to('cpu').item())

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
                        # Finding best threshold on the fly
                        ## Method 1: searching through a range of thresholds
                        best_thresh, best_f1 = find_best_threshold(vlabels, vlogits)
                        self.best_f1_thresh = best_thresh
                        ## Method 2: take p_k where k = ground(sum of probas)
                        
                        vmetrics['multilabel_accuracy_micro'].append(Fmetrics.classification.multilabel_accuracy(vlogits, vlabels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='micro'))
                        vmetrics['multilabel_accuracy_macro'].append(Fmetrics.classification.multilabel_accuracy(vlogits, vlabels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='macro'))
                        vmetrics['multilabel_f1_micro'].append(Fmetrics.classification.multilabel_f1_score(vlogits, vlabels, num_labels=self.num_labels, threshold=self.best_f1_thresh, average='micro'))
                    
                        self.tensorboard_writer.add_scalar("acc_micro_step/train", vmetrics['multilabel_accuracy_micro'][-1], val_steps)
                        self.tensorboard_writer.add_scalar("acc_macro_step/train", vmetrics['multilabel_accuracy_macro'][-1], val_steps)
                        self.tensorboard_writer.add_scalar("f1_micro_step/train", vmetrics['multilabel_f1_micro'][-1], val_steps)

                        # Log recall@K
                        self.tensorboard_writer.add_scalar("recall/train", retrieval_recall(vlogits, vlabels.to(int)), val_steps)
                        self.tensorboard_writer.add_scalar("recall@1/train", retrieval_recall(vlogits, vlabels.to(int), top_k=1), val_steps)
                        self.tensorboard_writer.add_scalar("recall@20/train", retrieval_recall(vlogits, vlabels.to(int), top_k=5), val_steps)
                        self.tensorboard_writer.add_scalar("recall@100/train", retrieval_recall(vlogits, vlabels.to(int), top_k=5), val_steps)
                        
                        # Log AUROC
                        self.tensorboard_writer.add_scalar("MultilabelAUROC_micro/train", multilabel_auroc(vlogits, vlabels.to(int), self.args.num_labels, average='micro'), val_steps)
                        self.tensorboard_writer.add_scalar("MultilabelAUROC_macro/train", multilabel_auroc(vlogits, vlabels.to(int), self.args.num_labels, average='macro'), val_steps)
                        
                        # Log mAP
                        self.tensorboard_writer.add_scalar("MultilabelAveragePrecision_micro/train", multilabel_average_precision(vlogits, vlabels.to(int), self.args.num_labels, average='micro'), val_steps)
                        self.tensorboard_writer.add_scalar("MultilabelAveragePrecision_macro/train", multilabel_average_precision(vlogits, vlabels.to(int), self.args.num_labels, average='macro'), val_steps)


                    if vstep % self.args.log_every_n_steps == 0:
                        # Log input batch images
                        log_input_imgs_multimodalities(vall_images, vidxs, vids, vstep, epoch_counter, n_samples=8, n_modalities=3, mode='val', log_images=self.log_images)
                        
                        # Log loss and moments step wise
                        log_loss_scheduler(vloss, self.scheduler, mode='val')
                        self.tensorboard_writer.add_scalar("Loss/val", vloss, val_steps)
                        
                        # Log accuracy step wise                    
                        vmetrics['multilabel_accuracy_micro'].append(Fmetrics.classification.multilabel_accuracy(vlogits, vlabels, num_labels=self.num_labels, threshold=0.1, average='micro'))
                        vmetrics['multilabel_accuracy_macro'].append(Fmetrics.classification.multilabel_accuracy(vlogits, vlabels, num_labels=self.num_labels, threshold=0.1, average='macro'))
                        vmetrics['multilabel_f1_micro'].append(Fmetrics.classification.multilabel_f1_score(vlogits, vlabels, num_labels=self.num_labels, threshold=0.1, average='micro'))
                        wandb.log({f"acc_micro_step/val/": vmetrics['multilabel_accuracy_micro'][-1],
                                   f"acc_macro_step/val/": vmetrics['multilabel_accuracy_macro'][-1],
                                   f"f1_micro_step/val/": vmetrics['multilabel_f1_micro'][-1]})
                        self.tensorboard_writer.add_scalar("acc_micro_step/val", vmetrics['multilabel_accuracy_micro'][-1], val_steps)
                        self.tensorboard_writer.add_scalar("acc_macro_step/val", vmetrics['multilabel_accuracy_macro'][-1], val_steps)
                        self.tensorboard_writer.add_scalar("f1_micro_step/val", vmetrics['multilabel_f1_micro'][-1], val_steps)

                    
                    val_steps += 1
                    if vstep >= max_iter:
                        break
                wandb.log({"Loss_epoch (batch avg)/val": np.array(running_vloss).mean()})
                wandb.log({f"acc_micro_epoch (batch_avg)/val/": np.array(vmetrics['multilabel_accuracy_micro']).mean(),
                           f"acc_macro_epoch (batch_avg)/val/": np.array(vmetrics['multilabel_accuracy_macro']).mean(),
                           f"f1_micro_epoch (batch_avg)/val/": np.array(vmetrics['multilabel_f1_micro']).mean()})
                self.tensorboard_writer.add_scalar("Loss_epoch (batch avg)/val", np.array(running_loss).mean(), epoch_counter)
                self.tensorboard_writer.add_scalar("acc_micro_epoch (batch_avg)/val", np.array(vmetrics['multilabel_accuracy_micro']).mean(), epoch_counter)
                self.tensorboard_writer.add_scalar("acc_macro_epoch (batch_avg)/val", np.array(vmetrics['multilabel_accuracy_macro']).mean(), epoch_counter)
                self.tensorboard_writer.add_scalar("f1_micro_epoch (batch_avg)/val", np.array(vmetrics['multilabel_f1_micro']).mean(), epoch_counter)
            
            self.scheduler.step()
            # Save model checkpoints
            save_checkpoint({
                'epoch': epoch_counter,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=(loss < best_train_loss), dirpath=self.writer.dir)
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.dir}.")
        logging.info("Training has finished.")

    def predict(self, test_dataloader: torch.utils.data.DataLoader):
        """Predict the model using SimCLR.

        Args:
            dataloader (torch.utils.data.DataLoader): pytorch dataloader for validation data
        """
        modalities_name = ['species', 'landscape', 'satellite']
        modalities_to_process = [b for b in modalities_name if b not in self.skip_modalities]
        self.model.eval()
        features_img, features_gps = [], []
        top1s, top5s = [], []
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

                    all_features_img.append(features_img)
                    all_features_gps.append(features_gps)
                    all_logits.append(logits)
                    ids.append(survey_ids)
                    idxs.append(inds)
                    
                # Log accuracy step wise
                for logits, modality_name in zip(all_logits, modalities_name):
                    vtop1, vtop5 = log_acc_topk_step(logits, labels, modality_name, topk=(1, 5), mode='test')
                    if all(topk is not None for topk in [vtop1, vtop5]):
                        top1s.append(vtop1)
                        top5s.append(vtop5)
                        
            # Log similarity matrix epoch wise
            sim_matrix_mean = mean_sim_matrices_over_modalities(sim_matrices)
            log_similarity_matrix_mean(sim_matrix_mean, 0, step, log_images=self.log_images, mode='test')
            for vtop1, vtop5, modality_name in zip(top1s, top5s, modalities_name):
                wandb.log({f"acc_epoch (batch avg)/test/top1_{modality_name}": vtop1,
                           f"acc_epoch (batch avg)/test/top5_{modality_name}": vtop5})
                print(f'Test accuracy for {modality_name} - Top-1: {vtop1:.4f}, Top-5: {vtop5:.4f}')
            wandb.log({"acc_epoch (batch avg)/test/top1": np.array(top1s).mean(),
                        "acc_epoch (batch avg)/test/top5": np.array(top5s).mean()})
            print(f'Test accuracy (mean) - Top-1: {np.array(top1s).mean():.4f}, Top-5: {np.array(top5s).mean():.4f}')
