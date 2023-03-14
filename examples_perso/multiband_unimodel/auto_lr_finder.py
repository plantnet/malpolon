import pytorch_lightning as pl    
from pathlib import Path

import torch
import torchmetrics

import pytorch_lightning as pl
from typing import Mapping, Union

from malpolon.models.standard_prediction_systems import GenericPredictionSystemLRFinder
from malpolon.models.utils import check_model

from transforms import *
from auto_plot import Autoplot
from pytopk import BalNoisedTopK
from pytopk import ImbalNoisedTopK

import os


import os
import shutil
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

class ClassificationSystem(GenericPredictionSystemLRFinder):
    def __init__(
        self,
        model: Union[torch.nn.Module, Mapping],
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        mode: str,
        factor: float,
        patience: int, 
        metric_to_track: str, 
        loss_type: str,
        k: int,
        epsilon: float,
        max_m: float,
        cls_num_list_train: list):

        num_outputs = model.modifiers.change_last_layer.num_outputs
        model = check_model(model)
                 
        if loss_type == 'BalNoisedTopK':
            loss =  BalNoisedTopK(k=k, epsilon=epsilon)
        elif loss_type == 'ImbalNoisedTopK':
            from init_elements import NormedLinear
            model.fc = NormedLinear(model.fc.in_features, model.fc.out_features)
            loss =  ImbalNoisedTopK(k=k, epsilon=epsilon, max_m=max_m, cls_num_list=cls_num_list_train)
        else :
            loss = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,)
        
        scheduler = {'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience),
                     'metric_to_track': metric_to_track}

        metrics = {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=1).to(device = "cuda"),
            "accuracy_macro":  torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=1).to(device = "cuda")}
        if num_outputs > 5 :
            metrics["top_5_accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=5).to(device = "cuda")
            metrics["top_5_accuracy_macro"] = torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=5).to(device = "cuda")
        if num_outputs > 10 :
            metrics["top_10_accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=10).to(device = "cuda")
            metrics["top_10_accuracy_macro"] = torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=10).to(device = "cuda")
        if num_outputs > 20 :            
            metrics["top_20_accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=20).to(device = "cuda")
            metrics["top_20_accuracy_macro"] = torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=20).to(device = "cuda")
        if num_outputs > 30 :            
            metrics["top_30_accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_outputs, top_k=30).to(device = "cuda")
            metrics["top_30_accuracy_macro"] = torchmetrics.Accuracy(task="multiclass",num_classes=num_outputs, average="macro", top_k=30).to(device = "cuda")            

        super().__init__(model, loss, optimizer, scheduler, metrics)



def Auto_lr_find(cfg, cfg_model, datamodule, cls_num_list_train):
    model = ClassificationSystem(cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, cls_num_list_train=cls_num_list_train)

    trainer = pl.Trainer(auto_lr_find=True)
            
    trainer.tune(model, datamodule=datamodule)

    lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule, min_lr=1e-15)

    # Results can be found in
    lr_finder.results

    # Plot with
    fig = lr_finder.plot(suggest=True)
    plt.title('Learning rate suggestion: '+ str(lr_finder.suggestion()))

    if not os.path.exists(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'auto_lr_finder/'):
        os.makedirs(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'auto_lr_finder/')
    
    fig.savefig(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'auto_lr_finder/' + 'auto_lr_finder' + '.png' )
    
    shutil.rmtree(os.getcwd())    