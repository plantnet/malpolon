import os

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from malpolon.logging import Summary

from pytorch_lightning.callbacks import LearningRateMonitor
from datamodule import MicroGeoLifeCLEF2022DataModule


def Transfer_learning_ia_biodiv(model, cfg, cfg_model, cls_num_list_train, patch_data_ext, logger):
    # chemin du chk
    chk_path ='/home/bbourel/Documents/IA/malpolon/examples_perso/multiband_unimodel/outputs/cnn_multi_band/2023-04-05_18-21-28_367789/checkpoint-epoch=05-step=7764-val_accuracy=0.0959.ckpt'
    
    # vérifier que le chargement des poids fonctionne
    #model_chk = model.load_from_checkpoint(chk_path, model=cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, cls_num_list_train=cls_num_list_train)
    #trainer.test(model_chk, datamodule=datamodule)

    # nouveau nombre d'outputs
    num_outputs_tf =47
       
    # adaptation des paramettres d'optimisation du model pour le transfer learning
    cfg.optimizer.loss.loss_type='PoissonNLLLoss'
    cfg.optimizer.SGD.lr=0.00001
    cfg.optimizer.scheduler.metric_to_track = 'val_loss'
    cfg.optimizer.scheduler.mode = 'min'
    cfg.optimizer.scheduler.factor = 0.1
    cfg.optimizer.scheduler.patience = 0
    cfg.optimizer.scheduler.threshold = 0.001
    cfg.optimizer.scheduler.cooldown = 1
    cfg.optimizer.scheduler.logging_interval = 'epoch'

                
    # chargement des poids et aplication de la modification des parametres d'optimisation du model pour le transfer learning
    model_chk_tf = model.load_from_checkpoint(chk_path, model=cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, cls_num_list_train=cls_num_list_train)
    
    # récupération du nombre de features du model d'origine
    #num_ftrs = model_chk_tf.model.fc.in_features
    num_ftrs = model_chk_tf.model.fc.out_features
    
    # notes :    
    # gelage les couches
    # nombre de couche -> len(list(model_chk_tf.model.named_children()))
    # détail de la couche 0 ->list(model_chk_tf.model.named_children())[0]
    # récuperer le nom de tout les layers
    #    for name, module in model_chk_tf.model.named_modules():
    #    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #        print(name)
    
    for param in model_chk_tf.model.parameters():
        param.requiers_grad = False

    # dégeler le layer 4
    # model_chk_tf.model.layer4.requires_grad_(True)



    '''
    from datamodule import ClassificationSystem
    model_chk_tf = ClassificationSystem(cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, cls_num_list_train=cls_num_list_train)
    num_ftrs = model_chk_tf.model.fc.in_features
    '''

    
    # remplacement de la dernière couche et modification du nombre de sortie
    # model_chk_tf.model.fc = torch.nn.Linear(num_ftrs, num_outputs_tf)
    #model_chk_tf.model.fc = torch.nn.Sequential(model_chk_tf.model.fc,torch.nn.Linear(num_ftrs, num_outputs_tf))
    model_chk_tf.model.fc = torch.nn.Sequential(model_chk_tf.model.fc, torch.nn.Linear(num_ftrs, num_outputs_tf),torch.nn.ReLU())

    # correction du nombre d'outputs pour les métriques
    import torchmetrics
    from custom_metrics import MetricChallangeIABiodiv
    model_chk_tf.metrics = {"metric_ia_biodiv": MetricChallangeIABiodiv().to(device = "cuda")}
    #                            "R2_score" : torchmetrics.R2Score(num_outputs=num_outputs_tf, multioutput='raw_values').to(device = "cuda")}
    
    # adaptation des paramettres pour le chargement du nouveau jeu de donnée
    data_tf = cfg.data.copy()
    data_tf.dataset_path = '/home/bbourel/Data/malpolon_datasets/Galaxy117-Sort_on_data_82'
    data_tf.csv_occurence_path = '/home/bbourel/Data/malpolon_datasets/Galaxy117-Sort_on_data_82/Galaxy117-Sort_on_data_82_n_vec.csv'
    #data_tf.csv_occurence_path = '/home/bbourel/Data/malpolon_datasets/Galaxy117-Sort_on_data_82/Galaxy117-Sort_on_data_82_log(n+1)_vec.csv'
    data_tf.csv_separator = ','
    data_tf.csv_col_occurence_id = 'SurveyID'
    data_tf.csv_col_class_id = 'species_abundance_vec' 
    data_tf.train_batch_size = 32
    data_tf.inference_batch_size = 256
    data_tf.num_workers = 8

    # configuration d'un datamodule pour le nouveau jeu de donnée sur la base de l'adaptation des paramettres pour le chargement du nouveau jeu de donnée 
    datamodule_tf = MicroGeoLifeCLEF2022DataModule(**data_tf,
                                                   patch_data_ext = patch_data_ext,
                                                   patch_data=cfg.patch.patch_data, 
                                                   patch_band_mean = cfg.patch.patch_band_mean,
                                                   patch_band_sd = cfg.patch.patch_band_sd)
            
    # adaptation des paramettres pour les callbacks
    cfg.callbacks.monitor = 'val_metric_ia_biodiv'
    cfg.callbacks.mode = 'min'
    cfg.callbacks.patience = 6

    # configuration des callbacks pour appliquer les adaptation des paramettres pour les callbacks
    callbacks_tf = [
        Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}-{step}-{val_metric_ia_biodiv:.4f}",
            monitor= cfg.callbacks.monitor,                    
            mode=cfg.callbacks.mode,),
        LearningRateMonitor(logging_interval=cfg.optimizer.scheduler.logging_interval), #epoch'),
        EarlyStopping(monitor=cfg.callbacks.monitor, min_delta=0.00, patience=cfg.callbacks.patience, verbose=False, mode=cfg.callbacks.mode),]              
          
    # configuration du trainer pour appliquer la configuration des callbacks
    trainer_tf = pl.Trainer(logger=logger, callbacks=callbacks_tf, **cfg.trainer)
    return model_chk_tf, datamodule_tf, trainer_tf